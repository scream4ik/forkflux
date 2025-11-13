from typing import TYPE_CHECKING, Optional, Sequence

from langchain.messages import AIMessage, HumanMessage
from langchain_core.exceptions import LangChainException
from openai import AuthenticationError

from .agents import AgentSession
from .exceptions import ManualOrchestratorException
from .state import AgentSessionState

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel

    from .constants import LLMModel


class ManualOrchestrator:
    main_task: str | None = None
    agents: dict[str, AgentSession[AIMessage, Optional["BaseModel"]]] = {}
    last_outputs: dict[str, str] = {}
    llm_api_key: str | None = None

    def set_llm_api_key(self, api_key: str) -> None:
        self.llm_api_key = api_key

    def set_main_task(self, main_task: str) -> None:
        self.main_task = main_task

    def add_agent(self, name: str, system_prompt: str, model: "LLMModel") -> None:
        if self.llm_api_key is None:
            raise ManualOrchestratorException("API key is not set")
        self.agents[name] = AgentSession(api_key=self.llm_api_key, system_prompt=system_prompt, model=model)

    def talk_to(self, agent_name: str, input_text: str, thread_id: str, context_from: str | None = None) -> str:
        if agent_name not in self.agents:
            raise ManualOrchestratorException(f"Agent {agent_name} not found")
        if self.main_task is None:
            raise ManualOrchestratorException("Main task not set")

        talk_to_input = None
        if context_from is not None and context_from in self.last_outputs:
            context_text = self.last_outputs[context_from]

            talk_to_input = f"""
            --- The context from the agent '{context_from}' ---
            {context_text}
            ------------------------------------

            YOUR TASK:
            {input_text}

            Don't forget about the main goal: {self.main_task}
            """

        messages: Sequence[HumanMessage] = [HumanMessage(content=talk_to_input or input_text)]
        config: "RunnableConfig" = {"configurable": {"thread_id": thread_id}}
        try:
            response = self.agents[agent_name].agent.invoke(
                input=AgentSessionState(agent_name=agent_name, messages=list(messages)), config=config  # type: ignore[arg-type]
            )
        except LangChainException:
            raise ManualOrchestratorException(f"Error while talking to agent {agent_name}")
        except AuthenticationError:
            raise ManualOrchestratorException("API key is invalid")

        response_content = response["messages"][-1].content
        self.last_outputs[agent_name] = response_content
        return response_content
