from typing import TYPE_CHECKING, Optional, Sequence

from langchain.messages import AIMessage, HumanMessage

from .agents import AgentSession
from .state import AgentSessionState

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel

    from .constants import LLM_AVAILABLE_MODELS


class ManualOrchestrator:
    def __init__(self, main_task: str) -> None:
        self.main_task = main_task
        self.agents: dict[str, AgentSession[AIMessage, Optional["BaseModel"]]] = {}
        self.last_outputs: dict[str, str] = {}

    def add_agent(self, name: str, system_prompt: str, model: "LLM_AVAILABLE_MODELS") -> None:
        self.agents[name] = AgentSession(system_prompt=system_prompt, model=model)

    def talk_to(self, agent_name: str, input_text: str, thread_id: str, context_from: str | None = None) -> str:
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")

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
        response = self.agents[agent_name].agent.invoke(
            input=AgentSessionState(agent_name=agent_name, messages=list(messages)), config=config  # type: ignore[arg-type]
        )

        response_content = response["messages"][-1].content
        self.last_outputs[agent_name] = response_content
        return response_content
