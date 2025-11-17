from typing import TYPE_CHECKING, Optional, Sequence

from langchain.messages import AIMessage, HumanMessage
from langchain_core.exceptions import LangChainException
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from openai import AuthenticationError

from .agents import AgentSession
from .constants import LLMModel
from .exceptions import ManualOrchestratorException
from .prompts import CONTEXT_WRAPPER_PROMPT
from .state import AgentSessionState

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel


class ManualOrchestrator:
    main_task: str | None = None
    agents: dict[str, AgentSession[AIMessage, Optional["BaseModel"]]] = {}
    openai_api_key: str | None = None
    google_api_key: str | None = None

    def set_llm_api_keys(self, openai_key: str | None = None, google_key: str | None = None) -> None:
        self.openai_api_key = openai_key
        self.google_api_key = google_key

    def set_main_task(self, main_task: str) -> None:
        self.main_task = main_task

    def add_agent(self, name: str, system_prompt: str, model: LLMModel) -> None:
        is_openai_model = model.startswith("gpt")
        key_to_use = self.openai_api_key if is_openai_model else self.google_api_key

        if not key_to_use:
            raise ManualOrchestratorException(f"API key for model '{model}' is not set.")

        summary_model_for_agent = LLMModel.GEMINI_2_5_FLASH if not is_openai_model else LLMModel.GPT_4O_MINI

        self.agents[name] = AgentSession(
            api_key=key_to_use, system_prompt=system_prompt, model=model, summary_model=summary_model_for_agent
        )

    def talk_to(self, agent_name: str, input_text: str, thread_id: str, context_from: str | None = None) -> str:
        if agent_name not in self.agents:
            raise ManualOrchestratorException(f"Agent {agent_name} not found")
        if self.main_task is None:
            raise ManualOrchestratorException("Main task not set")

        talk_to_input = None
        if context_from is not None:
            talk_to_input = CONTEXT_WRAPPER_PROMPT.format(main_task=self.main_task, context_text=input_text)

        messages: Sequence[HumanMessage] = [HumanMessage(content=talk_to_input or input_text)]
        config: "RunnableConfig" = {"configurable": {"thread_id": thread_id}}
        try:
            response = self.agents[agent_name].agent.invoke(
                input=AgentSessionState(agent_name=agent_name, messages=list(messages)), config=config  # type: ignore[arg-type]
            )
        except LangChainException:
            raise ManualOrchestratorException(f"Error while talking to agent {agent_name}")
        except (AuthenticationError, ChatGoogleGenerativeAIError):
            raise ManualOrchestratorException("API key is invalid")

        response_content = response["messages"][-1].content
        return response_content
