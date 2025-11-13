from typing import TYPE_CHECKING, Generic
import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware.types import ResponseT
from langchain.chat_models import init_chat_model
from langgraph.typing import ContextT
from langgraph.checkpoint.sqlite import SqliteSaver

from .middleware import LoggingMiddleware
from .state import AgentSessionState
from .config import get_settings

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

    from .constants import LLMModel


class AgentSession(Generic[ResponseT, ContextT]):
    agent: "CompiledStateGraph[AgentSessionState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]"
    settings = get_settings()

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model: "LLMModel",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        llm = init_chat_model(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        self.agent = create_agent(  # type: ignore[assignment]
            llm,
            system_prompt=system_prompt,
            middleware=[LoggingMiddleware()],
            state_schema=AgentSessionState,
            checkpointer=SqliteSaver(sqlite3.connect(self.settings.CHECKPOINT_STORAGE_PATH)),
        )
