from typing import TYPE_CHECKING, Generic

from langchain.agents import create_agent
from langchain.agents.middleware.types import ResponseT
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.typing import ContextT

from .config import get_settings
from .middleware import LoggingMiddleware
from .state import AgentSessionState

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
    from langgraph.graph.state import CompiledStateGraph

    from .constants import LLM_AVAILABLE_MODELS


class AgentSession(Generic[ResponseT, ContextT]):
    agent: "CompiledStateGraph[AgentSessionState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]"

    settings = get_settings()

    def __init__(
        self, system_prompt: str, model: "LLM_AVAILABLE_MODELS", temperature: float = 0.7, max_tokens: int | None = None
    ) -> None:
        llm = init_chat_model(
            model=model, temperature=temperature, max_tokens=max_tokens, api_key=self.settings.OPENAI_API_KEY
        )
        self.agent = create_agent(  # type: ignore[assignment]
            llm,
            system_prompt=system_prompt,
            middleware=[LoggingMiddleware()],
            state_schema=AgentSessionState,
            checkpointer=InMemorySaver(),
        )
