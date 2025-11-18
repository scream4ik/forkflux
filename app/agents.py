import sqlite3
from typing import TYPE_CHECKING, Generic, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.types import ResponseT
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.typing import ContextT
from pydantic import BaseModel, Field

from .config import get_settings
from .middleware import LoggingMiddleware
from .state import AgentSessionState

if TYPE_CHECKING:
    from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from .constants import LLMModel


class RefinementFeedback(BaseModel):
    critical_flaws: list[str] = Field(
        ...,
        description=(
            "A list of specific logical inconsistencies, factual errors, or security risks found in the input. "
            "Focus ONLY on objective failures. Do not include subjective style choices or nitpicks. "
            "If there are no critical errors, return an empty list."
        ),
    )

    suggestions: list[str] = Field(
        ...,
        description=(
            "A list of blunt, actionable steps. "
            "If the idea is viable: list technical/strategic improvements. "
            "If the idea is dead/flawed: The FIRST item MUST be 'KILL THIS IDEA' or 'ABANDON PROJECT'. "
            "Do not offer 'research' or 'minor fixes' for dead ideas."
        ),
    )

    sources: list[str] = Field(
        default_factory=list,
        description=(
            "A list of reliable sources (URLs) that support your critique. "
            "Format them as Markdown links: '[Title](URL)'. "
            "If you didn't use the search tool or found no sources, return an empty list."
        ),
    )

    chat_response: str = Field(
        ...,
        description=(
            "A professional, concise summary of the critique addressed to the human user. " "Do not output raw JSON."
        ),
    )


class AgentSession(Generic[ResponseT, ContextT]):
    agent: "CompiledStateGraph[AgentSessionState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]"
    settings = get_settings()

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model: "LLMModel",
        summary_model: "LLMModel",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        tools: Sequence["BaseTool"] | None = None,
    ) -> None:
        model_provider = "google_genai" if model.startswith("gemini") else "openai"

        llm = init_chat_model(
            model=model, model_provider=model_provider, temperature=temperature, max_tokens=max_tokens, api_key=api_key
        )
        summary_llm = init_chat_model(model=summary_model, model_provider=model_provider, api_key=api_key)
        self.agent = create_agent(  # type: ignore[assignment]
            llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                LoggingMiddleware(),  # type: ignore[list-item]
                SummarizationMiddleware(model=summary_llm, max_tokens_before_summary=4000, messages_to_keep=20),  # type: ignore[list-item]
            ],
            state_schema=AgentSessionState,
            checkpointer=SqliteSaver(sqlite3.connect(self.settings.CHECKPOINT_STORAGE_PATH, check_same_thread=False)),
            response_format=response_format,  # type: ignore[arg-type]
        )
