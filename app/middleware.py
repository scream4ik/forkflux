import logging
import sys
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ResponseT
from langchain.messages import AIMessage

from .state import AgentSessionState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


class LoggingMiddleware(AgentMiddleware[AgentSessionState[ResponseT], Any]):
    def before_model(self, state: AgentSessionState[ResponseT], runtime: Any) -> None:
        logger.info(f"Agent {state['agent_name']} is about to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentSessionState[ResponseT], runtime: Any) -> None:
        if isinstance(state["messages"][-1], AIMessage):
            logger.info(f"Agent {state['agent_name']} usage data: {state['messages'][-1].usage_metadata}")
        return None
