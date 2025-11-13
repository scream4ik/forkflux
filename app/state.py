from langchain.agents.middleware.types import AgentState, ResponseT


class AgentSessionState(AgentState[ResponseT]):
    agent_name: str
