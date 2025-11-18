from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.agents import RefinementFeedback
from app.constants import Agent, LLMModel
from app.orchestrator import ManualOrchestrator, ManualOrchestratorException


def test_add_agent_raises_error_if_api_key_is_not_set(mocker):
    orchestrator = ManualOrchestrator()
    orchestrator.llm_api_key = None

    with pytest.raises(ManualOrchestratorException, match="API key for model 'LLMModel.GPT_4_1' is not set."):
        orchestrator.add_agent(name=Agent.GENERATOR, system_prompt="Test prompt", model=LLMModel.GPT_4_1)


def test_talk_to_raises_error_if_agent_not_found(mocker):
    orchestrator = ManualOrchestrator()
    orchestrator.main_task = "A task"

    with pytest.raises(ManualOrchestratorException, match="Agent non_existent_agent not found"):
        orchestrator.talk_to(agent_name="non_existent_agent", input_text="Hello", thread_id="123")


def test_add_agent_configures_response_format_correctly(mocker):
    mock_session_cls = mocker.patch("app.orchestrator.AgentSession")

    orchestrator = ManualOrchestrator()
    orchestrator.set_llm_api_keys(openai_key="sk-test")

    orchestrator.add_agent(Agent.GENERATOR, "gen prompt", LLMModel.GPT_4_1)
    orchestrator.add_agent(Agent.CRITIC, "critic prompt", LLMModel.GPT_4_1)

    assert mock_session_cls.call_count == 2

    gen_call = mock_session_cls.mock_calls[0]
    assert gen_call.kwargs["response_format"] is None

    critic_call = mock_session_cls.mock_calls[1]
    assert critic_call.kwargs["response_format"] == RefinementFeedback


def test_talk_to_returns_string_for_generator(mocker):
    orchestrator = ManualOrchestrator()
    orchestrator.main_task = "Task"

    mock_session = MagicMock()
    mock_message = AIMessage(content="I am a text response")
    mock_session.agent.invoke.return_value = {"messages": [MagicMock(), mock_message]}

    orchestrator.agents[Agent.GENERATOR] = mock_session

    response = orchestrator.talk_to(Agent.GENERATOR, "Input", "thread-1")

    assert response == "I am a text response"
    assert isinstance(response, str)


def test_talk_to_returns_structured_object_for_critic(mocker):
    orchestrator = ManualOrchestrator()
    orchestrator.main_task = "Task"

    mock_session = MagicMock()

    mock_feedback_object = MagicMock(name="RefinementFeedbackObject")

    mock_response = {
        "messages": [MagicMock(content="Raw JSON string here")],
        "structured_response": mock_feedback_object,
    }

    mock_session.agent.invoke.return_value = mock_response
    orchestrator.agents[Agent.CRITIC] = mock_session

    response = orchestrator.talk_to(Agent.CRITIC, "Input", "thread-2")

    assert response == mock_feedback_object


def test_talk_to_unpacks_structured_context(mocker):
    orchestrator = ManualOrchestrator()
    orchestrator.main_task = "Build a wall"

    mock_session = MagicMock()
    mock_session.agent.invoke.return_value = {"messages": [AIMessage(content="OK")]}
    orchestrator.agents[Agent.GENERATOR] = mock_session

    structured_input = RefinementFeedback(
        critical_flaws=["Too expensive", "Ugly"], suggestions=["Cut costs", "Paint it"], chat_response="Ignored text"
    )

    orchestrator.talk_to(Agent.GENERATOR, structured_input, "thread-3", context_from="critic")

    call_args = mock_session.agent.invoke.call_args
    state_arg = call_args.kwargs["input"]
    last_message_content = state_arg["messages"][-1].content

    assert "CRITIQUE SUMMARY:" in last_message_content
    assert "- Too expensive" in last_message_content
    assert "Ignored text" not in last_message_content
