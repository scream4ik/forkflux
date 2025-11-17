from unittest.mock import MagicMock

import pytest

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


def test_talk_to_calls_correct_agent_invoke_method(mocker):
    mock_agent_session = MagicMock()
    mock_agent_session.agent.invoke.return_value = {"messages": [MagicMock(content="Mocked AI Response")]}

    orchestrator = ManualOrchestrator()
    orchestrator.main_task = "A task"
    orchestrator.agents[Agent.GENERATOR] = mock_agent_session

    response = orchestrator.talk_to(agent_name=Agent.GENERATOR, input_text="Test input", thread_id="thread-abc")

    assert response == "Mocked AI Response"

    mock_agent_session.agent.invoke.assert_called_once()

    call_args, call_kwargs = mock_agent_session.agent.invoke.call_args
    assert call_kwargs["config"]["configurable"]["thread_id"] == "thread-abc"
