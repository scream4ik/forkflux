from unittest.mock import MagicMock, call

from app.agents import AgentSession
from app.constants import LLMModel


def test_agent_session_initialization(mocker):
    mock_llm = MagicMock()
    mock_summary_llm = MagicMock()
    mock_init_chat_model = mocker.patch("app.agents.init_chat_model", side_effect=[mock_llm, mock_summary_llm])

    mock_logging_middleware_instance = MagicMock()
    mock_logging_middleware = mocker.patch(
        "app.agents.LoggingMiddleware", return_value=mock_logging_middleware_instance
    )

    mock_summarization_middleware_instance = MagicMock()
    mock_summarization_middleware = mocker.patch(
        "app.agents.SummarizationMiddleware", return_value=mock_summarization_middleware_instance
    )

    mock_sqlite_connection = MagicMock()
    mock_connect = mocker.patch("app.agents.sqlite3.connect", return_value=mock_sqlite_connection)
    mock_saver_instance = MagicMock()
    mock_sqlite_saver = mocker.patch("app.agents.SqliteSaver", return_value=mock_saver_instance)

    mock_compiled_agent = MagicMock()
    mock_create_agent = mocker.patch("app.agents.create_agent", return_value=mock_compiled_agent)

    session = AgentSession(
        api_key="test_api_key",
        system_prompt="Test Prompt",
        model=LLMModel.GPT_4_1,
        summary_model=LLMModel.GPT_4O_MINI,
        temperature=0.5,
        max_tokens=100,
    )

    assert mock_init_chat_model.call_count == 2
    expected_calls = [
        call(model=LLMModel.GPT_4_1, model_provider="openai", temperature=0.5, max_tokens=100, api_key="test_api_key"),
        call(model=LLMModel.GPT_4O_MINI, model_provider="openai", api_key="test_api_key"),
    ]
    mock_init_chat_model.assert_has_calls(expected_calls, any_order=False)

    mock_logging_middleware.assert_called_once()
    mock_summarization_middleware.assert_called_once_with(
        model=mock_summary_llm, max_tokens_before_summary=4000, messages_to_keep=20
    )

    mock_connect.assert_called_once_with(session.settings.CHECKPOINT_STORAGE_PATH, check_same_thread=False)
    mock_sqlite_saver.assert_called_once_with(mock_sqlite_connection)

    mock_create_agent.assert_called_once()
    call_args, call_kwargs = mock_create_agent.call_args
    assert call_args == (mock_llm,)
    assert call_kwargs["system_prompt"] == "Test Prompt"
    assert call_kwargs["checkpointer"] == mock_saver_instance
    assert call_kwargs["middleware"] == [mock_logging_middleware_instance, mock_summarization_middleware_instance]

    assert session.agent == mock_compiled_agent
