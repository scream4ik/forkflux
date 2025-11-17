from unittest.mock import MagicMock

from app.agents import AgentSession
from app.constants import LLMModel


def test_agent_session_initialization(mocker):
    mock_llm = MagicMock()
    mock_init_chat_model = mocker.patch("app.agents.init_chat_model", return_value=mock_llm)

    mock_sqlite_connection = MagicMock()
    mock_connect = mocker.patch("app.agents.sqlite3.connect", return_value=mock_sqlite_connection)

    mock_saver_instance = MagicMock()
    mock_sqlite_saver = mocker.patch("app.agents.SqliteSaver", return_value=mock_saver_instance)

    mock_compiled_agent = MagicMock()
    mock_create_agent = mocker.patch("app.agents.create_agent", return_value=mock_compiled_agent)

    session = AgentSession(
        api_key="test_api_key", system_prompt="Test Prompt", model=LLMModel.GPT_4_1, temperature=0.5, max_tokens=100
    )

    mock_init_chat_model.assert_called_once_with(
        model=LLMModel.GPT_4_1, temperature=0.5, max_tokens=100, api_key="test_api_key"
    )

    mock_connect.assert_called_once_with(session.settings.CHECKPOINT_STORAGE_PATH, check_same_thread=False)

    mock_sqlite_saver.assert_called_once_with(mock_sqlite_connection)

    mock_create_agent.assert_called_once()
    call_args, call_kwargs = mock_create_agent.call_args
    assert call_args == (mock_llm,)
    assert call_kwargs["system_prompt"] == "Test Prompt"
    assert call_kwargs["checkpointer"] == mock_saver_instance

    assert session.agent == mock_compiled_agent
