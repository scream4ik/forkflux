import json

import pytest

from app.agent_tools import safe_web_search


def test_safe_web_search_returns_valid_json_string(mocker):
    mock_tavily_class = mocker.patch("app.agent_tools._tavily_engine")

    mock_response = {
        "results": [{"title": "Test Title", "url": "https://test.com", "content": "Test Content", "score": 0.9}],
        "answer": "Direct answer",
    }
    mock_tavily_class.invoke.return_value = mock_response

    result = safe_web_search.invoke("test query")

    assert isinstance(result, str)

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        pytest.fail("Tool returned invalid JSON string")

    assert isinstance(parsed, list)
    assert len(parsed) == 2  # 1 answer + 1 result

    assert "score" not in parsed[1]
    assert parsed[1]["url"] == "https://test.com"
