import json

from langchain_core.tools import ToolException, tool
from langchain_tavily import TavilySearch
from pydantic import ValidationError

_tavily_engine = TavilySearch(max_results=3, include_answer=False)


@tool
def safe_web_search(query: str) -> str:
    """
    Useful for searching the internet for current events, facts, or checking specific claims.
    Input should be a search query.
    """
    safe_query = query[:395].strip()

    try:
        raw_response = _tavily_engine.invoke(safe_query)

        results = raw_response.get("results", [])
        answer = raw_response.get("answer")

        clean_output = []

        if answer:
            clean_output.append({"type": "summary", "content": answer})

        for r in results:
            clean_output.append(
                {"title": r.get("title", "No title"), "url": r.get("url", "No URL"), "content": r.get("content", "")}
            )

        return json.dumps(clean_output, ensure_ascii=False)

    except (ValidationError, ToolException) as e:
        return f"Search Error: {str(e)}"
