import os

import pytest
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache


@pytest.fixture(scope="session", autouse=True)
def set_llm_cache_for_session():
    cache_db_file = ".langchain.db"
    set_llm_cache(SQLiteCache(database_path=cache_db_file))
    os.environ["DEEPEVAL_CACHE_METRICS"] = "1"

    yield

    set_llm_cache(None)
    del os.environ["DEEPEVAL_CACHE_METRICS"]
