import pytest
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.prompts import CRITIC_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT


@pytest.fixture(scope="session", autouse=True)
def set_llm_cache_for_session():
    cache_db_file = ".langchain.db"
    set_llm_cache(SQLiteCache(database_path=cache_db_file))

    yield

    set_llm_cache(None)


@pytest.fixture(scope="session")
def critic_chain():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([("system", CRITIC_SYSTEM_PROMPT), ("human", "{input_text}")])
    return prompt_template | llm | StrOutputParser()


@pytest.fixture(scope="session")
def generator_chain():
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([("system", GENERATOR_SYSTEM_PROMPT), ("human", "{input_text}")])
    return prompt_template | llm | StrOutputParser()
