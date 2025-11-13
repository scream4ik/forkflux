from enum import Enum


class LLMModel(str, Enum):
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"


LLM_AVAILABLE_MODELS = [m.value for m in LLMModel]


class Agent(str, Enum):
    GENERATOR = "generator"
    CRITIC = "critic"
