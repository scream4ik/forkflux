from enum import Enum


class LLMModel(str, Enum):
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5_1 = "gpt-5.1"
    GPT_5_MINI = "gpt-5-mini"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"


LLM_AVAILABLE_MODELS = [m.value for m in LLMModel]


class Agent(str, Enum):
    GENERATOR = "generator"
    CRITIC = "critic"
