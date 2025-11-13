from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str


@lru_cache()
def get_settings():
    return Settings()  # type: ignore[call-arg]
