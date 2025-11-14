from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, env_file_encoding="utf-8", extra="ignore")

    CHECKPOINT_STORAGE_PATH: str = ".data/checkpoints/checkpointer.db"

    @field_validator("CHECKPOINT_STORAGE_PATH")
    @classmethod
    def _create_checkpoint_directory(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v


@lru_cache()
def get_settings():
    return Settings()
