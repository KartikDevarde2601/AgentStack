from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://agentstack:agentstack@localhost:5439/agentstack"
    temporal_address: str = "localhost:7233"
    temporal_namespace: str = "default"
    temporal_task_queue: str = "deep-research-queue"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    cors_origins: str = "http://localhost:5175,http://127.0.0.1:5175"
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def openai_mode(self) -> Literal["live", "demo"]:
        return "live" if self.openai_api_key.strip() else "demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()
