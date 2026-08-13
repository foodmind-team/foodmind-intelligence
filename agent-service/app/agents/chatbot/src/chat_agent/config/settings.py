"""Environment-backed Chat Agent settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOCAL_SERVICE_TOKEN = "local-chat-token"  # noqa: S105 - documented local-only sentinel
_LOCAL_BACKEND_TOKEN = "local-development-service-token"  # noqa: S105 - documented local-only sentinel
AGENTS_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    environment: Literal["local", "test", "ci", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    internal_service_token: SecretStr = SecretStr(_LOCAL_SERVICE_TOKEN)
    min_service_token_length: int = Field(default=16, ge=16, le=256)
    llm_enabled: bool = False
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-pro"
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_AGENT_LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0, le=180.0)
    llm_max_retries: int = Field(default=2, ge=0, le=4)
    llm_temperature: float = Field(default=1.3, ge=0.0, le=1.5)
    llm_thinking_enabled: bool = False
    llm_max_output_tokens: int = Field(default=2048, ge=128, le=8192)
    llm_connection_pool_size: int = Field(default=10, ge=1, le=100)
    max_active_requests: int = Field(default=20, ge=1, le=1000)
    max_queued_requests: int = Field(default=100, ge=0, le=10000)
    queue_timeout_seconds: float = Field(default=2.0, gt=0.0, le=30.0)
    backend_base_url: str = "http://127.0.0.1:8080"
    backend_service_token: SecretStr = SecretStr(_LOCAL_BACKEND_TOKEN)
    backend_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    backend_max_response_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)

    model_config = SettingsConfigDict(
        env_prefix="CHAT_AGENT_",
        env_file=AGENTS_ENV_FILE,
        extra="ignore",
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_security(self) -> "Settings":
        parsed = urlsplit(self.llm_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("LLM base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("LLM base URL must not contain credentials, query, or fragment")
        backend = urlsplit(self.backend_base_url)
        if backend.scheme not in {"http", "https"} or not backend.hostname:
            raise ValueError("Backend base URL must be an absolute HTTP(S) URL")
        if backend.username or backend.password or backend.query or backend.fragment:
            raise ValueError("Backend base URL must not contain credentials, query, or fragment")
        llm_key = self.llm_api_key.get_secret_value() if self.llm_api_key is not None else ""
        if self.llm_enabled and not llm_key:
            raise ValueError("an LLM API key is required when LLM use is enabled")
        if self.environment not in {"local", "test", "ci"}:
            service_token = self.internal_service_token.get_secret_value()
            if service_token == _LOCAL_SERVICE_TOKEN or len(service_token) < self.min_service_token_length:
                raise ValueError("a strong internal service token is required")
            backend_token = self.backend_service_token.get_secret_value()
            if backend_token == _LOCAL_BACKEND_TOKEN or len(backend_token) < self.min_service_token_length:
                raise ValueError("a strong Backend tool service token is required")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
