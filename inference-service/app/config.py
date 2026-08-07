from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["local", "test", "ci", "staging", "production"] = "local"
    internal_service_token: SecretStr = SecretStr("local-inference-only")
    min_service_token_length: int = Field(default=24, ge=16, le=256)
    model_package_dir: Path = Path("model-package")
    max_candidates: int = Field(default=100, ge=1, le=100)

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_security(self) -> Settings:
        token = self.internal_service_token.get_secret_value()
        if self.app_env not in {"local", "test", "ci"} and len(token) < self.min_service_token_length:
            raise ValueError("a strong inference service token is required")
        return self
