"""Strict environment-backed service settings."""

from functools import lru_cache
from ipaddress import ip_address
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _comma_separated(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


CommaSeparatedTuple = Annotated[tuple[str, ...], NoDecode, BeforeValidator(_comma_separated)]
_LOCAL_TOKEN = "local-development-only"  # noqa: S105 - explicit non-production sentinel
_LOCAL_INFERENCE_TOKEN = "local-inference-only"  # noqa: S105 - explicit non-production sentinel


class Settings(BaseSettings):
    """Recommendation Agent settings with a dedicated environment prefix."""

    app_env: Literal["local", "test", "ci", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    internal_service_token: SecretStr = SecretStr(_LOCAL_TOKEN)
    min_service_token_length: int = Field(default=24, ge=16, le=256)
    supported_contract_versions: CommaSeparatedTuple = ("recommendation-agent-v2",)
    max_request_bytes: int = Field(default=262_144, ge=1_024, le=1_048_576)
    max_response_bytes: int = Field(default=65_536, ge=1_024, le=262_144)
    max_candidates: int = Field(default=100, ge=1, le=100)
    max_active_requests: int = Field(default=20, ge=1, le=1_000)
    max_queued_requests: int = Field(default=100, ge=0, le=10_000)
    queue_timeout_seconds: float = Field(default=1.0, gt=0.0, le=30.0)
    shutdown_drain_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
    cors_allow_origins: CommaSeparatedTuple = ()
    inference_base_url: str = "http://127.0.0.1:9000"
    inference_endpoint_path: str = "/internal/v1/recommendations/score"
    inference_service_token: SecretStr = SecretStr(_LOCAL_INFERENCE_TOKEN)
    inference_connect_timeout_ms: int = Field(default=100, ge=1, le=5_000)
    inference_pool_timeout_ms: int = Field(default=50, ge=1, le=5_000)
    inference_total_timeout_ms: int = Field(default=700, ge=1, le=10_000)
    inference_max_response_bytes: int = Field(default=262_144, ge=1_024, le=1_048_576)
    inference_max_connections: int = Field(default=20, ge=1, le=1_000)
    deadline_guard_ms: int = Field(default=50, ge=1, le=5_000)
    min_deadline_budget_ms: int = Field(default=100, ge=1, le=10_000)
    accepted_inference_contract_version: Literal["recommendation-inference-v1"] = "recommendation-inference-v1"
    accepted_feature_schema_version: Literal["recommendation-features-v2"] = "recommendation-features-v2"
    accepted_model_key_version: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    accepted_model_version: Literal["hybrid-ranking-v1"] = "hybrid-ranking-v1"
    accepted_model_package_version: Literal["recommendation-package-v1"] = "recommendation-package-v1"

    model_config = SettingsConfigDict(
        env_prefix="RECOMMENDATION_AGENT_",
        extra="forbid",
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_deployment_security(self) -> "Settings":
        token = self.internal_service_token.get_secret_value()
        if "*" in self.cors_allow_origins:
            raise ValueError("wildcard CORS origins are forbidden")
        if self.app_env not in {"local", "test", "ci"}:
            if token == _LOCAL_TOKEN or len(token) < self.min_service_token_length:
                raise ValueError("a strong explicit internal service token is required")
        if self.supported_contract_versions != ("recommendation-agent-v2",):
            raise ValueError("only recommendation-agent-v2 is supported")
        parsed = urlsplit(self.inference_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("inference base URL must use HTTP(S) and include a host")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("inference base URL must not contain credentials, query, or fragment")
        if not self.inference_endpoint_path.startswith("/") or any(
            item in self.inference_endpoint_path for item in ("..", "?", "#")
        ):
            raise ValueError("inference endpoint path is invalid")
        if self.deadline_guard_ms >= self.min_deadline_budget_ms:
            raise ValueError("deadline guard must be smaller than the minimum budget")
        inference_token = self.inference_service_token.get_secret_value()
        if self.app_env not in {"local", "test", "ci"}:
            if parsed.scheme != "https" or not _private_hostname(parsed.hostname):
                raise ValueError("non-local inference origin must be private HTTPS")
            if inference_token == _LOCAL_INFERENCE_TOKEN or len(inference_token) < self.min_service_token_length:
                raise ValueError("a strong explicit inference service token is required")
        return self


def _private_hostname(hostname: str) -> bool:
    if hostname.endswith((".internal", ".svc", ".local")):
        return True
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return address.is_private or address.is_loopback


@lru_cache
def get_settings() -> Settings:
    """Return the process settings without exposing secrets in repr."""

    return Settings()
