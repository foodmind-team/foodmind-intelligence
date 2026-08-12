import pytest
from pydantic import SecretStr, ValidationError

from recommendation_agent.config.settings import Settings


def test_secret_is_redacted_from_repr() -> None:
    settings = Settings(internal_service_token=SecretStr("canary-secret-token"))
    assert "canary-secret-token" not in repr(settings)


def test_non_local_rejects_default_or_weak_token() -> None:
    for token in ("local-development-only", "short"):
        try:
            Settings(app_env="production", internal_service_token=SecretStr(token))
        except ValidationError:
            pass
        else:
            raise AssertionError("weak production token was accepted")


def test_wildcard_cors_is_rejected() -> None:
    try:
        Settings(cors_allow_origins=("*",))
    except ValidationError:
        pass
    else:
        raise AssertionError("wildcard CORS was accepted")


def test_shared_deepseek_key_and_service_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "shared-test-key")
    monkeypatch.delenv("RECOMMENDATION_AGENT_LLM_API_KEY", raising=False)
    shared = Settings(app_env="test")
    assert shared.llm_api_key is not None
    assert shared.llm_api_key.get_secret_value() == "shared-test-key"

    monkeypatch.setenv("RECOMMENDATION_AGENT_LLM_API_KEY", "service-test-key")
    overridden = Settings(app_env="test")
    assert overridden.llm_api_key is not None
    assert overridden.llm_api_key.get_secret_value() == "service-test-key"
