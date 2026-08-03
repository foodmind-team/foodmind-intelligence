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
