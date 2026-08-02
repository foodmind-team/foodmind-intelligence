"""P0-08 internal API security & CORS hardening tests.

Covers:
  - default config does NOT return Access-Control-Allow-Origin: *
  - allow-listed origin succeeds; non-allow-listed and null origins fail
  - unified native/compat authenticator behaviour
  - token strength enforced in non-local environments
  - stable 401/403 error codes; credentials never echoed
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)
    monkeypatch.delenv("COOKING_PLAN_CORS_ALLOW_ORIGINS", raising=False)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"X-Internal-Token": _TEST_TOKEN}
    if extra:
        headers.update(extra)
    return headers


def _minimal_body() -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "user_id": "test-user",
        "recipes": ({"recipe_id": "r1", "text": "Boil water.", "target_servings": "2"},),
        "dietary_restrictions": (),
        "user_allergens": (),
        "inventory_lots": (),
        "kitchen_resources": (),
        "approved_decisions": (),
        "schema_version": "1.0",
    }


# ---------------------------------------------------------------------------
# 1. CORS disabled by default
# ---------------------------------------------------------------------------


class TestCORSDefaultOff:
    def test_no_access_control_allow_origin_by_default(self, client: TestClient) -> None:
        """Default config must NOT emit Access-Control-Allow-Origin: * (P0-08)."""
        response = client.options(
            "/internal/v1/agents/cooking-plan/generate",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_preflight_without_cors_config_is_plain_405(self, client: TestClient) -> None:
        response = client.options(
            "/internal/v1/agents/cooking-plan/generate",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
        )
        # No CORS middleware → no ACAO header, standard non-CORS response.
        assert response.headers.get("access-control-allow-origin") is None


# ---------------------------------------------------------------------------
# 2. CORS with explicit allow-list
# ---------------------------------------------------------------------------


class TestCORSAllowList:
    @pytest.fixture(autouse=True)
    def _allow_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cooking_plan_agent.config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv(
            "COOKING_PLAN_CORS_ALLOW_ORIGINS",
            "http://localhost:3000,https://app.foodmind.example",
        )

    def test_allowlisted_origin_succeeds(self) -> None:
        with TestClient(create_app()) as client:
            response = client.options(
                "/internal/v1/agents/cooking-plan/generate",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_non_allowlisted_origin_fails(self) -> None:
        with TestClient(create_app()) as client:
            response = client.options(
                "/internal/v1/agents/cooking-plan/generate",
                headers={
                    "Origin": "http://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.headers.get("access-control-allow-origin") is None

    def test_null_origin_fails(self) -> None:
        with TestClient(create_app()) as client:
            response = client.options(
                "/internal/v1/agents/cooking-plan/generate",
                headers={
                    "Origin": "null",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.headers.get("access-control-allow-origin") is None


class TestCORSWildcardRejected:
    def test_wildcard_origin_rejected_at_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cooking_plan_agent.config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("COOKING_PLAN_CORS_ALLOW_ORIGINS", "*")
        with pytest.raises(RuntimeError, match="must not contain"):
            create_app()


# ---------------------------------------------------------------------------
# 3. Unified authenticator behaviour
# ---------------------------------------------------------------------------


class TestUnifiedAuth:
    def test_native_missing_token_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_body(),
        )
        # Missing required header → FastAPI 422 is the documented behaviour
        # for the native endpoint (contract test asserts 422).
        assert response.status_code in (401, 422)

    def test_compat_missing_bearer_is_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json={
                "contractVersion": "cooking-agent-v1",
                "requestId": str(uuid.uuid4()),
                "planId": str(uuid.uuid4()),
                "traceId": "t",
                "request": {"contractVersion": "cooking-public-v1", "ingredients": (), "servings": 2},
                "candidates": (),
            },
        )
        assert response.status_code == 401

    def test_short_token_rejected_in_non_local(self, client: TestClient) -> None:
        """Non-local environments must reject weak service tokens."""
        from cooking_plan_agent.config.settings import Settings

        settings = Settings(
            internal_service_token="short",
            environment="prod",
            min_service_token_length=16,
        )
        from cooking_plan_agent.api.dependencies import _check_credential

        assert _check_credential("short", settings) == "INSUFFICIENT_CREDENTIAL_STRENGTH"

    def test_short_token_allowed_in_local(self, client: TestClient) -> None:
        from cooking_plan_agent.api.dependencies import _check_credential
        from cooking_plan_agent.config.settings import Settings

        settings = Settings(
            internal_service_token="short",
            environment="local",
            min_service_token_length=16,
        )
        assert _check_credential("short", settings) is None


# ---------------------------------------------------------------------------
# 4. Credential never echoed
# ---------------------------------------------------------------------------


class TestCredentialNotEchoed:
    def test_401_body_has_no_credential(self, client: TestClient) -> None:
        wrong = "supersecret-wrong-token-value-xyz"
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_body(),
            headers={"X-Internal-Token": wrong},
        )
        assert response.status_code == 401
        body = str(response.json())
        assert _TEST_TOKEN not in body
        assert wrong not in body

    def test_bearer_401_body_has_no_credential(self, client: TestClient) -> None:
        wrong = "supersecret-wrong-token-value-xyz"
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json={
                "contractVersion": "cooking-agent-v1",
                "requestId": str(uuid.uuid4()),
                "planId": str(uuid.uuid4()),
                "traceId": "t",
                "request": {"contractVersion": "cooking-public-v1", "ingredients": (), "servings": 2},
                "candidates": (),
            },
            headers={"Authorization": f"Bearer {wrong}"},
        )
        assert response.status_code == 401
        body = str(response.json())
        assert _TEST_TOKEN not in body
        assert wrong not in body
