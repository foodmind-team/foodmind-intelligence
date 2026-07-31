"""Unit tests for api/dependencies.py — internal auth and correlation ID.

Tests cover:
  - _validate_correlation_id: boundary conditions and security rejections
  - require_internal_service: token match / mismatch / settings injection
  - extract_correlation_id: ID generation and request.state propagation
  - HTTP-level integration: dependencies wired into the real FastAPI app
"""

import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from cooking_plan_agent.api.dependencies import (
    _validate_correlation_id,
    extract_correlation_id,
    require_internal_service,
)
from cooking_plan_agent.config.settings import Settings, get_settings
from cooking_plan_agent.main import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-internal-secret-xyz"


def _make_request(x_request_id: str | None = None) -> Request:
    """Build a minimal Starlette Request scope for extract_correlation_id tests."""
    headers: list[tuple[bytes, bytes]] = []
    if x_request_id is not None:
        headers.append((b"x-request-id", x_request_id.encode()))
    scope: dict[str, str | list[tuple[bytes, bytes]]] = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Purge the @lru_cache on get_settings and inject the test token."""
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)


# ---------------------------------------------------------------------------
# _validate_correlation_id — pure-function unit tests (synchronous)
# ---------------------------------------------------------------------------


class TestValidateCorrelationId:
    """Security boundary tests for the private correlation ID validator."""

    # -- Happy path ----------------------------------------------------------

    def test_valid_uuid4_hex_passes(self) -> None:
        raw = "a1b2c3d4e5f6789012345678abcdef01"
        assert _validate_correlation_id(raw) == raw

    def test_valid_spring_style_passes(self) -> None:
        raw = "spring-req-20260729-abc-12345"
        assert _validate_correlation_id(raw) == raw

    def test_single_char_passes(self) -> None:
        assert _validate_correlation_id("x") == "x"

    def test_exactly_max_length_passes(self) -> None:
        raw = "a" * 128
        assert _validate_correlation_id(raw) == raw

    # -- Rejections ----------------------------------------------------------

    def test_empty_string_rejected(self) -> None:
        assert _validate_correlation_id("") is None

    def test_exceeds_max_length_rejected(self) -> None:
        assert _validate_correlation_id("a" * 129) is None

    # -- Log-injection defence -----------------------------------------------

    def test_newline_injected_rejected(self) -> None:
        assert _validate_correlation_id("abc\n123") is None

    def test_carriage_return_rejected(self) -> None:
        assert _validate_correlation_id("abc\r123") is None

    def test_tab_character_rejected(self) -> None:
        assert _validate_correlation_id("abc\tdef") is None

    def test_space_rejected(self) -> None:
        assert _validate_correlation_id("abc 123") is None

    # -- Other attack vectors ------------------------------------------------

    def test_path_traversal_rejected(self) -> None:
        assert _validate_correlation_id("../../../etc/passwd") is None

    def test_sql_metacharacters_rejected(self) -> None:
        assert _validate_correlation_id("1' OR '1'='1") is None

    def test_angle_brackets_rejected(self) -> None:
        assert _validate_correlation_id("<script>alert(1)</script>") is None

    def test_unicode_chinese_rejected(self) -> None:
        assert _validate_correlation_id("请求ID") is None

    def test_null_byte_rejected(self) -> None:
        assert _validate_correlation_id("abc\x00def") is None


# ---------------------------------------------------------------------------
# require_internal_service — async DI function unit tests
# ---------------------------------------------------------------------------


class TestRequireInternalService:
    """Authentication dependency: token validation and error behaviour."""

    @pytest.mark.asyncio
    async def test_matching_token_returns_none(self) -> None:
        settings = get_settings()
        result = await require_internal_service(_TEST_TOKEN, settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_mismatched_token_raises_401(self) -> None:
        settings = get_settings()
        with pytest.raises(HTTPException) as exc_info:
            await require_internal_service("wrong-token-zzz", settings)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == {"code": "INVALID_INTERNAL_CREDENTIAL"}

    @pytest.mark.asyncio
    async def test_empty_token_raises_401(self) -> None:
        settings = get_settings()
        with pytest.raises(HTTPException) as exc_info:
            await require_internal_service("", settings)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_token_comparison_is_case_sensitive(self) -> None:
        settings = get_settings()
        # Skip if the test token is all-uppercase — case comparison meaningless.
        if _TEST_TOKEN.upper() == _TEST_TOKEN:
            pytest.skip("Test token is all-uppercase; case test not meaningful")
        with pytest.raises(HTTPException):
            await require_internal_service(_TEST_TOKEN.upper(), settings)

    @pytest.mark.asyncio
    async def test_error_body_excludes_token_value(self) -> None:
        """The 401 body must never leak the expected or supplied token value."""
        settings = get_settings()
        with pytest.raises(HTTPException) as exc_info:
            await require_internal_service("some-bad-token", settings)
        detail_str = str(exc_info.value.detail)
        assert _TEST_TOKEN not in detail_str
        assert "some-bad-token" not in detail_str

    @pytest.mark.asyncio
    async def test_uses_injected_settings_not_global(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """require_internal_service compares against the injected Settings instance."""
        monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "global-token")
        custom = Settings(internal_service_token="custom-token")
        result = await require_internal_service("custom-token", custom)
        assert result is None


# ---------------------------------------------------------------------------
# extract_correlation_id — async DI function unit tests
# ---------------------------------------------------------------------------


class TestExtractCorrelationId:
    """Correlation ID extraction: generation, state propagation, security."""

    @pytest.mark.asyncio
    async def test_supplied_valid_id_is_returned(self) -> None:
        request = _make_request("spring-req-abc-001")
        result = await extract_correlation_id(request, "spring-req-abc-001")
        assert result == "spring-req-abc-001"

    @pytest.mark.asyncio
    async def test_supplied_id_is_written_to_state(self) -> None:
        request = _make_request("trace-me-999")
        await extract_correlation_id(request, "trace-me-999")
        assert request.state.correlation_id == "trace-me-999"

    @pytest.mark.asyncio
    async def test_missing_header_generates_uuid4_hex(self) -> None:
        request = _make_request(None)
        result = await extract_correlation_id(request)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    @pytest.mark.asyncio
    async def test_empty_header_generates_new_id(self) -> None:
        request = _make_request("")
        result = await extract_correlation_id(request, "")
        assert len(result) == 32

    @pytest.mark.asyncio
    async def test_malicious_header_generates_safe_id(self) -> None:
        request = _make_request("abc\ninjected")
        result = await extract_correlation_id(request, "abc\ninjected")
        assert result != "abc\ninjected"
        assert "\n" not in result
        assert len(result) == 32

    @pytest.mark.asyncio
    async def test_generated_id_is_written_to_state(self) -> None:
        request = _make_request(None)
        result = await extract_correlation_id(request)
        assert request.state.correlation_id == result
        assert len(request.state.correlation_id) == 32

    @pytest.mark.asyncio
    async def test_two_calls_produce_different_ids(self) -> None:
        r1 = await extract_correlation_id(_make_request(None))
        r2 = await extract_correlation_id(_make_request(None))
        assert r1 != r2

    @pytest.mark.asyncio
    async def test_return_value_and_state_are_consistent(self) -> None:
        request = _make_request("consistent-test-42")
        result = await extract_correlation_id(request, "consistent-test-42")
        assert result == "consistent-test-42"
        assert request.state.correlation_id == "consistent-test-42"


# ---------------------------------------------------------------------------
# HTTP-level integration tests — dependencies wired into the real FastAPI app
# ---------------------------------------------------------------------------


class TestDependenciesViaHttp:
    """End-to-end: dependencies exercised through the real FastAPI stack.

    Uses the TestClient context-manager form so FastAPI's lifespan runs
    and app.state.generate_plan_service is initialised before the test.
    """

    @staticmethod
    def _auth_headers(extra: dict | None = None) -> dict:
        h = {"X-Internal-Token": _TEST_TOKEN}
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def _minimal_request() -> dict:
        return {
            "request_id": str(uuid.uuid4()),
            "user_id": "test-user-001",
            "recipes": [
                {
                    "recipe_id": "r1",
                    "text": "Boil water. Add pasta. Cook for 10 minutes.",
                    "target_servings": "2",
                },
            ],
            "dietary_restrictions": [],
            "user_allergens": [],
            "time_limit_minutes": 60,
            "serving_time": "18:00",
            "inventory_lots": [],
            "kitchen_resources": [],
            "approved_decisions": [],
            "schema_version": "1.0",
        }

    def test_missing_auth_header_returns_422(self) -> None:
        with TestClient(create_app()) as client:
            resp = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=self._minimal_request(),
            )
            assert resp.status_code == 422

    def test_invalid_token_returns_stable_error(self) -> None:
        with TestClient(create_app()) as client:
            resp = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=self._minimal_request(),
                headers={"X-Internal-Token": "bad-token"},
            )
            assert resp.status_code == 401
            assert resp.json() == {"detail": {"code": "INVALID_INTERNAL_CREDENTIAL"}}

    def test_request_id_echoed_in_response_header(self) -> None:
        with TestClient(create_app()) as client:
            resp = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=self._minimal_request(),
                headers=self._auth_headers({"X-Request-ID": "e2e-test-req-42"}),
            )
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-ID") == "e2e-test-req-42"

    def test_request_id_generated_when_missing(self) -> None:
        with TestClient(create_app()) as client:
            resp = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=self._minimal_request(),
                headers=self._auth_headers(),
            )
            assert resp.status_code == 200
            assert "X-Request-ID" in resp.headers
            assert len(resp.headers["X-Request-ID"]) == 32

    def test_malicious_request_id_is_replaced(self) -> None:
        with TestClient(create_app()) as client:
            resp = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=self._minimal_request(),
                headers=self._auth_headers({"X-Request-ID": "abc\ndef\nghi"}),
            )
            echoed = resp.headers.get("X-Request-ID", "")
            assert "\n" not in echoed
            assert len(echoed) == 32  # freshly generated UUID, not the malicious input
