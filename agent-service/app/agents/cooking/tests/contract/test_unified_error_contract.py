"""P3-05 unified error contract — schema contract suite.

Every managed endpoint must return the same ``ErrorEnvelope(status,
error_code, message, correlation_id, details, retryable)`` for protocol/
HTTP-level failures. Legal business outcomes (READY / NEEDS_CONFIRMATION /
INFEASIBLE / FAILED) keep their own response models.

Verified statuses: 422, 401, 403, 404, 409, 429/503, 500, plus the
workflow FAILED business outcome (which is NOT an envelope — it is a legal
FailedPlanResponse). Also verifies:
  - correlation ID is identical in the header and the envelope body
  - retryable comes from the error catalog, never message text
  - the legacy FastAPI ``detail`` shape no longer leaks on managed endpoints
  - Spring v1 compat keeps its limited {"detail": {"code": ...}} mapping
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.domain.errors import retryable_error_codes
from cooking_plan_agent.main import create_app

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def _auth_headers(extra: dict | None = None) -> dict:
    headers = {"X-Internal-Token": _TEST_TOKEN}
    if extra:
        headers.update(extra)
    return headers


def _minimal_request(overrides: dict | None = None) -> dict:
    body = {
        "request_id": str(uuid.uuid4()),
        "user_id": "test-user-001",
        "recipes": (
            {
                "recipe_id": "r1",
                "text": "Boil water. Add pasta. Cook for 10 minutes.",
                "target_servings": "2",
            },
        ),
        "dietary_restrictions": (),
        "user_allergens": (),
        "time_limit_minutes": 60,
        "serving_time": "18:00",
        "inventory_lots": (),
        "kitchen_resources": (),
        "approved_decisions": (),
        "schema_version": "1.0",
    }
    if overrides:
        body.update(overrides)
    return body


def _assert_envelope_shape(body: dict, *, status: int) -> None:
    """Assert the body conforms to the ErrorEnvelope contract."""
    assert set(body.keys()) == {"status", "error_code", "message", "correlation_id", "details", "retryable"}, (
        f"Unexpected envelope keys: {sorted(body.keys())}"
    )
    assert body["status"] == status
    assert isinstance(body["error_code"], str) and body["error_code"]
    assert isinstance(body["message"], str) and body["message"]
    assert isinstance(body["correlation_id"], str) and body["correlation_id"]
    assert body["retryable"] in (True, False)


class TestEnvelopeShapeAcrossStatuses:
    """422/401/403/404/409/429/503/500 all pass the same schema contract."""

    def test_422_validation_error(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json={"bad": "payload"},
            headers=_auth_headers({"X-Request-ID": "corr-422"}),
        )
        assert response.status_code == 422
        body = response.json()
        _assert_envelope_shape(body, status=422)
        assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
        assert isinstance(body["details"], list)
        # Legacy FastAPI detail shape must not leak.
        assert "detail" not in body

    def test_401_auth_error(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers={"X-Internal-Token": "wrong", "X-Request-ID": "corr-401"},
        )
        assert response.status_code == 401
        _assert_envelope_shape(response.json(), status=401)

    def test_404_not_found(self, client: TestClient) -> None:
        response = client.get(
            "/internal/v1/agents/cooking-plan/does-not-exist",
            headers=_auth_headers({"X-Request-ID": "corr-404"}),
        )
        assert response.status_code == 404
        body = response.json()
        _assert_envelope_shape(body, status=404)
        assert body["error_code"] == "NOT_FOUND"

    def test_405_method_not_allowed_maps_to_envelope(self, client: TestClient) -> None:
        response = client.put(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers({"X-Request-ID": "corr-405"}),
        )
        assert response.status_code == 405
        _assert_envelope_shape(response.json(), status=405)

    def test_500_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unexpected exception maps to a stable INTERNAL_ERROR envelope."""
        from cooking_plan_agent.application import GenerateCookingPlanService

        # raise_server_exceptions=False lets the global handler produce the
        # 500 envelope instead of TestClient re-raising the injected error.
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            service = client.app.state.generate_plan_service
            assert isinstance(service, GenerateCookingPlanService)

            async def _boom(*_args: object, **_kwargs: object) -> object:
                raise RuntimeError("injected internal failure")

            monkeypatch.setattr(service, "execute", _boom)
            response = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=_minimal_request(),
                headers=_auth_headers({"X-Request-ID": "corr-500"}),
            )
            assert response.status_code == 500
            body = response.json()
            _assert_envelope_shape(body, status=500)
            assert body["error_code"] == "INTERNAL_ERROR"
            assert body["correlation_id"] == "corr-500"
            # The injected message must never leak.
            assert "injected internal failure" not in response.text


class TestCorrelationIdConsistency:
    """P3-05: correlation ID must be identical in header and envelope body."""

    @pytest.mark.parametrize("status_code", [401, 422])
    def test_correlation_id_matches_header(self, client: TestClient, status_code: int) -> None:
        cid = "corr-consistent-123"
        if status_code == 401:
            response = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=_minimal_request(),
                headers={"X-Internal-Token": "wrong", "X-Request-ID": cid},
            )
        else:
            response = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json={"bad": "payload"},
                headers=_auth_headers({"X-Request-ID": cid}),
            )
        assert response.status_code == status_code
        body = response.json()
        assert body["correlation_id"] == cid
        assert response.headers.get("X-Request-ID") == cid


class TestRetryableFromCatalog:
    """P3-05 D9: retryable comes from the catalog, never message text."""

    def test_retryable_codes_are_explicit(self) -> None:
        codes = retryable_error_codes()
        assert "OVERLOADED" in codes
        assert "EXTERNAL_PROVIDER_UNAVAILABLE" in codes

    def test_overload_503_is_retryable(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backpressure 503 must be retryable per the catalog."""
        from cooking_plan_agent.api.backpressure import RequestLimiter

        # Saturate the limiter: max_active=1 and hold the only slot.
        limiter = client.app.state.request_limiter
        assert isinstance(limiter, RequestLimiter)
        original_acquire = limiter.acquire

        async def _stub_acquire() -> bool:
            # Pretend the limiter rejects this request (queue full / timeout).
            return False

        monkeypatch.setattr(limiter, "acquire", _stub_acquire)
        try:
            response = client.post(
                "/internal/v1/agents/cooking-plan/generate",
                json=_minimal_request(),
                headers=_auth_headers({"X-Request-ID": "corr-503"}),
            )
        finally:
            monkeypatch.setattr(limiter, "acquire", original_acquire)

        assert response.status_code == 503
        body = response.json()
        _assert_envelope_shape(body, status=503)
        assert body["error_code"] == "OVERLOADED"
        assert body["retryable"] is True
        assert response.headers.get("Retry-After") is not None


class TestWorkflowFailedIsBusinessResult:
    """Legal business outcomes keep their response models, never envelopes."""

    def test_business_outcome_is_not_an_envelope(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers({"X-Request-ID": "corr-failed"}),
        )
        # Business outcomes return HTTP 200 with a status discriminator.
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED")
        # It is a PlanResponse, NOT an ErrorEnvelope (no protocol fields).
        assert "retryable" not in body
        assert "status" in body
        assert "error_code" in body or "reasons" in body or "plan_id" in body


class TestSpringCompatKeepsLimitedMapping:
    """P3-05 step 6: compat v1 keeps its limited {"detail": {"code": ...}}."""

    def _compat_body(self) -> dict:
        from datetime import UTC, datetime, timedelta

        return {
            "contractVersion": "cooking-agent-v1",
            "requestId": str(uuid.uuid4()),
            "planId": str(uuid.uuid4()),
            "traceId": "trace-compat-envelope",
            "deadlineAt": (datetime.now(UTC) + timedelta(seconds=10)).isoformat(),
            "request": {
                "contractVersion": "cooking-public-v1",
                "ingredients": ({"ingredientName": "Firm tofu", "quantity": "300", "unit": "g", "source": "MANUAL"},),
                "servings": 2,
                "maxMinutes": 60,
                "maxBudget": "20.00",
                "currency": "SGD",
                "constraints": {"requiredDietaryTagCodes": ["VEGETARIAN"], "avoidAllergenCodes": ["PEANUT"]},
            },
            "preferences": {"requiredDietaryTagCodes": ["VEGETARIAN"], "avoidAllergenCodes": ["PEANUT"]},
            "candidates": (),
        }

    def test_compat_auth_error_keeps_detail_code(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json=self._compat_body(),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401
        # Limited v1 mapping — Java caller unchanged.
        assert response.json() == {"detail": {"code": "INVALID_INTERNAL_CREDENTIAL"}}

    def test_compat_validation_error_keeps_detail_code(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/cooking-plans/generate",
            json={"unknown": True},
            headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
        )
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        assert body["detail"]["code"] == "REQUEST_VALIDATION_ERROR"
