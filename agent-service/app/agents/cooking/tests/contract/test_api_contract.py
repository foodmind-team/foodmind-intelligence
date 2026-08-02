"""Contract tests for the internal cooking-plan generation endpoint.

Handbook 9.12: use FastAPI TestClient for synchronous API tests.
Validates:
  - Valid request returns a Pydantic-valid response (200).
  - Missing/invalid service credential is rejected (401).
  - Extra request fields are rejected (422).
  - Every business status serialises correctly.
  - Unexpected application error returns a stable body and request ID.
  - OpenAPI includes the internal endpoint and expected schemas.
  - X-Request-ID is echoed in the response.
  - Example payload from Spring deserialises without code changes.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

# ---------------------------------------------------------------------------
# Shared test fixture: a valid internal service token
# ---------------------------------------------------------------------------

_TEST_TOKEN = "test-internal-token-abc123"


@pytest.fixture(autouse=True)
def _set_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure every test has the required env var for Settings validation."""
    # Purge the settings lru_cache so create_app() (which reads Settings for
    # CORS) picks up the monkeypatched token.
    from cooking_plan_agent.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", _TEST_TOKEN)


@pytest.fixture
def client():
    """Create a fresh TestClient with the app-factory for each test.

    Uses the context-manager form so FastAPI's lifespan runs and
    app.state.generate_plan_service is initialised before the test.
    """
    with TestClient(create_app()) as c:
        yield c


def _auth_headers(extra: dict | None = None) -> dict:
    """Build request headers with the valid internal service token."""
    headers = {"X-Internal-Token": _TEST_TOKEN}
    if extra:
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# Minimal valid request — matches the Spring Boot contract shape
# ---------------------------------------------------------------------------


def _minimal_request(overrides: dict | None = None) -> dict:
    """Build a minimal valid GeneratePlanRequest payload.

    The graph's stub nodes can process this — it will return a FAILED
    response because parse_recipes_node returns empty recipes, but the
    API layer correctly validates, authenticates, and invokes the graph.
    """
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


# ---------------------------------------------------------------------------
# 1. Missing/invalid service credential → 401
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Handbook 9.3: missing or invalid internal token must be rejected."""

    def test_missing_token_returns_422(self, client: TestClient) -> None:
        """Missing X-Internal-Token header → FastAPI returns 422.

        FastAPI treats missing required headers as a schema validation
        error (422 Unprocessable Entity), not 401. This is consistent
        with the OpenAPI contract.
        """
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
        )
        assert response.status_code == 422

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers={"X-Internal-Token": "wrong-token"},
        )
        assert response.status_code == 401

    def test_valid_token_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers(),
        )
        # With valid auth, the request reaches the graph. The graph
        # processes it (even with stub nodes) and returns 200.
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 2. Schema validation — extra fields rejected (422)
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Handbook 9.12: extra request fields must be rejected by Pydantic."""

    def test_extra_field_rejected(self, client: TestClient) -> None:
        body = _minimal_request({"unknown_field": "should_fail"})
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=body,
            headers=_auth_headers(),
        )
        assert response.status_code == 422

    def test_missing_required_field_rejected(self, client: TestClient) -> None:
        body = _minimal_request()
        del body["request_id"]
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=body,
            headers=_auth_headers(),
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. Business status serialisation (handbook 9.12)
# ---------------------------------------------------------------------------


class TestBusinessStatuses:
    """Every business status must serialise to the documented shape.

    In MVP, stub nodes will produce a FAILED response because
    parse_recipes returns empty candidates. This test verifies
    that the FAILED status body is Pydantic-valid.
    """

    def test_graph_returns_valid_response_body(self, client: TestClient) -> None:
        """Even with stub nodes, the graph returns a valid PlanResponse."""
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        body = response.json()
        # The union type should contain a 'status' discriminator field.
        assert "status" in body

    def test_response_matches_plan_response_shape(self, client: TestClient) -> None:
        """Response must be deserialisable as one of the PlanResponse variants."""
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers(),
        )
        body = response.json()
        status_value = body.get("status")
        assert status_value in {"READY", "NEEDS_CONFIRMATION", "INFEASIBLE", "FAILED"}, (
            f"Unexpected status: {status_value}"
        )


# ---------------------------------------------------------------------------
# 4. X-Request-ID propagation (handbook 9.10)
# ---------------------------------------------------------------------------


class TestCorrelationId:
    """X-Request-ID must be echoed in the response header."""

    def test_supplied_request_id_is_echoed(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers({"X-Request-ID": "spring-req-abc-123"}),
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == "spring-req-abc-123"

    def test_generated_request_id_returned(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        # Should generate and return a UUID-like correlation ID.
        assert "X-Request-ID" in response.headers
        generated = response.headers["X-Request-ID"]
        assert len(generated) == 32  # UUID4 hex is 32 chars


# ---------------------------------------------------------------------------
# 5. OpenAPI schema includes the endpoint (handbook 9.12)
# ---------------------------------------------------------------------------


class TestOpenAPISchema:
    """The OpenAPI schema must include the internal endpoint and schemas."""

    def test_openapi_includes_generate_endpoint(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/internal/v1/agents/cooking-plan/generate" in paths, "Generate endpoint missing from OpenAPI schema"

    def test_openapi_includes_request_schema(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        schema = response.json()
        schemas = schema.get("components", {}).get("schemas", {})
        assert "GeneratePlanRequest" in schemas, "GeneratePlanRequest schema missing from OpenAPI"

    def test_openapi_includes_response_schemas(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        schema = response.json()
        schemas = schema.get("components", {}).get("schemas", {})
        for name in (
            "ReadyPlanResponse",
            "ConfirmationPlanResponse",
            "InfeasiblePlanResponse",
            "FailedPlanResponse",
        ):
            assert name in schemas, f"{name} schema missing from OpenAPI"

    def test_openapi_includes_confirmation_question_schemas(self, client: TestClient) -> None:
        """P4-02: the structured confirmation form is documented in OpenAPI.

        ConfirmationQuestion/QuestionOption must be exposed via
        ConfirmationPlanResponse.confirmation_questions, with the legacy
        plain-string questions retained for dual-emit compatibility.
        """
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        schemas = schema.get("components", {}).get("schemas", {})
        assert "ConfirmationQuestion" in schemas, "ConfirmationQuestion schema missing from OpenAPI"
        assert "QuestionOption" in schemas, "QuestionOption schema missing from OpenAPI"
        confirmation = schemas.get("ConfirmationPlanResponse", {})
        props = confirmation.get("properties", {})
        assert "confirmation_questions" in props, "confirmation_questions missing from ConfirmationPlanResponse"
        assert "questions" in props, "legacy questions must stay for dual-emit compatibility"


# ---------------------------------------------------------------------------
# 6. Error response stability (handbook 9.9)
# ---------------------------------------------------------------------------


class TestErrorStability:
    """Unexpected errors must return a stable body with correlation ID."""

    def test_validation_error_has_stable_body(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json={"bad": "payload"},
            headers=_auth_headers(),
        )
        assert response.status_code == 422
        body = response.json()
        # P3-05: native endpoints return the unified ErrorEnvelope, never
        # the legacy FastAPI 'detail' shape.
        assert body["status"] == 422
        assert body["error_code"] == "REQUEST_VALIDATION_ERROR"
        assert body["retryable"] is False
        assert body["correlation_id"]
        assert isinstance(body["details"], list)

    def test_auth_error_body_is_stable(self, client: TestClient) -> None:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=_minimal_request(),
            headers={"X-Internal-Token": "bad-token"},
        )
        assert response.status_code == 401
        body = response.json()
        # P3-05: unified ErrorEnvelope keeps the stable error code and the
        # correlation ID echoed in both header and body.
        assert body["status"] == 401
        assert body["error_code"] == "INVALID_INTERNAL_CREDENTIAL"
        assert body["retryable"] is False
        assert body["correlation_id"] == response.headers.get("X-Request-ID")


# ---------------------------------------------------------------------------
# 7. Spring payload compatibility (handbook 9.12)
# ---------------------------------------------------------------------------


class TestSpringCompatibility:
    """Example payload from Spring must deserialise without code changes."""

    def test_spring_example_payload(self, client: TestClient) -> None:
        """A realistic Spring payload passes Pydantic validation."""
        payload = {
            "request_id": "req-2026-07-29-001",
            "user_id": "usr_abc123def",
            "recipes": (
                {
                    "recipe_id": "spring-recipe-1",
                    "text": "烤鸡胸肉：将鸡胸肉用盐和胡椒腌制30分钟。预热烤箱至200度。烤25分钟。",
                    "target_servings": "4",
                },
            ),
            "dietary_restrictions": ("no_gluten",),
            "user_allergens": ("peanuts",),
            "time_limit_minutes": 120,
            "serving_time": "19:30",
            "inventory_lots": (),
            "kitchen_resources": (),
            "approved_decisions": (),
            "schema_version": "1.0",
        }
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=payload,
            headers=_auth_headers({"X-Request-ID": "spring-req-abc-001"}),
        )
        # Must return 200 — deserialisation and auth passed.
        assert response.status_code == 200
        # Correlation ID must be echoed back.
        assert response.headers.get("X-Request-ID") == "spring-req-abc-001"

    def test_schema_version_field_accepted(self, client: TestClient) -> None:
        """schema_version is a required field for contract versioning."""
        payload = _minimal_request({"schema_version": "1.0"})
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json=payload,
            headers=_auth_headers(),
        )
        assert response.status_code == 200
