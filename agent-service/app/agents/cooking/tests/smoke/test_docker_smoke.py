"""Docker health smoke test — verify the container image starts and responds.

Validates app factory, health endpoints, internal auth, and OpenAPI schema
without requiring Docker.  All tests that require lifespan use the TestClient
context manager and provide required env vars.

Usage:
    pytest tests/smoke/test_docker_smoke.py -v
"""

from __future__ import annotations

import os
import uuid

from fastapi.testclient import TestClient

# Internal service token required by the lifespan's Settings validation.
# Tests use a dummy token — never commit real credentials.
_TEST_TOKEN = "test_smoke_token"


def _set_env() -> None:
    """Set required environment variables for the smoke test suite."""
    os.environ["COOKING_PLAN_INTERNAL_SERVICE_TOKEN"] = _TEST_TOKEN
    # Clear lru_cache so get_settings() reads the updated env var
    from cooking_plan_agent.config.settings import get_settings
    get_settings.cache_clear()


def test_app_factory_creates_valid_app():
    """The app factory returns a FastAPI instance with routes registered."""
    from cooking_plan_agent.main import create_app

    app = create_app()

    # Health endpoints are registered directly on the app
    routes = {r.path for r in app.routes if hasattr(r, "path") and isinstance(r.path, str)}
    assert "/health/live" in routes
    assert "/health/ready" in routes

    # OpenAPI spec includes the generate endpoint
    schema = app.openapi()
    assert "/internal/v1/agents/cooking-plan/generate" in schema["paths"]


def test_liveness_returns_200():
    """GET /health/live returns 200 {status: alive}."""
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


def test_readiness_returns_200():
    """GET /health/ready returns 200 when app is ready via lifespan."""
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["settings_validated"] is True
        assert data["checks"]["graph_compiled"] is True


def test_internal_endpoint_without_token_returns_422():
    """POST /generate without X-Internal-Token → 422 (missing required header)."""
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json={"request_id": "t1", "user_id": "u1", "recipes": []},
        )
        # FastAPI requires the header param → 422 validation error
        assert response.status_code == 422


def test_internal_endpoint_with_invalid_token_returns_401():
    """POST /generate with wrong X-Internal-Token returns 401."""
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json={"request_id": "t1", "user_id": "u1", "recipes": []},
            headers={"X-Internal-Token": "wrong_token"},
        )
        assert response.status_code == 401


def test_create_app_is_idempotent():
    """Calling create_app multiple times produces independent, equivalent apps."""
    from cooking_plan_agent.main import create_app

    app1 = create_app()
    app2 = create_app()
    assert app1 is not app2
    assert app1.title == app2.title


def test_openapi_schema_generation():
    """App generates a valid OpenAPI 3.x schema."""
    from cooking_plan_agent.main import create_app

    app = create_app()
    schema = app.openapi()

    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "FoodMind Cooking Plan Agent"
    assert "/health/live" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_correlation_id_middleware():
    """Correlation ID middleware echoes X-Request-ID in response header.

    The extract_correlation_id dependency is injected into the /generate
    endpoint via Depends, which stores the ID on request.state. The
    _add_correlation_id_header middleware then reads it from request.state
    and adds it to the response.

    Health endpoints do NOT use the correlation ID dependency, so the
    header won't appear for /health/live. We test via the generate endpoint.
    """
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    cid = uuid.uuid4().hex
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/agents/cooking-plan/generate",
            json={"request_id": "t1", "user_id": "u1", "recipes": []},
            headers={
                "X-Internal-Token": _TEST_TOKEN,
                "X-Request-ID": cid,
            },
        )
        assert response.headers.get("x-request-id") == cid


def test_shutdown_middleware_accepts_requests_after_lifespan():
    """After lifespan startup, requests are not rejected by shutdown middleware."""
    _set_env()
    from cooking_plan_agent.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health/live")
        assert response.status_code == 200
