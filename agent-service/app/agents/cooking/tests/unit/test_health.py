"""Health endpoint tests — /health/live and /health/ready (Handbook 12.4)."""

from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app


def test_liveness() -> None:
    """Liveness probe returns 200 with alive status.

    Handbook 12.4: must not call external providers.
    """
    # Use context-manager form so FastAPI lifespan runs and sets app.state.
    with TestClient(create_app()) as client:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


def test_readiness() -> None:
    """Readiness probe returns 200 when settings validated and graph compiled.

    Handbook 12.4: returns 503 if not ready (missing config, graph compile failure).
    """
    # Use context-manager form so FastAPI lifespan runs before the test.
    with TestClient(create_app()) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["settings_validated"] is True
        assert body["checks"]["graph_compiled"] is True
        assert body["checks"]["task_api_ready"] is True
        assert body["checks"]["shutting_down"] is False


def test_load_snapshot_reports_limiter_metrics() -> None:
    """P1-03: /health/load exposes active/queued/rejected metrics.

    The route must answer even while the business limiter is saturated —
    health probes bypass the lease dependency.
    """
    with TestClient(create_app()) as client:
        response = client.get("/health/load")
        assert response.status_code == 200
        body = response.json()
        assert "active" in body
        assert "queued" in body
        assert "rejected_total" in body
        assert "queue_wait_ms" in body
