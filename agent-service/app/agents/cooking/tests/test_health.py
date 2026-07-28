from fastapi.testclient import TestClient

from cooking_plan_agent.main import create_app

# Create a test client from the application factory.
client = TestClient(create_app())


def test_liveness() -> None:
    # The liveness endpoint should return a healthy status payload.
    client = TestClient(create_app())
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
