"""Independent Backend-style validation at the Agent v2 boundary."""

from conftest import AGENT_FIXTURES, REPOSITORY_ROOT, load_json
from fixtures.fake_inference import fixture_agent_client
from fixtures.scenarios import backend_validate

INFERENCE_FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


def test_backend_fixture_passes_agent_and_independent_validator() -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    inference = load_json(INFERENCE_FIXTURES / "valid-hybrid.json")
    with fixture_agent_client() as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={
                "Authorization": "Bearer e2e-agent-token",
                "X-Request-ID": request["requestId"],
                "X-Trace-ID": request["traceId"],
            },
        )

    assert response.status_code == 200
    assert fake.attempts == 1
    assert fake.last_request is not None
    assert fake.last_request["deadlineAt"] == request["deadlineAt"]
    assert len(response.request.content or b"") < 1_048_576
    assert backend_validate(request, inference, response.json())


def test_backend_auth_is_required_before_agent_or_inference_work() -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    with fixture_agent_client() as (client, fake):
        response = client.post("/internal/v1/recommendations/generate", json=request)

    assert response.status_code == 401
    assert response.json() == {
        "status": "failure",
        "error": {"code": "MISSING_AUTHORIZATION_HEADER", "retryable": False},
    }
    assert fake.attempts == 0
