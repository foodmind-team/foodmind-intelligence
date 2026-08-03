"""Agent consumer behavior for inference success and failure scenarios."""

import copy

import pytest
from conftest import AGENT_FIXTURES, load_json
from fixtures.fake_inference import Scenario, fixture_agent_client


@pytest.mark.parametrize(
    ("scenario", "status", "code"),
    [
        ("timeout", 504, "INFERENCE_TIMEOUT"),
        ("unavailable", 503, "INFERENCE_UNAVAILABLE"),
        ("non_2xx", 502, "INFERENCE_HTTP_ERROR"),
        ("malformed", 502, "INFERENCE_MALFORMED_RESPONSE"),
        ("feature_mismatch", 502, "FEATURE_VERSION_MISMATCH"),
        ("model_mismatch", 502, "MODEL_VERSION_MISMATCH"),
        ("package_mismatch", 502, "MODEL_PACKAGE_MISMATCH"),
        ("key_mismatch", 502, "MODEL_KEY_VERSION_MISMATCH"),
        ("unknown_candidate", 502, "UNKNOWN_CANDIDATE"),
        ("duplicate_candidate", 502, "DUPLICATE_CANDIDATE"),
        ("missing_candidate", 502, "MISSING_CANDIDATE"),
        ("invalid_probability", 502, "INVALID_PROBABILITY"),
        ("invalid_evidence", 502, "INVALID_EVIDENCE"),
    ],
)
def test_inference_failure_is_typed_and_attempted_once(scenario: Scenario, status: int, code: str) -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    with fixture_agent_client(scenario) as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={"Authorization": "Bearer e2e-agent-token"},
        )

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["status"] == "failure"
    assert fake.attempts == 1
    assert "fixture" not in response.text


def test_oversized_inference_body_is_rejected_without_parsing() -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    with fixture_agent_client("oversized", inference_max_response_bytes=1024) as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={"Authorization": "Bearer e2e-agent-token"},
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "INFERENCE_RESPONSE_TOO_LARGE"
    assert fake.attempts == 1


@pytest.mark.parametrize("fixture", ["valid-normal.json", "valid-cold-start.json", "valid-sparse-group.json"])
def test_supported_fixture_succeeds_with_one_inference_attempt(fixture: str) -> None:
    request = load_json(AGENT_FIXTURES / fixture)
    with fixture_agent_client() as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={"Authorization": "Bearer e2e-agent-token"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert fake.attempts == 1


def test_expired_deadline_performs_zero_inference_attempts() -> None:
    request = copy.deepcopy(load_json(AGENT_FIXTURES / "valid-normal.json"))
    request["decisionAt"] = "2020-01-01T00:00:00Z"
    request["deadlineAt"] = "2020-01-01T00:00:01Z"
    with fixture_agent_client() as (client, fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={"Authorization": "Bearer e2e-agent-token"},
        )

    assert response.status_code == 408
    assert response.json()["error"]["code"] == "DEADLINE_EXPIRED"
    assert fake.attempts == 0
