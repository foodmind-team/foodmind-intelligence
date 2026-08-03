"""Backend fallback remains constant across private Agent failure modes."""

import copy

import pytest
from conftest import AGENT_FIXTURES, load_json
from fixtures.fake_inference import Scenario, fixture_agent_client
from fixtures.scenarios import FALLBACK, backend_call


@pytest.mark.parametrize(
    "scenario",
    [
        "timeout",
        "unavailable",
        "non_2xx",
        "malformed",
        "oversized",
        "feature_mismatch",
        "model_mismatch",
        "package_mismatch",
        "key_mismatch",
        "unknown_candidate",
        "duplicate_candidate",
        "missing_candidate",
        "invalid_probability",
        "invalid_evidence",
    ],
)
def test_private_failure_maps_to_same_safe_backend_fallback(scenario: Scenario) -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    response_limit = 1024 if scenario == "oversized" else 262_144
    with fixture_agent_client(scenario, inference_max_response_bytes=response_limit) as (client, fake):
        result = backend_call(client, request)

    assert result == FALLBACK
    assert fake.attempts == 1
    assert "fixture" not in str(result)


def test_expired_deadline_maps_to_fallback_without_inference() -> None:
    request = copy.deepcopy(load_json(AGENT_FIXTURES / "valid-normal.json"))
    request["decisionAt"] = "2020-01-01T00:00:00Z"
    request["deadlineAt"] = "2020-01-01T00:00:01Z"
    with fixture_agent_client() as (client, fake):
        result = backend_call(client, request)

    assert result == FALLBACK
    assert fake.attempts == 0
