"""Release scenario matrix for deterministic, grounded Agent output."""

import copy

import pytest
from conftest import AGENT_FIXTURES, REPOSITORY_ROOT, load_json
from fixtures.fake_inference import fixture_agent_client
from fixtures.scenarios import backend_validate

INFERENCE_FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"


@pytest.mark.parametrize(
    ("request_fixture", "inference_fixture"),
    [
        ("valid-normal.json", "valid-hybrid.json"),
        ("valid-cold-start.json", "valid-cold-start.json"),
    ],
)
def test_replay_is_deterministic_and_grounded(request_fixture: str, inference_fixture: str) -> None:
    request = load_json(AGENT_FIXTURES / request_fixture)
    inference = load_json(INFERENCE_FIXTURES / inference_fixture)
    with fixture_agent_client() as (client, _fake):
        values = [
            client.post(
                "/internal/v1/recommendations/generate",
                json=request,
                headers={"Authorization": "Bearer e2e-agent-token"},
            ).json()
            for _ in range(2)
        ]

    values[1]["agentTraceId"] = values[0]["agentTraceId"]
    assert values[0] == values[1]
    assert backend_validate(request, inference, values[0])


def test_sparse_group_omits_unsupported_group_result_and_reason() -> None:
    request = load_json(AGENT_FIXTURES / "valid-sparse-group.json")
    with fixture_agent_client() as (client, fake):
        responses = [
            client.post(
                "/internal/v1/recommendations/generate",
                json=request,
                headers={"Authorization": "Bearer e2e-agent-token"},
            )
            for _ in range(2)
        ]

    assert all(response.status_code == 200 for response in responses)
    values = [response.json() for response in responses]
    values[1]["agentTraceId"] = values[0]["agentTraceId"]
    assert values[0] == values[1]
    assert fake.last_response is not None
    assert backend_validate(request, fake.last_response, values[0])
    recommendations = values[0]["recommendations"]
    assert all(item["recommendationType"] != "Group" for item in recommendations)
    assert all("GROUP_POPULAR" not in item["reasons"] for item in recommendations)


@pytest.mark.parametrize(
    "mutation",
    [
        {"candidateId": "candidate-outside-request"},
        {"rank": 2},
        {"recommendationType": "EXPLORATORY"},
        {"modelScore": -99.0},
        {"reasons": ["USER_CF_PROXY"]},
        {"explanation": "It matches your saved preferences."},
        {"explanation": "Guaranteed safest meal"},
        {"modelVersion": "hybrid-ranking-v0"},
    ],
)
def test_independent_backend_validator_rejects_malicious_agent_output(mutation: dict[str, object]) -> None:
    request = load_json(AGENT_FIXTURES / "valid-normal.json")
    inference = load_json(INFERENCE_FIXTURES / "valid-hybrid.json")
    with fixture_agent_client() as (client, _fake):
        response = client.post(
            "/internal/v1/recommendations/generate",
            json=request,
            headers={"Authorization": "Bearer e2e-agent-token"},
        ).json()
    malicious = copy.deepcopy(response)
    target = malicious if "modelVersion" in mutation else malicious["recommendations"][0]
    target.update(mutation)

    assert not backend_validate(request, inference, malicious)
