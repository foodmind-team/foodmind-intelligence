import json

import pytest
from conftest import AGENT_FIXTURES, GOLDEN_FIXTURES, load_json
from pydantic import ValidationError

from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse


@pytest.mark.parametrize("name", ["valid-normal.json", "valid-cold-start.json", "valid-sparse-group.json"])
def test_valid_request_fixtures_round_trip_camel_case(name: str) -> None:
    raw = (AGENT_FIXTURES / name).read_text(encoding="utf-8")
    parsed = AgentRequest.model_validate_json(raw)
    assert parsed.model_dump(mode="json", by_alias=True) == json.loads(raw)
    assert "modelUserKey" not in repr(parsed)


def test_golden_success_round_trips() -> None:
    raw = (GOLDEN_FIXTURES / "expected-agent-response.json").read_text(encoding="utf-8")
    parsed = AgentResponse.model_validate_json(raw)
    assert parsed.model_dump(mode="json", by_alias=True) == json.loads(raw)


@pytest.mark.parametrize("name", ["invalid-model-key-version.json", "invalid-over-100-candidates.json"])
def test_invalid_request_fixtures_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        AgentRequest.model_validate_json((AGENT_FIXTURES / name).read_text(encoding="utf-8"))


def test_unknown_field_coercion_and_duplicate_ids_are_rejected() -> None:
    payload = load_json(AGENT_FIXTURES / "valid-normal.json")
    payload["unknown"] = True
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(payload)
    payload.pop("unknown")
    payload["candidates"][0]["evidence"]["wantToTry"] = 1
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(payload)
    payload = load_json(AGENT_FIXTURES / "valid-normal.json")
    payload["candidates"][1]["candidateId"] = payload["candidates"][0]["candidateId"]
    with pytest.raises(ValidationError):
        AgentRequest.model_validate(payload)


@pytest.mark.parametrize(
    "name",
    [
        "invalid-unsupported-reason.json",
        "invalid-over-3-results.json",
        "invalid-non-contiguous-ranks.json",
        "invalid-unsafe-template.json",
    ],
)
def test_invalid_response_fixtures_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        AgentResponse.model_validate_json((AGENT_FIXTURES / name).read_text(encoding="utf-8"))
