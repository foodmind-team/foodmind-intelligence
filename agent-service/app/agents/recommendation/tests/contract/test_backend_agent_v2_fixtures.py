import hashlib
import json

import pytest
from conftest import AGENT_FIXTURES, GOLDEN_FIXTURES, REPOSITORY_ROOT
from pydantic import ValidationError

from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse

BACKEND_V1_MANIFEST = REPOSITORY_ROOT / "contracts/internal/agent/recommendation/v1/backend-fixture-manifest.json"


def test_backend_success_fixture_is_strict_and_checksum_bound() -> None:
    body = (GOLDEN_FIXTURES / "expected-agent-response.json").read_bytes()
    expected = next(
        line.split("  ", 1)[0]
        for line in (GOLDEN_FIXTURES / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        if line.endswith("expected-agent-response.json")
    )
    assert hashlib.sha256(body).hexdigest() == expected
    AgentResponse.model_validate_json(body)


def test_backend_v1_fixture_identity_is_pinned_without_copying_or_retirement() -> None:
    manifest = json.loads(BACKEND_V1_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["repository"] == "foodmind-backend"
    assert manifest["revision"] == "7ea2b90c1451d689c59d4ea37d337b4552220f44"
    assert manifest["version"] == "recommendation-agent-v1"
    assert manifest["featureSchemaVersion"] == "recommendation-features-v1"
    assert manifest["copyStoredInIntelligence"] is False
    assert manifest["retirementApproved"] is False
    assert len(manifest["files"]) == 4
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


@pytest.mark.parametrize(
    "name",
    [
        "invalid-unknown-candidate.json",
        "invalid-unsupported-reason.json",
        "invalid-over-3-results.json",
        "invalid-non-contiguous-ranks.json",
        "invalid-unsafe-template.json",
    ],
)
def test_backend_negative_output_fixtures_are_rejected_or_cross_contract_invalid(name: str) -> None:
    body = (AGENT_FIXTURES / name).read_bytes()
    if name == "invalid-unknown-candidate.json":
        response = AgentResponse.model_validate_json(body)
        request_ids = {candidate.candidate_id for candidate in __import_request().candidates}
        assert not {item.candidate_id for item in response.recommendations}.issubset(request_ids)
    else:
        with pytest.raises(ValidationError):
            AgentResponse.model_validate_json(body)


def __import_request() -> AgentRequest:
    return AgentRequest.model_validate_json((AGENT_FIXTURES / "valid-normal.json").read_bytes())
