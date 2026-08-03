import json

from conftest import AGENT_FIXTURES

from recommendation_agent.domain.errors import ErrorCode
from recommendation_agent.schemas.agent_v2 import AgentFailure


def test_canonical_failure_fixtures_parse() -> None:
    for name in ("failure-inference-unavailable.json", "failure-deadline-exhausted.json"):
        AgentFailure.model_validate_json((AGENT_FIXTURES / name).read_text(encoding="utf-8"))


def test_failure_policy_covers_every_stable_error_code() -> None:
    catalog = json.loads((AGENT_FIXTURES / "failure-policy-cases.json").read_text(encoding="utf-8"))
    assert {case["code"] for case in catalog["cases"]} == {code.value for code in ErrorCode}
