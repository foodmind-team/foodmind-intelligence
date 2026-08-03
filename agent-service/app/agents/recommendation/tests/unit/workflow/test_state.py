import json

from workflow_helpers import canonical_request

from recommendation_agent.workflow.state import RecommendationState, serialize_state


def test_state_serialization_is_json_safe_and_excludes_sensitive_values() -> None:
    request = canonical_request()
    state: RecommendationState = {
        "request": request,
        "agent_trace_id": "agent-test-trace",
        "deadline_expiry": 102.0,
        "inference_calls": 0,
        "node_trace": (),
    }
    rendered = json.dumps(serialize_state(state), sort_keys=True)
    assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in rendered
    assert "meal_key_00000001" not in rendered
    assert "preferenceMatch" not in rendered
    assert "candidate-a" in rendered
