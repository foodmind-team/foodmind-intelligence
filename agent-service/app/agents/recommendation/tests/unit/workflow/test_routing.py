from recommendation_agent.domain.errors import ErrorCode
from recommendation_agent.workflow.routing import route_after_node, route_after_success_builder
from recommendation_agent.workflow.state import FailureRecord, RecommendationState


def test_routes_are_forward_terminal_decisions() -> None:
    healthy: RecommendationState = {}
    failed: RecommendationState = {"failure": FailureRecord(ErrorCode.INVALID_REQUEST, 400)}
    assert route_after_node(healthy) == "continue"
    assert route_after_node(failed) == "failure"
    assert route_after_success_builder(healthy) == "success"
    assert route_after_success_builder(failed) == "failure"
