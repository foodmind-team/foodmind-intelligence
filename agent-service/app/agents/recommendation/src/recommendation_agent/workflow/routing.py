"""Explicit forward-only routing decisions."""

from typing import Literal

from recommendation_agent.workflow.state import RecommendationState


def route_after_node(state: RecommendationState) -> Literal["continue", "failure"]:
    return "failure" if "failure" in state else "continue"


def route_after_success_builder(state: RecommendationState) -> Literal["success", "failure"]:
    return "failure" if "failure" in state else "success"
