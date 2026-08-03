"""Pure similarity penalties over approved request facts."""

from recommendation_agent.policy.diversity import DiversityPolicy
from recommendation_agent.schemas.agent_v2 import Candidate


def similarity_penalty(candidate: Candidate, selected: Candidate, policy: DiversityPolicy) -> float:
    """Return a bounded penalty; missing or different facts never imply similarity."""

    if (
        candidate.model_offering_key == selected.model_offering_key
        or candidate.model_meal_key == selected.model_meal_key
    ):
        return policy.repeated_category_penalty + policy.repeated_cuisine_penalty
    penalty = 0.0
    if candidate.evidence.category_code == selected.evidence.category_code:
        penalty += policy.repeated_category_penalty
    if candidate.evidence.cuisine_code == selected.evidence.cuisine_code:
        penalty += policy.repeated_cuisine_penalty
    return min(penalty, policy.repeated_category_penalty + policy.repeated_cuisine_penalty)
