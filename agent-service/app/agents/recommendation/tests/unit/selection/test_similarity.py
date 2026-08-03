from workflow_helpers import canonical_request

from recommendation_agent.policy.diversity import DIVERSITY_POLICY
from recommendation_agent.selection.similarity import similarity_penalty


def test_similarity_uses_only_approved_exact_and_categorical_facts() -> None:
    request = canonical_request()
    first, second = request.candidates[:2]
    assert similarity_penalty(first, second, DIVERSITY_POLICY) == 0.0
    same_category = second.model_copy(
        update={"evidence": second.evidence.model_copy(update={"category_code": "NOODLES"})}
    )
    assert similarity_penalty(first, same_category, DIVERSITY_POLICY) == 0.06
    same_offering = second.model_copy(update={"model_offering_key": first.model_offering_key})
    assert similarity_penalty(first, same_offering, DIVERSITY_POLICY) == 0.1
