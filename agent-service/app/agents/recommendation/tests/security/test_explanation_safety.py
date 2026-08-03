import pytest

from recommendation_agent.domain.errors import AgentError
from recommendation_agent.reasons.renderer import validate_explanation


@pytest.mark.parametrize(
    "canary",
    [
        "This is safe.",
        "This is allergen-free.",
        "This is healthy.",
        "A guaranteed best choice.",
        "<script>alert(1)</script>",
        "line one\nline two",
        "medical outcome",
    ],
)
def test_unsafe_claim_markup_and_control_canaries_are_rejected(canary: str) -> None:
    with pytest.raises(AgentError):
        validate_explanation(canary)


def test_observational_cleanliness_template_is_allowed() -> None:
    assert validate_explanation("A recent cleanliness observation is recorded.")
