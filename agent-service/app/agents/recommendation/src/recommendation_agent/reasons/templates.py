"""Versioned fixed explanation fragments with no interpolation surface."""

from recommendation_agent.domain.models import ReasonCode
from recommendation_agent.policy.versions import TEMPLATE_VERSION

TEMPLATES: dict[ReasonCode, str] = {
    ReasonCode.USER_CF: "People with similar preferences also liked this.",
    ReasonCode.ITEM_CF: "It resembles meals you liked.",
    ReasonCode.PREFERENCE_MATCH: "It matches your saved preferences.",
    ReasonCode.WANT_TO_TRY: "You marked this as Want to Try.",
    ReasonCode.GROUP_POPULAR: "It matches preferences shared by the group.",
    ReasonCode.CONTEXT_MATCH: "It matches the current meal context.",
    ReasonCode.CLEANLINESS_OBSERVED: "A recent cleanliness observation is recorded.",
}

if set(TEMPLATES) != set(ReasonCode):
    raise RuntimeError(f"{TEMPLATE_VERSION} must define exactly one template per reason")
