"""Immutable diversity policy constants and startup validation."""

from dataclasses import dataclass

from recommendation_agent.policy.versions import DIVERSITY_POLICY_VERSION


@dataclass(frozen=True, slots=True)
class DiversityPolicy:
    version: str = DIVERSITY_POLICY_VERSION
    lead_tie_band: float = 0.03
    personal_score_threshold: float = 0.60
    personal_neighbor_support: int = 3
    preference_match_threshold: float = 0.70
    novelty_bonus_multiplier: float = 0.08
    novelty_bonus_cap: float = 0.08
    repeated_category_penalty: float = 0.06
    repeated_cuisine_penalty: float = 0.04
    group_rate_threshold: float = 0.60
    group_member_threshold: int = 2
    max_results: int = 3

    def validate(self) -> None:
        ratios = (
            self.lead_tie_band,
            self.personal_score_threshold,
            self.preference_match_threshold,
            self.novelty_bonus_multiplier,
            self.novelty_bonus_cap,
            self.repeated_category_penalty,
            self.repeated_cuisine_penalty,
            self.group_rate_threshold,
        )
        if not all(0 <= value <= 1 for value in ratios):
            raise RuntimeError("diversity policy ratio is outside 0..1")
        if self.novelty_bonus_multiplier > self.novelty_bonus_cap:
            raise RuntimeError("novelty multiplier exceeds the frozen cap")
        if self.personal_neighbor_support < 1 or self.group_member_threshold < 1 or self.max_results != 3:
            raise RuntimeError("diversity count policy is invalid")


DIVERSITY_POLICY = DiversityPolicy()
DIVERSITY_POLICY.validate()
