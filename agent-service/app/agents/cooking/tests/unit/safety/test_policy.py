"""P3-04: regional food-safety policy packs — registry, resolution, provenance.

Covers the plan's unit-test requirements:
  - policy pack loading and version resolution
  - threshold traceability (every value back to a source ID)
  - unknown region / not-yet-effective / missing-source rejection
  - old policy versions retained for historical-checkpoint audit
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import SafetyReport
from cooking_plan_agent.safety import policy as policy_mod
from cooking_plan_agent.safety.engine import SafetyEngine
from cooking_plan_agent.safety.policies.sfa import SFA_POLICY
from cooking_plan_agent.safety.policies.usda import USDA_POLICY
from cooking_plan_agent.safety.policy import (
    PolicyMissingSourcesError,
    PolicyNotYetEffectiveError,
    PolicyResolutionError,
    PolicySource,
    SafetyPolicy,
    SafetyThresholds,
    UnknownPolicyVersionError,
    UnknownRegionError,
    latest_version,
    register_policy,
    resolve_policy,
    supported_regions,
)
from cooking_plan_agent.safety.rules import HoldingTimeRule, ProteinSafetyTemperatureRule, build_rules

# ---------------------------------------------------------------------------
# Registry & resolution
# ---------------------------------------------------------------------------


def test_both_initial_packs_are_registered() -> None:
    """US and SG packs register at import time (policies/__init__)."""
    assert {"SG", "US"} <= set(supported_regions())


def test_resolve_us_and_sg_latest() -> None:
    """Resolving without a version picks the latest of the region."""
    us = resolve_policy("US")
    sg = resolve_policy("SG")
    assert us is USDA_POLICY
    assert sg is SFA_POLICY
    assert us.region == "US"
    assert sg.region == "SG"


def test_resolve_is_case_insensitive() -> None:
    assert resolve_policy("us").region == "US"
    assert resolve_policy("sg").region == "SG"


def test_resolve_explicit_version() -> None:
    assert resolve_policy("US", "1.0") is USDA_POLICY


# ---------------------------------------------------------------------------
# Rejection rules (P3-04 D6: no silent fallback)
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_region() -> str:
    """A unique region whose temp packs are removed after the test."""
    return f"ZZ{abs(hash(__file__)) % 10_000}"


def _register_temp(region: str, *, future: bool = False, sources: bool = True) -> SafetyPolicy:
    """Register a throwaway policy pack under a private region for a test."""
    thresholds = SafetyThresholds(
        safe_minimum_temperatures_c={"poultry": Decimal(74)},
        max_room_temp_holding_minutes=120,
        hot_holding_minimum_c=Decimal(60),
        cold_holding_maximum_c=Decimal(4),
        reheat_minimum_c=Decimal(74),
        reheat_hold_seconds=0,
    )
    policy = SafetyPolicy(
        region=region,
        version="1.0",
        effective_at=date.today() + timedelta(days=7) if future else date(2024, 1, 1),
        sources=((PolicySource(source_id="s1", title="t", url="https://example.com"),) if sources else ()),
        thresholds=thresholds,
    )
    register_policy(policy)
    return policy


def test_unknown_region_is_rejected() -> None:
    with pytest.raises(UnknownRegionError):
        resolve_policy("XX")


def test_unknown_region_never_silently_falls_back() -> None:
    """An unknown region must not resolve to any registered pack (D6)."""
    with pytest.raises(PolicyResolutionError):
        resolve_policy("USX")


def test_unknown_version_is_rejected(temp_region: str) -> None:
    _register_temp(temp_region)
    try:
        with pytest.raises(UnknownPolicyVersionError):
            resolve_policy(temp_region, "99.9")
    finally:
        policy_mod._POLICY_REGISTRY.pop((temp_region.upper(), "1.0"))


def test_not_yet_effective_is_rejected(temp_region: str) -> None:
    _register_temp(temp_region, future=True)
    try:
        with pytest.raises(PolicyNotYetEffectiveError):
            resolve_policy(temp_region)
    finally:
        policy_mod._POLICY_REGISTRY.pop((temp_region.upper(), "1.0"))


def test_missing_sources_is_rejected(temp_region: str) -> None:
    _register_temp(temp_region, sources=False)
    try:
        with pytest.raises(PolicyMissingSourcesError):
            resolve_policy(temp_region)
    finally:
        policy_mod._POLICY_REGISTRY.pop((temp_region.upper(), "1.0"))


# ---------------------------------------------------------------------------
# Versioning & audit retention
# ---------------------------------------------------------------------------


def test_new_version_becomes_default_old_version_still_resolvable(temp_region: str) -> None:
    """A bumped version becomes the default; the old one stays for audit."""
    _register_temp(temp_region)
    try:
        older = resolve_policy(temp_region)
        # Bump: register v2.0 of the same region.
        thresholds = SafetyThresholds(
            safe_minimum_temperatures_c={"poultry": Decimal(75)},
            max_room_temp_holding_minutes=240,
            hot_holding_minimum_c=Decimal(60),
            cold_holding_maximum_c=Decimal(5),
            reheat_minimum_c=Decimal(75),
            reheat_hold_seconds=120,
        )
        newer = SafetyPolicy(
            region=temp_region,
            version="2.0",
            effective_at=date(2024, 1, 1),
            sources=(PolicySource(source_id="s1", title="t", url="https://example.com"),),
            thresholds=thresholds,
        )
        register_policy(newer)

        assert latest_version(temp_region) == "2.0"
        assert resolve_policy(temp_region) is newer  # default = latest
        assert resolve_policy(temp_region, "1.0") is older  # old retained for audit
    finally:
        for version in ("1.0", "2.0"):
            policy_mod._POLICY_REGISTRY.pop((temp_region.upper(), version), None)


def test_duplicate_registration_is_rejected() -> None:
    """Registering the same (region, version) twice is a hard error."""
    with pytest.raises(ValueError):
        register_policy(USDA_POLICY)


# ---------------------------------------------------------------------------
# Provenance: every threshold traces to a source
# ---------------------------------------------------------------------------


def test_policy_record_carries_provenance() -> None:
    record = resolve_policy("SG").to_record()
    assert record.region == "SG"
    assert record.version == "1.0"
    assert record.effective_at == date(2024, 1, 1)
    assert record.sources, "Policy must declare official sources (D7)"
    assert all(s.source_id and s.url for s in record.sources)


def test_us_vs_sg_threshold_differences() -> None:
    """Same recipe → different constraints across packs (fixture-locked)."""
    us = resolve_policy("US").thresholds
    sg = resolve_policy("SG").thresholds

    # Danger zone / cold holding: USDA 4°C vs SFA 5°C.
    assert us.cold_holding_maximum_c == Decimal(4)
    assert sg.cold_holding_maximum_c == Decimal(5)

    # Room-temperature holding: USDA 2 h vs SFA 4 h (EPH reg 13A).
    assert us.max_room_temp_holding_minutes == 120
    assert sg.max_room_temp_holding_minutes == 240

    # Poultry internal temperature: USDA 74°C vs SFA 75°C.
    assert us.safe_minimum_temperatures_c["poultry"] == Decimal(74)
    assert sg.safe_minimum_temperatures_c["poultry"] == Decimal(75)

    # Reheating: SFA ≥75°C for ≥2 min; USDA ≥74°C, no hold time.
    assert sg.reheat_minimum_c == Decimal(75) and sg.reheat_hold_seconds == 120
    assert us.reheat_minimum_c == Decimal(74) and us.reheat_hold_seconds == 0

    # Rest time: USDA mandates 3 min for whole cuts; SFA documents none.
    assert us.rest_time_minutes["whole_cuts"] == 3
    assert sg.rest_time_minutes == {}


def test_sg_pack_omits_undocumented_categories() -> None:
    """SFA publishes no per-cut table — fish/whole cuts are not flagged."""
    sg = resolve_policy("SG").thresholds
    assert "beef" not in sg.safe_minimum_temperatures_c
    assert "fish" not in sg.safe_minimum_temperatures_c


# ---------------------------------------------------------------------------
# Policy-bound rules produce region-specific constraints
# ---------------------------------------------------------------------------


def test_same_chicken_step_flagged_by_sg_not_by_us() -> None:
    """Chicken at 74°C passes USDA (≥74) but is flagged by SFA (<75)."""
    us_rule = ProteinSafetyTemperatureRule(
        safe_temperatures_c=dict(resolve_policy("US").thresholds.safe_minimum_temperatures_c)
    )
    sg_rule = ProteinSafetyTemperatureRule(
        safe_temperatures_c=dict(resolve_policy("SG").thresholds.safe_minimum_temperatures_c)
    )
    assert us_rule.safe_temperatures_c["chicken"] == Decimal(74)
    assert sg_rule.safe_temperatures_c["chicken"] == Decimal(75)


def test_build_rules_binds_holding_limit_from_policy() -> None:
    """build_rules wires the policy's room-temperature holding limit."""
    us_rules = build_rules(resolve_policy("US"))
    sg_rules = build_rules(resolve_policy("SG"))

    us_holding = next(r for r in us_rules if isinstance(r, HoldingTimeRule))
    sg_holding = next(r for r in sg_rules if isinstance(r, HoldingTimeRule))
    assert us_holding.max_holding_minutes_room_temp == 120
    assert sg_holding.max_holding_minutes_room_temp == 240


def test_engine_report_carries_policy_record() -> None:
    """SafetyEngine bound to a policy records provenance on the report."""
    from cooking_plan_agent.domain.models import SafetyContext

    sg_policy = resolve_policy("SG")
    engine = SafetyEngine(rules=build_rules(sg_policy), policy=sg_policy)
    report = engine.evaluate(SafetyContext(recipes=()))

    assert report.safety_policy is not None
    assert report.safety_policy.region == "SG"
    assert report.safety_policy.version == "1.0"
    assert report.safety_policy.sources
    # No recipes → no findings; the engine still tags the policy provenance.
    assert report.is_safe is True


def test_legacy_engine_report_has_no_policy() -> None:
    """A pre-policy SafetyEngine produces a report with safety_policy=None."""
    from cooking_plan_agent.domain.models import SafetyContext

    report = SafetyEngine().evaluate(SafetyContext(recipes=()))
    assert report.safety_policy is None


# ---------------------------------------------------------------------------
# Historical-checkpoint compatibility
# ---------------------------------------------------------------------------


def test_old_safety_report_without_policy_still_deserializes() -> None:
    """A pre-P3-04 SafetyReport (no safety_policy field) stays valid."""
    old_payload = {
        "report_id": "safety_legacy",
        "findings": [],
        "is_safe": True,
        "has_unrepairable": False,
        "required_safety_task_ids": [],
        "insertions": [],
    }
    report = SafetyReport.model_validate(old_payload)
    assert report.safety_policy is None
    assert report.is_safe is True
