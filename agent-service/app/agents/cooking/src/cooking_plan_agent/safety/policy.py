"""Regional food-safety policy packs (P3-04).

Migrates hard-coded temperature / holding / rest constants out of the safety
rules into versioned, source-backed, region-selectable policy packs.

Design contract (development plan P3-04, data-flow rules D6/D7):
  - Region selection is EXPLICIT: the request region overrides the deployment
    default (Settings.safety_policy_region). An unknown region is rejected —
    never silently falls back to another pack.
  - Thresholds come ONLY from approved, reviewed packs in this directory.
    LLM or web-search results can never modify them (D7).
  - Policies are versioned and immutable. New guidance ships as a NEW version;
    old versions stay registered so historical checkpoints remain auditable.
  - A policy that is not yet effective, has no sources, or belongs to an
    unknown region can never be applied to a plan (cannot enter READY).

Resolution semantics (resolve_policy):
  1. region must be registered — otherwise UnknownRegionError.
  2. version defaults to the latest registered version of that region.
     An explicit version must exist — otherwise UnknownPolicyVersionError.
  3. effective_at must be <= today — otherwise PolicyNotYetEffectiveError.
  4. sources must be non-empty — otherwise PolicyMissingSourcesError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from cooking_plan_agent.domain.models import PolicySourceRef, SafetyPolicyRecord

# =============================================================================
# Schema
# =============================================================================


@dataclass(frozen=True)
class PolicySource:
    """An official source backing one or more thresholds (D7 traceability)."""

    source_id: str
    title: str
    url: str


@dataclass(frozen=True)
class SafetyThresholds:
    """Region-specific safety thresholds consumed by the safety rules.

    Every value is locked by unit-test fixtures that assert the exact number
    and its source provenance (plan P3-04 verification).

    ``safe_minimum_temperatures_c`` keys the per-protein safe minimum internal
    cooking temperatures in °C (same protein categories as the keyword matcher
    in safety/rules.py). Categories absent from the map are "not documented by
    this authority" and are NOT flagged by ProteinSafetyTemperatureRule.
    """

    safe_minimum_temperatures_c: dict[str, Decimal]
    # Max time perishable food may sit in the danger zone (room temperature).
    max_room_temp_holding_minutes: int
    # Hot holding: keep cooked food at or above this temperature.
    hot_holding_minimum_c: Decimal
    # Cold holding: keep chilled food at or below this temperature.
    cold_holding_maximum_c: Decimal
    # Reheating: reach at least this internal temperature …
    reheat_minimum_c: Decimal
    # … and hold it for at least this long (seconds). 0 = no hold requirement.
    reheat_hold_seconds: int
    # Post-cooking rest time for protein categories that need it (minutes).
    rest_time_minutes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyPolicy:
    """An immutable, versioned, source-backed safety policy pack.

    ``region`` is a stable ISO-3166 alpha-2 code ("US", "SG"). ``version``
    follows MAJOR.MINOR and is bumped (never mutated) when thresholds change.
    """

    region: str
    version: str
    effective_at: date
    sources: tuple[PolicySource, ...]
    thresholds: SafetyThresholds

    def to_record(self) -> SafetyPolicyRecord:
        """Project this policy into the serialisable domain record.

        The record is what travels in workflow state and terminal responses —
        the full SafetyPolicy (with rule-config thresholds) stays in-process.
        """
        return SafetyPolicyRecord(
            region=self.region,
            version=self.version,
            effective_at=self.effective_at,
            sources=tuple(PolicySourceRef(source_id=s.source_id, title=s.title, url=s.url) for s in self.sources),
        )


# =============================================================================
# Registry & resolution
# =============================================================================


class PolicyResolutionError(ValueError):
    """Base class for policy resolution failures (P3-04 D6: no silent fallback)."""


class UnknownRegionError(PolicyResolutionError):
    """The requested region has no registered policy pack."""


class UnknownPolicyVersionError(PolicyResolutionError):
    """The requested version is not registered for the region."""


class PolicyNotYetEffectiveError(PolicyResolutionError):
    """The policy's effective_at date is in the future."""


class PolicyMissingSourcesError(PolicyResolutionError):
    """The policy declares no official sources (unverifiable thresholds)."""


# Registered packs keyed by (region, version). Regions MUST be upper-case.
_POLICY_REGISTRY: dict[tuple[str, str], SafetyPolicy] = {}


def register_policy(policy: SafetyPolicy) -> None:
    """Register a policy pack (called once at import time by policies/__init__).

    Registration replaces nothing silently: a duplicate (region, version) pair
    raises ValueError so packs cannot be overwritten by accident.
    """
    key = (policy.region.upper(), policy.version)
    if key in _POLICY_REGISTRY:
        raise ValueError(f"Policy already registered: region={key[0]} version={key[1]}")
    _POLICY_REGISTRY[key] = policy


def supported_regions() -> tuple[str, ...]:
    """Return the sorted regions that have at least one registered pack."""
    return tuple(sorted({region for region, _ in _POLICY_REGISTRY}))


def latest_version(region: str) -> str | None:
    """Return the highest registered version for a region (None if unknown)."""
    versions = [ver for reg, ver in _POLICY_REGISTRY if reg == region.upper()]
    if not versions:
        return None
    return max(versions)


def resolve_policy(region: str, version: str | None = None) -> SafetyPolicy:
    """Resolve the active policy for a region, honouring the P3-04 reject rules.

    Args:
        region: Explicit region (ISO alpha-2, case-insensitive). Never None —
                callers must supply the request region or the deployment
                default; an unknown region is a hard error (D6).
        version: Optional explicit version. Defaults to the latest registered
                 version of the region.

    Returns:
        The resolved SafetyPolicy.

    Raises:
        UnknownRegionError: region has no registered pack.
        UnknownPolicyVersionError: explicit version not registered.
        PolicyNotYetEffectiveError: policy not yet effective.
        PolicyMissingSourcesError: policy has no official sources.
    """
    region_key = region.upper()
    if region_key not in {r for r, _ in _POLICY_REGISTRY}:
        raise UnknownRegionError(
            f"Unknown safety-policy region: {region!r}. Supported regions: {', '.join(supported_regions()) or '(none)'}"
        )

    if version is None:
        version = latest_version(region_key) or ""
    key = (region_key, version)
    policy = _POLICY_REGISTRY.get(key)
    if policy is None:
        raise UnknownPolicyVersionError(
            f"No safety policy for region {region_key!r} version {version!r}. "
            f"Available versions: {[v for r, v in _POLICY_REGISTRY if r == region_key]}"
        )

    if policy.effective_at > date.today():
        raise PolicyNotYetEffectiveError(
            f"Safety policy {region_key}@{policy.version} is not effective until {policy.effective_at.isoformat()}"
        )

    if not policy.sources:
        raise PolicyMissingSourcesError(
            f"Safety policy {region_key}@{policy.version} declares no official sources — cannot be applied"
        )

    return policy
