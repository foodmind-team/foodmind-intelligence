"""Multi-source evidence reconciliation (handbook 10.7).

Reconciles CookingEvidence from multiple sources into consensus values.
Uses median for durations, majority voting for heat levels, and flags
disagreement when MAD exceeds the configured threshold.
"""

import statistics
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import CookingEvidence, ReconciledEvidence


def _median_duration(
    values: list[int],
) -> int:
    """Return the median of a list of ints (rounds .5 down)."""
    if not values:
        return 0
    return int(statistics.median(values))


def _median_absolute_deviation(
    values: list[int],
    median: int,
) -> float:
    """Return MAD: median of absolute deviations from the median.

    Robust measure of spread — insensitive to outliers (handbook 10.7).
    """
    if not values or median == 0:
        return 0.0
    deviations = [abs(v - median) for v in values]
    return float(statistics.median(deviations))


def _majority_heat(
    levels: list[HeatLevel],
) -> HeatLevel | None:
    """Return the most common heat level, or None if empty."""
    if not levels:
        return None
    # Count occurrences, return the most frequent
    from collections import Counter
    counts = Counter(levels)
    # Tie-breaking: prefer HIGH > MEDIUM > LOW > NONE
    max_count = max(counts.values())
    candidates = [l for l, c in counts.items() if c == max_count]
    priority = {HeatLevel.HIGH: 4, HeatLevel.MEDIUM: 3, HeatLevel.LOW: 2, HeatLevel.NONE: 1}
    return max(candidates, key=lambda l: priority.get(l, 0))


def reconcile(
    evidence_items: tuple[CookingEvidence, ...],
    disagreement_threshold: float = 0.5,
) -> ReconciledEvidence:
    """Reconcile multiple CookingEvidence items into consensus.

    Duration: median of all reported values (handbook 10.7 formula).
    If MAD > threshold * median, flag needs_confirmation.

    Heat level: majority vote, weighted by source count.
    Preserve dissenting sources in evidence_items metadata.

    Args:
        evidence_items: Evidence from one or more sources.
        disagreement_threshold: MAD/median threshold for flagging.

    Returns:
        ReconciledEvidence with consensus values and disagreement flag.
    """
    if not evidence_items:
        return ReconciledEvidence(
            source_count=0,
            needs_confirmation=True,
            evidence_items=(),
        )

    # --- Duration reconciliation (handbook 10.7) ---
    dur_mins: list[int] = []
    dur_maxs: list[int] = []
    for e in evidence_items:
        if e.duration_min_minutes is not None:
            dur_mins.append(e.duration_min_minutes)
        if e.duration_max_minutes is not None:
            dur_maxs.append(e.duration_max_minutes)

    reconciled_min: int | None = None
    reconciled_max: int | None = None
    needs_confirmation = False

    if dur_mins:
        med_min = _median_duration(dur_mins)
        mad_min = _median_absolute_deviation(dur_mins, med_min)
        if mad_min > disagreement_threshold * med_min:
            needs_confirmation = True
        reconciled_min = med_min

    if dur_maxs:
        med_max = _median_duration(dur_maxs)
        mad_max = _median_absolute_deviation(dur_maxs, med_max)
        if mad_max > disagreement_threshold * med_max:
            needs_confirmation = True
        reconciled_max = med_max

    # --- Heat level reconciliation (majority vote) ---
    heat_levels: list[HeatLevel] = [
        e.heat_level for e in evidence_items if e.heat_level is not None
    ]
    reconciled_heat = _majority_heat(heat_levels)

    # --- Temperature: take the average if multiple sources agree ---
    temps: list[Decimal] = [
        e.explicit_temperature_c
        for e in evidence_items
        if e.explicit_temperature_c is not None
    ]
    reconciled_temp: Decimal | None = None
    if temps:
        reconciled_temp = sum(temps) / len(temps)
        reconciled_temp = reconciled_temp.quantize(Decimal("0.1"))

    # If no data at all, flag confirmation
    if reconciled_heat is None and reconciled_min is None and reconciled_max is None and reconciled_temp is None:
        needs_confirmation = True

    return ReconciledEvidence(
        heat_level=reconciled_heat,
        duration_min_minutes=reconciled_min,
        duration_max_minutes=reconciled_max,
        explicit_temperature_c=reconciled_temp,
        source_count=len(evidence_items),
        needs_confirmation=needs_confirmation,
        evidence_items=evidence_items,
    )
