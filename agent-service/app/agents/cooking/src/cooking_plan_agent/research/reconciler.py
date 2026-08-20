# =============================================================================
# 多源证据调和模块（research/reconciler）
# -----------------------------------------------------------------------------
# 将来自多个来源的 CookingEvidence 调和为共识值：
#   - 时长用中位数（手册 10.7）
#   - 火力等级用多数投票
#   - MAD 超过阈值时标记分歧
# =============================================================================

"""Multi-source evidence reconciliation (handbook 10.7).

多源证据调和（手册 10.7）。

Reconciles CookingEvidence from multiple sources into consensus values.
Uses median for durations, majority voting for heat levels, and flags
disagreement when MAD exceeds the configured threshold.

将来自多个来源的 CookingEvidence 调和为共识值：
时长采用中位数，火力等级采用多数投票，
当 MAD 超过配置阈值时标记分歧。
"""

import statistics
from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import CookingEvidence, ReconciledEvidence


def _median_duration(
    values: list[int],
) -> int:
    """Return the median of a list of ints (rounds .5 down).

    返回整数列表的中位数（.5 向下取整）。
    """
    if not values:
        return 0
    return int(statistics.median(values))


def _median_absolute_deviation(
    values: list[int],
    median: int,
) -> float:
    """Return MAD: median of absolute deviations from the median.

    返回 MAD：偏离中位数的绝对偏差的中位数。

    Robust measure of spread — insensitive to outliers (handbook 10.7).

    稳健的离散度度量 —— 对离群值不敏感（手册 10.7）。
    """
    if not values or median == 0:
        return 0.0
    deviations = [abs(v - median) for v in values]
    return float(statistics.median(deviations))


def _majority_heat(
    levels: list[HeatLevel],
) -> HeatLevel | None:
    """Return the most common heat level, or None if empty.

    返回出现次数最多的火力等级；列表为空时返回 None。
    """
    if not levels:
        return None
    # Count occurrences, return the most frequent
    # 统计出现次数，返回最频繁的等级
    from collections import Counter

    counts = Counter(levels)
    # Tie-breaking: prefer HIGH > MEDIUM > LOW > NONE
    # 平票裁决：优先 HIGH > MEDIUM > LOW > NONE
    max_count = max(counts.values())
    candidates = [level for level, c in counts.items() if c == max_count]
    priority = {HeatLevel.HIGH: 4, HeatLevel.MEDIUM: 3, HeatLevel.LOW: 2, HeatLevel.NONE: 1}
    return max(candidates, key=lambda level: priority.get(level, 0))


def reconcile(
    evidence_items: tuple[CookingEvidence, ...],
    disagreement_threshold: float = 0.5,
) -> ReconciledEvidence:
    """Reconcile multiple CookingEvidence items into consensus.

    将多条 CookingEvidence 证据调和为共识。

    Duration: median of all reported values (handbook 10.7 formula).
    If MAD > threshold * median, flag needs_confirmation.

    时长：取所有报告值的中位数（手册 10.7 公式）。
    若 MAD > 阈值 × 中位数，则标记 needs_confirmation。

    Heat level: majority vote, weighted by source count.
    Preserve dissenting sources in evidence_items metadata.

    火力等级：按来源数量加权的多数投票。
    在 evidence_items 元数据中保留少数派来源。

    Args:
        evidence_items: Evidence from one or more sources.
            来自一个或多个来源的证据。
        disagreement_threshold: MAD/median threshold for flagging.
            用于标记分歧的 MAD/中位数阈值。

    Returns:
        ReconciledEvidence with consensus values and disagreement flag.
            带有共识值和分歧标记的 ReconciledEvidence。
    """
    if not evidence_items:
        return ReconciledEvidence(
            source_count=0,
            needs_confirmation=True,
            evidence_items=(),
        )

    # --- Duration reconciliation (handbook 10.7) ---
    # --- 时长调和（手册 10.7） ---
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
    # --- 火力等级调和（多数投票） ---
    heat_levels: list[HeatLevel] = [e.heat_level for e in evidence_items if e.heat_level is not None]
    reconciled_heat = _majority_heat(heat_levels)

    # --- Temperature: take the average if multiple sources agree ---
    # --- 温度：多个来源一致时取平均值 ---
    temps: list[Decimal] = [e.explicit_temperature_c for e in evidence_items if e.explicit_temperature_c is not None]
    reconciled_temp: Decimal | None = None
    if temps:
        # sum with a Decimal start keeps pure Decimal arithmetic; dividing by a
        # Decimal avoids mixing float precision into financial-style quantities.
        # 以 Decimal 0 为起点求和保持纯 Decimal 运算；除以 Decimal 避免
        # 把浮点精度混入类财务数量。
        reconciled_temp = (sum(temps, Decimal(0)) / Decimal(len(temps))).quantize(Decimal("0.1"))

    # If no data at all, flag confirmation
    # 若完全没有数据，则标记需要确认
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
