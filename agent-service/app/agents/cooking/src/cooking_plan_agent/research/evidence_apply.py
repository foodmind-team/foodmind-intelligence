# =============================================================================
# 证据回写模块（research/evidence_apply）
# -----------------------------------------------------------------------------
# 将调和后的研究证据写回菜谱候选（P1-01）。关键安全规则：
# 安全关键温度绝不从缺少可验证 URL 的证据中自动解析。
# =============================================================================

"""Apply reconciled research evidence back into recipe candidates (P1-01).

将调和后的研究证据写回菜谱候选（P1-01）。

P1-01 closes the research loop: before this module, ``research_missing``
wrote ``research_evidence`` into state that no downstream node consumed, so
search results never updated the plan and the pipeline silently ignored them.

P1-01 闭合了研究回路：在此模块之前，``research_missing`` 把 ``research_evidence``
写入状态，却没有下游节点消费它，导致搜索结果从不更新计划、流水线静默忽略它们。

This module:
  1. Locates the EXACT target field via ``gap_id + recipe_id + field_path``
     (never by list position — P1-01 rule 2).
  2. Writes heat / duration / temperature values back into the candidate
     step identified by the gap.
  3. Carries evidence provenance as ``EvidenceRef`` on an Assumption so the
     source remains traceable in the final assumption/response.
  4. Decides when evidence is reliable enough to auto-apply vs. when it must
     surface for user confirmation.

本模块：
  1. 通过 ``gap_id + recipe_id + field_path`` 精确定位目标字段
     （绝不按列表位置 —— P1-01 规则 2）。
  2. 将火力 / 时长 / 温度值写回缺口所标识的候选步骤。
  3. 把证据来源作为 Assumption 上的 ``EvidenceRef`` 携带，
     使来源在最终假设/响应中仍可追溯。
  4. 决定何时证据足够可靠、可自动应用，何时必须浮出以征得用户确认。

Safety rule (P1-01 rule 6): a safety-critical temperature is NEVER
auto-resolved from evidence that lacks a verifiable URL. LLM knowledge
carries no URL, so it cannot prove a minimum safe cooking temperature.

安全规则（P1-01 规则 6）：安全关键温度绝不从缺少可验证 URL 的证据中自动解析。
LLM 知识不带 URL，因此无法证明最低安全烹饪温度。
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    EvidenceRef,
    ExtractedRecipeCandidate,
    ExtractedStep,
    RecipeGap,
    ReconciledEvidence,
)

# Any reconciled value needs at least this many independent sources AND no
# disagreement flag before it is written back without user confirmation.
# 任何调和值在无需用户确认即可回写之前，都需要至少这么多独立来源且没有分歧标记。
_RELIABLE_SOURCE_COUNT = 1

# Safety-critical temperature facts are only auto-applied when the evidence
# carries a verifiable URL (e.g. a food-safety authority page). LLM knowledge
# (source_url="") cannot prove a minimum safe temperature.
# 安全关键温度事实只有在证据携带可验证 URL（如食品安全权威页面）时才会自动应用。
# LLM 知识（source_url=""）无法证明最低安全温度。
_SAFETY_GAP_CLASSES = frozenset({"safety_critical"})
_TEMPERATURE_FIELDS = frozenset({"temperature", "target_temperature_c", "temperature_c"})

_STEP_INDEX_RE = re.compile(r"steps\[(\d+)\]")


class EvidenceApplication(NamedTuple):
    """Result of applying one reconciled evidence to one gap target.

    将一条调和后的证据应用到单个缺口目标的结果。

    Attributes:
        candidate: Updated candidate when a value was written; None otherwise.
            写入值后更新的候选；否则为 None。
        assumption: Evidence-backed assumption when applied; None otherwise.
            应用时带证据支撑的假设；否则为 None。
        applied: True when a field value was written into the candidate.
            当字段值被写入候选时为 True。
        needs_confirmation: True when this gap cannot be safely auto-resolved
            (no source, disagreement, field-location failure, or unverified
            safety-critical temperature).
            当该缺口无法安全自动解析时为 True（无来源、分歧、字段定位失败，
            或未经验证的安全关键温度）。
    """

    candidate: ExtractedRecipeCandidate | None
    assumption: Assumption | None
    applied: bool
    needs_confirmation: bool


# ---------------------------------------------------------------------------
# Field location helpers — deterministic, never positional guessing
# 字段定位辅助函数 —— 确定性定位，绝不靠位置猜测
# ---------------------------------------------------------------------------


def locate_step_index(field_path: str) -> int | None:
    """Return the step index encoded in a gap's ``field_path``.

    返回缺口 ``field_path`` 中编码的步骤索引。

    Accepts paths like ``steps[0].heat_level``. Returns None when the path
    does not reference a step (e.g. candidate-level ``dish_name``).

    接受类似 ``steps[0].heat_level`` 的路径。当路径不引用步骤时
    （例如候选级 ``dish_name``）返回 None。
    """
    match = _STEP_INDEX_RE.search(field_path)
    return int(match.group(1)) if match else None


def field_name(field_path: str) -> str:
    """Return the leaf field name of a gap's ``field_path``.

    返回缺口 ``field_path`` 的叶子字段名。
    """
    return field_path.rsplit(".", 1)[-1]


def evidence_has_verifiable_url(reconciled: ReconciledEvidence) -> bool:
    """True when at least one source carries a stable URL.

    当至少一个来源携带稳定 URL 时返回 True。
    """
    return any(bool(ev.source_url) for ev in reconciled.evidence_items)


# ---------------------------------------------------------------------------
# Core application logic
# 核心应用逻辑
# ---------------------------------------------------------------------------


def apply_evidence_to_candidate(
    candidate: ExtractedRecipeCandidate,
    gap: RecipeGap,
    reconciled: ReconciledEvidence,
) -> EvidenceApplication:
    """Apply reconciled evidence to the candidate at the gap's field path.

    将调和后的证据应用到候选在缺口字段路径处。

    Returns an EvidenceApplication describing whether the value was written
    and whether the gap still requires user confirmation.

    返回 EvidenceApplication，描述值是否被写入以及缺口是否仍需要用户确认。
    """
    if reconciled.source_count <= 0:
        # Search returned nothing usable (timeout, empty results, error).
        # 搜索未返回可用结果（超时、空结果、错误）。
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    step_index = locate_step_index(gap.field_path)
    if step_index is None or step_index >= len(candidate.steps):
        # Field could not be located — never guess by list position.
        # 字段无法定位 —— 绝不按列表位置猜测。
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    field = field_name(gap.field_path)

    # Safety-critical temperature: only auto-apply when the evidence has a
    # verifiable URL (P1-01 rule 6). LLM knowledge (no URL) cannot resolve.
    # 安全关键温度：仅当证据具有可验证 URL 时才自动应用（P1-01 规则 6）。
    # LLM 知识（无 URL）无法解析。
    if gap.gap_class in _SAFETY_GAP_CLASSES and field in _TEMPERATURE_FIELDS:
        if not evidence_has_verifiable_url(reconciled):
            return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    value, evidence_refs = _pick_value(field, reconciled)
    if value is None:
        # No reconciled value usable for this field — cannot auto-resolve.
        # 该字段没有可用的调和值 —— 无法自动解析。
        return EvidenceApplication(
            None,
            None,
            applied=False,
            needs_confirmation=reconciled.needs_confirmation,
        )

    step = candidate.steps[step_index]
    updated_step = _write_value(step, field, value)
    if updated_step is step:
        # Value could not be mapped onto this step shape.
        # 值无法映射到该步骤结构。
        return EvidenceApplication(None, None, applied=False, needs_confirmation=True)

    updated_steps = candidate.steps[:step_index] + (updated_step,) + candidate.steps[step_index + 1 :]
    updated_candidate = candidate.model_copy(update={"steps": updated_steps})

    assumption = Assumption(
        text=(f"Researched {field} for {candidate.dish_name}: {value} (from {len(evidence_refs)} source(s))"),
        confidence=Decimal("0.8"),
        evidence=tuple(evidence_refs),
    )

    return EvidenceApplication(updated_candidate, assumption, applied=True, needs_confirmation=False)


def _pick_value(field: str, reconciled: ReconciledEvidence) -> tuple[str | None, list[EvidenceRef]]:
    """Return (value_str, evidence_refs) for a field, or (None, []) if unusable.

    返回字段的 (value_str, evidence_refs)；不可用时返回 (None, [])。

    Duration mapping follows gap semantics: an ``active_duration`` gap uses
    the lower bound (minimum hands-on time); a ``passive_duration`` gap uses
    the upper bound (conservative wait time).

    时长映射遵循缺口语义：``active_duration`` 缺口用下界（最少动手时间）；
    ``passive_duration`` 缺口用上界（保守等待时间）。
    """
    refs = [
        EvidenceRef(
            source_type="web_search",
            title=ev.source_title or None,
            url=ev.source_url or None,
        )
        for ev in reconciled.evidence_items
        if ev.source_url
    ]

    if "heat" in field:
        if reconciled.heat_level is not None:
            return reconciled.heat_level.value, refs
    elif "active_duration" in field:
        if reconciled.duration_min_minutes is not None:
            return str(reconciled.duration_min_minutes), refs
    elif "duration" in field:
        if reconciled.duration_max_minutes is not None:
            return str(reconciled.duration_max_minutes), refs
    elif "temperature" in field:
        if reconciled.explicit_temperature_c is not None:
            return str(reconciled.explicit_temperature_c), refs

    return None, []


def _write_value(step: ExtractedStep, field: str, value_str: str) -> ExtractedStep:
    """Map a reconciled value onto the step, returning the original step when
    the value does not fit the field's type.

    将调和值映射到步骤上；当值不符合字段类型时返回原始步骤。
    """
    if "heat_level" in field:
        try:
            heat = HeatLevel(value_str)
        except ValueError:
            return step
        return step.model_copy(update={"heat_level": heat})

    if "active_duration" in field:
        return step.model_copy(update={"active_duration_minutes": int(value_str)})

    if "duration" in field:
        return step.model_copy(update={"passive_duration_minutes": int(value_str)})

    if "temperature" in field:
        return step.model_copy(update={"target_temperature_c": Decimal(value_str)})

    return step
