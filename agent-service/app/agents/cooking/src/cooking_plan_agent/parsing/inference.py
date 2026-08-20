# =============================================================================
# 本地烹饪知识推断模块（parsing/inference）
# -----------------------------------------------------------------------------
# 实现手册 4.11：用确定性、基于规则的烹饪知识填补检测到的菜谱缺口。
# 无 LLM 依赖 —— 这是联网研究之前的“本地推断兜底”。
# 设计原则：
#   - 只有推断规则，绝不调用 provider
#   - 所有决策确定且可解释
#   - 置信度反映推断的可靠性
#   - 安全关键缺口绝不以高置信度推断
# 核心函数：
#   - infer_local                 ：对候选的缺口集应用本地知识填充
#   - infer_gap                   ：按 field_path 分派到具体推断（火力/时长/温度/资源）
#   - infer_deterministic_default ：非安全字段的最后兜底默认值
#   - merge_inference             ：把填充结果回写进候选
# =============================================================================

"""Local cooking knowledge inference — handbook 4.11.

本地烹饪知识推断 —— 手册 4.11。

Fills detected recipe gaps using deterministic, rule-based cooking knowledge.
No LLM dependency — this is the local inference fallback before web research.

用确定性、基于规则的烹饪知识填补检测到的菜谱缺口。
无 LLM 依赖 —— 这是联网研究之前的本地推断兜底。

Design principles:
  - Only inference rules, NO provider calls
  - All decisions are deterministic and explainable
  - Confidence scores reflect the reliability of the inference
  - Safety-critical gaps are NEVER inferred with high confidence

设计原则：
  - 只有推断规则，绝不调用 provider
  - 所有决策确定且可解释
  - 置信度反映推断的可靠性
  - 安全关键缺口绝不以高置信度推断
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedRecipeCandidate,
    RecipeGap,
)
from cooking_plan_agent.parsing.gaps import GapClass

# =============================================================================
# InferenceResult — output of the local inference step
# 推断结果 —— 本地推断步骤的输出
# =============================================================================


class InferenceResult(NamedTuple):
    """本地推断填充菜谱缺口的结果。

    Result of applying local inference to fill recipe gaps.

    Attributes:
        filled_gaps: Gaps that were successfully filled by local rules.
            filled_gaps：被本地规则成功填充的缺口。
        unresolved_gaps: Critical gaps that local inference could not fill.
            unresolved_gaps：本地推断无法填充的关键缺口。
        assumptions: Explanatory assumptions made during inference.
            assumptions：推断过程中做出的解释性假设。
    """

    filled_gaps: tuple[RecipeGap, ...]
    unresolved_gaps: tuple[RecipeGap, ...]
    assumptions: tuple[Assumption, ...]


# =============================================================================
# Heat level inference rules
# 火力档位推断规则
# =============================================================================

# Technique → default heat level mapping
# These are standard culinary conventions, not guesses.
# 技法 → 默认火力档位映射。这些是标准烹饪惯例，不是猜测。
_TECHNIQUE_HEAT_MAP: dict[str, tuple[HeatLevel, str]] = {
    "stir_fry": (HeatLevel.HIGH, "stir-frying typically requires high heat"),
    "deep_fry": (HeatLevel.HIGH, "deep-frying requires high heat for proper oil temperature"),
    "boil": (HeatLevel.HIGH, "boiling water requires high heat to reach 100°C"),
    "simmer": (HeatLevel.LOW, "simmering is done at low heat, just below boiling"),
    "steam": (HeatLevel.HIGH, "steaming requires high heat to produce steam"),
    "bake": (HeatLevel.MEDIUM, "baking typically uses medium heat (180°C/350°F)"),
    "roast": (HeatLevel.HIGH, "roasting typically requires high heat (200-220°C)"),
    "grill": (HeatLevel.HIGH, "grilling requires high heat for proper searing"),
    "sauté": (HeatLevel.MEDIUM, "sautéing is done at medium to medium-high heat"),
    "sear": (HeatLevel.HIGH, "searing requires high heat for Maillard reaction"),
    "braise": (HeatLevel.LOW, "braising is done at low heat for tender results"),
    "poach": (HeatLevel.LOW, "poaching requires low, gentle heat"),
    "heat": (HeatLevel.HIGH, "heating oil/pan typically uses high heat initially"),
}

# =============================================================================
# Duration inference rules
# 时长推断规则
# =============================================================================

# Technique → default duration range in minutes
# These are conservative defaults based on common cooking practice.
# 技法 → 默认时长区间（分钟）。这些是基于常见烹饪实践的保守默认值。
_TECHNIQUE_DURATION_MAP: dict[str, tuple[int, int, str]] = {
    "stir_fry": (3, 8, "stir-frying typically takes 3-8 minutes"),
    "deep_fry": (3, 10, "deep-frying typically takes 3-10 minutes depending on food size"),
    "boil": (8, 20, "boiling typically takes 8-20 minutes for most ingredients"),
    "simmer": (20, 60, "simmering typically takes 20-60 minutes"),
    "steam": (8, 15, "steaming typically takes 8-15 minutes"),
    "bake": (20, 45, "baking typically takes 20-45 minutes"),
    "roast": (30, 90, "roasting typically takes 30-90 minutes"),
    "grill": (5, 15, "grilling typically takes 5-15 minutes per side"),
    "sauté": (3, 10, "sautéing typically takes 3-10 minutes"),
    "sear": (2, 5, "searing typically takes 2-5 minutes per side"),
    "braise": (60, 180, "braising is a slow cooking method (1-3 hours)"),
    "poach": (5, 15, "poaching typically takes 5-15 minutes"),
}

# =============================================================================
# Temperature inference rules
# 温度推断规则
# =============================================================================

# Technique → typical target temperature in Celsius
# 技法 → 典型目标温度（摄氏度）
_TECHNIQUE_TEMPERATURE_MAP: dict[str, tuple[Decimal, str]] = {
    "bake": (Decimal(180), "standard baking temperature is 180°C (350°F)"),
    "roast": (Decimal(200), "standard roasting temperature is 200°C (400°F)"),
    "deep_fry": (Decimal(180), "standard deep-frying oil temperature is 180°C"),
    "grill": (Decimal(220), "grilling surface temperature ~220°C"),
}

# =============================================================================
# Resource inference rules
# 资源推断规则
# =============================================================================

# Technique → default required resources
# 技法 → 默认所需资源
_TECHNIQUE_RESOURCES: dict[str, tuple[str, ...]] = {
    "stir_fry": ("stove", "wok", "spatula"),
    "deep_fry": ("stove", "pot"),
    "boil": ("stove", "pot"),
    "simmer": ("stove", "pot"),
    "steam": ("stove", "steamer"),
    "bake": ("oven",),
    "roast": ("oven",),
    "grill": ("stove", "pan"),
    "sauté": ("stove", "pan", "spatula"),
    "sear": ("stove", "pan"),
    "braise": ("stove", "pot"),
    "poach": ("stove", "pot"),
    "marinate": ("mixing_bowl",),
}

# =============================================================================
# Public API
# 公共 API
# =============================================================================


def infer_local(
    candidate: ExtractedRecipeCandidate,
    gaps: tuple[RecipeGap, ...],
) -> InferenceResult:
    """应用本地烹饪知识填充检测到的缺口。

    Apply local cooking knowledge to fill detected gaps.

    Processes each gap and attempts to fill it with deterministic rules.
    Critical and safety_critical gaps that can be filled by local knowledge
    are resolved. Safety-critical gaps that CANNOT be reliably inferred
    remain unresolved and must be routed to confirmation or web research.

    逐个处理缺口，尝试用确定性规则填充。本地知识能填充的 critical / safety_critical
    缺口被解决；无法可靠推断的 safety_critical 缺口保持未解决，须路由到确认或联网研究。

    Args:
        candidate: The extracted recipe candidate with gaps.
            candidate：带缺口的提取菜谱候选。
        gaps: Detected gaps from find_recipe_gaps().
            gaps：来自 find_recipe_gaps() 的检测缺口。

    Returns:
        InferenceResult with filled/unresolved gaps and assumptions made.
        含已填充 / 未解决缺口与所做假设的 InferenceResult。
    """
    filled: list[RecipeGap] = []
    unresolved: list[RecipeGap] = []
    assumptions: list[Assumption] = []

    # Detect the cooking technique for the recipe (from step analysis)
    # 从步骤分析检测菜谱的主要烹饪技法
    technique = _detect_primary_technique(candidate)

    for gap in gaps:
        # Only attempt to fill critical and safety_critical gaps
        # 只尝试填充 critical 与 safety_critical 缺口
        if gap.gap_class not in (GapClass.CRITICAL, GapClass.SAFETY_CRITICAL):
            unresolved.append(gap)
            continue

        result = infer_gap(candidate, gap, technique)
        if result is not None:
            filled.append(result[0])
            assumptions.append(result[1])
        else:
            unresolved.append(gap)

    return InferenceResult(
        filled_gaps=tuple(filled),
        unresolved_gaps=tuple(unresolved),
        assumptions=tuple(assumptions),
    )


def infer_gap(
    candidate: ExtractedRecipeCandidate,
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """用本地烹饪知识填充单个缺口。

    Fill a single gap using local cooking knowledge.

    Pure dispatch on ``gap.field_path``. Returns None when the gap cannot be
    inferred locally (e.g. a safety-critical temperature), leaving the caller
    to decide whether to keep it unresolved.

    纯按 gap.field_path 分派。当缺口无法本地推断（如安全关键温度）时返回 None，
    由调用方决定是否保持未解决。

    ``technique`` is the recipe's primary cooking technique, detected once by
    the caller via ``_detect_primary_technique`` so the detection cost is not
    repeated per gap.

    technique 是菜谱的主要烹饪技法，由调用方通过 _detect_primary_technique 检测一次，
    避免每个缺口重复检测。
    """
    if "heat_level" in gap.field_path:
        return _infer_heat(gap, candidate, technique)
    if "duration" in gap.field_path.lower():
        return _infer_duration(gap, technique)
    if "temperature" in gap.field_path.lower():
        if gap.gap_class == GapClass.SAFETY_CRITICAL:
            # NEVER infer safety-critical temperatures locally
            # 绝不在本地推断安全关键温度
            return None
        return _infer_temperature(gap, technique)
    if "resource" in gap.field_path.lower():
        return _infer_resources(gap, technique)
    return None


def infer_deterministic_default(
    candidate: ExtractedRecipeCandidate,
    gap: RecipeGap,
) -> tuple[RecipeGap, Assumption] | None:
    """为非安全字段返回一个保守的最后兜底值。

    Return a conservative last-resort value for a non-safety field.

    This is deliberately used only after model/research inference produced
    no usable value. Safety-critical gaps are never defaulted.

    刻意仅在模型 / 研究推断未产生可用值之后使用。安全关键缺口绝不默认填充。
    """
    if gap.gap_class == GapClass.SAFETY_CRITICAL:
        return None
    step_index = _extract_step_index(gap.field_path)
    if step_index is None or step_index >= len(candidate.steps):
        return None

    instruction = candidate.steps[step_index].instruction.lower()
    field = gap.field_path.rsplit(".", 1)[-1].lower()
    value: str | None = None
    reason = "conservative general cooking default"

    if "duration" in field:
        duration_rules = (
            (("quickly", "briefly", "coat", "remove", "garnish", "serve", "快速", "翻炒", "捞出", "装盘"), 2),
            (("simmer", "braise", "stew", "tender", "炖", "焖", "红烧"), 45),
            (("bake", "roast", "烤", "烘"), 30),
            (("boil", "steam", "煮", "蒸"), 10),
            (("fry", "sauté", "saute", "stir", "炒", "煎", "炸"), 5),
        )
        value = str(
            next((minutes for terms, minutes in duration_rules if any(term in instruction for term in terms)), 5)
        )
        reason = "instruction-keyword duration default"
    elif "heat_level" in field:
        if any(term in instruction for term in ("simmer", "braise", "stew", "low heat", "炖", "焖", "小火")):
            value = HeatLevel.LOW.value
        elif any(
            term in instruction for term in ("boil", "fry", "sear", "stir", "high heat", "煮沸", "炸", "爆炒", "大火")
        ):
            value = HeatLevel.HIGH.value
        else:
            value = HeatLevel.MEDIUM.value
        reason = "instruction-keyword heat default"
    elif "temperature" in field:
        value = "180"
        reason = "standard non-safety oven temperature default"

    if value is None:
        return None
    confidence = Decimal("0.4")
    return (
        gap.model_copy(update={"current_value": value, "confidence": confidence}),
        Assumption(
            text=f"Used {value} for {field} after model inference returned no usable value ({reason})",
            confidence=confidence,
        ),
    )


# =============================================================================
# Merge inference results back into the candidate
# 把推断结果回写进候选
# =============================================================================


def merge_inference(
    candidate: ExtractedRecipeCandidate,
    result: InferenceResult,
) -> ExtractedRecipeCandidate:
    """把已填充的缺口应用回候选，生成更新后的候选。

    Apply filled gaps back into the candidate, producing an updated candidate.

    Only fields that were successfully filled by local inference are applied.
    Unresolved gaps are left as-is — downstream routing handles them.

    仅应用本地推断成功填充的字段。未解决缺口保持原样 —— 由下游路由处理。

    Args:
        candidate: The original extracted candidate.
            candidate：原始提取候选。
        result: InferenceResult from infer_local().
            result：来自 infer_local() 的 InferenceResult。

    Returns:
        Updated ExtractedRecipeCandidate with inferred values applied.
        已应用推断值的更新 ExtractedRecipeCandidate。
    """
    updated_steps = list(candidate.steps)

    for gap in result.filled_gaps:
        step_idx = _extract_step_index(gap.field_path)
        if step_idx is not None and step_idx < len(updated_steps):
            step = updated_steps[step_idx]

            # Apply the inferred values  应用推断值
            field = gap.field_path.rsplit(".", 1)[-1] if "." in gap.field_path else gap.field_path

            if "heat_level" in field and gap.current_value:
                try:
                    heat = HeatLevel(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "heat_level": heat,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "passive_duration" in field and gap.current_value:
                try:
                    minutes = int(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "passive_duration_minutes": minutes,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "active_duration" in field and gap.current_value:
                try:
                    minutes = int(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "active_duration_minutes": minutes,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except ValueError:
                    pass

            elif "temperature" in field and gap.current_value:
                try:
                    temp = Decimal(gap.current_value)
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "target_temperature_c": temp,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )
                except (ValueError, TypeError):
                    pass

            elif "resource" in field and gap.current_value:
                # Resource values are comma-separated  资源值以逗号分隔
                resources = tuple(r.strip() for r in gap.current_value.split(",") if r.strip())
                if resources:
                    updated_steps[step_idx] = step.model_copy(
                        update={
                            "resources_hint": resources,
                            "extraction_source": "RULE_INFERRED",
                            "confidence": min(step.confidence, gap.confidence),
                        }
                    )

    return candidate.model_copy(update={"steps": tuple(updated_steps)})


# Technique keyword → technique name mapping for Chinese text
# Used by _detect_primary_technique to bridge extractor output and inference maps.
# 中文文本的技法关键词 → 技法名映射。用于 _detect_primary_technique 桥接提取器输出与推断映射。
_CHINESE_TECHNIQUE_KEYWORDS: dict[str, str] = {
    "焯水": "boil",
    "煮": "boil",
    "烧开": "boil",
    "煮沸": "boil",
    "炒": "stir_fry",
    "翻炒": "stir_fry",
    "爆炒": "stir_fry",
    "大火炒": "stir_fry",
    "煸炒": "stir_fry",
    "煎": "sauté",
    "炖": "simmer",
    "焖": "simmer",
    "煲": "simmer",
    "慢炖": "simmer",
    "小火炖": "simmer",
    "蒸": "steam",
    "炸": "deep_fry",
    "油炸": "deep_fry",
    "烤": "bake",
    "烘烤": "bake",
    "烤箱": "bake",
    "腌制": "marinate",
    "腌": "marinate",
    "烧": "braise",
    "红烧": "braise",
}

# Common English spellings whose surface form differs from the canonical
# inference-map key. Keep these aliases next to the language-specific lookup
# so extraction and local gap filling agree on the same technique.
# 表面形式与规范推断映射键不同的常见英文拼写。把这些别名放在语言特定查找旁边，
# 使提取与本地缺口填充对同一技法达成一致。
_ENGLISH_TECHNIQUE_ALIASES: dict[str, str] = {
    "pan-fry": "sauté",
    "pan fry": "sauté",
    "pan-fried": "sauté",
    "pan fried": "sauté",
    "saute": "sauté",
    "sauteed": "sauté",
    "sauté": "sauté",
    "sautéed": "sauté",
}


# =============================================================================
# Internal inference helpers
# 内部推断辅助函数
# =============================================================================


def _detect_primary_technique(candidate: ExtractedRecipeCandidate) -> str:
    """从菜谱步骤检测主要烹饪技法。

    Detect the primary cooking technique from the recipe's steps.

    Searches step instruction text for both English and Chinese technique
    keywords. Returns the most frequently mentioned technique, or 'general'
    if none are found.

    在步骤指令文本中搜索中英文技法关键词。返回出现最频繁的技法，若未找到则返回 'general'。
    """
    heating_steps = [s for s in candidate.steps if s.category == "heating"]
    if not heating_steps:
        return "general"

    from collections import Counter

    technique_counts: Counter[str] = Counter()

    for step in heating_steps:
        instruction = step.instruction.lower()
        # Check English technique names  检查英文技法名
        for tech in _TECHNIQUE_HEAT_MAP:
            term = tech.replace("_", " ")
            if term in instruction or tech.replace("_", "-") in instruction:
                technique_counts[tech] += 1

        for keyword, tech in _ENGLISH_TECHNIQUE_ALIASES.items():
            if keyword in instruction:
                technique_counts[tech] += 1

        # Check Chinese technique keywords  检查中文技法关键词
        for zh_keyword, tech in _CHINESE_TECHNIQUE_KEYWORDS.items():
            if zh_keyword in step.instruction:
                technique_counts[tech] += 1

    if technique_counts:
        return technique_counts.most_common(1)[0][0]

    return "general"


def _infer_heat(
    gap: RecipeGap,
    candidate: ExtractedRecipeCandidate,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """从技法推断缺失的火力档位。"""
    if technique not in _TECHNIQUE_HEAT_MAP:
        return None

    heat, explanation = _TECHNIQUE_HEAT_MAP[technique]
    confidence = Decimal("0.7")  # Moderate — technique-level inference  中等 —— 技法级推断

    filled_gap = gap.model_copy(
        update={
            "current_value": heat.value,
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming {heat.value} heat for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_duration(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """从技法推断缺失的时长。"""
    if technique not in _TECHNIQUE_DURATION_MAP:
        return None

    dur_min, dur_max, explanation = _TECHNIQUE_DURATION_MAP[technique]
    # Use midpoint as the inferred passive duration
    # 用区间中点作为推断的被动时长
    inferred = (dur_min + dur_max) // 2
    confidence = Decimal("0.5")  # Low confidence — duration varies a lot  低置信度 —— 时长差异大

    filled_gap = gap.model_copy(
        update={
            "current_value": str(inferred),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming ~{inferred} minutes for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_temperature(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """从技法推断缺失的温度。"""
    if technique not in _TECHNIQUE_TEMPERATURE_MAP:
        return None

    temp, explanation = _TECHNIQUE_TEMPERATURE_MAP[technique]
    confidence = Decimal("0.6")

    filled_gap = gap.model_copy(
        update={
            "current_value": str(temp),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming {temp}°C for {technique}: {explanation}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _infer_resources(
    gap: RecipeGap,
    technique: str,
) -> tuple[RecipeGap, Assumption] | None:
    """从技法推断缺失的资源提示。"""
    if technique not in _TECHNIQUE_RESOURCES:
        return None

    resources = _TECHNIQUE_RESOURCES[technique]
    confidence = Decimal("0.75")

    filled_gap = gap.model_copy(
        update={
            "current_value": ", ".join(resources),
            "confidence": confidence,
        }
    )
    assumption = Assumption(
        text=f"Assuming required equipment for {technique}: {', '.join(resources)}",
        confidence=confidence,
    )
    return filled_gap, assumption


def _extract_step_index(field_path: str) -> int | None:
    """从 'steps[2].heat_level' 这类字段路径提取步骤索引。"""
    import re

    match = re.search(r"steps\[(\d+)\]", field_path)
    if match:
        return int(match.group(1))
    return None
