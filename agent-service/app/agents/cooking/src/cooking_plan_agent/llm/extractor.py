# =============================================================================
# LLM 菜谱提取器模块（llm/extractor）
# -----------------------------------------------------------------------------
# 实现 RecipeExtractor 协议，用本地 LLM（JSON 模式输出）把自由文本菜谱
# 转换为结构化的 ExtractedRecipeCandidate。当启用 LLM 时取代规则提取器，
# 同时保持与 workflow/context.py 的协议契约完全一致，工作流图无需改动。
# 兜底：若 LLM 调用失败或其输出未通过 schema 校验，则回退到规则提取器，
#       使管线优雅降级。
# 关键点：
#   - _to_candidate / _to_ingredient / _to_step 等映射函数做“防御式转换”，
#     容忍 LLM 缺失键 / 非法值，任何坏值都降级为安全默认值。
#   - PARSE_PROMPT_VERSION 由系统提示词哈希派生，提示词一改旧缓存键自动失效。
# =============================================================================

"""LLM-backed recipe extractor implementing the RecipeExtractor Protocol.

基于 LLM 的菜谱提取器，实现 RecipeExtractor 协议。

Converts free-form recipe text into a structured ExtractedRecipeCandidate
using a local LLM with JSON-mode output. This replaces the rule-based
extractor when LLM is enabled, while preserving the exact Protocol contract
(workflow/context.py) so the workflow graph does not change.

用本地 LLM（JSON 模式输出）把自由文本菜谱转为结构化 ExtractedRecipeCandidate。
启用 LLM 时取代规则提取器，同时保持与 workflow/context.py 的协议契约一致，
工作流图无需改动。

Fallback: if the LLM call fails or its output fails schema validation, the
rule-based extractor is used so the pipeline degrades gracefully.

兜底：若 LLM 调用失败或其输出未通过 schema 校验，则使用规则提取器，使管线优雅降级。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
)
from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.normalisation.names import clean_dish_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt — instructs the LLM to emit a JSON object matching
# ExtractedRecipeCandidate (snake_case fields). Bounded and deterministic.
# 提取提示词 —— 指示 LLM 输出与 ExtractedRecipeCandidate 匹配的 JSON 对象
# （snake_case 字段）。有界且确定性。
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a recipe structuring and completion assistant. Convert the "
    "user-provided cooking text into a practical, schedulable recipe. "
    "Respond with a SINGLE JSON object only — "
    "no prose, no markdown fences. The object must use exactly these fields:\n"
    '{"dish_name": string, "original_servings": number, "source_language": '
    '"zho"|"eng"|"und", "ingredients": [{"raw_text": string, "name": string, '
    '"quantity": number|null, "unit": string|null, "preparation": string|null, '
    '"extraction_source": "EXPLICIT"|"LLM_INFERRED", "confidence": number}], '
    '"steps": [{"instruction": string, "category": "general"|"heating"|'
    '"preparation"|"resting"|"mixing", "active_duration_minutes": number|null, '
    '"passive_duration_minutes": number|null, "heat_level": "NONE"|"LOW"|"MEDIUM"|'
    '"HIGH", "target_temperature_c": number|null, "resources_hint": [string], '
    '"extraction_source": "EXPLICIT"|"LLM_INFERRED", "confidence": number}], '
    '"inferred_fields": [string]}\n'
    "Rules: preserve every explicit fact. When operational details are omitted, "
    "use conservative culinary common sense to infer the values needed to execute "
    "and schedule the recipe: servings, ingredient quantities/units when necessary, "
    "step category, active/passive duration, heat level, target temperature, and "
    "equipment. Add every inferred candidate-level field path to inferred_fields. "
    "Set an ingredient or step extraction_source to LLM_INFERRED when any of its "
    "values were inferred, and give it a calibrated confidence from 0 to 1. "
    "For a step cooking raw animal protein, target_temperature_c means a conservative "
    "safe internal food temperature, not the oven or pan setting. For other baking, "
    "roasting, or frying steps it may represent the appliance or oil temperature. "
    "Use null only when no reasonable culinary inference is possible. Quantity must "
    "be a positive number when given. "
    "dish_name must be a SHORT conventional dish title only. Ignore webpage introductions, video references, "
    "recipe notes, nutrition text, and component headings such as seasoning mix, sauce, marinade, or topping. "
    "Those sections belong to the same finished dish. Infer the title only from explicit wording in the source "
    "(for example, 'This Jambalaya' means 'Jambalaya'); never copy a full introductory sentence. Strip quantities, units, "
    "parenthetical notes, and preparation instructions (e.g. 'Fresh Shrimp', not "
    "'Fresh shrimp (remove head, tail, and thread)')."
)

_ENGLISH_OUTPUT_RULE = (
    " Translate every user-visible text field into clear English before returning it. "
    "dish_name, ingredient raw_text/name/preparation, step instruction/category, and resource hints "
    "must be English even when the source is written in another language. Preserve quantities, units, "
    "temperatures, times, and proper nouns accurately. source_language must describe the original input language."
)
# ↑ 可选英文输出规则：把所有用户可见文本字段翻译成英文，但保留数量 / 单位 / 温度 / 时间 / 专有名词

# Stable cache tag (P1-06 rule 2): changing the prompt changes this digest, so
# cached parse artifacts keyed on the old prompt are never reused.
# 稳定缓存标签（P1-06 规则 2）：提示词一变，这个摘要就变，从而旧提示词对应的
# 缓存解析产物永不被复用。
PARSE_PROMPT_VERSION = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


class LLMRecipeExtractor:
    """通过本地 LLM 提取结构化菜谱候选。

    Extract structured recipe candidates via a local LLM.

    Implements the async extract() contract expected by the workflow
    (RecipeExtractor Protocol in workflow/context.py).

    实现工作流期望的异步 extract() 契约（workflow/context.py 中的 RecipeExtractor 协议）。
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        translate_to_english: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = client
        self._system_prompt = _SYSTEM_PROMPT + (_ENGLISH_OUTPUT_RULE if translate_to_english else "")
        # ↑ 根据 translate_to_english 决定是否追加英文输出规则
        self._timeout_seconds = timeout_seconds

    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        """用 LLM 把菜谱文本解析为结构化候选。

        Parse recipe text into a structured candidate using the LLM.

        Args:
            source_text: Raw recipe text (preprocessed).
                source_text：原始菜谱文本（已预处理）。

        Returns:
            An ExtractedRecipeCandidate with extraction_source="LLM" on
            success. On LLM failure, falls back to the rule-based extractor
            so the pipeline degrades gracefully (source="RULE_BASED").
            成功时返回 extraction_source="LLM" 的 ExtractedRecipeCandidate。
            LLM 失败时回退到规则提取器，使管线优雅降级（source="RULE_BASED"）。
        """
        try:
            data = await asyncio.wait_for(
                self._client.chat_json(
                    [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": source_text},
                    ]
                ),
                timeout=self._timeout_seconds,
            )
            return self._to_candidate(source_text, data)
        except (TimeoutError, LLMError, ValidationError, TypeError, ValueError):
            # Degrade to rule-based parsing — never block the workflow.
            # 降级到规则解析 —— 绝不阻塞工作流。
            logger.warning("LLM extraction failed — falling back to rule-based")
            return await self._rule_based_extract(source_text)

    # ------------------------------------------------------------------
    # Fallback: built-in rule-based extractor (same path as llm disabled)
    # 兜底：内置规则提取器（与 LLM 禁用时同路径）
    # ------------------------------------------------------------------

    @staticmethod
    async def _rule_based_extract(source_text: str) -> ExtractedRecipeCandidate:
        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor

        return await RuleExtractor().extract(source_text)

    # ------------------------------------------------------------------
    # Mapping LLM JSON → domain model (defensive: tolerate missing keys)
    # LLM JSON → 领域模型映射（防御式：容忍缺失键）
    # ------------------------------------------------------------------

    @staticmethod
    def _to_candidate(source_text: str, data: dict[str, Any]) -> ExtractedRecipeCandidate:
        ingredients = tuple(
            LLMRecipeExtractor._to_ingredient(item) for item in data.get("ingredients") or [] if isinstance(item, dict)
        )
        steps = tuple(
            LLMRecipeExtractor._to_step(i, item)
            for i, item in enumerate(data.get("steps") or [], start=1)
            if isinstance(item, dict)
        )
        dish_name = clean_dish_name(str(data.get("dish_name") or "Untitled Recipe"))[:80]
        try:
            servings = Decimal(str(data.get("original_servings") or 2))
        except (InvalidOperation, TypeError, ValueError):
            servings = Decimal(2)

        return ExtractedRecipeCandidate(
            recipe_id=f"recipe_{dish_name[:40]}",
            dish_name=dish_name,
            original_servings=servings,
            source_language=str(data.get("source_language") or "und"),
            ingredients=ingredients,
            steps=steps,
            extraction_source="LLM",
            inferred_fields=tuple(
                str(field).strip()
                for field in (data.get("inferred_fields") or [])
                if isinstance(field, str) and field.strip()
            ),
        )

    @staticmethod
    def _to_ingredient(item: dict[str, Any]) -> ExtractedIngredient:
        raw_text = str(item.get("raw_text") or "").strip()
        name = str(item.get("name") or "").strip()
        quantity_raw = item.get("quantity")
        quantity = None
        try:
            if quantity_raw is not None and str(quantity_raw).strip():
                q = Decimal(str(quantity_raw))
                if q > 0:
                    quantity = q
        except (InvalidOperation, ValueError, TypeError):
            quantity = None
        unit = str(item.get("unit") or "").strip() or None
        prep = str(item.get("preparation") or "").strip() or None
        return ExtractedIngredient(
            raw_text=raw_text or name,
            name=name or "unknown",
            quantity=quantity,
            unit=unit,
            preparation=prep,
            extraction_source=LLMRecipeExtractor._to_extraction_source(item.get("extraction_source")),
            confidence=LLMRecipeExtractor._to_confidence(item.get("confidence")),
        )

    @staticmethod
    def _to_step(index: int, item: dict[str, Any]) -> ExtractedStep:
        instruction = str(item.get("instruction") or "").strip()
        heat = str(item.get("heat_level") or "NONE").upper()
        if heat not in {"NONE", "LOW", "MEDIUM", "HIGH"}:
            heat = "NONE"
        return ExtractedStep(
            step_number=index,
            instruction=instruction,
            category=str(item.get("category") or "general"),
            active_duration_minutes=LLMRecipeExtractor._to_int(item.get("active_duration_minutes")),
            passive_duration_minutes=LLMRecipeExtractor._to_int(item.get("passive_duration_minutes")),
            heat_level=HeatLevel(heat),
            target_temperature_c=LLMRecipeExtractor._to_decimal(item.get("target_temperature_c")),
            resources_hint=tuple(str(r) for r in (item.get("resources_hint") or []) if isinstance(r, str)),
            extraction_source=LLMRecipeExtractor._to_extraction_source(item.get("extraction_source")),
            confidence=LLMRecipeExtractor._to_confidence(item.get("confidence")),
        )

    @staticmethod
    def _to_extraction_source(value: Any) -> str:
        """把提取来源归一化为 EXPLICIT / LLM_INFERRED 二值。"""
        return "LLM_INFERRED" if str(value or "").upper() == "LLM_INFERRED" else "EXPLICIT"

    @staticmethod
    def _to_confidence(value: Any) -> Decimal:
        """把置信度转换为 [0, 1] 区间内的 Decimal，坏值回退 0.8。"""
        try:
            confidence = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0.8")
        return min(Decimal(1), max(Decimal(0), confidence))

    @staticmethod
    def _to_int(value: Any) -> int | None:
        """安全转换正整数；非法或非正值返回 None。"""
        try:
            if value is not None:
                v = int(value)
                return v if v > 0 else None
        except (TypeError, ValueError):
            pass
        return None

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        """安全转换正 Decimal；非法或非正值返回 None。"""
        try:
            if value is not None:
                d = Decimal(str(value))
                return d if d > 0 else None
        except (InvalidOperation, ValueError, TypeError):
            pass
        return None
