# =============================================================================
# LLM 知识研究员模块（llm/researcher）
# -----------------------------------------------------------------------------
# 实现 RecipeResearcher 协议，用本地 LLM 的“烹饪知识”填补菜谱缺口，
# 而非联网搜索。工作流节点（research_missing_node）调用 research()，
# 并把任何失败视为 needs_confirmation —— 绝不当作不安全猜测。
# 证据结果携带 source_type="LLM_KNOWLEDGE"，使下游追踪能区分
# “模型推导事实”与“联网来源事实”。
# 关键点：
#   - 提示词“有界”：只回答特定缺口，绝不臆造完整菜谱或以虚假自信给出安全关键值；
#   - resolve_gap 走“更丰富”路径，保留模型的数值，让工作流直接回写菜谱，
#     而非让用户填写内部调度字段。
# =============================================================================

"""LLM-backed knowledge researcher — fills recipe gaps from model knowledge.

基于 LLM 的知识研究员 —— 用模型知识填补菜谱缺口。

Implements the RecipeResearcher Protocol (workflow/context.py) using the
local LLM's culinary knowledge instead of web search. The workflow node
(research_missing_node) calls research() and treats any failure as
needs_confirmation — never an unsafe guess.

用本地 LLM 的烹饪知识（而非联网搜索）实现 RecipeResearcher 协议。
工作流节点 research_missing_node 调用 research()，并把任何失败视为
needs_confirmation —— 绝不当作不安全猜测。

Evidence results carry source_type="LLM_KNOWLEDGE" so downstream tracing
can distinguish model-derived facts from web-sourced ones.

证据结果携带 source_type="LLM_KNOWLEDGE"，使下游追踪能区分模型推导事实与联网来源事实。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    CookingEvidence,
    EvidenceQuery,
    EvidenceResult,
    RecipeGap,
    ReconciledEvidence,
)
from cooking_plan_agent.llm.client import LLMClient
from cooking_plan_agent.research.query_builder import build_minimal_query
from cooking_plan_agent.research.reconciler import reconcile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Research prompt — bounded: answer only the specific gap, never invent
# a full recipe or safety-critical values with false confidence.
# 研究提示词 —— 有界：只回答特定缺口，绝不以虚假自信臆造完整菜谱或安全关键值。
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a culinary knowledge assistant. Answer a focused cooking question "
    "with a JSON object only (no prose, no markdown):\n"
    '{"facts": [{"source_title": string, "fact": string, "value": number|string, '
    '"unit": string|null, "confidence": number between 0 and 1}]}\n'
    "Rules: give exactly one best conservative estimate when a reasonable "
    "culinary inference is possible; confidence must reflect how standard the "
    "answer is; if the answer is safety-critical and uncertain, use "
    "confidence below 0.5; never fabricate sources or precise figures you "
    "cannot support. For durations, value must be a positive number and unit "
    "must be minutes. For heat, value must be LOW, MEDIUM, or HIGH. For "
    "temperature, value must be Celsius."
)
# ↑ 提示词：只输出一个保守估计，置信度反映标准程度；安全关键且不确定时置信度须 < 0.5


class LLMKnowledgeResearcher:
    """用本地 LLM 烹饪知识回答菜谱缺口查询。"""

    # Web research is capped to two external queries per dish. Local model
    # inference may resolve every operational gap because it does not fan out
    # to third-party search providers or expose recipe context externally.
    # 联网研究每道菜最多两次外部查询。本地模型推理可解决每个操作缺口，
    # 因为它不会 fan-out 到第三方搜索提供方，也不会把菜谱上下文暴露到外部。
    max_gap_queries: int | None = None

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def research(self, query: EvidenceQuery) -> list[EvidenceResult]:
        """用 LLM 知识回答一个结构化证据查询。

        Answer a structured evidence query using LLM knowledge.

        Args:
            query: Structured question (heat/duration/temperature gap).
                query：结构化问题（火力 / 时长 / 温度缺口）。

        Returns:
            One or more EvidenceResult items, each tagged
            source_type="LLM_KNOWLEDGE".
            一个或多个 EvidenceResult 条目，每个都打上 source_type="LLM_KNOWLEDGE"。

        Raises:
            LLMError: If the LLM call fails (workflow treats as confirmation).
            LLMError：LLM 调用失败时抛出（工作流视作需确认）。
        """
        user_prompt = f"Query: {query.query_text}\nContext dish: {query.recipe_context}\nGap type: {query.gap_type}"
        data = await self._client.chat_json(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        results: list[EvidenceResult] = []
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            results.append(self._to_result(query, item))
        return results

    async def resolve_gap(self, gap: RecipeGap, dish_name: str = "") -> ReconciledEvidence:
        """把单个操作缺口推断为结构化烹饪证据。

        Infer one operational recipe gap as structured cooking evidence.

        The generic ``research()`` contract returns display-oriented strings.
        This richer path preserves the model's numeric value so the workflow
        can write it back into the recipe instead of asking the user to fill
        internal scheduling fields.

        通用 ``research()`` 契约返回面向展示的字符串。本“更丰富”路径保留模型的数值，
        使工作流能把它回写进菜谱，而非让用户填写内部调度字段。
        """
        query = EvidenceQuery(
            query_text=build_minimal_query(gap, dish_name),
            gap_type=gap.gap_class,
            recipe_context=dish_name,
            target_fields=(gap.field_path,),
        )
        data = await self._chat_facts(query)
        evidence = tuple(
            item
            for fact in (data.get("facts") or [])[:1]
            if isinstance(fact, dict) and (item := self._to_cooking_evidence(gap, fact)) is not None
        )
        return reconcile(evidence)

    # ------------------------------------------------------------------
    # Mapping LLM JSON → EvidenceResult
    # LLM JSON → EvidenceResult 映射
    # ------------------------------------------------------------------

    @staticmethod
    def _to_result(query: EvidenceQuery, item: dict[str, Any]) -> EvidenceResult:
        title = str(item.get("source_title") or "Local culinary knowledge").strip()
        fact = str(item.get("fact") or "").strip()
        value = item.get("value")
        unit = str(item.get("unit") or "").strip() or None
        try:
            confidence = Decimal(str(item.get("confidence") or "0.5"))
            confidence = min(max(confidence, Decimal(0)), Decimal(1))
        except (ValueError, TypeError):
            confidence = Decimal("0.5")

        return EvidenceResult(
            source_title=title,
            source_url="",  # LLM knowledge has no URL — traced via source_type
            # ↑ LLM 知识没有 URL —— 通过 source_type 追踪
            snippet=fact,
            confidence=confidence,
            extracted_fact=fact,
            fact_type=query.gap_type,
            fact_value=f"{value}{unit or ''}",
        )

    async def _chat_facts(self, query: EvidenceQuery) -> dict[str, Any]:
        user_prompt = (
            f"Query: {query.query_text}\nContext dish: {query.recipe_context}"
            f"\nGap type: {query.gap_type}\nTarget field: {', '.join(query.target_fields)}"
        )
        return await self._client.chat_json(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

    @staticmethod
    def _to_cooking_evidence(gap: RecipeGap, item: dict[str, Any]) -> CookingEvidence | None:
        """把 LLM 事实转换为 CookingEvidence（按缺口字段类型分支，非法值返回 None）。"""
        field = gap.field_path.lower()
        raw_value = item.get("value")
        fact = str(item.get("fact") or "LLM culinary estimate").strip()
        heat_level: HeatLevel | None = None
        duration_minutes: int | None = None
        temperature_c: Decimal | None = None
        try:
            if "duration" in field:
                minutes = int(Decimal(str(raw_value)))
                if minutes <= 0:
                    return None
                duration_minutes = minutes
            elif "heat" in field:
                heat = HeatLevel(str(raw_value).strip().upper())
                if heat == HeatLevel.NONE:
                    return None
                heat_level = heat
            elif "temperature" in field:
                temperature = Decimal(str(raw_value))
                if temperature <= 0:
                    return None
                temperature_c = temperature
            else:
                return None
        except (ArithmeticError, TypeError, ValueError):
            return None

        return CookingEvidence(
            operation=gap.description[:120] or "cooking",
            source_url="",
            source_title="LLM culinary inference",
            source_excerpt=fact[:500],
            heat_level=heat_level,
            duration_min_minutes=duration_minutes,
            duration_max_minutes=duration_minutes,
            explicit_temperature_c=temperature_c,
        )
