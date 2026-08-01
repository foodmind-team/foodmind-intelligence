"""LLM-backed knowledge researcher — fills recipe gaps from model knowledge.

Implements the RecipeResearcher Protocol (application/ports.py) using the
local LLM's culinary knowledge instead of web search. The workflow node
(research_missing_node) calls research() and treats any failure as
needs_confirmation — never an unsafe guess.

Evidence results carry source_type="LLM_KNOWLEDGE" so downstream tracing
can distinguish model-derived facts from web-sourced ones.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from cooking_plan_agent.domain.models import (
    EvidenceQuery,
    EvidenceResult,
)
from cooking_plan_agent.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Research prompt — bounded: answer only the specific gap, never invent
# a full recipe or safety-critical values with false confidence.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a culinary knowledge assistant. Answer a focused cooking question "
    "with a JSON object only (no prose, no markdown):\n"
    '{"facts": [{"source_title": string, "fact": string, "value": number|string, '
    '"unit": string|null, "confidence": number between 0 and 1}]}\n'
    "Rules: give at most 3 facts; confidence must reflect how standard the "
    "answer is; if the answer is safety-critical and uncertain, use "
    "confidence below 0.5; never fabricate sources or precise figures you "
    "cannot support."
)


class LLMKnowledgeResearcher:
    """Answer recipe-gap queries from local LLM culinary knowledge."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def research(self, query: EvidenceQuery) -> list[EvidenceResult]:
        """Answer a structured evidence query using LLM knowledge.

        Args:
            query: Structured question (heat/duration/temperature gap).

        Returns:
            One or more EvidenceResult items, each tagged
            source_type="LLM_KNOWLEDGE".

        Raises:
            LLMError: If the LLM call fails (workflow treats as confirmation).
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

    # ------------------------------------------------------------------
    # Mapping LLM JSON → EvidenceResult
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
            snippet=fact,
            confidence=confidence,
            extracted_fact=fact,
            fact_type=query.gap_type,
            fact_value=f"{value}{unit or ''}",
        )
