"""Unit tests for LLM-backed adapters (mock provider — offline deterministic).

Covers:
  - LLMRecipeExtractor: maps LLM JSON to ExtractedRecipeCandidate, tolerant
    of missing fields, extracts structure correctly.
  - LLMKnowledgeResearcher: maps LLM JSON facts to EvidenceResult.
  - LLMPlanExplainer: returns LLM explanation or deterministic fallback.
All tests inject a fake LLMClient — no real network, CI-safe.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from cooking_plan_agent.domain.models import EvidenceQuery, RecipeGap
from cooking_plan_agent.llm import (
    LLMKnowledgeResearcher,
    LLMPlanExplainer,
    LLMRecipeExtractor,
)
from cooking_plan_agent.llm.client import LLMError


class FakeLLMClient:
    """Stub client returning a fixed JSON payload per call."""

    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self._payload = payload
        self.calls: list[list[dict[str, str]]] = []

    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        self.calls.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.calls.append(messages)
        if isinstance(self._payload, Exception):
            raise self._payload
        return str(self._payload)


class SlowLLMClient:
    async def chat_json(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
        await asyncio.sleep(60)
        return {}


# ---------------------------------------------------------------------------
# LLMRecipeExtractor
# ---------------------------------------------------------------------------


class TestLLMRecipeExtractor:
    @pytest.mark.asyncio
    async def test_extracts_full_recipe(self) -> None:
        client = FakeLLMClient(
            {
                "dish_name": "番茄炒蛋",
                "original_servings": 2,
                "source_language": "zho",
                "inferred_fields": ["steps[0].resources_hint"],
                "ingredients": [
                    {"raw_text": "鸡蛋 3个", "name": "鸡蛋", "quantity": 3, "unit": "个"},
                    {"raw_text": "番茄 2个", "name": "番茄", "quantity": 2, "unit": "个"},
                ],
                "steps": [
                    {
                        "instruction": "热油炒蛋",
                        "category": "heating",
                        "heat_level": "HIGH",
                        "active_duration_minutes": 3,
                        "resources_hint": ["stove", "wok"],
                        "extraction_source": "LLM_INFERRED",
                        "confidence": 0.72,
                    },
                ],
            }
        )
        extractor = LLMRecipeExtractor(client)  # type: ignore[arg-type]

        candidate = await extractor.extract("番茄炒蛋：3个鸡蛋…")

        assert candidate.dish_name == "番茄炒蛋"
        assert candidate.original_servings == Decimal(2)
        assert candidate.extraction_source == "LLM"
        assert len(candidate.ingredients) == 2
        assert candidate.ingredients[0].name == "鸡蛋"
        assert candidate.ingredients[0].quantity == Decimal(3)
        assert candidate.steps[0].heat_level.value == "HIGH"
        assert candidate.steps[0].active_duration_minutes == 3
        assert candidate.steps[0].extraction_source == "LLM_INFERRED"
        assert candidate.steps[0].confidence == Decimal("0.72")
        assert candidate.inferred_fields == ("steps[0].resources_hint",)
        assert "culinary common sense" in client.calls[0][0]["content"]

    @pytest.mark.asyncio
    async def test_tolerates_missing_fields(self) -> None:
        client = FakeLLMClient(
            {
                "dish_name": "",
                "ingredients": [{"name": "salt"}],
                "steps": [],
            }
        )
        extractor = LLMRecipeExtractor(client)  # type: ignore[arg-type]

        candidate = await extractor.extract("some text")

        assert candidate.dish_name == "Untitled Recipe"
        assert candidate.original_servings == Decimal(2)
        assert len(candidate.steps) == 0
        # Missing quantity defaults to None — not a positive Decimal
        assert candidate.ingredients[0].quantity is None

    @pytest.mark.asyncio
    async def test_falls_back_to_rule_based_on_llm_failure(self) -> None:
        client = FakeLLMClient(LLMError("boom"))
        extractor = LLMRecipeExtractor(client)  # type: ignore[arg-type]

        candidate = await extractor.extract("1. Boil water.\n2. Add pasta.")

        # Rule-based fallback — never blocks the workflow
        assert candidate.extraction_source == "RULE_BASED"

    @pytest.mark.asyncio
    async def test_falls_back_without_waiting_for_slow_llm(self) -> None:
        extractor = LLMRecipeExtractor(SlowLLMClient(), timeout_seconds=0.01)  # type: ignore[arg-type]

        candidate = await extractor.extract("1. Boil water.\n2. Add pasta.")

        assert candidate.extraction_source == "RULE_BASED"


# ---------------------------------------------------------------------------
# LLMKnowledgeResearcher
# ---------------------------------------------------------------------------


class TestLLMKnowledgeResearcher:
    @pytest.mark.asyncio
    async def test_maps_facts_to_evidence(self) -> None:
        client = FakeLLMClient(
            {
                "facts": [
                    {
                        "source_title": "Standard practice",
                        "fact": "stir-fry at high heat",
                        "value": "HIGH",
                        "confidence": 0.8,
                    },
                ],
            }
        )
        researcher = LLMKnowledgeResearcher(client)  # type: ignore[arg-type]
        query = EvidenceQuery(
            query_text="heat level?",
            gap_type="critical",
            recipe_context="stir-fry",
        )

        results = await researcher.research(query)

        assert len(results) == 1
        assert results[0].fact_value == "HIGH"
        assert results[0].confidence == Decimal("0.8")
        assert results[0].fact_type == "critical"

    @pytest.mark.asyncio
    async def test_empty_facts_yields_empty_results(self) -> None:
        client = FakeLLMClient({"facts": []})
        researcher = LLMKnowledgeResearcher(client)  # type: ignore[arg-type]
        query = EvidenceQuery(query_text="q", gap_type="critical", recipe_context="dish")

        results = await researcher.research(query)

        assert results == []

    @pytest.mark.asyncio
    async def test_resolves_duration_gap_to_structured_evidence(self) -> None:
        client = FakeLLMClient(
            {
                "facts": [
                    {
                        "source_title": "Model knowledge",
                        "fact": "Coating pork in the pan takes about one minute.",
                        "value": 1,
                        "unit": "minutes",
                        "confidence": 0.82,
                    },
                ],
            }
        )
        researcher = LLMKnowledgeResearcher(client)  # type: ignore[arg-type]
        gap = RecipeGap(
            gap_id="gap-duration",
            recipe_id="braised-pork",
            field_path="steps[3].passive_duration_minutes",
            gap_class="critical",
            description="Missing duration for coating pork in the pan",
            confidence=Decimal("0.2"),
        )

        resolved = await researcher.resolve_gap(gap, "Braised pork belly")

        assert resolved.duration_min_minutes == 1
        assert resolved.duration_max_minutes == 1
        assert resolved.source_count == 1
        assert resolved.needs_confirmation is False
        assert resolved.evidence_items[0].source_title == "LLM culinary inference"
        assert resolved.evidence_items[0].source_url == ""


# ---------------------------------------------------------------------------
# LLMPlanExplainer
# ---------------------------------------------------------------------------


class TestLLMPlanExplainer:
    @pytest.mark.asyncio
    async def test_returns_llm_explanation(self) -> None:
        client = FakeLLMClient({"explanation": "Parallel cooking saves time."})
        explainer = LLMPlanExplainer(client)  # type: ignore[arg-type]

        text = await explainer.explain({"makespan_minutes": 45, "dish_completions": []})

        assert text == "Parallel cooking saves time."

    @pytest.mark.asyncio
    async def test_falls_back_on_failure(self) -> None:
        client = FakeLLMClient(LLMError("down"))
        explainer = LLMPlanExplainer(client)  # type: ignore[arg-type]

        text = await explainer.explain({"makespan_minutes": 45, "dish_completions": []})

        assert "45" in text
