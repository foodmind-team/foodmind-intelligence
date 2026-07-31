"""Provider-neutral researcher (handbook 10.2).

Orchestrates the full research pipeline:
  gap → query → search → filter → extract → reconcile → ReconciledEvidence

Implements the RecipeResearcher Protocol from application/ports.py.
Concrete providers are injected — this module depends only on the Protocol.
"""

import asyncio
import logging
from typing import Protocol

from cooking_plan_agent.config.settings import Settings
from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    CookingEvidence,
    EvidenceQuery,
    EvidenceResult,
    RecipeGap,
    ReconciledEvidence,
    SearchDocument,
)
from cooking_plan_agent.research.config import DomainAllowList
from cooking_plan_agent.research.domain_filter import (
    classify_documents,
    filter_by_domain,
)
from cooking_plan_agent.research.evidence_extractor import extract_evidence
from cooking_plan_agent.research.query_builder import build_minimal_query
from cooking_plan_agent.research.reconciler import reconcile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider Protocol (concrete providers implement this)
# ---------------------------------------------------------------------------


class SearchProvider(Protocol):
    """Protocol for concrete search providers (Fake, Tavily, Brave, etc.).

    Each provider normalises its results into SearchDocument,
    hiding provider-specific SDK details from the rest of the pipeline.
    """

    async def search(
        self,
        query: str,
        max_results: int,
    ) -> tuple[SearchDocument, ...]: ...


# ---------------------------------------------------------------------------
# Researcher — implements RecipeResearcher from application/ports.py
# ---------------------------------------------------------------------------


class Researcher:
    """Bounded web research orchestrator.

    Wires query_builder, search provider, domain_filter, evidence_extractor,
    and reconciler into a single pipeline.

    The RecipeResearcher Protocol in application/ports.py requires:
      async research(query: EvidenceQuery) -> list[EvidenceResult]
    This class wraps the full pipeline to satisfy that contract while
    returning ReconciledEvidence (richer than EvidenceResult).
    """

    def __init__(
        self,
        provider: SearchProvider,
        allow_list: DomainAllowList,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._allow_list = allow_list
        self._settings = settings

    async def search(
        self,
        query: EvidenceQuery,
        max_results: int | None = None,
    ) -> tuple[SearchDocument, ...]:
        """Execute a bounded search against the provider.

        Respects per-query timeout (Settings.research_timeout_seconds),
        result count bounds, and domain allow-list filtering.
        On timeout or failure, returns empty tuple.
        """
        max_r = max_results or self._settings.research_max_results_per_query

        try:
            docs = await asyncio.wait_for(
                self._provider.search(query.query_text, max_r),
                timeout=self._settings.research_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "Research timed out after %.1fs for query: %s",
                self._settings.research_timeout_seconds,
                query.query_text[:80],
            )
            return ()
        except Exception:
            logger.exception("Search provider failed for query: %s", query.query_text[:80])
            return ()

        # Filter results through domain allow-list (handbook 10.4)
        return filter_by_domain(docs, self._allow_list)

    async def research(
        self,
        query: EvidenceQuery,
    ) -> list[EvidenceResult]:
        """Full research pipeline for one gap.

        Satisfies the RecipeResearcher Protocol from application/ports.py.
        Returns EvidenceResult items for each piece of evidence found.

        Pipeline:
          1. Search via provider
          2. Filter via domain allow-list
          3. Extract CookingEvidence from each document
          4. Reconcile into consensus
          5. Map back to EvidenceResult for compatibility
        """
        docs = await self.search(query)

        if not docs:
            return []

        # Classify documents into safety/technique tiers (handbook 10.4)
        safety_docs, technique_docs = classify_documents(docs, self._allow_list)

        # Safety sources have precedence (handbook 10.8). Process them first,
        # then supplement with technique sources if needed.
        all_evidence: list[CookingEvidence] = []

        for doc in safety_docs:
            evidence = extract_evidence(doc, dish_name=query.recipe_context)
            if evidence is not None:
                all_evidence.append(evidence)

        for doc in technique_docs:
            evidence = extract_evidence(doc, dish_name=query.recipe_context)
            if evidence is not None:
                all_evidence.append(evidence)

        if not all_evidence:
            return []

        # Reconcile evidence from multiple sources (handbook 10.7)
        reconciled = reconcile(
            tuple(all_evidence),
            disagreement_threshold=self._settings.research_disagreement_threshold,
        )

        # Map to EvidenceResult for Protocol compatibility
        return _reconciled_to_results(reconciled)

    # ------------------------------------------------------------------
    # Convenience: full gap resolution
    # ------------------------------------------------------------------

    async def resolve_gap(
        self,
        gap: RecipeGap,
        dish_name: str = "",
    ) -> ReconciledEvidence:
        """Resolve a single RecipeGap through the full research pipeline.

        Returns ReconciledEvidence directly — this is the primary API
        called from the workflow node.
        """
        query_text = build_minimal_query(gap, dish_name)
        query = EvidenceQuery(
            query_text=query_text,
            gap_type=gap.gap_class,
            recipe_context=dish_name,
            target_fields=(gap.field_path,),
        )

        docs = await self.search(query)

        if not docs:
            return ReconciledEvidence(
                source_count=0,
                needs_confirmation=True,
            )

        safety_docs, technique_docs = classify_documents(docs, self._allow_list)

        all_evidence: list[CookingEvidence] = []
        for doc in safety_docs + technique_docs:
            evidence = extract_evidence(doc, dish_name=dish_name)
            if evidence is not None:
                all_evidence.append(evidence)

        if not all_evidence:
            return ReconciledEvidence(
                source_count=0,
                needs_confirmation=True,
            )

        return reconcile(
            tuple(all_evidence),
            disagreement_threshold=self._settings.research_disagreement_threshold,
        )


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _reconciled_to_results(reconciled: ReconciledEvidence) -> list[EvidenceResult]:
    """Map ReconciledEvidence back to EvidenceResult for Protocol compatibility."""
    results: list[EvidenceResult] = []
    from decimal import Decimal

    for ev in reconciled.evidence_items:
        fact_parts: list[str] = []
        if ev.heat_level and ev.heat_level != HeatLevel.NONE:
            fact_parts.append(f"heat={ev.heat_level.value}")
        if ev.duration_min_minutes is not None:
            fact_parts.append(f"duration_min={ev.duration_min_minutes}min")
        if ev.duration_max_minutes is not None:
            fact_parts.append(f"duration_max={ev.duration_max_minutes}min")
        if ev.explicit_temperature_c is not None:
            fact_parts.append(f"temp={ev.explicit_temperature_c}C")

        fact_str = ", ".join(fact_parts) if fact_parts else "no cooking data"
        fact_type = "heat_level" if ev.heat_level and ev.heat_level != HeatLevel.NONE else "duration"

        results.append(EvidenceResult(
            source_title=ev.source_title,
            source_url=ev.source_url,
            snippet=ev.source_excerpt,
            confidence=Decimal("0.7"),  # Rule-based extraction: moderate confidence
            extracted_fact=fact_str,
            fact_type=fact_type,
            fact_value=fact_str,
        ))

    return results
