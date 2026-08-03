"""Workflow node implementations for a single pipeline stage.

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.
"""

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.models import (
    Assumption,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState


async def research_missing_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Query web research for low-confidence critical gaps (handbook 10).

    For each critical gap in state, query the RecipeResearcher from
    WorkflowContext. Gaps that are heat or duration related are prioritised.
    Returns evidence dict keyed by gap_id for traceability.

    On any failure (timeout, provider error, no results), the evidence
    stays empty and routing falls back to confirmation — never unsafe guess.
    """
    researcher = runtime.context.recipe_researcher
    if researcher is None:
        # No researcher wired — nothing to do (MVP without research)
        return {}

    gaps = state.get("gaps", ())
    if not gaps:
        return {}

    # Only research critical gaps that are heat/duration/temperature related
    # (handbook 10.1: "only for missing cooking heat or duration")
    _researchable_fields = {"heat_level", "duration", "temperature", "target_temperature_c"}

    researchable_gaps = [
        g
        for g in gaps
        if g.gap_class in ("critical", "safety_critical")
        and any(f in g.field_path.lower() for f in _researchable_fields)
    ]

    if not researchable_gaps:
        return {}

    # Extract dish name from recipe candidates for query context
    candidates = state.get("extracted_candidates", ())
    dish_name = candidates[0].dish_name if candidates else ""

    # Resolve each gap (handbook 10.9: at most 2 queries per dish)
    # For MVP, we use the Researcher directly rather than the Protocol
    # since the Protocol's research() signature returns list[EvidenceResult]
    from cooking_plan_agent.config.settings import get_settings
    from cooking_plan_agent.domain.models import ReconciledEvidence
    from cooking_plan_agent.infrastructure.cache import (
        RESEARCH_SAFETY_POLICY_VERSION,
        _stable_digest,
        build_research_cache_key,
    )
    from cooking_plan_agent.research.query_builder import build_minimal_query
    from cooking_plan_agent.research.researcher import Researcher

    settings = get_settings()
    # P1-06 cache is optional — getattr keeps duck-typed contexts working.
    cache = getattr(runtime.context, "cache", None)

    # P1-06 research cache key: query + provider tag + allow-list + safety
    # policy version (+ model for LLM-backed researchers).
    allow_list_fingerprint = _stable_digest(*sorted(set(settings.allowed_research_domains)))
    provider_tag = type(researcher).__name__
    model_tag = settings.llm_model

    async def _resolve_uncached(gap: object, query_text: str) -> ReconciledEvidence:
        if isinstance(researcher, Researcher):
            try:
                return await researcher.resolve_gap(gap, dish_name)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 — any failure → confirmation
                return ReconciledEvidence(source_count=0, needs_confirmation=True)
        # Non-Researcher RecipeResearcher — Protocol research() path.
        from cooking_plan_agent.domain.models import EvidenceQuery

        query = EvidenceQuery(
            query_text=query_text,
            gap_type=gap.gap_class,  # type: ignore[attr-defined]
            recipe_context=dish_name,
        )
        try:
            results = await researcher.research(query)
            if results:
                return ReconciledEvidence(source_count=len(results), needs_confirmation=False)
            return ReconciledEvidence(source_count=0, needs_confirmation=True)
        except Exception:  # noqa: BLE001 — any failure → confirmation
            return ReconciledEvidence(source_count=0, needs_confirmation=True)

    async def _resolve(gap: object) -> ReconciledEvidence:
        query_text = build_minimal_query(gap, dish_name)  # type: ignore[arg-type]
        if cache is None:
            return await _resolve_uncached(gap, query_text)
        key = build_research_cache_key(
            query_text,
            provider_tag=provider_tag,
            allow_list_fingerprint=allow_list_fingerprint,
            safety_policy_version=RESEARCH_SAFETY_POLICY_VERSION,
            model=model_tag,
        )
        value = await cache.get_or_compute(
            key,
            settings.cache_ttl_seconds,
            lambda: _resolve_uncached(gap, query_text),
        )
        from typing import cast

        return cast(ReconciledEvidence, value)

    research_evidence: dict[str, ReconciledEvidence] = {}
    for gap in researchable_gaps[:2]:  # At most 2 queries (handbook 10.9)
        research_evidence[gap.gap_id] = await _resolve(gap)

    return {"research_evidence": research_evidence}


async def apply_research_evidence_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Write reconciled research evidence back into candidates (P1-01).

    Sits between ``research_missing`` and IR validation so search results
    actually update the plan — research is no longer a write-only bypass.

    For each gap with evidence:
      1. Locates the exact target via ``gap_id + recipe_id + field_path``
         (never by list position).
      2. Applies reliable values (heat / duration / temperature) to the
         candidate step and records an EvidenceRef-backed Assumption.
      3. Marks the gap resolved; only unresolved gaps stay in state.

    Anything that cannot be safely auto-applied — no source, disagreement,
    field-location failure, or a safety-critical temperature without a
    verifiable URL — sets ``needs_confirmation`` so routing surfaces the
    user confirmation instead of silently guessing (P1-01 rules 5 & 6).
    """
    from cooking_plan_agent.research.evidence_apply import apply_evidence_to_candidate

    research_evidence = state.get("research_evidence", {})
    if not research_evidence:
        # No research ran — leave gaps untouched; downstream routing handles them.
        return {}

    candidates = list(state.get("extracted_candidates", ()))
    gaps = state.get("gaps", ())

    applied_gap_ids: set[str] = set()
    assumptions: list[Assumption] = []
    needs_confirmation = False

    for gap in gaps:
        reconciled = research_evidence.get(gap.gap_id)
        if reconciled is None:
            # Gap not targeted by research — stays unresolved.
            continue

        # Locate the recipe by stable recipe_id, never by list position.
        candidate_idx = next(
            (i for i, candidate in enumerate(candidates) if candidate.recipe_id == gap.recipe_id),
            None,
        )
        if candidate_idx is None:
            needs_confirmation = True  # recipe-level location failure
            continue

        result = apply_evidence_to_candidate(candidates[candidate_idx], gap, reconciled)
        if result.applied and result.candidate is not None:
            candidates[candidate_idx] = result.candidate
            applied_gap_ids.add(gap.gap_id)
            if result.assumption is not None:
                assumptions.append(result.assumption)
            # Even applied values that came from conflicting evidence (MAD
            # over threshold) must surface for user confirmation — never
            # silently adopt a disputed value (P1-01 rule 5).
            if reconciled.needs_confirmation:
                needs_confirmation = True
        else:
            needs_confirmation = needs_confirmation or result.needs_confirmation

    remaining_gaps = tuple(g for g in gaps if g.gap_id not in applied_gap_ids)
    if any(g.gap_class in ("critical", "safety_critical") for g in remaining_gaps):
        needs_confirmation = True

    return {
        "extracted_candidates": tuple(candidates),
        "gaps": remaining_gaps,
        "research_assumptions": tuple(assumptions),
        "needs_confirmation": needs_confirmation,
    }
