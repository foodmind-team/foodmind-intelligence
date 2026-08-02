"""LangGraph workflow nodes — thin wrappers around domain services.

Per handbook 8.4: each node calls ONE application/domain service and
returns only CHANGED state fields. No in-place state mutation.
No broad exception catching that masks errors as partial success.
"""

# LangGraph runtime type — context is injected by the framework
import logging

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    Assumption,
    CookingTask,
    ExtractedRecipeCandidate,
    FeasibilityReport,
    IngredientDemand,
    RecipeGap,
    RecipeIR,
    RepairOption,
    SafetyContext,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

logger = logging.getLogger(__name__)

# ============================================================================
# Input & parsing nodes
# ============================================================================


def _solver_timeout() -> float:
    """Return the configured CP-SAT solver timeout in seconds (P0-05)."""
    from cooking_plan_agent.config.settings import get_settings

    return get_settings().solver_timeout_seconds


async def validate_input_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Validate the incoming GeneratePlanRequest (P0-03, P0-05).

    Checks (in order so the FIRST violation wins):
      1. Non-empty recipes
      2. recipe_id uniqueness
      3. Recipe count vs Settings.max_recipe_count
      4. Per-recipe text byte size vs Settings.max_recipe_text_bytes
      5. Total serialised request size vs Settings.max_request_bytes
      6. schema_version in Settings.supported_schema_versions
      7. time_limit_minutes is non-negative

    Returns an empty dict on success or {'error': WorkflowError} on failure.
    The graph routes any error straight to the FAILED terminal.
    """
    from cooking_plan_agent.config.settings import get_settings

    request = state["request"]
    settings = get_settings()

    # 1. Non-empty recipes — keep the original request error code (P0-03:
    #    must not be overwritten by downstream SCHEDULE_INFEASIBLE).
    if not request.recipes:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message="Request contains no recipes",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 2. Duplicate recipe IDs
    seen: set[str] = set()
    for recipe in request.recipes:
        if recipe.recipe_id in seen:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.DUPLICATE_RECIPE_ID.value,
                    message=f"Duplicate recipe_id: {recipe.recipe_id}",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }
        seen.add(recipe.recipe_id)

    # 3. Recipe count cap
    if len(request.recipes) > settings.max_recipe_count:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.TOO_MANY_RECIPES.value,
                message=(
                    f"Request contains {len(request.recipes)} recipes, max allowed is {settings.max_recipe_count}"
                ),
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 4. Per-recipe text byte cap
    for recipe in request.recipes:
        text_bytes = len(recipe.text.encode("utf-8"))
        if text_bytes > settings.max_recipe_text_bytes:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.RECIPE_TEXT_TOO_LARGE.value,
                    message=(
                        f"Recipe '{recipe.recipe_id}' text is {text_bytes} bytes, "
                        f"max allowed is {settings.max_recipe_text_bytes}"
                    ),
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 5. Total serialised request size
    try:
        total_bytes = len(request.model_dump_json().encode("utf-8"))
    except Exception:  # noqa: BLE001 — serialisation must not crash validation
        total_bytes = 0
    if total_bytes > settings.max_request_bytes:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.REQUEST_TOO_LARGE.value,
                message=f"Request is {total_bytes} bytes, max allowed is {settings.max_request_bytes}",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 6. schema_version support
    if request.schema_version not in settings.supported_schema_versions:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.UNSUPPORTED_SCHEMA_VERSION.value,
                message=(
                    f"schema_version '{request.schema_version}' is not supported; "
                    f"supported versions: {sorted(settings.supported_schema_versions)}"
                ),
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # 7. Negative time limit
    if request.time_limit_minutes is not None and request.time_limit_minutes < 0:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_TIME_LIMIT.value,
                message=f"time_limit_minutes must be >= 0, got {request.time_limit_minutes}",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }

    # --- P0-05 time semantics ---

    # 8. serving_time must be a valid HH:MM string (legacy field).
    if request.serving_time is not None:
        parts = request.serving_time.split(":")
        valid = (
            len(parts) == 2
            and parts[0].isdigit()
            and parts[1].isdigit()
            and 0 <= int(parts[0]) <= 23
            and 0 <= int(parts[1]) <= 59
        )
        if not valid:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_SERVING_TIME.value,
                    message=f"serving_time must be HH:MM, got {request.serving_time!r}",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 9. serving_at must carry a timezone (never treated as local wall-clock).
    if request.serving_at is not None:
        if request.serving_at.tzinfo is None or request.serving_at.utcoffset() is None:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_SERVING_TIME.value,
                    message="serving_at must include a timezone offset",
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

    # 10. Approved decisions (P0-06): structural + revision validation.
    if request.approved_decisions:
        from cooking_plan_agent.repair.options import validate_approved_decisions

        issues = validate_approved_decisions(request.approved_decisions, request.plan_revision)
        if issues:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_APPROVED_DECISION.value,
                    message="; ".join(issues),
                    correlation_id=request.request_id,
                    node_name="validate_input",
                )
            }

        # Apply decisions as a pure transformation → the downstream pipeline
        # (safety, feasibility, scheduling) re-runs against the RESOLVED
        # request, never bypassing hard rules (P0-06 rule 6).
        from cooking_plan_agent.repair.options import apply_approved_decisions_structured

        resolved = apply_approved_decisions_structured(request, request.approved_decisions)
        if resolved is not request:
            return {"request": resolved}

    return {}


async def parse_recipes_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Extract structured candidates from each recipe's raw text.

    Uses the RecipeExtractor from WorkflowContext (rule-based or LLM-backed).
    Each recipe in the request is individually extracted, then aggregated.

    P1-02: multi-recipe extraction runs via ``asyncio.gather`` capped by a
    configurable Semaphore (llm_max_concurrency) and bounded by an overall
    envelope timeout (llm_overall_timeout_seconds) so a single request
    cannot exhaust provider quota or hang the event loop.
    Falls back to rule-based extraction if no extractor is configured.
    """
    import asyncio

    from cooking_plan_agent.config.settings import get_settings

    request = state["request"]

    # Compat layer injects pre-parsed structured candidates (snapshots).
    # Use them directly — never re-invoke the LLM for compat requests
    # (P0-02 rule 4).
    if request.preparsed_candidates:
        return {"extracted_candidates": request.preparsed_candidates}

    extractor = runtime.context.recipe_extractor
    settings = get_settings()

    candidates: list[ExtractedRecipeCandidate] = []

    if extractor is not None:
        # Use configured extractor (LLM or rule-based from WorkflowContext)
        semaphore = asyncio.Semaphore(max(1, settings.llm_max_concurrency))
        # P1-06 cache is optional — getattr keeps duck-typed contexts working.
        cache = getattr(runtime.context, "cache", None)

        async def _extract_one(text: str) -> ExtractedRecipeCandidate:
            async with semaphore:
                if cache is None:
                    return await extractor.extract(text)
                # P1-06: reuse stable parse artifacts. Key includes text hash,
                # parser type, model, prompt version, language, schema version
                # so model/prompt upgrades invalidate old entries.
                from typing import cast

                from cooking_plan_agent.infrastructure.cache import build_parse_cache_key
                from cooking_plan_agent.llm.extractor import PARSE_PROMPT_VERSION

                key = build_parse_cache_key(
                    text,
                    parser_type=type(extractor).__name__,
                    model=settings.llm_model,
                    prompt_version=PARSE_PROMPT_VERSION,
                    schema_version=request.schema_version,
                )
                value = await cache.get_or_compute(
                    key,
                    settings.cache_ttl_seconds,
                    lambda: extractor.extract(text),
                )
                return cast(ExtractedRecipeCandidate, value)

        try:
            # Bound the whole batch: a slow provider must not hold the request
            # open beyond llm_overall_timeout_seconds.
            extracted = await asyncio.wait_for(
                asyncio.gather(*(_extract_one(r.text) for r in request.recipes if r.text)),
                timeout=settings.llm_overall_timeout_seconds,
            )
            candidates = list(extracted)
        except Exception as exc:  # noqa: BLE001 — per-node error → error state
            # P2-03: public text comes from the catalog; keep only the
            # exception type as controlled diagnostic context.
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.EXTERNAL_PROVIDER_UNAVAILABLE.value,
                    message="Recipe extraction failed",
                    correlation_id=request.request_id,
                    node_name="parse_recipes",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }
    else:
        # No extractor configured — use built-in rule-based extractor
        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor

        rule_extractor = RuleExtractor()
        for recipe in request.recipes:
            if recipe.text:
                candidate = await rule_extractor.extract(recipe.text)
                candidates.append(candidate)

    return {"extracted_candidates": tuple(candidates)}


async def detect_gaps_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Identify missing/inferred fields in extracted candidates.

    Runs gap detection (find_recipe_gaps + classify_recipe_gap) on every
    extracted candidate. Aggregates all gaps across recipes.
    """
    from cooking_plan_agent.parsing.gaps import find_recipe_gaps

    candidates = state.get("extracted_candidates", ())
    if not candidates:
        return {"gaps": ()}

    all_gaps: list[RecipeGap] = []
    for candidate in candidates:
        gaps = find_recipe_gaps(candidate)
        all_gaps.extend(gaps)

    return {"gaps": tuple(all_gaps)}


async def infer_local_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Apply local cooking rules to fill detected gaps.

    Runs infer_local on each candidate with its gaps, then merges inference
    results back into updated candidates. Unresolved critical gaps are left
    in state for routing decisions.
    """
    from cooking_plan_agent.parsing.inference import infer_local as local_infer
    from cooking_plan_agent.parsing.inference import merge_inference

    candidates = state.get("extracted_candidates", ())
    gaps = state.get("gaps", ())

    if not candidates or not gaps:
        return {}

    # Partition gaps by recipe_id for per-candidate inference
    gaps_by_recipe: dict[str, list[RecipeGap]] = {}
    for gap in gaps:
        gaps_by_recipe.setdefault(gap.recipe_id, []).append(gap)

    updated_candidates: list[ExtractedRecipeCandidate] = []
    all_unresolved: list[RecipeGap] = []
    all_assumptions: list[Assumption] = []

    for candidate in candidates:
        recipe_gaps = tuple(gaps_by_recipe.get(candidate.recipe_id, ()))
        if not recipe_gaps:
            updated_candidates.append(candidate)
            continue

        result = local_infer(candidate, recipe_gaps)
        updated = merge_inference(candidate, result)
        updated_candidates.append(updated)
        all_unresolved.extend(result.unresolved_gaps)
        all_assumptions.extend(result.assumptions)

    return {
        "extracted_candidates": tuple(updated_candidates),
        "gaps": tuple(all_unresolved),  # Only keep unresolved gaps for routing
    }


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


async def validate_recipe_ir_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Build validated RecipeIR objects from candidates.

    Converts each ExtractedRecipeCandidate → RecipeIR via build_recipe_ir,
    then runs semantic validation (validate_recipe_ir_semantics). Produces
    parsed_recipes in state for downstream scheduling nodes.

    If semantic validation fails with errors, returns a workflow error
    that will be routed to FAILED terminal.
    """
    from cooking_plan_agent.parsing.ir_builder import (
        attach_research_assumptions,
        build_recipe_ir,
        validate_recipe_ir_semantics,
    )

    candidates = state.get("extracted_candidates", ())
    if not candidates:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message="No extracted recipe candidates to validate",
                correlation_id=state["request"].request_id,
                node_name="validate_recipe_ir",
            )
        }

    request = state["request"]
    # The extractor produces candidates in the same order as the request's
    # non-empty recipes. Pair each candidate with its request recipe input so
    # the caller's recipe_id overrides the extractor's internal ID and the
    # target_servings drives ingredient scaling (P0-04 rules 2 & 5).
    request_recipes = tuple(r for r in request.recipes if r.text)
    pairs = tuple(zip(candidates, request_recipes, strict=True))

    # Build RecipeIR from each candidate
    recipes: list[RecipeIR] = []
    for candidate, recipe_input in pairs:
        try:
            recipe_ir = build_recipe_ir(
                candidate,
                request_recipe_id=recipe_input.recipe_id,
                target_servings=recipe_input.target_servings,
            )
            recipes.append(recipe_ir)
        except (ValueError, TypeError) as exc:
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                    message="Failed to build RecipeIR",
                    correlation_id=state["request"].request_id,
                    node_name="validate_recipe_ir",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }

    # P0-06: apply substitute_ingredient decisions as a RecipeIR patch so
    # safety (allergen) and feasibility checks re-run against the NEW
    # ingredient — hard rules are never bypassed (rule 6).
    if request.approved_decisions:
        from cooking_plan_agent.repair.options import apply_ingredient_substitutions_patch

        recipes = list(apply_ingredient_substitutions_patch(tuple(recipes), request.approved_decisions))

    # P1-01: attach evidence-backed research assumptions so provenance is
    # traceable in the final assumption/response.
    recipes = list(attach_research_assumptions(tuple(recipes), state.get("research_assumptions", ())))

    # Semantic validation
    report = validate_recipe_ir_semantics(tuple(recipes))
    if not report.passed:
        error_details = "; ".join(i.message for i in report.issues if i.severity == "error")
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INVALID_RECIPE_TEXT.value,
                message=f"Recipe semantic validation failed: {error_details}",
                correlation_id=state["request"].request_id,
                node_name="validate_recipe_ir",
            )
        }

    return {"parsed_recipes": tuple(recipes)}


# ============================================================================
# Safety & feasibility nodes
# ============================================================================


async def validate_safety_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Evaluate all safety rules against the recipe set under a regional policy.

    P3-04:
      1. Resolve the regional food-safety policy — the request's explicit
         ``region`` wins over the deployment default. An unknown region,
         unknown version, not-yet-effective policy, or source-less policy is a
         hard error (D6) routed to FAILED — never a silent fallback.
      2. Build the rule set from the resolved policy's thresholds.
      3. Evaluate and return the SafetyReport plus the policy record so
         terminal responses can carry region/version/sources.

    Handbook 5.7: safety_validator node — first hard gate after parsing.
    """
    from cooking_plan_agent.config.settings import get_settings
    from cooking_plan_agent.safety.engine import SafetyEngine
    from cooking_plan_agent.safety.policy import PolicyResolutionError, resolve_policy
    from cooking_plan_agent.safety.rules import build_rules

    request = state["request"]
    settings = get_settings()
    # Explicit selection: request region overrides the deployment default.
    region = request.region or settings.safety_policy_region

    try:
        policy = resolve_policy(region, settings.safety_policy_version)
    except PolicyResolutionError as exc:
        # P2-03: only the exception type is retained as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SAFETY_POLICY_UNAVAILABLE.value,
                message="Safety policy resolution failed",
                correlation_id=request.request_id,
                node_name="validate_safety",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }

    # Rules are bound to the resolved policy so thresholds always match the
    # region recorded on the plan. A context engine already bound to the same
    # policy is reused (DI); otherwise one is built for the policy.
    engine = runtime.context.safety_engine
    if engine is None or getattr(engine, "policy", None) != policy:
        engine = SafetyEngine(rules=build_rules(policy), policy=policy)

    parsed_recipes = state.get("parsed_recipes", ())

    context = SafetyContext(
        recipes=parsed_recipes,
        dietary_restrictions=request.dietary_restrictions,
        user_allergens=request.user_allergens,
        inventory_lots=request.inventory_lots,
        cooking_date=request.cooking_date,
    )

    report = engine.evaluate(context)
    return {"safety_report": report, "safety_policy": policy.to_record()}


async def check_feasibility_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Check inventory sufficiency and resource compatibility.

    Ingredient check: aggregates all RecipeIR.ingredients then runs FEFO
    allocation against request.inventory_lots.

    Resource pre-check: inspects RecipeStep.resources_hint for required
    resource types and verifies at least one compatible resource exists.
    Full resource-capacity checking (per CookingTask) is deferred to
    merge_preparation + build_task_graph stages.
    """
    from cooking_plan_agent.inventory.feasibility import check_all_inventory

    request = state["request"]
    parsed_recipes = state.get("parsed_recipes", ())

    # --- Ingredient feasibility ---
    all_ingredients: list[IngredientDemand] = []
    for recipe in parsed_recipes:
        all_ingredients.extend(recipe.ingredients)

    ingredient_report = check_all_inventory(
        requirements=tuple(all_ingredients),
        lots=request.inventory_lots,
        cooking_date=request.cooking_date,
    )

    # --- Resource pre-check (from step hints, pre-decomposition) ---
    missing_resources: list[str] = []
    if request.kitchen_resources:
        available_types = {r.resource_type.lower() for r in request.kitchen_resources if r.available}
        for recipe in parsed_recipes:
            for step in recipe.steps:
                for hint in step.resources_hint:
                    if hint.lower() not in available_types:
                        if hint not in missing_resources:
                            missing_resources.append(hint)

    is_feasible = ingredient_report.is_feasible and len(missing_resources) == 0

    # --- Generate repair options when infeasible ---
    repair_options: tuple[RepairOption, ...] = ()
    if not is_feasible:
        from cooking_plan_agent.repair.options import (
            propose_dish_replacements,
            propose_equipment_alternatives,
            propose_ingredient_substitutions,
            propose_portion_adjustments,
            rank_repair_options,
        )

        recipe_names = tuple(r.dish_name for r in parsed_recipes)

        opts = list(propose_ingredient_substitutions(ingredient_report.ingredient_shortages))
        opts.extend(propose_portion_adjustments(ingredient_report.ingredient_shortages))
        opts.extend(propose_equipment_alternatives(tuple(missing_resources)))
        opts.extend(propose_dish_replacements(ingredient_report.ingredient_shortages, recipe_names))
        repair_options = rank_repair_options(tuple(opts))

    return {
        "feasibility_report": FeasibilityReport(
            report_id=ingredient_report.report_id,
            ingredient_shortages=ingredient_report.ingredient_shortages,
            missing_resources=tuple(sorted(missing_resources)),
            is_feasible=is_feasible,
        ),
        "repair_options": repair_options,
    }


async def build_confirmation_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render NEEDS_CONFIRMATION response with assumptions and repair options.

    Delegates to rendering.responses.render_confirmation_response.
    """
    from cooking_plan_agent.rendering.responses import render_confirmation_response

    response = render_confirmation_response(state)
    return {"response": response}


# ============================================================================
# Preparation & scheduling nodes
# ============================================================================


def _wire_prep_consumption(
    recipe_tasks: list[CookingTask],
    demand_final_states: dict[str, str],
) -> list[CookingTask]:
    """Let each recipe's first task consume the final states of its demands.

    ``demand_final_states`` keys are ``"recipe_id:index"``. The first
    decomposed task of every affected recipe consumes all of that recipe's
    prep outputs, so ``build_task_graph`` creates real prep → recipe edges.
    This is a pure transform — no mutation of frozen CookingTask objects.
    Returns a mutable list because the caller still wires safety-task
    reverse dependencies into it (P0-07).
    """
    if not demand_final_states:
        return list(recipe_tasks)

    states_by_recipe: dict[str, list[str]] = {}
    for demand_id, state in demand_final_states.items():
        recipe_id = demand_id.split(":", 1)[0]
        states_by_recipe.setdefault(recipe_id, []).append(state)

    seen: set[str] = set()
    updated: list[CookingTask] = []
    for task in recipe_tasks:
        if task.dish_id in states_by_recipe and task.dish_id not in seen:
            seen.add(task.dish_id)
            extra = tuple(dict.fromkeys(states_by_recipe[task.dish_id]))
            task = task.model_copy(update={"consumes_states": task.consumes_states + extra})
        updated.append(task)
    return updated


async def merge_preparation_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Decompose recipe steps + merge shared preparation into CookingTasks.

    Iterates over validated RecipeIR steps, calls decompose_step for each,
    and collects the resulting CookingTasks. Preparation merging via prep_trie
    is deferred to MVP+1 (requires ingredient-demand-to-operation-chain bridge).

    Safety tasks are generated from the safety report when present.
    """
    from uuid import uuid4

    from cooking_plan_agent.domain.enums import WorkMode
    from cooking_plan_agent.domain.models import CookingTask, ResourceNeed, TaskDependency
    from cooking_plan_agent.preparation.decompose import decompose_step

    parsed_recipes = state.get("parsed_recipes", ())
    if not parsed_recipes:
        return {"recipe_tasks": (), "prep_tasks": (), "safety_tasks": ()}

    all_recipe_tasks: list[CookingTask] = []
    # Track the last task of each recipe to chain subsequent steps
    recipe_last_task: dict[str, str] = {}
    # P0-07: per-recipe map step_number → (first_task_id, last_task_id) so
    # safety insertions can anchor between exact recipe steps.
    step_task_anchors: dict[str, dict[int, tuple[str, str]]] = {}

    for recipe in parsed_recipes:
        last_task_id: str | None = None
        anchors: dict[int, tuple[str, str]] = {}
        for step in recipe.steps:
            tasks = decompose_step(recipe.recipe_id, step)
            if not tasks:
                continue

            # If this is not the first step in the recipe, add a dependency
            # from the previous step's last task to this step's first task
            if last_task_id is not None and tasks:
                first = tasks[0]
                dep = TaskDependency(predecessor_id=last_task_id)
                tasks = (first.model_copy(update={"dependencies": first.dependencies + (dep,)}),) + tasks[1:]

            all_recipe_tasks.extend(tasks)
            last_task_id = tasks[-1].task_id
            anchors[step.step_number] = (tasks[0].task_id, tasks[-1].task_id)

        step_task_anchors[recipe.recipe_id] = anchors
        if last_task_id is not None:
            recipe_last_task[recipe.recipe_id] = last_task_id

    # --- P2-01: shared preparation merging via prep_trie ---
    # Merges identical prep operations across recipes (one wash for N dishes)
    # and wires each recipe's first task to consume the prep outputs. When
    # the feature is disabled, prep_tasks stays empty (per-recipe prep).
    prep_tasks: tuple[CookingTask, ...] = ()
    prep_observations: tuple[str, ...] = ()
    from cooking_plan_agent.config.settings import get_settings

    if get_settings().shared_prep_enabled:
        from cooking_plan_agent.normalisation.errors import InvalidQuantityError
        from cooking_plan_agent.preparation.prep_trie import build_shared_prep_tasks

        demands = tuple((recipe.recipe_id, demand) for recipe in parsed_recipes for demand in recipe.ingredients)
        try:
            shared = build_shared_prep_tasks(demands)
        except InvalidQuantityError as exc:
            # D1: conservation failure must never produce a half-built task
            # graph — terminate to FAILED via INTERNAL_ERROR. P2-03: keep
            # only the exception type as diagnostic context.
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INTERNAL_ERROR.value,
                    message="Preparation quantity conservation failed",
                    correlation_id=state["request"].request_id,
                    node_name="merge_preparation",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }
        prep_tasks = shared.tasks
        prep_observations = shared.observations
        all_recipe_tasks = _wire_prep_consumption(all_recipe_tasks, shared.demand_final_states)

    # --- Safety tasks (P0-07: anchored insertions from the safety report) ---
    safety_task_list: list[CookingTask] = []
    safety_report = state.get("safety_report")
    if safety_report is not None:
        # 1. Structured insertions with exact step anchors.
        for insertion in safety_report.insertions:
            anchors = step_task_anchors.get(insertion.recipe_id, {})
            after_pair = anchors.get(insertion.after_step_number) if insertion.after_step_number is not None else None
            before_pair = (
                anchors.get(insertion.before_step_number) if insertion.before_step_number is not None else None
            )
            _after_first, after_last = after_pair if after_pair is not None else (None, None)
            before_first, _before_last = before_pair if before_pair is not None else (None, None)

            task_id = f"safety_{insertion.insertion_id}_{uuid4().hex[:8]}"
            deps: list[TaskDependency] = []
            if after_last is not None:
                # raw task → sanitise task
                deps.append(TaskDependency(predecessor_id=after_last))
            resources = tuple(ResourceNeed(quantity=1, resource_type=r) for r in insertion.required_resources)
            task = CookingTask(
                task_id=task_id,
                dish_id=insertion.recipe_id,
                instruction=insertion.task_instruction,
                duration_minutes=insertion.duration_minutes,
                work_mode=WorkMode.ACTIVE,
                category="safety",
                dependencies=tuple(deps),
                resources=resources,
                safety_tags=(insertion.rule_id,),
            )
            safety_task_list.append(task)

            # 2. sanitise task → RTE task (reverse dependency on the RTE head)
            if before_first is not None:
                rte_task = next((t for t in all_recipe_tasks if t.task_id == before_first), None)
                if rte_task is not None:
                    rte_dep = TaskDependency(predecessor_id=task_id)
                    updated = rte_task.model_copy(update={"dependencies": rte_task.dependencies + (rte_dep,)})
                    for idx, t in enumerate(all_recipe_tasks):
                        if t.task_id == before_first:
                            all_recipe_tasks[idx] = updated
                            break

        # 3. Fallback: legacy bare task IDs (no anchors) — keep the old
        #    behaviour for rules that still emit IDs only.
        anchored_ids = {s.insertion_id for s in safety_report.insertions}
        for task_id in safety_report.required_safety_task_ids:
            if any(s.rule_id.lower() in task_id for s in safety_report.insertions):
                continue  # already materialised via insertion above
            task = CookingTask(
                task_id=task_id,
                dish_id="_safety",
                instruction=f"Safety task: {task_id}",
                duration_minutes=3,
                work_mode=WorkMode.ACTIVE,
                category="safety",
                safety_tags=(task_id,),
            )
            safety_task_list.append(task)
        _ = anchored_ids  # keep for clarity

    return {
        "recipe_tasks": tuple(all_recipe_tasks),
        "prep_tasks": prep_tasks,
        "safety_tasks": tuple(safety_task_list),
        "prep_observations": prep_observations,
    }


async def build_task_graph_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Build the task DAG from recipe, prep, and safety tasks.

    Wired to existing preparation/task_graph.py.

    Lazy-imports build_task_graph inside the function so the module
    can be imported even when preparation dependencies are missing.
    """
    from cooking_plan_agent.preparation.task_graph import build_task_graph

    recipe_tasks = state.get("recipe_tasks", ())
    prep_tasks = state.get("prep_tasks", ())
    safety_tasks = state.get("safety_tasks", ())

    # Defensive: if merge_preparation returned nothing, skip building
    if not recipe_tasks and not prep_tasks:
        return {}

    try:
        graph = build_task_graph(
            recipe_tasks=recipe_tasks,
            prep_tasks=prep_tasks,
            safety_tasks=safety_tasks,
        )
        return {"task_graph": graph}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Cycle detection or invalid dependencies -> workflow error. P2-03:
        # only the exception type is retained as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.TASK_GRAPH_CYCLE.value,
                message="Task graph construction failed",
                correlation_id=state["request"].request_id,
                node_name="build_task_graph",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


async def solve_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Solve the CP-SAT scheduling problem.

    Wired to existing scheduling/orchestrator.py.
    schedule() returns tuple[ScheduleResult, VerificationReport] — we store
    only the result; verification is done independently in verify_schedule_node.

    Error semantics (P1-04): ``SCHEDULE_INFEASIBLE`` means ONLY that the
    solver proved no solution exists for a VALID model. Everything else uses
    a distinct code:
      - MODEL_INVALID → SCHEDULE_MODEL_INVALID (model construction bug)
      - UNKNOWN       → SCHEDULE_UNKNOWN (solver hit its limit, undetermined)
      - missing task graph → INTERNAL_ERROR (invariant break, never INFEASIBLE)
      - ValueError/TypeError during solve → SCHEDULE_MODEL_INVALID
      - RuntimeError from the solver → INTERNAL_ERROR
    """
    import asyncio

    from cooking_plan_agent.domain.enums import SolverStatus
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.orchestrator import ScheduleOrchestrator

    task_graph = state.get("task_graph")
    request = state["request"]
    if task_graph is None:
        # Missing DAG is an internal invariant failure, not a business
        # infeasibility — solve must never run without a task graph.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="No task graph available for scheduling — internal invariant violated",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=request.kitchen_resources,
        requested_time_limit_minutes=request.time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        # CP-SAT solving is CPU-bound — run it in a worker thread so the
        # event loop stays responsive (P1-02). The verifier is synchronous
        # and stays inside the solve call; it is not moved to a thread.
        # P3-03: ScheduleOrchestrator runs the lexicographic phases
        # (makespan → holding → context switch); Phase 4 stays gated.
        # The depth is configurable for rollback (solver_optimization_level).
        from cooking_plan_agent.config.settings import get_settings

        orchestrator = ScheduleOrchestrator()
        result, _ = await asyncio.to_thread(
            orchestrator.solve,
            problem,
            get_settings().solver_optimization_level,
        )
    except (ValueError, TypeError) as exc:
        # Model-construction phase: bad variable shapes, contradictory
        # constraints → the model was never valid. P2-03: keep only the
        # exception type as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="Scheduling model construction failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }
    except RuntimeError as exc:
        # Solver-internal failure (runtime) — not a business outcome.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="Scheduling solver failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }

    # Map solver status to a stable, distinct error code. Only INFEASIBLE is a
    # business outcome (routes to render_infeasible_response); MODEL_INVALID
    # and UNKNOWN are FAILED responses (P1-04).
    status = result.status
    if status == SolverStatus.MODEL_INVALID:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="The scheduling model is invalid — likely a data inconsistency",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }
    if status == SolverStatus.UNKNOWN:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_UNKNOWN.value,
                message="The solver could not determine feasibility within the time limit",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    return {"schedule_result": result}


async def verify_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Independent verification of solver output.

    Wired to existing scheduling/verifier.py.
    verify() signature: verify(problem: SchedulingProblem, result: ScheduleResult)

    Verification is done in a SEPARATE node (not inside solve_schedule) so that:
    - verification can be skipped/instrumented independently
    - the verifier catches bugs in the solver itself
    """
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

    schedule_result = state.get("schedule_result")
    if schedule_result is None:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No schedule result to verify",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No task graph available for verification",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
        requested_time_limit_minutes=state["request"].time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, schedule_result)
        return {"verification_report": report}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Verification failure is an invariant break — the solver output
        # must never reach the client. P2-03: public text comes from the
        # catalog; keep only the exception type as diagnostic context.
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="Schedule verification failed",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


# ============================================================================
# P4-01: schedule explanation (between verify and READY render)
# ============================================================================


def _build_schedule_summary(state: PlanState) -> dict[str, object]:
    """Build the compact, non-sensitive summary the explainer consumes (D3/D4).

    Only facts already present in the verified schedule are included:
    makespan minutes, per-dish completion minutes, and the maximum number of
    concurrently ACTIVE tasks (parallel groups). No recipe text, inventory,
    or user identity is ever included.
    """
    from cooking_plan_agent.rendering.builder import build_dish_completion_summary

    schedule = state.get("schedule_result")
    makespan: int = (schedule.makespan_minutes or 0) if schedule is not None else 0

    dish_completions: list[dict[str, object]] = []
    if schedule is not None:
        task_graph = state.get("task_graph")
        tasks = task_graph.tasks if task_graph is not None else ()
        for entry in build_dish_completion_summary(schedule, tasks):
            # builder emits "dish_id"; the explainer consumes "dish".
            raw_completion = entry.get("completion_minute")
            dish_completions.append(
                {
                    "dish": str(entry.get("dish_id") or "?"),
                    "completion_minute": int(raw_completion) if isinstance(raw_completion, int) else 0,
                }
            )

    return {
        "makespan_minutes": makespan,
        "dish_completions": dish_completions,
        "parallel_groups": _max_parallel_active(state),
    }


def _max_parallel_active(state: PlanState) -> int:
    """Maximum number of concurrently ACTIVE tasks across the timeline (D3).

    A simple sweep over (start, end) events of ACTIVE tasks gives the peak
    concurrency. Falls back to 0 when no schedule/timeline is available.
    """
    from cooking_plan_agent.domain.enums import WorkMode
    from cooking_plan_agent.rendering.builder import build_timeline

    schedule = state.get("schedule_result")
    if schedule is None:
        return 0

    task_graph = state.get("task_graph")
    tasks = task_graph.tasks if task_graph is not None else ()
    events: list[tuple[int, int]] = []  # (minute, +1 start / -1 end)
    for entry in build_timeline(schedule, tasks):
        if entry.get("work_mode") != WorkMode.ACTIVE.value:
            continue
        raw_start = entry.get("start_minute")
        raw_end = entry.get("end_minute")
        start = int(raw_start) if isinstance(raw_start, int) else 0
        end = int(raw_end) if isinstance(raw_end, int) else start
        events.append((start, 1))
        events.append((end, -1))
    # Half-open intervals [start, end): at a shared boundary the ending task
    # is already done before the starting task begins, so -1 sorts before +1.
    events.sort(key=lambda event: (event[0], event[1]))
    current = 0
    peak = 0
    for _minute, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _deterministic_explanation(summary: dict[str, object]) -> str:
    """Deterministic fallback: re-states only verified schedule facts (D3).

    Used when the LLM explainer is absent or fails. The content is always
    derived from the summary — no new claims are introduced.
    """
    makespan = summary.get("makespan_minutes")
    parts = [f"Plan completes in approximately {makespan} minutes."]
    raw_completions = summary.get("dish_completions")
    completions = raw_completions if isinstance(raw_completions, list) else []
    if completions:
        parts.append(
            "Dishes finish at: "
            + ", ".join(
                f"{entry.get('dish', '?')} at {entry.get('completion_minute', '?')} min" for entry in completions
            )
        )
    return " ".join(parts)


async def explain_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Attach a short, additive explanation to a verified schedule (P4-01).

    Placed between verify_schedule and render_ready_response. The node NEVER
    writes a WorkflowError: an absent explainer, LLM timeout, malformed output
    or any exception degrades to a deterministic summary, so the verified
    READY response is never blocked (P2-02 fault matrix).

    Returns state fields:
      - explanation: prose or None (feature disabled).
      - explanation_source: "llm" | "deterministic" | "disabled".
    """
    from cooking_plan_agent.config.settings import get_settings

    if not get_settings().explanation_enabled:
        return {"explanation": None, "explanation_source": "disabled"}

    summary = _build_schedule_summary(state)
    explainer = runtime.context.explainer
    if explainer is not None:
        try:
            text = await explainer.explain(summary)
            if isinstance(text, str) and text.strip():
                return {"explanation": text, "explanation_source": "llm"}
        except Exception:  # noqa: BLE001 — additive capability must never fail READY
            logger.warning("Schedule explanation failed — using deterministic fallback")

    return {
        "explanation": _deterministic_explanation(summary),
        "explanation_source": "deterministic",
    }


# ============================================================================
# Terminal response nodes
# ============================================================================


async def render_ready_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render READY response with verified schedule, mise en place, and checklist.

    Delegates to rendering.responses.render_ready_response.
    """
    from cooking_plan_agent.rendering.responses import render_ready_response

    response = render_ready_response(state)
    return {"response": response}


async def render_infeasible_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render INFEASIBLE response with ordered reasons from all sources.

    Delegates to rendering.responses.render_infeasible_response.
    """
    from cooking_plan_agent.rendering.responses import render_infeasible_response

    response = render_infeasible_response(state)
    return {"response": response}


async def render_failed_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render FAILED response with stable error code and correlation ID.

    Delegates to rendering.responses.render_failed_response.

    P2-03: emits a structured diagnostic log line (error code, node,
    correlation ID, controlled diagnostics) so failures can be traced via
    correlation ID without writing the raw internal message to the log.
    """
    from cooking_plan_agent.rendering.responses import render_failed_response

    error = state.get("error")
    if error is not None:
        logger.warning(
            "Workflow FAILED | error_code=%s | node=%s | correlation_id=%s | recoverable=%s | diagnostics=%s",
            error.error_code,
            error.node_name,
            error.correlation_id,
            error.recoverable,
            error.diagnostics,
        )

    response = render_failed_response(state)
    return {"response": response}
