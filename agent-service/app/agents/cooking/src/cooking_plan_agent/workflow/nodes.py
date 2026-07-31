"""LangGraph workflow nodes — thin wrappers around domain services.

Per handbook 8.4: each node calls ONE application/domain service and
returns only CHANGED state fields. No in-place state mutation.
No broad exception catching that masks errors as partial success.
"""

# LangGraph runtime type — context is injected by the framework
from langgraph.runtime import Runtime

from cooking_plan_agent.domain.models import (
    Assumption,
    ExtractedRecipeCandidate,
    FeasibilityReport,
    IngredientDemand,
    RecipeGap,
    RecipeIR,
    SafetyContext,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

# ============================================================================
# Input & parsing nodes
# ============================================================================


async def validate_input_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Validate the incoming GeneratePlanRequest.

    Checks: non-empty recipes, reasonable serving/task counts, schema version.
    STUB: passes through. Full validation when contract is finalized.

    Returns an empty dict on success or {'error': WorkflowError} on failure.
    The graph does NOT have an error edge from validate_input, so an error
    here will propagate as a runtime exception.
    """
    request = state["request"]
    if not request.recipes:
        return {
            "error": WorkflowError(
                error_code="INVALID_RECIPE_TEXT",
                message="Request contains no recipes",
                correlation_id=request.request_id,
                node_name="validate_input",
            )
        }
    return {}


async def parse_recipes_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Extract structured candidates from each recipe's raw text.

    Uses the RecipeExtractor from WorkflowContext (rule-based or LLM-backed).
    Each recipe in the request is individually extracted, then aggregated.
    Falls back to rule-based extraction if no extractor is configured.
    """
    request = state["request"]
    extractor = runtime.context.recipe_extractor

    candidates: list[ExtractedRecipeCandidate] = []

    if extractor is not None:
        # Use configured extractor (LLM or rule-based from WorkflowContext)
        for recipe in request.recipes:
            try:
                text = recipe.get("text", "")
                if text:
                    candidate = await extractor.extract(text)
                    candidates.append(candidate)
            except Exception as exc:  # noqa: BLE001 — per-node error → error state
                return {
                    "error": WorkflowError(
                        error_code="EXTERNAL_PROVIDER_UNAVAILABLE",
                        message=f"Recipe extraction failed: {exc}",
                        correlation_id=request.request_id,
                        node_name="parse_recipes",
                    )
                }
    else:
        # No extractor configured — use built-in rule-based extractor
        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor

        rule_extractor = RuleExtractor()
        for recipe in request.recipes:
            text = recipe.get("text", "")
            if text:
                candidate = await rule_extractor.extract(text)
                candidates.append(candidate)

    return {"extracted_candidates": tuple(candidates)}


async def detect_gaps_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
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
) -> dict:
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
) -> dict:
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
    from cooking_plan_agent.domain.models import ReconciledEvidence
    from cooking_plan_agent.research.researcher import Researcher

    research_evidence: dict[str, ReconciledEvidence] = {}

    if isinstance(researcher, Researcher):
        for gap in researchable_gaps[:2]:  # At most 2 queries (handbook 10.9)
            try:
                reconciled = await researcher.resolve_gap(gap, dish_name)
                research_evidence[gap.gap_id] = reconciled
            except Exception:  # noqa: BLE001 — any failure → confirmation, not unsafe guess
                # Search failure routes to confirmation (handbook 10.9)
                research_evidence[gap.gap_id] = ReconciledEvidence(
                    source_count=0,
                    needs_confirmation=True,
                )
    else:
        # Non-Researcher RecipeResearcher — fallback to Protocol's research()
        for gap in researchable_gaps[:2]:
            from cooking_plan_agent.domain.models import EvidenceQuery
            from cooking_plan_agent.research.query_builder import build_minimal_query

            query_text = build_minimal_query(gap, dish_name)
            query = EvidenceQuery(
                query_text=query_text,
                gap_type=gap.gap_class,
                recipe_context=dish_name,
            )
            try:
                results = await researcher.research(query)
                if results:
                    # Map results to ReconciledEvidence as best-effort
                    research_evidence[gap.gap_id] = ReconciledEvidence(
                        source_count=len(results),
                        needs_confirmation=False,
                    )
                else:
                    research_evidence[gap.gap_id] = ReconciledEvidence(
                        source_count=0,
                        needs_confirmation=True,
                    )
            except Exception:  # noqa: BLE001 — any failure → confirmation
                research_evidence[gap.gap_id] = ReconciledEvidence(
                    source_count=0,
                    needs_confirmation=True,
                )

    return {"research_evidence": research_evidence}


async def validate_recipe_ir_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Build validated RecipeIR objects from candidates.

    Converts each ExtractedRecipeCandidate → RecipeIR via build_recipe_ir,
    then runs semantic validation (validate_recipe_ir_semantics). Produces
    parsed_recipes in state for downstream scheduling nodes.

    If semantic validation fails with errors, returns a workflow error
    that will be routed to FAILED terminal.
    """
    from cooking_plan_agent.parsing.ir_builder import (
        build_recipe_ir,
        validate_recipe_ir_semantics,
    )

    candidates = state.get("extracted_candidates", ())
    if not candidates:
        return {
            "error": WorkflowError(
                error_code="INVALID_RECIPE_TEXT",
                message="No extracted recipe candidates to validate",
                correlation_id=state["request"].request_id,
                node_name="validate_recipe_ir",
            )
        }

    # Build RecipeIR from each candidate
    recipes: list[RecipeIR] = []
    for candidate in candidates:
        try:
            recipe_ir = build_recipe_ir(candidate)
            recipes.append(recipe_ir)
        except (ValueError, TypeError) as exc:
            return {
                "error": WorkflowError(
                    error_code="INVALID_RECIPE_TEXT",
                    message=f"Failed to build RecipeIR: {exc}",
                    correlation_id=state["request"].request_id,
                    node_name="validate_recipe_ir",
                )
            }

    # Semantic validation
    report = validate_recipe_ir_semantics(tuple(recipes))
    if not report.passed:
        error_details = "; ".join(i.message for i in report.issues if i.severity == "error")
        return {
            "error": WorkflowError(
                error_code="INVALID_RECIPE_TEXT",
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
) -> dict:
    """Evaluate all safety rules against the recipe set.

    Builds a SafetyContext from request + parsed_recipes, then delegates
    to the SafetyRuleEngine from WorkflowContext. When no engine is
    configured (backwards-compat), returns a pass-through safe report.

    Handbook 5.7: safety_validator node — first hard gate after parsing.
    """
    safety_engine = runtime.context.safety_engine
    if safety_engine is None:
        # No engine wired — pass-through safe (MVP backwards-compat)
        return {
            "safety_report": SafetyReport(
                report_id="stub-safety",
                findings=(),
                is_safe=True,
                has_unrepairable=False,
                required_safety_task_ids=(),
            )
        }

    parsed_recipes = state.get("parsed_recipes", ())
    request = state["request"]

    context = SafetyContext(
        recipes=parsed_recipes,
        dietary_restrictions=request.dietary_restrictions,
        user_allergens=request.user_allergens,
        inventory_lots=request.inventory_lots,
        cooking_date=None,  # MVP: no cooking date from request yet
    )

    report = safety_engine.evaluate(context)
    return {"safety_report": report}


async def check_feasibility_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
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
        cooking_date=None,  # MVP: no cooking_date from request yet
    )

    # --- Resource pre-check (from step hints, pre-decomposition) ---
    missing_resources: list[str] = []
    if request.kitchen_resources:
        available_types = {
            r.resource_type.lower()
            for r in request.kitchen_resources
            if r.available
        }
        for recipe in parsed_recipes:
            for step in recipe.steps:
                for hint in step.resources_hint:
                    if hint.lower() not in available_types:
                        if hint not in missing_resources:
                            missing_resources.append(hint)

    is_feasible = ingredient_report.is_feasible and len(missing_resources) == 0

    # --- Generate repair options when infeasible ---
    repair_options: tuple = ()
    if not is_feasible:
        from cooking_plan_agent.repair.options import (
            propose_dish_replacements,
            propose_equipment_alternatives,
            propose_ingredient_substitutions,
            propose_portion_adjustments,
            rank_repair_options,
        )

        recipe_names = tuple(r.dish_name for r in parsed_recipes)

        opts = list(
            propose_ingredient_substitutions(ingredient_report.ingredient_shortages)
        )
        opts.extend(
            propose_portion_adjustments(ingredient_report.ingredient_shortages)
        )
        opts.extend(
            propose_equipment_alternatives(tuple(missing_resources))
        )
        opts.extend(
            propose_dish_replacements(ingredient_report.ingredient_shortages, recipe_names)
        )
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
) -> dict:
    """Render NEEDS_CONFIRMATION response with assumptions and repair options.

    Delegates to rendering.responses.render_confirmation_response.
    """
    from cooking_plan_agent.rendering.responses import render_confirmation_response

    response = render_confirmation_response(state)
    return {"response": response}


# ============================================================================
# Preparation & scheduling nodes
# ============================================================================


async def merge_preparation_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Decompose recipe steps + merge shared preparation into CookingTasks.

    Uses existing preparation/decompose.py and preparation/prep_trie.py.
    STUB: returns empty task sets. Full wiring requires ingredient demand
    to operation chain extraction bridge.

    Returns three task tuples — downstream nodes concatenate them before
    building the DAG.
    """
    return {
        "recipe_tasks": (),
        "prep_tasks": (),
        "safety_tasks": (),
    }


async def build_task_graph_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
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
        # Cycle detection or invalid dependencies -> workflow error
        return {
            "error": WorkflowError(
                error_code="TASK_GRAPH_CYCLE",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="build_task_graph",
            )
        }


async def solve_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Solve the CP-SAT scheduling problem.

    Wired to existing scheduling/orchestrator.py.
    schedule() returns tuple[ScheduleResult, VerificationReport] — we store
    only the result; verification is done independently in verify_schedule_node.

    Lazy-imports inside the function avoid coupling to OR-Tools at import time
    (OR-Tools is a heavy C++ dependency).
    """
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.orchestrator import schedule as solve_schedule_fn

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_INFEASIBLE",
                message="No task graph available for scheduling",
                correlation_id=state["request"].request_id,
                node_name="solve_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
    )

    try:
        # schedule() returns (ScheduleResult, VerificationReport)
        result, _ = solve_schedule_fn(problem)
        return {"schedule_result": result}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_INFEASIBLE",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="solve_schedule",
            )
        }


async def verify_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
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
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message="No schedule result to verify",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message="No task graph available for verification",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
    )

    try:
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, schedule_result)
        return {"verification_report": report}
    except (ValueError, TypeError, RuntimeError) as exc:
        return {
            "error": WorkflowError(
                error_code="SCHEDULE_VERIFICATION_FAILED",
                message=str(exc),
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }


# ============================================================================
# Terminal response nodes
# ============================================================================


async def render_ready_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render READY response with verified schedule, mise en place, and checklist.

    Delegates to rendering.responses.render_ready_response.
    """
    from cooking_plan_agent.rendering.responses import render_ready_response

    response = render_ready_response(state)
    return {"response": response}


async def render_infeasible_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render INFEASIBLE response with ordered reasons from all sources.

    Delegates to rendering.responses.render_infeasible_response.
    """
    from cooking_plan_agent.rendering.responses import render_infeasible_response

    response = render_infeasible_response(state)
    return {"response": response}


async def render_failed_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    """Render FAILED response with stable error code and correlation ID.

    Delegates to rendering.responses.render_failed_response.
    """
    from cooking_plan_agent.rendering.responses import render_failed_response

    response = render_failed_response(state)
    return {"response": response}
