"""Workflow node implementations for a single pipeline stage.

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.
"""

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    FeasibilityReport,
    IngredientDemand,
    RepairOption,
    SafetyContext,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState


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
    from cooking_plan_agent.normalisation.names import (
        normalise_essential_resource,
        normalise_resource_type,
    )

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
    # resources_hint is a soft hint from extraction (LLM free text or keyword
    # regex). Only hints that resolve to an ESSENTIAL equipment type gate
    # feasibility — consumables/containers/unknown labels (e.g. 剪刀、厨房纸、
    # 碗、锅盖) never block a plan on their own, so an operation that only
    # needs a knife isn't reported infeasible because its text also mentions
    # other tools.
    missing_resources: list[str] = []
    if request.kitchen_resources:
        available_types = {normalise_resource_type(r.resource_type) for r in request.kitchen_resources if r.available}
        for recipe in parsed_recipes:
            for step in recipe.steps:
                for hint in step.resources_hint:
                    canonical = normalise_essential_resource(hint)
                    if canonical is not None and canonical not in available_types:
                        if canonical not in missing_resources:
                            missing_resources.append(canonical)

    is_feasible = ingredient_report.is_feasible and len(missing_resources) == 0

    # --- Generate repair options when infeasible ---
    repair_options: tuple[RepairOption, ...] = ()
    if not is_feasible:
        # 削减份量以用户请求的份量为基准（取各菜谱 target_servings 的最大值），
        # 而非固定默认 2——否则 4 人份菜单会错误地建议「from 2 to 1」。
        from decimal import Decimal as _Decimal

        from cooking_plan_agent.repair.options import (
            propose_equipment_alternatives,
            propose_ingredient_substitutions,
            propose_portion_adjustments,
            rank_repair_options,
        )

        base_servings: _Decimal = max(
            (r.target_servings for r in parsed_recipes),
            default=_Decimal(2),
        )

        opts = list(propose_ingredient_substitutions(ingredient_report.ingredient_shortages))
        opts.extend(propose_portion_adjustments(ingredient_report.ingredient_shortages, base_servings))
        opts.extend(propose_equipment_alternatives(tuple(missing_resources)))
        repair_options = rank_repair_options(tuple(opts))

    return {
        "feasibility_report": FeasibilityReport(
            report_id=ingredient_report.report_id,
            ingredient_shortages=ingredient_report.ingredient_shortages,
            missing_resources=tuple(sorted(missing_resources)),
            is_feasible=is_feasible,
            # 透传完整分配结果，保证 READY 消耗清单可用（不丢弃满足食材的 FEFO 分配）。
            ingredient_results=ingredient_report.ingredient_results,
        ),
        "repair_options": repair_options,
    }


async def build_confirmation_response_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Render NEEDS_CONFIRMATION response with assumptions and repair options.

    Delegates to rendering.responses.render_confirmation_response.

    P4-02: the rendered response (which carries the structured
    ``confirmation_questions``) is ALSO written to the state's
    ``confirmation_context`` field, so downstream/async consumers can read
    the field-level confirmation form without re-rendering.
    """
    from cooking_plan_agent.rendering.responses import render_confirmation_response

    response = render_confirmation_response(state)
    return {"response": response, "confirmation_context": response}
