"""Contract adapter — maps Spring Boot v1 DTOs to/from internal domain models.

P0-02 design decision: the compat layer is the ONLY place that understands
Java DTO details.  Core domain services and the workflow never import
``api.compat_models``.  This adapter is the decoupling boundary:

  - ``build_internal_request``      CompatCookingRequest → GeneratePlanRequest
  - ``to_compat_response``          PlanResponse → CompatCookingResponse
  - ``deadline_budget_seconds``     deadlineAt → execution budget (fast fail)

Key contract rules honoured here (CookingPlanResultValidator equivalents):

  * The agent MUST return the request's own servings, requestId, planId,
    traceId and contractVersion unchanged.
  * sourceRecipeId MUST be one of the supplied candidate recipeIds.
  * A non-READY terminal state maps to ``status="FAILED"`` (the Java adapter
    then treats it as AGENT_UNAVAILABLE — a safe terminal state).
  * Candidate snapshots are mapped to internal structured candidates so the
    workflow never calls the LLM for compat requests.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from cooking_plan_agent.api.compat_models import (
    CONTRACT_VERSION,
    CompatCandidateRequest,
    CompatCookingRequest,
    CompatCookingResponse,
    CompatIngredientResponse,
    CompatStepResponse,
)
from cooking_plan_agent.domain.models import (
    ExtractedIngredient,
    ExtractedRecipeCandidate,
    ExtractedStep,
    GeneratePlanRequest,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    PlanResponse,
    ReadyPlanResponse,
    RecipeInput,
)

# Batch-size caps enforced on the Java side (CookingPlanResultValidator).
_MAX_INGREDIENTS = 50
_MAX_STEPS = 30
_MAX_WARNINGS = 10

# Warning allow-list members we may emit (subset of the Java enum).
_BUDGET_ESTIMATE_ONLY = "BUDGET_ESTIMATE_ONLY"


def _compatibility_kitchen_resources() -> tuple[KitchenResourceSnapshot, ...]:
    """Supply the standard-kitchen assumption required by the frozen v1 DTO.

    The Backend v1 contract predates explicit kitchen-resource snapshots. The
    compatibility workflow must therefore retain its original semantics: it
    evaluates controlled recipe candidates against a small standard kitchen,
    while native v2 callers continue to send their actual resource inventory.
    """
    from cooking_plan_agent.normalisation.names import ESSENTIAL_RESOURCE_TYPES

    return tuple(
        KitchenResourceSnapshot(
            resource_id=f"compat-{resource_type}",
            resource_type=resource_type,
            capacity=Decimal(4 if resource_type == "stove" else 1),
            capacity_unit="burners" if resource_type == "stove" else None,
        )
        for resource_type in sorted(ESSENTIAL_RESOURCE_TYPES)
    )


# ===========================================================================
# Request direction — CompatCookingRequest → GeneratePlanRequest
# ===========================================================================


def _to_extracted_candidate(candidate: CompatCandidateRequest) -> ExtractedRecipeCandidate:
    """Map one candidate snapshot to an internal ExtractedRecipeCandidate.

    The snapshot is the serialised RecipeCandidate (name, ingredients,
    steps, dietary tags…), so this is a pure structure-to-structure map —
    no LLM, no text parsing (P0-02 rule 4).
    """
    snapshot = candidate.snapshot

    ingredients = tuple(
        ExtractedIngredient(
            raw_text=ing.ingredientName,
            name=ing.ingredientName,
            quantity=ing.quantity,
            unit=ing.unit,
            confidence=Decimal("1.0"),
            extraction_source="SNAPSHOT",
        )
        for ing in snapshot.ingredients
        if ing.ingredientName
    )

    steps = tuple(
        ExtractedStep(
            step_number=step.stepNo,
            instruction=step.instruction,
            extraction_source="SNAPSHOT",
        )
        for step in snapshot.steps
    )

    return ExtractedRecipeCandidate(
        recipe_id=str(candidate.recipeId),
        dish_name=snapshot.name,
        original_servings=Decimal(snapshot.defaultServings),
        source_language="en",
        ingredients=ingredients,
        steps=steps,
        extraction_source="SNAPSHOT",
    )


def _to_inventory_lots(compat: CompatCookingRequest) -> tuple[InventoryLotSnapshot, ...]:
    """Map request.ingredients (user-owned items) to inventory lots.

    The Java caller lists the ingredients the user already has; treating
    them as inventory lets the internal feasibility check use them instead
    of flagging a shortage.
    """
    lots: list[InventoryLotSnapshot] = []
    for item in compat.request.ingredients:
        lots.append(
            InventoryLotSnapshot(
                lot_id=f"compat-{item.ingredientName}-{len(lots)}",
                item_id=item.ingredientName,
                canonical_name=item.ingredientName,
                on_hand=item.quantity,
                reserved=Decimal(0),
                unit=item.unit,
            )
        )
    return tuple(lots)


def build_internal_request(compat: CompatCookingRequest) -> GeneratePlanRequest:
    """Build the internal GeneratePlanRequest from a compat request.

    Recipes are carried as ``preparsed_candidates`` so the workflow skips
    LLM extraction; the ``recipes`` field is populated with lightweight
    descriptors for traceability/logging only.
    """
    request_snapshot = compat.request
    candidates = tuple(
        _to_extracted_candidate(c) for c in compat.candidates if c.snapshot.ingredients or c.snapshot.steps
    )

    # Lightweight descriptors mirroring the native request shape.  The
    # workflow ignores them when preparsed_candidates is non-empty.
    recipes = tuple(
        RecipeInput(
            recipe_id=str(c.recipeId),
            text=c.snapshot.name or "Compatibility candidate",
            target_servings=Decimal(str(request_snapshot.servings)),
        )
        for c in compat.candidates
    )

    return GeneratePlanRequest(
        request_id=str(compat.requestId),
        user_id=str(compat.planId),
        recipes=recipes,
        dietary_restrictions=request_snapshot.constraints.requiredDietaryTagCodes,
        user_allergens=request_snapshot.constraints.avoidAllergenCodes,
        time_limit_minutes=request_snapshot.maxMinutes,
        inventory_lots=_to_inventory_lots(compat),
        kitchen_resources=_compatibility_kitchen_resources(),
        approved_decisions=(),
        schema_version="1.0",
        preparsed_candidates=candidates,
    )


def selected_recipe_id(compat: CompatCookingRequest) -> UUID | None:
    """Return the primary candidate recipeId (first candidate).

    MVP rule: the agent selects the first controlled candidate.  This must
    be one of the supplied recipeIds (Java UNKNOWN_RECIPE check).
    """
    if not compat.candidates:
        return None
    return compat.candidates[0].recipeId


# ===========================================================================
# Deadline / execution budget
# ===========================================================================


def deadline_budget_seconds(deadline_at: datetime | None, now: datetime) -> float | None:
    """Return the remaining execution budget for a request, or None if absent.

    If the deadline has already passed, returns 0.0 so the caller can fast
    fail before invoking the workflow (P0-02 rule 7).
    """
    if deadline_at is None:
        return None
    remaining = (deadline_at - now).total_seconds()
    return max(0.0, remaining)


# ===========================================================================
# Response direction — PlanResponse → CompatCookingResponse
# ===========================================================================


def _build_ingredients(
    compat: CompatCookingRequest,
    response: ReadyPlanResponse,
) -> tuple[CompatIngredientResponse, ...]:
    """Map the READY plan's ingredients to the v1 response shape.

    Ingredients come from the selected candidate's snapshot (authoritative
    structured source).  Items covered by the completion checklist are
    AVAILABLE; everything else is TO_BUY.
    """
    selected = compat.candidates[0] if compat.candidates else None
    if selected is None:
        return ()

    available_names = {item.ingredient_name.lower().strip() for item in response.completion_checklist}

    result: list[CompatIngredientResponse] = []
    for seq, ing in enumerate(selected.snapshot.ingredients, start=1):
        if not ing.ingredientName:
            continue
        if len(result) >= _MAX_INGREDIENTS:
            break
        availability: str = "AVAILABLE" if ing.ingredientName.lower().strip() in available_names else "TO_BUY"
        result.append(
            CompatIngredientResponse(
                sequenceNo=seq,
                ingredientName=ing.ingredientName,
                quantity=ing.quantity,
                unit=ing.unit,
                availability=availability,  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def _build_steps(compat: CompatCookingRequest) -> tuple[CompatStepResponse, ...]:
    """Map the selected candidate's steps to the v1 response shape."""
    selected = compat.candidates[0] if compat.candidates else None
    if selected is None:
        return ()

    result: list[CompatStepResponse] = []
    for step in selected.snapshot.steps:
        if not step.instruction:
            continue
        if len(result) >= _MAX_STEPS:
            break
        result.append(
            CompatStepResponse(
                stepNo=len(result) + 1,
                instruction=step.instruction,
            )
        )
    return tuple(result)


def to_compat_response(
    compat: CompatCookingRequest,
    response: PlanResponse,
    source_recipe_id: UUID | None,
    agent_trace_id: str | None = None,
) -> CompatCookingResponse:
    """Map an internal PlanResponse to the v1 CompatCookingResponse.

    READY  → ``status="SUCCEEDED"`` with ingredients/steps/totalMinutes.
    Other → ``status="FAILED"`` (Java adapter maps to AGENT_UNAVAILABLE).

    contractVersion / requestId / planId / traceId are always echoed back
    unchanged (Java SCHEMA_MISMATCH / UNSUPPORTED_VERSION checks).
    """
    base = CompatCookingResponse(
        contractVersion=compat.contractVersion,
        requestId=compat.requestId,
        planId=compat.planId,
        traceId=compat.traceId,
        agentTraceId=agent_trace_id or uuid4().hex,
        status="FAILED",
        servings=compat.request.servings,
    )

    if not isinstance(response, ReadyPlanResponse):
        # Non-READY terminal state → FAILED; keep correlation ID for triage.
        return base

    steps = _build_steps(compat)
    ingredients = _build_ingredients(compat, response)
    # Cost/currency come from the selected candidate when the caller asked
    # for a budget; otherwise leave them empty (validator allows nulls).
    selected = compat.candidates[0] if compat.candidates else None
    estimated_cost = selected.snapshot.estimatedCost if selected is not None else None
    currency = selected.snapshot.currency if selected is not None else None

    return base.model_copy(
        update={
            "status": "SUCCEEDED",
            "sourceRecipeId": source_recipe_id,
            "totalMinutes": response.makespan_minutes,
            "estimatedCost": estimated_cost,
            "currency": currency,
            "ingredients": ingredients,
            "steps": steps,
            "warnings": _build_warnings(compat, estimated_cost),
        }
    )


def _build_warnings(compat: CompatCookingRequest, estimated_cost: Decimal | None) -> tuple[object, ...]:
    """Build an allow-list-compliant warnings list.

    Currently a single BUDGET_ESTIMATE_ONLY note when a cost is present
    and the caller supplied a budget — mirrors the Java fixture.  All codes
    must be in the validator's allow-list.
    """
    warnings: list[object] = []
    if estimated_cost is not None and compat.request.maxBudget is not None:
        from cooking_plan_agent.api.compat_models import CompatWarningResponse

        warnings.append(
            CompatWarningResponse(
                sequenceNo=1,
                warningCode=_BUDGET_ESTIMATE_ONLY,  # type: ignore[arg-type]
                message="Costs are an estimate, not a live price.",
            )
        )
    return tuple(warnings[:_MAX_WARNINGS])


def is_contract_supported(contract_version: str) -> bool:
    """Return True when the caller's contract version is supported."""
    return contract_version == CONTRACT_VERSION
