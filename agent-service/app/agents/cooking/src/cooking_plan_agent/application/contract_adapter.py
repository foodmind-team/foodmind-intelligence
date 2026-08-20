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
# 以下上限由 Java 侧 CookingPlanResultValidator 强制，此处保持一致
_MAX_INGREDIENTS = 50
_MAX_STEPS = 30
_MAX_WARNINGS = 10

# Warning allow-list members we may emit (subset of the Java enum).
# 允许发出的告警码白名单（Java 枚举的子集）
_BUDGET_ESTIMATE_ONLY = "BUDGET_ESTIMATE_ONLY"


def _compatibility_kitchen_resources() -> tuple[KitchenResourceSnapshot, ...]:
    """Supply the standard-kitchen assumption required by the frozen v1 DTO.

    The Backend v1 contract predates explicit kitchen-resource snapshots. The
    compatibility workflow must therefore retain its original semantics: it
    evaluates controlled recipe candidates against a small standard kitchen,
    while native v2 callers continue to send their actual resource inventory.
    """
    from cooking_plan_agent.normalisation.names import ESSENTIAL_RESOURCE_TYPES

    # 兼容路径固定假设一套"标准厨房"（stove 4 个炉眼、其余 1 份），
    # 因为冻结的 v1 DTO 早于显式厨房资源快照；v2 原生调用方仍传真实库存。
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

    # 食材：纯结构到结构映射，置信度固定 1.0，来源标记为 SNAPSHOT（无 LLM、无文本解析）
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

    # 组装内部候选模型：原始份数、语言固定 en、来源 SNAPSHOT
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
        # 把调用方已有食材视为库存，让内部可行性检查直接使用而不再报缺货
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
    # 仅保留带食材或步骤的候选快照，避免空快照进入工作流
    candidates = tuple(
        _to_extracted_candidate(c) for c in compat.candidates if c.snapshot.ingredients or c.snapshot.steps
    )

    # 轻量描述符：镜像原生请求结构；preparsed_candidates 非空时工作流会忽略它
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
    # MVP 规则：选择第一个受控候选；必须命中调用方提供的 recipeId（Java UNKNOWN_RECIPE 校验）
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
    # 已过截止时间则返回 0.0，让调用方在进入工作流前快速失败（P0-02 rule 7）
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

    # 完成清单里的食材视为 AVAILABLE，其余为 TO_BUY
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
        # 重新编号：stepNo 取当前结果长度 + 1，保证从 1 连续
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

    # 非 READY 终态一律映射为 FAILED（Java 侧再视为 AGENT_UNAVAILABLE，安全终态）
    if not isinstance(response, ReadyPlanResponse):
        # 保留关联 ID 以便排查
        return base

    steps = _build_steps(compat)
    ingredients = _build_ingredients(compat, response)
    # 仅当调用方要求预算时，从选中候选取费用/币种；否则留空（校验器允许 null）
    selected = compat.candidates[0] if compat.candidates else None
    estimated_cost = selected.snapshot.estimatedCost if selected is not None else None
    currency = selected.snapshot.currency if selected is not None else None

    # READY → SUCCEEDED，回填 sourceRecipeId、总时长、费用、食材、步骤、告警
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
    # 有费用且调用方提供了预算时，发一条"费用仅为估算"的告警（须在白名单内）
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
    # 仅支持冻结的 CONTRACT_VERSION，其余一律拒绝（UNSUPPORTED_VERSION）
    return contract_version == CONTRACT_VERSION
