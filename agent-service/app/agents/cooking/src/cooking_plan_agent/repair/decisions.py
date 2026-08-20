"""Validation and application of user-confirmed repair decisions."""

from __future__ import annotations

import re
from decimal import Decimal

from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    ConfirmationQuestion,
    GeneratePlanRequest,
    QuestionAnswer,
    QuestionResponseType,
    RecipeIR,
    RepairOption,
)


def apply_approved_decisions(
    request: GeneratePlanRequest,
    approved_ids: tuple[str, ...],
    available_options: tuple[RepairOption, ...],
) -> dict[str, object]:
    """Apply user-approved repair decisions to produce a resolved plan input.

    This is a planning step — it modifies the request context (e.g. removes
    dietary restrictions, adjusts time limits) and records which decisions
    were approved. Downstream nodes use the approved_decisions field.

    Current MVP: passes approved decision IDs through to request context.
    Full ingredient/recipe mutation is deferred to the rendering layer
    where per-dish context is available.

    Args:
        request: The original GeneratePlanRequest.
        approved_ids: IDs of RepairOptions the user approved.
        available_options: All options that were presented.

    Returns:
        Dict with keys: 'request' (updated GeneratePlanRequest or dict of
        modifications) and 'applied_count' (int).
    """
    approved_set = set(approved_ids)
    applied: list[str] = []

    modifications: dict[str, object] = {}

    for opt in available_options:
        if opt.option_id not in approved_set:
            continue

        applied.append(opt.option_id)

        if opt.option_type == "extend_time":
            # Extract the proposed new time limit from the option description
            import re

            match = re.search(r"to (\d+) minutes", opt.description)
            if match:
                modifications["time_limit_minutes"] = int(match.group(1))

        elif opt.option_type == "reduce_servings":
            import re

            match = re.search(r"from ([\d.]+) to ([\d.]+)", opt.description)
            if match:
                modifications["target_servings"] = Decimal(match.group(2))

        # Other option types (substitute, equipment, dish replacement, purchase)
        # are deferred to rendering layer for now.

    return {
        "applied_count": len(applied),
        "applied_ids": tuple(applied),
        "modifications": modifications,
    }


# =============================================================================
# 5.26  Structured decision loop (P0-06)
#      结构化决策循环（P0-06）
# =============================================================================

# The decision kinds the confirmation loop supports (P0-06 rule 5).
# "purchase" (外出采购) is confirmable end-to-end: selecting it emits a
# structured decision the client echoes back; applying it is a no-op on
# the schedule inputs — the client buys the missing ingredients, updates
# the inventory snapshot in the backend, and resubmits the request.
SUPPORTED_DECISION_TYPES = frozenset(
    {
        "reduce_servings",
        "extend_time",
        "substitute_ingredient",
        "alternative_equipment",
        "replace_dish",
        "purchase",
        "provide_gap_value",
    }
)

_PURCHASE_BUNDLE_OPTION_ID = "repair_purchase_bundle"


def build_approved_decisions(
    repair_options: tuple[RepairOption, ...],
    plan_revision: str | None,
) -> tuple[ApprovedDecision, ...]:
    """Convert presented RepairOptions into structured, submittable decisions.

    将已展示的 RepairOption 转换为结构化、可提交的决策。

    Every supported option becomes an ApprovedDecision using the option's
    machine-readable payload. Human-facing descriptions are never parsed
    back into business data.

    每个受支持的选项都使用其机器可读的负载成为 ApprovedDecision。
    面向人类的描述绝不会被反向解析为业务数据。
    """
    decisions: list[ApprovedDecision] = []
    purchase_items: list[dict[str, object]] = []
    for option in repair_options:
        if option.option_type not in SUPPORTED_DECISION_TYPES:
            continue
        if option.option_type == "purchase":
            payload = option.payload
            if all(payload.get(key) is not None for key in ("ingredient_name", "quantity", "unit")):
                purchase_items.append(
                    {
                        "ingredient_name": payload["ingredient_name"],
                        "quantity": payload["quantity"],
                        "unit": payload["unit"],
                    }
                )
            continue
        decisions.append(
            ApprovedDecision(
                option_id=option.option_id,
                option_type=option.option_type,
                payload=dict(option.payload),
                plan_revision=plan_revision,
            )
        )
    if purchase_items:
        decisions.append(
            ApprovedDecision(
                option_id=_PURCHASE_BUNDLE_OPTION_ID,
                option_type="purchase",
                payload={"items": tuple(purchase_items)},
                plan_revision=plan_revision,
            )
        )
    return tuple(decisions)


def validate_approved_decisions(
    decisions: tuple[ApprovedDecision, ...],
    current_plan_revision: str | None,
) -> tuple[str, ...]:
    """Validate a client's resubmitted decisions (P0-06 rule 3).

    校验客户端重新提交的决策（P0-06 规则 3）。

    Checks:
      - option_type is one of the six supported kinds
      - payload is not conflicting (mutually exclusive decision kinds)
      - option_id is non-empty
      - plan_revision matches the confirmation the client is answering
        (stale confirmation rejected)

    检查项：
      - option_type 是六种受支持种类之一
      - payload 不冲突（互斥的决策种类）
      - option_id 非空
      - plan_revision 与客户端正在回答的确认一致（拒绝过期确认）

    Returns a tuple of issue strings. Empty = all valid.

    返回问题字符串元组。空 = 全部有效。
    """
    issues: list[str] = []
    seen_option_ids: set[str] = set()
    # Most repair kinds are scoped to a particular ingredient, dish, or
    # resource, so several decisions of the same type can be valid in one
    # submission (for example, purchasing three different missing
    # ingredients). Only the two plan-wide scalar changes are inherently
    # mutually exclusive with another decision of the same kind.
    # 大多数修复种类作用于特定食材、菜品或资源，因此一次提交中可以有多个
    # 同类型的有效决策（例如采购三种不同的缺失食材）。只有两个计划级标量变更
    # 本质上与同类的另一个决策互斥。
    seen_global_types: set[str] = set()

    for decision in decisions:
        if not decision.option_id.strip():
            issues.append("decision has empty option_id")
        if decision.option_id in seen_option_ids:
            issues.append(f"duplicate option_id: {decision.option_id}")
        seen_option_ids.add(decision.option_id)

        if decision.option_type not in SUPPORTED_DECISION_TYPES:
            issues.append(
                f"unsupported option_type {decision.option_type!r}; supported: {sorted(SUPPORTED_DECISION_TYPES)}"
            )
        elif decision.option_type in {"reduce_servings", "extend_time"}:
            if decision.option_type in seen_global_types:
                issues.append(f"conflicting decisions of type {decision.option_type}")
            seen_global_types.add(decision.option_type)

        if decision.plan_revision is not None and current_plan_revision is not None:
            if decision.plan_revision != current_plan_revision:
                issues.append(f"stale plan_revision {decision.plan_revision!r}, current is {current_plan_revision!r}")

    return tuple(issues)


def apply_approved_decisions_structured(
    request: GeneratePlanRequest,
    decisions: tuple[ApprovedDecision, ...],
) -> GeneratePlanRequest:
    """Apply approved decisions to produce a resolved request (P0-06 rule 4).

    应用已批准的决策以产出已解析的请求（P0-06 规则 4）。

    Pure transformation: never mutates the input request. Returns a new
    GeneratePlanRequest with the applicable constraints updated:
      - reduce_servings   → target_servings of every recipe
      - extend_time       → time_limit_minutes
      - alternative_equipment → kitchen resource snapshot adjusted
      - replace_dish      → recipe removed from the request
      - purchase          → no request mutation; the backend must persist
        real inventory and submit a fresh inventory snapshot
      - provide_gap_value → patch one field on a pre-parsed recipe candidate

    纯变换：从不修改输入请求。返回一个更新了适用约束的新 GeneratePlanRequest：
      - reduce_servings   → 每个菜谱的 target_servings
      - extend_time       → time_limit_minutes
      - alternative_equipment → 调整厨房资源快照
      - replace_dish      → 从请求中移除菜谱
      - purchase          → 不修改请求；后端必须持久化真实库存并提交新库存快照
      - provide_gap_value → 修补预解析菜谱候选上的一个字段
    """
    new_request = request
    new_kitchen: list[object] = list(request.kitchen_resources)

    for decision in decisions:
        payload = decision.payload
        if decision.option_type == "reduce_servings" and payload.get("servings") is not None:
            servings = Decimal(str(payload["servings"]))
            new_recipes = tuple(
                r.model_copy(update={"target_servings": servings}) if r.target_servings != servings else r
                for r in new_request.recipes
            )
            new_request = new_request.model_copy(update={"recipes": new_recipes})

        elif decision.option_type == "extend_time" and payload.get("time_limit_minutes") is not None:
            new_request = new_request.model_copy(update={"time_limit_minutes": int(str(payload["time_limit_minutes"]))})

        elif decision.option_type == "replace_dish" and payload.get("recipe_id"):
            target = str(payload["recipe_id"])
            new_recipes = tuple(r for r in new_request.recipes if r.recipe_id != target)
            if len(new_recipes) == len(new_request.recipes):
                # No-op replace of an unknown dish is tolerated but unused.
                # 对未知菜品的空操作替换被容忍但不使用。
                continue
            new_request = new_request.model_copy(update={"recipes": new_recipes})

        elif decision.option_type == "alternative_equipment" and payload.get("resource_type"):
            from cooking_plan_agent.domain.models import KitchenResourceSnapshot

            target_type = str(payload["resource_type"]).lower()
            alternative = str(payload.get("alternative", "")).lower()
            if not alternative:
                continue
            # Replace resources of the target type with an alternative type.
            # 用替代类型替换目标类型的资源。
            kept = [
                r
                for r in new_kitchen
                if not isinstance(r, KitchenResourceSnapshot) or r.resource_type.lower() != target_type
            ]
            kept.append(
                KitchenResourceSnapshot(
                    resource_id=f"alt-{alternative}",
                    resource_type=alternative,
                    capacity=Decimal(1),
                )
            )
            new_kitchen = kept

        elif decision.option_type == "purchase":
            # Purchase approval is an external workflow boundary. The Agent
            # never fabricates inventory; Backend persists checked purchases,
            # queries inventory again, then submits a fresh request.
            # 采购批准是外部工作流边界。Agent 绝不伪造库存；Backend 持久化
            # 已核对的采购、重新查询库存，然后提交新请求。
            continue

        elif decision.option_type == "provide_gap_value":
            new_request = _apply_gap_value(new_request, payload)

    new_request = new_request.model_copy(update={"kitchen_resources": tuple(new_kitchen)})
    return new_request


def _apply_gap_value(request: GeneratePlanRequest, payload: dict[str, object]) -> GeneratePlanRequest:
    """Apply one validated user answer to a pre-parsed recipe field.

    将一个已校验的用户答案应用到预解析菜谱字段。
    """
    field_path = str(payload.get("field_path") or "")
    raw_value = str(payload.get("value") or "").strip()
    if not field_path.startswith("recipe.") or not raw_value:
        return request

    recipe_and_field = field_path.removeprefix("recipe.").split(".", 1)
    if len(recipe_and_field) != 2:
        return request
    recipe_id, relative_path = recipe_and_field
    step_match = re.fullmatch(r"steps\[(\d+)]\.(\w+)", relative_path)
    if step_match is None:
        return request

    step_index = int(step_match.group(1))
    field_name = step_match.group(2)
    candidates = list(request.preparsed_candidates)
    for candidate_index, candidate in enumerate(candidates):
        if candidate.recipe_id != recipe_id or step_index >= len(candidate.steps):
            continue
        step = candidate.steps[step_index]
        update: dict[str, object] = {
            "extraction_source": "USER_CONFIRMED",
            "confidence": Decimal(1),
        }
        try:
            if field_name == "heat_level":
                from cooking_plan_agent.domain.enums import HeatLevel

                update[field_name] = HeatLevel(raw_value.upper())
            elif field_name in {"active_duration_minutes", "passive_duration_minutes"}:
                duration = int(raw_value)
                if duration <= 0:
                    return request
                update[field_name] = duration
            elif field_name == "target_temperature_c":
                temperature = Decimal(raw_value)
                if temperature <= 0:
                    return request
                update[field_name] = temperature
            elif field_name == "resources_hint":
                resources = tuple(part.strip() for part in raw_value.split(",") if part.strip())
                if not resources:
                    return request
                update[field_name] = resources
            else:
                return request
        except (ArithmeticError, TypeError, ValueError):
            return request

        steps = list(candidate.steps)
        steps[step_index] = step.model_copy(update=update)
        candidates[candidate_index] = candidate.model_copy(update={"steps": tuple(steps)})
        return request.model_copy(update={"preparsed_candidates": tuple(candidates)})
    return request


def apply_ingredient_substitutions_patch(
    recipes: tuple[RecipeIR, ...],
    decisions: tuple[ApprovedDecision, ...],
) -> tuple[RecipeIR, ...]:
    """Patch RecipeIR ingredients per substitute_ingredient decisions.

    按 substitute_ingredient 决策修补 RecipeIR 的食材。

    Pure transformation: each decision with option_type
    ``substitute_ingredient`` renames the target ingredient's canonical
    name to the substitute so safety (allergen) and feasibility checks
    re-run against the NEW ingredient (P0-06 rule 6).

    纯变换：每个 option_type 为 ``substitute_ingredient`` 的决策把目标食材的
    规范名称重命名为替代品，使安全（过敏原）与可行性检查针对新食材重新执行
    （P0-06 规则 6）。
    """
    substitutes = {
        (d.payload.get("recipe_id"), d.payload.get("ingredient")): d.payload.get("substitute")
        for d in decisions
        if d.option_type == "substitute_ingredient"
        and d.payload.get("recipe_id")
        and d.payload.get("ingredient")
        and d.payload.get("substitute")
    }
    if not substitutes:
        return recipes

    patched: list[RecipeIR] = []
    for recipe in recipes:
        changed = False
        new_ingredients = list(recipe.ingredients)
        for i, ingredient in enumerate(recipe.ingredients):
            key = (recipe.recipe_id, ingredient.canonical_name)
            substitute = substitutes.get(key)
            if substitute is not None:
                new_ingredients[i] = ingredient.model_copy(
                    update={
                        "canonical_name": str(substitute),
                        "raw_name": str(substitute),
                    }
                )
                changed = True
        if changed:
            recipe = recipe.model_copy(update={"ingredients": tuple(new_ingredients)})
        patched.append(recipe)
    return tuple(patched)


# =============================================================================
# P4-02  Structured confirmation answers → ApprovedDecision mapping
#      结构化确认答案 → ApprovedDecision 映射
# =============================================================================

# Bounded free-text answer length (P4-02 rule 5: bound length/types).
# 有界的自由文本答案长度（P4-02 规则 5：约束长度/类型）。
_MAX_TEXT_ANSWER_LENGTH = 500


class ConfirmationAnswersError(ValueError):
    """Raised when a set of confirmation answers is invalid (P4-02).

    当一组确认答案无效时抛出（P4-02）。

    Carries the individual issues (unknown question_id, invalid option,
    missing required answer, duplicate answer, over-length text) so the
    caller can produce field-level fix guidance (P2-04 fault matrix).

    携带具体问题（未知 question_id、无效选项、缺失必答、重复答案、超长文本），
    使调用方能生成字段级修复指引（P2-04 故障矩阵）。
    """

    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


def answers_to_approved_decisions(
    questions: tuple[ConfirmationQuestion, ...],
    answers: tuple[QuestionAnswer, ...],
    plan_revision: str | None,
    presented_decisions: tuple[ApprovedDecision, ...] = (),
) -> tuple[ApprovedDecision, ...]:
    """Validate client answers and map them losslessly to ApprovedDecision.

    校验客户端答案并无损映射为 ApprovedDecision。

    Validation (P4-02 rule 4 / P2-04 fault matrix):
      - every answer's question_id must exist in the presented questions;
      - no duplicate answers for the same question;
      - every required question must be answered;
      - CHOICE answers must hit exactly one of the question's option values;
      - TEXT answers must be non-empty and bounded in length.

    校验（P4-02 规则 4 / P2-04 故障矩阵）：
      - 每个答案的 question_id 必须存在于已展示的问题中；
      - 同一问题不得有重复答案；
      - 每个必答问题都必须被回答；
      - CHOICE 答案必须恰好命中问题的某个选项值；
      - TEXT 答案必须非空且长度有界。

    Mapping (D9): CHOICE answers that select a presented repair decision
    emit that exact ApprovedDecision. TEXT gap answers emit a bounded
    ``provide_gap_value`` decision so the next parse pass can apply the
    confirmed value instead of asking the same question again.

    映射（D9）：选择已展示修复决策的 CHOICE 答案发出该精确的 ApprovedDecision。
    TEXT 缺口答案发出有界的 ``provide_gap_value`` 决策，使下一轮解析能应用
    已确认的值，而不是再次询问同一问题。

    Args:
        questions: The ConfirmationQuestions presented to the client.
            展示给客户端的 ConfirmationQuestion。
        answers: The client's submitted QuestionAnswers.
            客户端提交的 QuestionAnswer。
        plan_revision: The revision of the confirmation being answered.
            正在回答的确认的 revision。
        presented_decisions: The ApprovedDecisions carried by the
            confirmation response (used to map option values verbatim).
            确认响应携带的 ApprovedDecision（用于按原样映射选项值）。

    Returns:
        The decisions to resubmit in the next request's
        ``approved_decisions`` field.
            要在下一次请求的 ``approved_decisions`` 字段中重新提交的决策。

    Raises:
        ConfirmationAnswersError: With field-level fix guidance when any
            answer fails validation.
            当任一答案校验失败时，抛出带字段级修复指引的 ConfirmationAnswersError。
    """
    issues: list[str] = []
    by_id: dict[str, ConfirmationQuestion] = {q.question_id: q for q in questions}
    answered_ids: set[str] = set()

    for answer in answers:
        question = by_id.get(answer.question_id)
        if question is None:
            issues.append(f"unknown question_id: {answer.question_id}")
            continue
        if answer.question_id in answered_ids:
            issues.append(f"duplicate answer for question_id: {answer.question_id}")
        answered_ids.add(answer.question_id)

        value = answer.value.strip()
        if question.response_type == QuestionResponseType.CHOICE:
            valid_values = {option.value for option in question.options}
            if value not in valid_values:
                issues.append(
                    f"invalid option for question {answer.question_id!r}: {answer.value!r}; "
                    f"allowed: {sorted(valid_values)}"
                )
        else:
            if not value:
                issues.append(f"empty answer for question {answer.question_id!r}")
            elif len(value) > _MAX_TEXT_ANSWER_LENGTH:
                issues.append(
                    f"answer for question {answer.question_id!r} exceeds {_MAX_TEXT_ANSWER_LENGTH} characters"
                )

    # Required questions must all be answered.
    # 所有必答问题都必须被回答。
    for question in questions:
        if question.required and question.question_id not in answered_ids:
            issues.append(f"missing required answer for question {question.question_id!r}")

    if issues:
        raise ConfirmationAnswersError(tuple(issues))

    # Lossless mapping: an answer selects a presented decision verbatim
    # (by option_id); payload is never rebuilt from prose (D9).
    # 无损映射：答案按原样（通过 option_id）选择已展示的决策；
    # 负载绝不从自然语言重建（D9）。
    decisions_by_option_id: dict[str, ApprovedDecision] = {d.option_id: d for d in presented_decisions}
    mapped: list[ApprovedDecision] = []
    for answer in answers:
        decision = decisions_by_option_id.get(answer.value)
        if decision is not None and plan_revision is not None and decision.plan_revision != plan_revision:
            # Keep the decision's payload/type; only rebind the revision the
            # client is answering. This is a metadata update, not a payload
            # rewrite (D9).
            # 保留决策的 payload/type；仅重绑定客户端正在回答的 revision。
            # 这是元数据更新，不是负载重写（D9）。
            decision = decision.model_copy(update={"plan_revision": plan_revision})
        if decision is not None:
            mapped.append(decision)
            continue
        question = by_id[answer.question_id]
        if question.response_type == QuestionResponseType.TEXT:
            mapped.append(
                ApprovedDecision(
                    option_id=f"answer:{answer.question_id}",
                    option_type="provide_gap_value",
                    payload={"field_path": question.field_path, "value": answer.value.strip()},
                    plan_revision=plan_revision,
                )
            )
    return tuple(mapped)
