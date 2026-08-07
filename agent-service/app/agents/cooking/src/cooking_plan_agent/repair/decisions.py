"""Validation and application of user-confirmed repair decisions."""

from __future__ import annotations

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
# =============================================================================

# The six decision kinds the confirmation loop supports (P0-06 rule 5).
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
    }
)


def build_approved_decisions(
    repair_options: tuple[RepairOption, ...],
    plan_revision: str | None,
) -> tuple[ApprovedDecision, ...]:
    """Convert presented RepairOptions into structured, submittable decisions.

    Every supported option becomes an ApprovedDecision whose payload is
    populated from the option's structured description. The client can
    resubmit these verbatim; the server re-validates them (P0-06 rule 2).
    """
    import re as _re

    decisions: list[ApprovedDecision] = []
    for option in repair_options:
        if option.option_type not in SUPPORTED_DECISION_TYPES:
            continue
        payload: dict[str, object] = {}
        if option.option_type == "extend_time":
            match = _re.search(r"to (\d+) minutes", option.description)
            if match:
                payload["time_limit_minutes"] = int(match.group(1))
        elif option.option_type == "reduce_servings":
            match = _re.search(r"from ([\d.]+) to ([\d.]+)", option.description)
            if match:
                # 削减后的新份量始终为整数（to_integral_value），故保留 int 语义
                payload["servings"] = int(Decimal(match.group(2)))
        decisions.append(
            ApprovedDecision(
                option_id=option.option_id,
                option_type=option.option_type,
                payload=payload,
                plan_revision=plan_revision,
            )
        )
    return tuple(decisions)


def validate_approved_decisions(
    decisions: tuple[ApprovedDecision, ...],
    current_plan_revision: str | None,
) -> tuple[str, ...]:
    """Validate a client's resubmitted decisions (P0-06 rule 3).

    Checks:
      - option_type is one of the six supported kinds
      - payload is not conflicting (mutually exclusive decision kinds)
      - option_id is non-empty
      - plan_revision matches the confirmation the client is answering
        (stale confirmation rejected)

    Returns a tuple of issue strings. Empty = all valid.
    """
    issues: list[str] = []
    seen_option_ids: set[str] = set()
    # Most repair kinds are scoped to a particular ingredient, dish, or
    # resource, so several decisions of the same type can be valid in one
    # submission (for example, purchasing three different missing
    # ingredients). Only the two plan-wide scalar changes are inherently
    # mutually exclusive with another decision of the same kind.
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

    Pure transformation: never mutates the input request. Returns a new
    GeneratePlanRequest with the applicable constraints updated:
      - reduce_servings   → target_servings of every recipe
      - extend_time       → time_limit_minutes
      - substitute_ingredient → recorded in approved payload for the IR
        builder (ingredient substitution applied downstream as a patch)
      - alternative_equipment → kitchen resource snapshot adjusted
      - replace_dish      → recipe removed from the request
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
                continue
            new_request = new_request.model_copy(update={"recipes": new_recipes})

        elif decision.option_type == "alternative_equipment" and payload.get("resource_type"):
            from cooking_plan_agent.domain.models import KitchenResourceSnapshot

            target_type = str(payload["resource_type"]).lower()
            alternative = str(payload.get("alternative", "")).lower()
            if not alternative:
                continue
            # Replace resources of the target type with an alternative type.
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

        # substitute_ingredient is handled as a patch by the IR builder
        # (payload: {recipe_id, ingredient, substitute}) — see
        # apply_ingredient_substitutions_patch.

        elif decision.option_type == "purchase":
            # 外出采购：决策本身不改变排程输入（agent 无法代购）。
            # 用户购买后由后端更新库存快照（inventory_lots）并重新提交请求；
            # 若库存未变，重跑仍会返回 NEEDS_CONFIRMATION（可再次选择）。
            pass  # no-op：继续到最终 model_copy（保留 approved_decisions）

    new_request = new_request.model_copy(update={"kitchen_resources": tuple(new_kitchen)})
    return new_request


def apply_ingredient_substitutions_patch(
    recipes: tuple[RecipeIR, ...],
    decisions: tuple[ApprovedDecision, ...],
) -> tuple[RecipeIR, ...]:
    """Patch RecipeIR ingredients per substitute_ingredient decisions.

    Pure transformation: each decision with option_type
    ``substitute_ingredient`` renames the target ingredient's canonical
    name to the substitute so safety (allergen) and feasibility checks
    re-run against the NEW ingredient (P0-06 rule 6).
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
# =============================================================================

# Bounded free-text answer length (P4-02 rule 5: bound length/types).
_MAX_TEXT_ANSWER_LENGTH = 500


class ConfirmationAnswersError(ValueError):
    """Raised when a set of confirmation answers is invalid (P4-02).

    Carries the individual issues (unknown question_id, invalid option,
    missing required answer, duplicate answer, over-length text) so the
    caller can produce field-level fix guidance (P2-04 fault matrix).
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

    Validation (P4-02 rule 4 / P2-04 fault matrix):
      - every answer's question_id must exist in the presented questions;
      - no duplicate answers for the same question;
      - every required question must be answered;
      - CHOICE answers must hit exactly one of the question's option values;
      - TEXT answers must be non-empty and bounded in length.

    Mapping (D9): only CHOICE answers that select a presented repair
    decision emit an ApprovedDecision — the EXACT object that was
    presented (looked up by option_id), so the payload is preserved
    verbatim with zero rewriting. Gap/assumption answers are validated
    but have no ApprovedDecision carrier yet (contract v2).

    Args:
        questions: The ConfirmationQuestions presented to the client.
        answers: The client's submitted QuestionAnswers.
        plan_revision: The revision of the confirmation being answered.
        presented_decisions: The ApprovedDecisions carried by the
            confirmation response (used to map option values verbatim).

    Returns:
        The decisions to resubmit in the next request's
        ``approved_decisions`` field.

    Raises:
        ConfirmationAnswersError: With field-level fix guidance when any
            answer fails validation.
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
    for question in questions:
        if question.required and question.question_id not in answered_ids:
            issues.append(f"missing required answer for question {question.question_id!r}")

    if issues:
        raise ConfirmationAnswersError(tuple(issues))

    # Lossless mapping: an answer selects a presented decision verbatim
    # (by option_id); payload is never rebuilt from prose (D9).
    decisions_by_option_id: dict[str, ApprovedDecision] = {d.option_id: d for d in presented_decisions}
    mapped: list[ApprovedDecision] = []
    for answer in answers:
        decision = decisions_by_option_id.get(answer.value)
        if decision is None:
            continue
        if plan_revision is not None and decision.plan_revision != plan_revision:
            # Keep the decision's payload/type; only rebind the revision the
            # client is answering. This is a metadata update, not a payload
            # rewrite (D9).
            decision = decision.model_copy(update={"plan_revision": plan_revision})
        mapped.append(decision)
    return tuple(mapped)
