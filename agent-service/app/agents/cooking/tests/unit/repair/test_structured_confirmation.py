"""P4-02 structured confirmation question tests.

Covers the full P4-02 contract:
  - structured ConfirmationQuestion generation (gaps / assumptions / repairs)
  - stable question IDs for identical input (D6)
  - one required question per blocking gap
  - lossless answer → ApprovedDecision mapping (D9)
  - negative cases: illegal option, missing required answer, unknown
    question_id, duplicate answer, over-length text, extra field
  - legacy plain-string questions dual-emit compatibility
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    Assumption,
    ConfirmationPlanResponse,
    ConfirmationQuestion,
    GeneratePlanRequest,
    IngredientDemand,
    QuestionAnswer,
    QuestionOption,
    QuestionResponseType,
    RecipeGap,
    RecipeIR,
    RecipeStep,
    RepairOption,
)
from cooking_plan_agent.rendering.responses import render_confirmation_response
from cooking_plan_agent.repair.options import (
    ConfirmationAnswersError,
    answers_to_approved_decisions,
    build_approved_decisions,
)
from cooking_plan_agent.workflow.state import PlanState

# =============================================================================
# Helpers
# =============================================================================


def _state(**overrides) -> PlanState:
    base = PlanState(
        request=GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
        ),
    )
    base.update(overrides)  # type: ignore[arg-type]
    return base


def _repair_option(
    option_id: str = "repair_servings_1_abc",
    option_type: str = "reduce_servings",
    description: str = "Reduce servings from 2 to 1",
) -> RepairOption:
    payload: dict[str, object]
    if option_type == "purchase":
        import re

        match = re.search(r"Purchase\s+([\d.]+)\s+(\S+)\s+of\s+'([^']+)'", description)
        payload = (
            {
                "ingredient_name": match.group(3),
                "quantity": int(Decimal(match.group(1))),
                "unit": match.group(2),
            }
            if match
            else {}
        )
    elif option_type == "reduce_servings":
        payload = {"servings": 1}
    else:
        payload = {}
    return RepairOption(
        option_id=option_id,
        option_type=option_type,
        description=description,
        changes=("Scale down",),
        effects=("Fixed",),
        payload=payload,
        revalidation_status="validated",
    )


def _gap(
    recipe_id: str = "r1",
    field_path: str = "recipe.r1.step_1.heat",
    gap_class: str = "critical",
) -> RecipeGap:
    return RecipeGap(
        gap_id=f"gap_{recipe_id}_{field_path}",
        recipe_id=recipe_id,
        field_path=field_path,
        gap_class=gap_class,
        description="Missing heat level",
        confidence=Decimal("0.3"),
    )


def _assumption(text: str = "Assumed 200C for baking", confidence: str = "0.4") -> Assumption:
    return Assumption(text=text, confidence=Decimal(confidence))


# =============================================================================
# Structured question generation
# =============================================================================


class TestStructuredQuestionGeneration:
    def test_purchase_and_reduce_options_collapse_into_one_strategy_question(self) -> None:
        """缺料场景不再逐个 repair 确认，而是聚合成一个高层策略题。"""
        options = (
            _repair_option(
                option_id="repair_purchase_broccoli",
                option_type="purchase",
                description="Purchase 90 g of 'Broccoli' (no known substitute available)",
            ),
            _repair_option(
                option_id="repair_purchase_tomato",
                option_type="purchase",
                description="Purchase 200 g of 'Canned tomatoes' (no known substitute available)",
            ),
            _repair_option(),
        )

        response = render_confirmation_response(_state(repair_options=options))

        strategy_questions = [q for q in response.confirmation_questions if q.field_path == "repair_strategy"]
        assert len(strategy_questions) == 1
        question = strategy_questions[0]
        assert question.question_id == "repair:strategy"
        assert question.response_type == QuestionResponseType.CHOICE
        assert question.required is True
        assert len(question.options) == 2
        option_values = {o.value for o in question.options}
        assert "repair_servings_1_abc" in option_values
        assert "repair_purchase_bundle" in option_values
        assert any("Reduce" in option.label for option in question.options)
        assert any("Buy" in option.label for option in question.options)

        purchase_decisions = [d for d in response.decisions if d.option_type == "purchase"]
        assert len(purchase_decisions) == 1
        assert purchase_decisions[0].option_id == "repair_purchase_bundle"
        assert purchase_decisions[0].payload == {
            "items": (
                {"ingredient_name": "Broccoli", "quantity": 90, "unit": "g"},
                {"ingredient_name": "Canned tomatoes", "quantity": 200, "unit": "g"},
            )
        }

    def test_single_repair_option_still_collapses_to_strategy_question(self) -> None:
        """Even a single plan-level repair (e.g. only reduce_servings) is
        collapsed into the strategy question — never a per-item Apply/Do-not-
        apply list. The user always decides at plan level."""
        options = (_repair_option(),)
        response = render_confirmation_response(_state(repair_options=options))

        assert isinstance(response, ConfirmationPlanResponse)
        assert response.confirmation_questions

        strategy_questions = [q for q in response.confirmation_questions if q.field_path == "repair_strategy"]
        assert len(strategy_questions) == 1
        question = strategy_questions[0]
        assert question.question_id == "repair:strategy"
        assert question.response_type == QuestionResponseType.CHOICE
        option_values = {o.value for o in question.options}
        assert "repair_servings_1_abc" in option_values
        # No per-item repair questions remain.
        assert not any(q.field_path == "repair_options" for q in response.confirmation_questions)

    def test_blocking_gaps_produce_one_required_question_each(self) -> None:
        """Every unresolved critical gap → exactly one required TEXT question,
        keyed by stable recipe_id + field_path (one-to-one, D6)."""
        gaps = (
            _gap(recipe_id="r1", field_path="recipe.r1.step_1.heat"),
            _gap(recipe_id="r1", field_path="recipe.r1.step_2.temperature"),
            _gap(recipe_id="r1", field_path="recipe.r1.step_3.duration", gap_class="cosmetic"),
        )
        response = render_confirmation_response(_state(gaps=gaps))

        gap_questions = [q for q in response.confirmation_questions if q.field_path.startswith("recipe.")]
        # cosmetic gap is NOT blocking → no question.
        assert len(gap_questions) == 2
        for question in gap_questions:
            assert question.response_type == QuestionResponseType.TEXT
            assert question.required is True
            assert question.question_id.startswith("gap:")

    def test_question_ids_are_stable_for_identical_input(self) -> None:
        """Same input twice → identical question IDs (D6)."""
        gaps = (
            _gap(),
            _gap(recipe_id="r1", field_path="recipe.r1.step_2.temperature"),
        )
        state = _state(gaps=gaps, repair_options=(_repair_option(),))

        first = render_confirmation_response(state).confirmation_questions
        second = render_confirmation_response(state).confirmation_questions

        assert [q.question_id for q in first] == [q.question_id for q in second]
        assert len({q.question_id for q in first}) == len(first)  # no collisions

    def test_backend_preprocessed_request_keeps_unresolved_gap_questions(self) -> None:
        """When the backend preprocesses recipes (preparsed_candidates set),
        accepted inference assumptions are not re-asked, but any genuinely
        unresolved blocking gap stays actionable."""
        from cooking_plan_agent.domain.models import ExtractedIngredient, ExtractedRecipeCandidate, ExtractedStep

        gaps = (
            _gap(),
            _gap(recipe_id="r1", field_path="recipe.r1.step_2.temperature"),
        )
        assumptions_recipe = RecipeIR(
            recipe_id="r1",
            dish_name="Dish",
            original_servings=2,
            target_servings=2,
            source_language="en",
            ingredients=(
                IngredientDemand(
                    canonical_name="salt",
                    raw_name="salt",
                    quantity=Decimal(1),
                    unit="g",
                    confidence=Decimal("1.0"),
                ),
            ),
            steps=(RecipeStep(step_number=1, instruction="Bake"),),
            assumptions=(_assumption(confidence="0.4"),),
        )
        candidate = ExtractedRecipeCandidate(
            recipe_id="r1",
            dish_name="Dish",
            original_servings=Decimal(2),
            source_language="en",
            ingredients=(ExtractedIngredient(raw_text="salt", name="salt", quantity=Decimal(1), unit="g"),),
            steps=(ExtractedStep(step_number=1, instruction="Bake"),),
        )
        request = GeneratePlanRequest(
            request_id="req-1",
            user_id="user-1",
            recipes=({"recipe_id": "r1", "text": "test", "target_servings": 2},),
            preparsed_candidates=(candidate,),
        )
        state = _state(gaps=gaps, parsed_recipes=(assumptions_recipe,), repair_options=(_repair_option(),))
        state["request"] = request  # type: ignore[arg-type]

        response = render_confirmation_response(state)

        question_ids = [q.question_id for q in response.confirmation_questions]
        assert any(qid.startswith("gap:") for qid in question_ids)
        assert not any(qid.startswith("assumption:") for qid in question_ids)
        # Strategy-level repair questions still surface.
        assert any(qid.startswith("repair:") for qid in question_ids)

    def test_low_confidence_assumptions_are_accepted_without_questions(self) -> None:
        """Operational assumptions remain auditable but are not user decisions."""
        recipe = RecipeIR(
            recipe_id="r1",
            dish_name="Dish",
            original_servings=2,
            target_servings=2,
            source_language="en",
            ingredients=(
                IngredientDemand(
                    canonical_name="salt",
                    raw_name="salt",
                    quantity=Decimal(1),
                    unit="g",
                    confidence=Decimal("1.0"),
                ),
            ),
            steps=(RecipeStep(step_number=1, instruction="Bake"),),
            assumptions=(_assumption(confidence="0.4"), _assumption(text="Solid", confidence="0.9")),
        )
        response = render_confirmation_response(_state(parsed_recipes=(recipe,)))

        assert response.confirmation_questions == ()
        assert [assumption.text for assumption in response.assumptions] == [
            "Assumed 200C for baking",
            "Solid",
        ]

    def test_research_assumptions_are_accepted_without_questions(self) -> None:
        """Research provenance is retained without a technical confirmation."""
        research = (_assumption(text="Conflicting oven temp", confidence="0.8"),)
        response = render_confirmation_response(_state(research_assumptions=research))

        assert response.confirmation_questions == ()
        assert response.assumptions[0].text == "Conflicting oven temp"

    def test_legacy_questions_are_derived_from_structured(self) -> None:
        """Legacy questions dual-emit ``f"{prompt} ({question_id})"`` so old
        clients stay readable (P4-02)."""
        response = render_confirmation_response(_state(repair_options=(_repair_option(),)))
        assert len(response.questions) == len(response.confirmation_questions)
        for question, legacy in zip(response.confirmation_questions, response.questions, strict=True):
            assert legacy == f"{question.prompt} ({question.question_id})"

    def test_empty_state_keeps_legacy_fallback(self) -> None:
        """No field-level question is meaningful for an empty confirmation —
        the legacy fallback stays for validation compatibility."""
        response = render_confirmation_response(_state())
        assert response.confirmation_questions == ()
        assert response.questions == ("Would you like to proceed with these options?",)


# =============================================================================
# answers_to_approved_decisions — mapping & negative cases
# =============================================================================


class TestAnswersToApprovedDecisions:
    def _questions_and_decisions(
        self,
    ) -> tuple[tuple[ConfirmationQuestion, ...], tuple[ApprovedDecision, ...]]:
        """One repair option → strategy question plus presented decisions."""
        options = (_repair_option(),)
        decisions = build_approved_decisions(options, "req-1:v1")
        state = _state(repair_options=options)
        questions = render_confirmation_response(state).confirmation_questions
        return questions, decisions

    def test_choice_answer_maps_decision_verbatim(self) -> None:
        """Selecting the apply option returns the EXACT presented decision —
        payload is preserved verbatim (D9)."""
        questions, decisions = self._questions_and_decisions()
        decision = decisions[0]
        answers = (QuestionAnswer(question_id="repair:strategy", value=decision.option_id),)

        mapped = answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)

        assert len(mapped) == 1
        assert mapped[0].option_id == decision.option_id
        assert mapped[0].option_type == decision.option_type
        assert mapped[0].payload == decision.payload  # verbatim, no rewriting
        assert mapped[0].payload == {"servings": 1}

    def test_strategy_answer_maps_to_aggregated_purchase_decision(self) -> None:
        options = (
            _repair_option(
                option_id="repair_purchase_broccoli",
                option_type="purchase",
                description="Purchase 90 g of 'Broccoli' (no known substitute available)",
            ),
            _repair_option(
                option_id="repair_purchase_tomato",
                option_type="purchase",
                description="Purchase 200 g of 'Tomato' (no known substitute available)",
            ),
            _repair_option(),
        )
        response = render_confirmation_response(_state(repair_options=options))
        answers = (QuestionAnswer(question_id="repair:strategy", value="repair_purchase_bundle"),)

        mapped = answers_to_approved_decisions(
            response.confirmation_questions,
            answers,
            "req-1:v1",
            presented_decisions=response.decisions,
        )

        assert len(mapped) == 1
        assert mapped[0].option_type == "purchase"
        assert mapped[0].payload == {
            "items": (
                {"ingredient_name": "Broccoli", "quantity": 90, "unit": "g"},
                {"ingredient_name": "Tomato", "quantity": 200, "unit": "g"},
            )
        }

    def test_no_answers_maps_to_no_decisions(self) -> None:
        # A non-required question (repair option) submitted with no answers
        # maps to no decisions — it may be skipped.
        optional_question = ConfirmationQuestion(
            question_id="repair:r1",
            field_path="repair_options",
            prompt="Apply the repair option 'x'?",
            response_type=QuestionResponseType.CHOICE,
            options=(QuestionOption(value="d1", label="Apply"), QuestionOption(value="__skip__", label="Do not apply")),
            required=False,
        )
        decision = ApprovedDecision(option_id="d1", option_type="reduce_servings", payload={}, plan_revision="req-1:v1")
        mapped = answers_to_approved_decisions((optional_question,), (), "req-1:v1", presented_decisions=(decision,))
        assert mapped == ()

    def test_invalid_option_rejected(self) -> None:
        questions, decisions = self._questions_and_decisions()
        answers = (QuestionAnswer(question_id="repair:strategy", value="not-an-option"),)

        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)
        assert any("invalid option" in issue for issue in exc_info.value.issues)

    def test_unknown_question_id_rejected(self) -> None:
        questions, decisions = self._questions_and_decisions()
        answers = (QuestionAnswer(question_id="ghost-question", value="x"),)

        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)
        assert any("unknown question_id" in issue for issue in exc_info.value.issues)

    def test_missing_required_answer_rejected(self) -> None:
        """A required (gap/assumption) question that is not answered is
        rejected even when every submitted answer itself is valid."""
        gap_question = ConfirmationQuestion(
            question_id="gap:abc",
            field_path="recipe.r1.step_1.heat",
            prompt="Missing heat level. Please provide the correct value.",
            response_type=QuestionResponseType.TEXT,
            required=True,
        )
        unrelated = (QuestionAnswer(question_id="other", value="x"),)
        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions((gap_question,), unrelated, "req-1:v1")
        assert any("missing required answer" in issue for issue in exc_info.value.issues)

    def test_duplicate_answer_rejected(self) -> None:
        questions, decisions = self._questions_and_decisions()
        answers = (
            QuestionAnswer(question_id="repair:strategy", value=decisions[0].option_id),
            QuestionAnswer(question_id="repair:strategy", value="__skip__"),
        )

        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)
        assert any("duplicate answer" in issue for issue in exc_info.value.issues)

    def test_empty_text_answer_rejected(self) -> None:
        gap_question = ConfirmationQuestion(
            question_id="gap:abc",
            field_path="recipe.r1.step_1.heat",
            prompt="Please provide the value.",
            response_type=QuestionResponseType.TEXT,
            required=True,
        )
        answers = (QuestionAnswer(question_id="gap:abc", value="   "),)
        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions((gap_question,), answers, "req-1:v1")
        assert any("empty answer" in issue for issue in exc_info.value.issues)

    def test_overlength_text_answer_rejected(self) -> None:
        gap_question = ConfirmationQuestion(
            question_id="gap:abc",
            field_path="recipe.r1.step_1.heat",
            prompt="Please provide the value.",
            response_type=QuestionResponseType.TEXT,
            required=True,
        )
        answers = (QuestionAnswer(question_id="gap:abc", value="x" * 501),)
        with pytest.raises(ConfirmationAnswersError) as exc_info:
            answers_to_approved_decisions((gap_question,), answers, "req-1:v1")
        assert any("exceeds" in issue for issue in exc_info.value.issues)

    def test_extra_field_rejected_at_model_boundary(self) -> None:
        """QuestionAnswer is a StrictModel — unknown fields fail fast."""
        with pytest.raises(ValidationError):
            QuestionAnswer.model_validate({"question_id": "q1", "value": "v", "extra_field": "x"})

    def test_plan_revision_rebound_on_presented_decision(self) -> None:
        """A presented decision answering a newer revision is rebound to the
        current one — a metadata update, never a payload rewrite (D9)."""
        questions, decisions = self._questions_and_decisions()
        decision = decisions[0]
        answers = (QuestionAnswer(question_id="repair:strategy", value=decision.option_id),)

        mapped = answers_to_approved_decisions(questions, answers, "req-1:v2", presented_decisions=decisions)
        assert mapped[0].plan_revision == "req-1:v2"
        assert mapped[0].payload == decision.payload

    def test_gap_answer_maps_to_structured_value_decision(self) -> None:
        """A TEXT gap answer survives resubmission as a field patch."""
        questions = (
            ConfirmationQuestion(
                question_id="gap:abc",
                field_path="recipe.r1.step_1.heat",
                prompt="Missing heat level.",
                response_type=QuestionResponseType.TEXT,
                required=True,
            ),
            ConfirmationQuestion(
                question_id="assumption:def",
                field_path="recipe.r1.assumptions",
                prompt="Accept?",
                response_type=QuestionResponseType.CHOICE,
                options=(
                    QuestionOption(value="accept", label="Accept"),
                    QuestionOption(value="provide_alternative", label="Alternative"),
                ),
                required=True,
            ),
        )
        answers = (
            QuestionAnswer(question_id="gap:abc", value="medium"),
            QuestionAnswer(question_id="assumption:def", value="accept"),
        )
        mapped = answers_to_approved_decisions(questions, answers, "req-1:v1")
        assert len(mapped) == 1
        assert mapped[0].option_type == "provide_gap_value"
        assert mapped[0].payload == {"field_path": "recipe.r1.step_1.heat", "value": "medium"}


# =============================================================================
# Round-trip serialisation (OpenAPI / contract surface)
# =============================================================================


class TestStructuredConfirmationSerialisation:
    def test_response_serialises_and_reparses_questions(self) -> None:
        """confirmation_questions survive a JSON round-trip — the schema is
        exposed in OpenAPI via ConfirmationPlanResponse."""
        response = render_confirmation_response(_state(repair_options=(_repair_option(),)))
        assert response.confirmation_questions

        raw = response.model_dump_json()
        reparsed = ConfirmationPlanResponse.model_validate_json(raw)

        assert len(reparsed.confirmation_questions) == len(response.confirmation_questions)
        reparsed_q = reparsed.confirmation_questions[0]
        original_q = response.confirmation_questions[0]
        assert reparsed_q.question_id == original_q.question_id
        assert reparsed_q.response_type == QuestionResponseType.CHOICE
        assert reparsed_q.options[0].value == original_q.options[0].value
