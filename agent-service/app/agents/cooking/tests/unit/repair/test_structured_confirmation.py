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
    return RepairOption(
        option_id=option_id,
        option_type=option_type,
        description=description,
        changes=("Scale down",),
        effects=("Fixed",),
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
    def test_repair_options_produce_choice_questions(self) -> None:
        """Each supported RepairOption becomes one CHOICE question whose apply
        option value is the presented decision's option_id (D9)."""
        options = (_repair_option(),)
        response = render_confirmation_response(_state(repair_options=options))

        assert isinstance(response, ConfirmationPlanResponse)
        assert response.confirmation_questions

        repair_questions = [q for q in response.confirmation_questions if q.field_path == "repair_options"]
        assert len(repair_questions) == 1
        question = repair_questions[0]
        assert question.question_id == "repair:repair_servings_1_abc"
        assert question.response_type == QuestionResponseType.CHOICE
        assert question.required is False  # repair options may be skipped
        option_values = {o.value for o in question.options}
        assert "repair_servings_1_abc" in option_values
        assert "__skip__" in option_values
        # Apply option carries the exact decision payload reference.
        apply_option = next(o for o in question.options if o.value == "repair_servings_1_abc")
        assert apply_option.suggested is True

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

    def test_low_confidence_assumptions_produce_choice_questions(self) -> None:
        """Assumptions below the confidence threshold surface as required
        CHOICE questions (accept suggested value / provide alternative)."""
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

        assumption_questions = [q for q in response.confirmation_questions if q.question_id.startswith("assumption:")]
        # Only the low-confidence assumption surfaces.
        assert len(assumption_questions) == 1
        question = assumption_questions[0]
        assert question.response_type == QuestionResponseType.CHOICE
        assert question.required is True
        option_values = {o.value for o in question.options}
        assert option_values == {"accept", "provide_alternative"}
        assert question.suggested_value == "Assumed 200C for baking"

    def test_research_assumptions_always_surface(self) -> None:
        """Research-backed assumptions surface regardless of confidence — the
        graph only routed to confirmation because they warranted it (P1-01)."""
        research = (_assumption(text="Conflicting oven temp", confidence="0.8"),)
        response = render_confirmation_response(_state(research_assumptions=research))

        assumption_questions = [q for q in response.confirmation_questions if q.question_id.startswith("assumption:")]
        assert len(assumption_questions) == 1

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
        """One repair option → its CHOICE question plus presented decisions."""
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
        answers = (QuestionAnswer(question_id="repair:repair_servings_1_abc", value=decision.option_id),)

        mapped = answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)

        assert len(mapped) == 1
        assert mapped[0].option_id == decision.option_id
        assert mapped[0].option_type == decision.option_type
        assert mapped[0].payload == decision.payload  # verbatim, no rewriting
        assert mapped[0].payload == {"servings": 1}

    def test_skip_option_produces_no_decision(self) -> None:
        questions, decisions = self._questions_and_decisions()
        answers = (QuestionAnswer(question_id="repair:repair_servings_1_abc", value="__skip__"),)

        mapped = answers_to_approved_decisions(questions, answers, "req-1:v1", presented_decisions=decisions)
        assert mapped == ()

    def test_invalid_option_rejected(self) -> None:
        questions, decisions = self._questions_and_decisions()
        answers = (QuestionAnswer(question_id="repair:repair_servings_1_abc", value="not-an-option"),)

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
            QuestionAnswer(question_id="repair:repair_servings_1_abc", value=decisions[0].option_id),
            QuestionAnswer(question_id="repair:repair_servings_1_abc", value="__skip__"),
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
        answers = (QuestionAnswer(question_id="repair:repair_servings_1_abc", value=decision.option_id),)

        mapped = answers_to_approved_decisions(questions, answers, "req-1:v2", presented_decisions=decisions)
        assert mapped[0].plan_revision == "req-1:v2"
        assert mapped[0].payload == decision.payload

    def test_gap_and_assumption_answers_validate_without_decisions(self) -> None:
        """TEXT gap answers and assumption CHOICE answers validate cleanly;
        they carry no ApprovedDecision yet (contract v2)."""
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
        assert answers_to_approved_decisions(questions, answers, "req-1:v1") == ()


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
