"""Deterministic recovery for LLM-derived workflow failures."""

from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pytest

from cooking_plan_agent.application.service import GenerateCookingPlanService
from cooking_plan_agent.domain.errors import DomainErrorCode, public_message_for
from cooking_plan_agent.domain.models import (
    ExtractedRecipeCandidate,
    FailedPlanResponse,
    GeneratePlanRequest,
    InfeasiblePlanResponse,
    RecipeInput,
)
from cooking_plan_agent.parsing.extractor import RecipeExtractor as DeterministicRecipeExtractor
from cooking_plan_agent.workflow.context import WorkflowContext


class _ExternalExtractor:
    async def extract(self, source_text: str) -> ExtractedRecipeCandidate:
        raise AssertionError("The fake graph should not call the extractor")


def _request(*, preparsed: bool = False) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        request_id="fallback-request",
        user_id="qa-user",
        recipes=(RecipeInput(recipe_id="kung-pao", text="Kung Pao Chicken", target_servings=Decimal(2)),),
        preparsed_candidates=(
            ExtractedRecipeCandidate(
                recipe_id="kung-pao",
                dish_name="Kung Pao Chicken",
                original_servings=Decimal(2),
                source_language="eng",
                ingredients=(),
                steps=(),
            ),
        )
        if preparsed
        else (),
    )


def _internal_failure() -> FailedPlanResponse:
    return FailedPlanResponse(
        error_code=DomainErrorCode.INTERNAL_ERROR.value,
        correlation_id="fallback-request",
        message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
    )


def _recovered() -> InfeasiblePlanResponse:
    return InfeasiblePlanResponse(plan_id="fallback-request", reasons=("safe deterministic result",))


class _FakeGraph:
    def __init__(self, *, raise_primary: bool = False) -> None:
        self.raise_primary = raise_primary
        self.calls: list[tuple[WorkflowContext, dict[str, Any]]] = []

    async def ainvoke(
        self,
        initial_state: dict[str, object],
        *,
        context: WorkflowContext,
        config: dict[str, Any],
    ) -> dict[str, object]:
        self.calls.append((context, config))
        if not isinstance(context.recipe_extractor, DeterministicRecipeExtractor):
            if self.raise_primary:
                raise RuntimeError("unsafe structured candidate")
            return {"response": _internal_failure()}
        return {"response": _recovered()}

    async def astream(
        self,
        initial_state: dict[str, object],
        *,
        context: WorkflowContext,
        config: dict[str, Any],
        stream_mode: list[str],
    ) -> AsyncIterator[tuple[str, dict[str, object]]]:
        self.calls.append((context, config))
        if not isinstance(context.recipe_extractor, DeterministicRecipeExtractor):
            if self.raise_primary:
                raise RuntimeError("unsafe structured candidate")
            yield "values", {"response": _internal_failure()}
            return
        yield "updates", {"parse_recipes": {}}
        yield "values", {"response": _recovered()}


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_primary", [False, True])
async def test_execute_retries_internal_llm_failure_with_deterministic_parser(raise_primary: bool) -> None:
    graph = _FakeGraph(raise_primary=raise_primary)
    service = GenerateCookingPlanService(
        graph=graph,  # type: ignore[arg-type]
        context=WorkflowContext(recipe_extractor=_ExternalExtractor()),
    )

    response = await service.execute(_request(), thread_id="task-thread")

    assert response.status == "INFEASIBLE"
    assert len(graph.calls) == 2
    assert isinstance(graph.calls[1][0].recipe_extractor, DeterministicRecipeExtractor)
    assert graph.calls[1][1]["configurable"] == {"thread_id": "task-thread__deterministic"}


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_primary", [False, True])
async def test_streaming_execute_retries_and_reports_fallback_progress(raise_primary: bool) -> None:
    graph = _FakeGraph(raise_primary=raise_primary)
    service = GenerateCookingPlanService(
        graph=graph,  # type: ignore[arg-type]
        context=WorkflowContext(recipe_extractor=_ExternalExtractor()),
    )
    progress: list[tuple[str, int]] = []

    response = await service.execute_with_progress(
        _request(),
        thread_id="task-thread",
        on_progress=lambda node, step: _capture_progress(progress, node, step),
    )

    assert response.status == "INFEASIBLE"
    assert progress == [("parse_recipes", 1)]
    assert len(graph.calls) == 2
    assert isinstance(graph.calls[1][0].recipe_extractor, DeterministicRecipeExtractor)


@pytest.mark.asyncio
async def test_preparsed_contract_does_not_retry_with_a_different_parser() -> None:
    graph = _FakeGraph()
    service = GenerateCookingPlanService(
        graph=graph,  # type: ignore[arg-type]
        context=WorkflowContext(recipe_extractor=_ExternalExtractor()),
    )

    response = await service.execute(_request(preparsed=True))

    assert response.status == "FAILED"
    assert len(graph.calls) == 1


async def _capture_progress(progress: list[tuple[str, int]], node: str, step: int) -> None:
    progress.append((node, step))
