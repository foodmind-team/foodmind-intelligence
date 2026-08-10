"""Application service — the entry use case for cooking plan generation.

Per handbook 9.4: this service wraps the compiled LangGraph graph and
provides a single execute() method that the API route calls. The route
does NOT call individual nodes directly.

CPU-bound solver concern (handbook 9.7):
  LLM and HTTP calls are asynchronous I/O. CP-SAT solving is CPU-bound
  and must not block the event loop for long requests. For the bounded MVP,
  the solver timeout is kept short (default 5 seconds). If future workloads
  become large, move optimisation to a controlled worker queue rather than
  spawning unbounded threads.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from cooking_plan_agent.domain.errors import DomainErrorCode, public_message_for
from cooking_plan_agent.domain.models import (
    ConfirmationAnswersRequest,
    ExtractedRecipeCandidate,
    FailedPlanResponse,
    GeneratePlanRequest,
    PlanResponse,
    PreprocessRecipesRequest,
    PreprocessRecipesResponse,
    RecipeGap,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

logger = logging.getLogger(__name__)


class GenerateCookingPlanService:
    """Application service that orchestrates the cooking plan generation workflow.

    Constructed once at app startup (FastAPI lifespan) and stored on
    app.state for route-level dependency injection. Stateless — all
    request-specific data flows through the graph invoke.

    Handbook 9.4: the route calls this use case; the route does not call
    individual nodes.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[PlanState, WorkflowContext],
        context: WorkflowContext,
    ) -> None:
        """Initialise the service with a compiled graph and immutable context.

        Args:
            graph: A compiled LangGraph StateGraph ready for ainvoke().
            context: Frozen WorkflowContext carrying injectable services
                     (RecipeExtractor, RecipeResearcher, etc.).
        """
        self._graph = graph
        self._context = context

    async def preprocess(
        self,
        body: PreprocessRecipesRequest,
    ) -> PreprocessRecipesResponse:
        """Parse stored recipe text and fill gaps with deterministic local rules."""
        import asyncio

        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor
        from cooking_plan_agent.parsing.gaps import find_recipe_gaps
        from cooking_plan_agent.parsing.inference import (
            GapClass,
            InferenceResult,
            _detect_primary_technique,
            _infer_duration,
            _infer_heat,
            _infer_resources,
            _infer_temperature,
            merge_inference,
        )

        extractor = RuleExtractor()

        def _fill_gap(candidate, gap):
            technique = _detect_primary_technique(candidate)
            if "heat_level" in gap.field_path:
                return _infer_heat(gap, candidate, technique)
            if "duration" in gap.field_path.lower():
                return _infer_duration(gap, technique)
            if "temperature" in gap.field_path.lower():
                if gap.gap_class == GapClass.SAFETY_CRITICAL:
                    return None
                return _infer_temperature(gap, technique)
            if "resource" in gap.field_path.lower():
                return _infer_resources(gap, technique)
            return None

        async def _process_one(recipe) -> ExtractedRecipeCandidate:
            candidate = await extractor.extract(recipe.text)
            filled: list[RecipeGap] = []
            for gap in find_recipe_gaps(candidate):
                result = _fill_gap(candidate, gap)
                if result is not None:
                    filled.append(result[0])
            merged = merge_inference(
                candidate,
                InferenceResult(
                    filled_gaps=tuple(filled),
                    unresolved_gaps=(),
                    assumptions=(),
                ),
            )
            return merged.model_copy(update={"recipe_id": recipe.recipe_id})

        candidates = await asyncio.gather(*(_process_one(recipe) for recipe in body.recipes))
        return PreprocessRecipesResponse(recipes=tuple(candidates))

    async def execute(
        self,
        request: GeneratePlanRequest,
        thread_id: str | None = None,
        progress_callback: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> PlanResponse:
        """Run the cooking plan generation workflow for the given request.

        Args:
            request: A validated GeneratePlanRequest from the Spring Boot
                     internal endpoint.
            thread_id: Optional LangGraph thread ID for checkpoint persistence
                (P2-06). When provided, node-boundary state is stored under
                this thread so a restarted process can resume it. When None,
                the graph runs stateless.
            progress_callback: Optional safe node-boundary observer. Receives
                only the public node name and completed-node count.

        Returns:
            PlanResponse: One of ReadyPlanResponse, ConfirmationPlanResponse,
                          InfeasiblePlanResponse, or FailedPlanResponse.

        Raises:
            WorkflowException: If the graph encounters a domain error that
                produces a FAILED terminal state. This is caught by the
                global exception handler and mapped to a stable response.
        """
        initial_state: PlanState = {"request": request}

        invoke_config: RunnableConfig = {"recursion_limit": 30}
        if thread_id is not None:
            invoke_config["configurable"] = {"thread_id": thread_id}

        result: PlanState
        if progress_callback is None:
            result = cast(
                PlanState,
                await self._graph.ainvoke(
                    initial_state,
                    context=self._context,
                    config=invoke_config,
                ),
            )
        else:
            # ``updates`` reliably emits each completed node even when the
            # production graph uses a persistent checkpointer; ``values``
            # provides the authoritative graph state. Only the node name and
            # a monotonic counter leave this service: prompts, model output,
            # update payloads, and hidden reasoning are never exposed.
            result = initial_state
            completed_steps = 0
            async for stream_mode, payload in self._graph.astream(
                initial_state,
                context=self._context,
                config=invoke_config,
                stream_mode=["updates", "values"],
            ):
                if stream_mode == "values":
                    result = cast(PlanState, payload)
                    continue
                if not isinstance(payload, dict):
                    continue
                for node_name in payload:
                    if not isinstance(node_name, str):
                        continue
                    completed_steps += 1
                    await progress_callback(node_name, completed_steps)

        response = result.get("response")
        if not isinstance(response, PlanResponse):
            # Graph reached END without a valid terminal response — defensive fallback.
            logger.error(
                "Graph completed without a valid response field | request_id=%s | state_keys=%s",
                request.request_id,
                list(result.keys()),
            )
            return FailedPlanResponse(
                status="FAILED",
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                correlation_id=request.request_id,
                # P2-03: public text resolves through the message catalog.
                message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
            )

        return response

    async def continue_after_confirmation(
        self,
        body: ConfirmationAnswersRequest,
        thread_id: str | None = None,
    ) -> PlanResponse:
        """Resume a paused NEEDS_CONFIRMATION dialog with user answers (P5-4).

        Re-enters the same checkpoint thread with the user's answers so
        ``apply_confirmation`` consumes them and the graph resumes toward
        READY (or another confirmation turn). The original GeneratePlanRequest
        lives in the checkpoint state, so the client only resubmits answers.

        Requires ``confirmation_dialog_enabled`` AND a checkpointer; otherwise
        the graph was compiled with the terminal-confirmation edge and there
        is nothing to resume — the service returns a FAILED response rather
        than silently re-running.
        """
        # P5-4 预检：无 checkpointer（interrupt 必需）→ 稳定 FAILED，绝不静默重跑。
        if self._graph.checkpointer is None:
            return FailedPlanResponse(
                status="FAILED",
                error_code=DomainErrorCode.CONFIRMATION_DIALOG_UNAVAILABLE.value,
                correlation_id=body.plan_id,
                message=public_message_for(DomainErrorCode.CONFIRMATION_DIALOG_UNAVAILABLE.value),
            )

        invoke_config: RunnableConfig = {"recursion_limit": 30}
        if thread_id is not None:
            invoke_config["configurable"] = {"thread_id": thread_id}

        from langgraph.types import Command

        result = await self._graph.ainvoke(
            Command(resume=[answer.model_dump(mode="json") for answer in body.answers]),
            context=self._context,
            config=invoke_config,
        )

        response = result.get("response")
        if isinstance(response, PlanResponse):
            return response
        # Resumed but no terminal response — defensive FAILED fallback.
        logger.error(
            "Confirmation resume completed without a response | plan_id=%s | state_keys=%s",
            body.plan_id,
            list(result.keys()),
        )
        return FailedPlanResponse(
            status="FAILED",
            error_code=DomainErrorCode.INTERNAL_ERROR.value,
            correlation_id=body.plan_id,
            message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
        )
