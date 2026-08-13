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
    Assumption,
    ConfirmationAnswersRequest,
    ConfirmationPlanResponse,
    ExtractedRecipeCandidate,
    FailedPlanResponse,
    GeneratePlanRequest,
    PlanResponse,
    PreprocessRecipesRequest,
    PreprocessRecipesResponse,
    RecipeGap,
    RecipeInput,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], Awaitable[None]]


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
        """Structure recipe text and fill practical gaps before planning.

        The configured extractor is LLM-backed in production, so extraction
        and common-sense completion happen in one bounded structured-output
        call. Deterministic inference remains a graceful fallback for fields
        the model leaves empty or when the LLM adapter falls back entirely.
        """
        import asyncio

        from cooking_plan_agent.parsing.extractor import RecipeExtractor as RuleExtractor
        from cooking_plan_agent.parsing.gaps import GapClass, find_recipe_gaps
        from cooking_plan_agent.parsing.inference import (
            InferenceResult,
            _detect_primary_technique,
            _infer_duration,
            _infer_heat,
            _infer_resources,
            _infer_temperature,
            merge_inference,
        )

        extractor = self._context.recipe_extractor or RuleExtractor()

        def _fill_gap(
            candidate: ExtractedRecipeCandidate,
            gap: RecipeGap,
        ) -> tuple[RecipeGap, Assumption] | None:
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

        async def _process_one(recipe: RecipeInput) -> ExtractedRecipeCandidate:
            candidate = await extractor.extract(recipe.text)
            filled: list[RecipeGap] = []
            unresolved: list[RecipeGap] = []
            for gap in find_recipe_gaps(candidate):
                result = _fill_gap(candidate, gap)
                if result is not None:
                    filled.append(result[0])
                else:
                    unresolved.append(gap)
            merged = merge_inference(
                candidate,
                InferenceResult(
                    filled_gaps=tuple(filled),
                    unresolved_gaps=tuple(unresolved),
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
    ) -> PlanResponse:
        """Run the cooking plan generation workflow for the given request.

        Args:
            request: A validated GeneratePlanRequest from the Spring Boot
                     internal endpoint.
            thread_id: Optional LangGraph thread ID for checkpoint persistence
                (P2-06). When provided, node-boundary state is stored under
                this thread so a restarted process can resume it. When None,
                the graph runs stateless.

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

        result = await self._graph.ainvoke(
            initial_state,
            context=self._context,
            config=invoke_config,
        )

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

        if isinstance(response, ConfirmationPlanResponse) and not response.confirmation_questions:
            logger.error(
                "Rejected non-actionable confirmation response | request_id=%s",
                request.request_id,
            )
            return FailedPlanResponse(
                status="FAILED",
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                correlation_id=request.request_id,
                message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
            )

        return response

    async def execute_with_progress(
        self,
        request: GeneratePlanRequest,
        thread_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> PlanResponse:
        """Run the graph while publishing real node-boundary progress.

        ``updates`` reports the nodes that actually completed in LangGraph;
        ``values`` carries the authoritative full state. This keeps progress
        truthful without estimating work from elapsed wall-clock time.
        """
        initial_state: PlanState = {"request": request}
        invoke_config: RunnableConfig = {"recursion_limit": 30}
        if thread_id is not None:
            invoke_config["configurable"] = {"thread_id": thread_id}

        final_state: dict[str, object] = {}
        completed_steps = 0
        async for mode, chunk in self._graph.astream(
            initial_state,
            context=self._context,
            config=invoke_config,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final_state = cast(dict[str, object], chunk)
                continue
            for node in chunk:
                if node == "__interrupt__":
                    continue
                completed_steps += 1
                if on_progress is not None:
                    await on_progress(node, completed_steps)

        response = final_state.get("response")
        if isinstance(response, PlanResponse):
            if isinstance(response, ConfirmationPlanResponse) and not response.confirmation_questions:
                logger.error(
                    "Rejected non-actionable confirmation response | request_id=%s",
                    request.request_id,
                )
                return FailedPlanResponse(
                    status="FAILED",
                    error_code=DomainErrorCode.INTERNAL_ERROR.value,
                    correlation_id=request.request_id,
                    message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
                )
            return response

        logger.error(
            "Graph stream completed without a valid response | request_id=%s | state_keys=%s",
            request.request_id,
            list(final_state.keys()),
        )
        return FailedPlanResponse(
            status="FAILED",
            error_code=DomainErrorCode.INTERNAL_ERROR.value,
            correlation_id=request.request_id,
            message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
        )

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
            if isinstance(response, ConfirmationPlanResponse) and not response.confirmation_questions:
                logger.error(
                    "Rejected non-actionable confirmation response after resume | plan_id=%s",
                    body.plan_id,
                )
                return FailedPlanResponse(
                    status="FAILED",
                    error_code=DomainErrorCode.INTERNAL_ERROR.value,
                    correlation_id=body.plan_id,
                    message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
                )
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
