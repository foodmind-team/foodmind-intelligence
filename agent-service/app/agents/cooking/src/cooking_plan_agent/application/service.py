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

from langgraph.graph.state import CompiledStateGraph

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    FailedPlanResponse,
    GeneratePlanRequest,
    PlanResponse,
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

    def __init__(self, graph: CompiledStateGraph, context: WorkflowContext) -> None:
        """Initialise the service with a compiled graph and immutable context.

        Args:
            graph: A compiled LangGraph StateGraph ready for ainvoke().
            context: Frozen WorkflowContext carrying injectable services
                     (RecipeExtractor, RecipeResearcher, etc.).
        """
        self._graph = graph
        self._context = context

    async def execute(self, request: GeneratePlanRequest) -> PlanResponse:
        """Run the cooking plan generation workflow for the given request.

        Args:
            request: A validated GeneratePlanRequest from the Spring Boot
                     internal endpoint.

        Returns:
            PlanResponse: One of ReadyPlanResponse, ConfirmationPlanResponse,
                          InfeasiblePlanResponse, or FailedPlanResponse.

        Raises:
            WorkflowException: If the graph encounters a domain error that
                produces a FAILED terminal state. This is caught by the
                global exception handler and mapped to a stable response.
        """
        initial_state: PlanState = {"request": request}

        result = await self._graph.ainvoke(
            initial_state,
            context=self._context,
            config={"recursion_limit": 30},
        )

        response = result.get("response")
        if response is None:
            # Graph reached END without setting a response — defensive fallback.
            logger.error(
                "Graph completed without a response field | request_id=%s | state_keys=%s",
                request.request_id,
                list(result.keys()),
            )
            return FailedPlanResponse(
                status="FAILED",
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                correlation_id=request.request_id,
                message="Workflow completed without producing a terminal response.",
            )

        return response
