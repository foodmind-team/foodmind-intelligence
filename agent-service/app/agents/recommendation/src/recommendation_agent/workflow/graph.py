"""Compilation and execution of the fixed acyclic Recommendation Agent graph."""

import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse
from recommendation_agent.time.budget import DeadlineBudget
from recommendation_agent.workflow.context import WorkflowContext
from recommendation_agent.workflow.nodes import WorkflowNodes
from recommendation_agent.workflow.routing import route_after_node, route_after_success_builder
from recommendation_agent.workflow.state import FailureRecord, RecommendationState

NODE_ORDER = (
    "validate_envelope",
    "score_once",
    "validate_compatibility",
    "select_results",
    "derive_reasons",
    "render_explanations",
    "build_success",
)
MAX_WORKFLOW_STEPS = 8


class BoundedRecommendationWorkflow:
    """A precompiled graph with fixed forward edges and a single inference node."""

    def __init__(self, context: WorkflowContext) -> None:
        self.context = context
        self.nodes = WorkflowNodes(context)
        self.compiled = self._compile()
        self._assert_bounded_shape()

    def _compile(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        graph = StateGraph(RecommendationState)

        def add_instrumented(
            name: str,
            node: Callable[[RecommendationState], Awaitable[dict[str, Any]]],
        ) -> None:
            # LangGraph's overload omits ordinary async callables that it accepts at runtime.
            graph.add_node(name, self._instrument(name, node))  # type: ignore[call-overload]

        add_instrumented("validate_envelope", self.nodes.validate_envelope)
        add_instrumented("score_once", self.nodes.score_once)
        add_instrumented("validate_compatibility", self.nodes.validate_compatibility)
        add_instrumented("select_results", self.nodes.select_results)
        add_instrumented("derive_reasons", self.nodes.derive_reasons)
        add_instrumented("render_explanations", self.nodes.render_explanations)
        add_instrumented("build_success", self.nodes.build_success)
        add_instrumented("build_failure", self.nodes.build_failure)
        graph.add_edge(START, "validate_envelope")
        for current, following in zip(NODE_ORDER, NODE_ORDER[1:], strict=False):
            graph.add_conditional_edges(
                current,
                route_after_node,
                {"continue": following, "failure": "build_failure"},
            )
        graph.add_conditional_edges(
            "build_success",
            route_after_success_builder,
            {"success": END, "failure": "build_failure"},
        )
        graph.add_edge("build_failure", END)
        return graph.compile()

    def _instrument(
        self,
        name: str,
        node: Callable[[RecommendationState], Awaitable[dict[str, Any]]],
    ) -> Callable[[RecommendationState], Awaitable[RecommendationState]]:
        async def instrumented(state: RecommendationState) -> RecommendationState:
            started = perf_counter()
            try:
                return cast(RecommendationState, await node(state))
            finally:
                self.context.metrics.record_stage(stage=name, duration_seconds=perf_counter() - started)

        return instrumented

    def _assert_bounded_shape(self) -> None:
        graph = self.compiled.get_graph()
        nodes = set(graph.nodes)
        expected = {"__start__", "__end__", *NODE_ORDER, "build_failure"}
        if nodes != expected:
            raise RuntimeError("compiled workflow node set changed")
        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        for edge in graph.edges:
            adjacency[edge.source].add(edge.target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise RuntimeError("compiled workflow contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for target in adjacency[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)

        visit("__start__")

    async def run(self, request: AgentRequest, *, agent_trace_id: str) -> AgentResponse:
        started = perf_counter()
        state: RecommendationState = {
            "request": request,
            "agent_trace_id": agent_trace_id,
            "inference_calls": 0,
            "node_trace": (),
        }
        budget: DeadlineBudget | None = None
        try:
            budget = DeadlineBudget.from_absolute(
                request.deadline_at,
                clock=self.context.clock,
                minimum_seconds=self.context.settings.min_deadline_budget_ms / 1000.0,
            )
            state["deadline_expiry"] = budget.monotonic_expiry
        except AgentError as error:
            state["failure"] = FailureRecord(error.code, error.http_status, error.retryable)

        try:
            if budget is None:
                result = await self.compiled.ainvoke(state)
            else:
                async with asyncio.timeout(budget.remaining()):
                    result = await self.compiled.ainvoke(state)
            final_state = cast(RecommendationState, result)
        except TimeoutError:
            final_state = await self._terminal_failure(state, ErrorCode.DEADLINE_EXHAUSTED, 504)
        except Exception:  # noqa: BLE001 - graph boundary maps unknown failures without exposing details
            final_state = await self._terminal_failure(state, ErrorCode.INTERNAL_ERROR, 500)

        if "response" in final_state:
            if len(final_state.get("node_trace", ())) > MAX_WORKFLOW_STEPS:
                raise RuntimeError("workflow step bound exceeded")
            response = final_state["response"]
            self.context.metrics.record_request(
                result="success",
                duration_seconds=perf_counter() - started,
                candidates=len(request.candidates),
                outputs=len(response.recommendations),
            )
            return response
        failure = final_state.get("failure") or FailureRecord(ErrorCode.INTERNAL_ERROR, 500)
        failure_response = final_state.get("failure_response")
        content = failure_response.model_dump(mode="json", by_alias=True) if failure_response is not None else None
        self.context.metrics.record_failure(failure.code)
        self.context.metrics.record_request(
            result="failure",
            duration_seconds=perf_counter() - started,
            candidates=len(request.candidates),
            outputs=0,
        )
        raise AgentError(
            failure.code,
            http_status=failure.http_status,
            retryable=failure.retryable,
            failure_content=content,
        )

    async def _terminal_failure(
        self,
        state: RecommendationState,
        code: ErrorCode,
        http_status: int,
    ) -> RecommendationState:
        failed_state: RecommendationState = {
            **state,
            "failure": FailureRecord(code, http_status),
        }
        update = await self.nodes.build_failure(failed_state)
        failed_state.update(cast(RecommendationState, update))
        return failed_state

    async def aclose(self) -> None:
        """Inference lifecycle is owned by the app; the graph owns no resource."""
