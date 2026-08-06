"""LangGraph graph builder — assembles nodes and edges per handbook 8.7.

The graph is the last thing built. It only orchestrates tested services.
Keep terminal response construction explicit — the graph must end with
exactly one valid response status.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.nodes import (
    apply_research_evidence_node,
    build_confirmation_response_node,
    build_task_graph_node,
    check_feasibility_node,
    detect_gaps_node,
    explain_schedule_node,
    infer_local_node,
    merge_preparation_node,
    parse_recipes_node,
    render_failed_response_node,
    render_infeasible_response_node,
    render_ready_response_node,
    repair_schedule_node,
    research_missing_node,
    solve_schedule_node,
    validate_input_node,
    validate_recipe_ir_node,
    validate_safety_node,
    verify_schedule_node,
)
from cooking_plan_agent.workflow.routing import (
    route_after_feasibility,
    route_after_gap_detection,
    route_after_local_inference,
    route_after_repair,
    route_after_research,
    route_after_safety,
    route_after_solve,
    route_after_verification,
    route_on_workflow_error,
)
from cooking_plan_agent.workflow.state import PlanState


def build_cooking_plan_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[PlanState, WorkflowContext]:
    """Build and compile the cooking-plan generation workflow graph.

    Args:
        checkpointer: Optional LangGraph saver for node-boundary persistence
            (P2-06). When None the graph runs stateless — the pre-P2-06
            behaviour. Injected at startup so no connection is created at
            module import time.

    Returns:
        A compiled graph ready for ainvoke() with PlanState input
        and WorkflowContext as runtime context.
    """
    builder = StateGraph(PlanState, context_schema=WorkflowContext)

    # ------------------------------------------------------------------
    # 8.5 Register all nodes (16 core + P4-01 explain_schedule)
    # ------------------------------------------------------------------
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("parse_recipes", parse_recipes_node)
    builder.add_node("detect_gaps", detect_gaps_node)
    builder.add_node("infer_local", infer_local_node)
    # research_missing runs only when web research is enabled (P1-01)
    builder.add_node("research_missing", research_missing_node)
    # P1-01: consumes research_evidence and applies it back to candidates
    builder.add_node("apply_research_evidence", apply_research_evidence_node)
    builder.add_node("validate_recipe_ir", validate_recipe_ir_node)
    builder.add_node("validate_safety", validate_safety_node)
    builder.add_node("check_feasibility", check_feasibility_node)
    # build_confirmation_response is a terminal node reachable from multiple paths
    builder.add_node("build_confirmation_response", build_confirmation_response_node)
    builder.add_node("merge_preparation", merge_preparation_node)
    builder.add_node("build_task_graph", build_task_graph_node)
    builder.add_node("solve_schedule", solve_schedule_node)
    builder.add_node("verify_schedule", verify_schedule_node)
    # P5-3: 反思修复节点 —— 验证失败后降级重试（唯一的 back-edge 起点）。
    builder.add_node("repair_schedule", repair_schedule_node)
    # P4-01: additive schedule explanation (verify → explain → READY render).
    builder.add_node("explain_schedule", explain_schedule_node)
    builder.add_node("render_ready_response", render_ready_response_node)
    builder.add_node("render_infeasible_response", render_infeasible_response_node)
    builder.add_node("render_failed_response", render_failed_response_node)

    # ------------------------------------------------------------------
    # 8.7 Fixed edges — linear pipeline sections
    # ------------------------------------------------------------------
    builder.add_edge(START, "validate_input")

    # ------------------------------------------------------------------
    # 8.6 P0-03 error short-circuit — every error-capable node routes to
    # FAILED the moment a WorkflowError is written. __continue__ carries
    # the happy path forward.
    # ------------------------------------------------------------------
    builder.add_conditional_edges(
        "validate_input",
        route_on_workflow_error,
        {
            "render_failed_response": "render_failed_response",
            "__continue__": "parse_recipes",
        },
    )
    builder.add_conditional_edges(
        "parse_recipes",
        route_on_workflow_error,
        {
            "render_failed_response": "render_failed_response",
            "__continue__": "detect_gaps",
        },
    )

    # ------------------------------------------------------------------
    # 8.6 Conditional edges
    # ------------------------------------------------------------------

    # detect_gaps: gaps exist -> infer_local; no gaps -> skip to validation
    builder.add_conditional_edges(
        "detect_gaps",
        route_after_gap_detection,
        {
            "infer_local": "infer_local",
            "validate_recipe_ir": "validate_recipe_ir",
        },
    )

    # infer_local: resolved -> validate; unresolved critical -> confirm;
    #   researchable (future) -> research_missing
    builder.add_conditional_edges(
        "infer_local",
        route_after_local_inference,
        {
            "research_missing": "research_missing",
            "build_confirmation_response": "build_confirmation_response",
            "validate_recipe_ir": "validate_recipe_ir",
        },
    )

    # research_missing -> apply_research_evidence (P1-01): evidence is
    # written back into candidates, then routed to IR or confirmation.
    builder.add_edge("research_missing", "apply_research_evidence")
    builder.add_conditional_edges(
        "apply_research_evidence",
        route_after_research,
        {
            "build_confirmation_response": "build_confirmation_response",
            "validate_recipe_ir": "validate_recipe_ir",
        },
    )
    builder.add_conditional_edges(
        "validate_recipe_ir",
        route_on_workflow_error,
        {
            "render_failed_response": "render_failed_response",
            "__continue__": "validate_safety",
        },
    )

    # validate_safety: policy-resolution error -> FAILED (P3-04);
    # unrepairable safety issue -> INFEASIBLE; else -> feasibility
    builder.add_conditional_edges(
        "validate_safety",
        route_after_safety,
        {
            "check_feasibility": "check_feasibility",
            "render_infeasible_response": "render_infeasible_response",
            "render_failed_response": "render_failed_response",
        },
    )

    # check_feasibility: feasible -> merge; infeasible+repairable -> confirm;
    #   infeasible+unrepairable -> INFEASIBLE
    builder.add_conditional_edges(
        "check_feasibility",
        route_after_feasibility,
        {
            "merge_preparation": "merge_preparation",
            "build_confirmation_response": "build_confirmation_response",
            "render_infeasible_response": "render_infeasible_response",
        },
    )

    # ------------------------------------------------------------------
    # 8.7 Fixed edges — preparation & scheduling pipeline
    # ------------------------------------------------------------------
    # Linear chain: merge -> task graph -> CP-SAT solve -> verify
    builder.add_edge("merge_preparation", "build_task_graph")
    builder.add_conditional_edges(
        "build_task_graph",
        route_on_workflow_error,
        {
            "render_failed_response": "render_failed_response",
            "__continue__": "solve_schedule",
        },
    )

    # solve_schedule: OPTIMAL/FEASIBLE -> verify; INFEASIBLE -> infeasible;
    #   MODEL_INVALID/error -> FAILED
    builder.add_conditional_edges(
        "solve_schedule",
        route_after_solve,
        {
            "verify_schedule": "verify_schedule",
            "render_infeasible_response": "render_infeasible_response",
            "render_failed_response": "render_failed_response",
        },
    )

    # verify_schedule: passes -> explain (P4-01) then READY; fails -> repair
    # loop (P5-3); error -> FAILED.
    builder.add_conditional_edges(
        "verify_schedule",
        route_after_verification,
        {
            "explain_schedule": "explain_schedule",
            "repair_schedule": "repair_schedule",
            "render_failed_response": "render_failed_response",
        },
    )

    # P5-3: 反思修复循环 —— 唯一的 back-edge。retrying -> 重新求解；
    # gave_up -> FAILED。循环次数由 repair_attempts <= max_attempts 保证终止。
    builder.add_conditional_edges(
        "repair_schedule",
        route_after_repair,
        {
            "solve_schedule": "solve_schedule",
            "render_failed_response": "render_failed_response",
        },
    )

    # P4-01: the explanation is additive — the verified schedule renders
    # READY regardless of whether an explanation could be produced.
    builder.add_edge("explain_schedule", "render_ready_response")

    # ------------------------------------------------------------------
    # 8.7 Terminal edges — every response node ends the graph
    # ------------------------------------------------------------------
    builder.add_edge("render_ready_response", END)
    builder.add_edge("build_confirmation_response", END)
    builder.add_edge("render_infeasible_response", END)
    builder.add_edge("render_failed_response", END)

    return builder.compile(checkpointer=checkpointer)
