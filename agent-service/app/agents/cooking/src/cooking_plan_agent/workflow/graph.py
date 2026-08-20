# =============================================================================
# LangGraph 图构建器（workflow/graph）
# -----------------------------------------------------------------------------
# 按手册 8.7 组装节点与边。图是最后构建的部分，只编排已测试的服务。
# 保持终态响应构建显式 —— 图必须以且仅以一个有效响应状态结束。
# =============================================================================

"""LangGraph graph builder — assembles nodes and edges per handbook 8.7.

LangGraph 图构建器 —— 按手册 8.7 组装节点与边。

The graph is the last thing built. It only orchestrates tested services.
Keep terminal response construction explicit — the graph must end with
exactly one valid response status.

图是最后构建的部分，只编排已测试的服务。保持终态响应构建显式 ——
图必须以且仅以一个有效响应状态结束。
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.nodes import (
    agent_controller_node,
    apply_confirmation_node,
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
    run_tool_node,
    solve_schedule_node,
    validate_input_node,
    validate_recipe_ir_node,
    validate_safety_node,
    verify_schedule_node,
)
from cooking_plan_agent.workflow.routing import (
    route_after_confirmation,
    route_after_controller,
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

    构建并编译烹饪计划生成工作流图。

    Args:
        checkpointer: Optional LangGraph saver for node-boundary persistence
            (P2-06). When None the graph runs stateless — the pre-P2-06
            behaviour. Injected at startup so no connection is created at
            module import time.

    Args（参数）：
        checkpointer：可选的 LangGraph 保存器，用于节点边界持久化（P2-06）。
            为 None 时图以无状态方式运行 —— 即 P2-06 之前的行为。
            在启动时注入，因此模块导入时不会创建连接。

    Returns:
        A compiled graph ready for ainvoke() with PlanState input
        and WorkflowContext as runtime context.

    Returns（返回值）：
        一个已编译的图，可直接以 PlanState 作为输入、
        以 WorkflowContext 作为运行时上下文调用 ainvoke()。
    """
    builder = StateGraph(PlanState, context_schema=WorkflowContext)

    # ------------------------------------------------------------------
    # 8.5 Register all nodes (16 core + P4-01 explain_schedule)
    # 8.5 注册所有节点（16 个核心节点 + P4-01 explain_schedule）
    # ------------------------------------------------------------------
    builder.add_node("validate_input", validate_input_node)
    builder.add_node("parse_recipes", parse_recipes_node)
    builder.add_node("detect_gaps", detect_gaps_node)
    builder.add_node("infer_local", infer_local_node)
    # research_missing runs only when web research is enabled (P1-01)
    # research_missing 仅在启用联网研究时运行（P1-01）
    builder.add_node("research_missing", research_missing_node)
    # P1-01: consumes research_evidence and applies it back to candidates
    # P1-01：消费 research_evidence 并将其应用回候选
    builder.add_node("apply_research_evidence", apply_research_evidence_node)
    builder.add_node("validate_recipe_ir", validate_recipe_ir_node)
    builder.add_node("validate_safety", validate_safety_node)
    builder.add_node("check_feasibility", check_feasibility_node)
    # build_confirmation_response is a terminal node reachable from multiple paths
    # build_confirmation_response 是可通过多条路径到达的终态节点
    builder.add_node("build_confirmation_response", build_confirmation_response_node)
    builder.add_node("merge_preparation", merge_preparation_node)
    builder.add_node("build_task_graph", build_task_graph_node)
    builder.add_node("solve_schedule", solve_schedule_node)
    builder.add_node("verify_schedule", verify_schedule_node)
    # P5-3: 反思修复节点 —— 验证失败后降级重试（唯一的 back-edge 起点）。
    builder.add_node("repair_schedule", repair_schedule_node)
    # P5-2: ReAct 控制器节点 —— LLM 编排（软决策），失败回退确定性 DAG。
    builder.add_node("agent_controller", agent_controller_node)
    builder.add_node("run_tool", run_tool_node)
    # P5-4: 确认对话中间态 —— 消费用户 answers 后续接重排。
    builder.add_node("apply_confirmation", apply_confirmation_node)
    # P4-01: additive schedule explanation (verify → explain → READY render).
    # P4-01：加法式排程解释（verify → explain → READY 渲染）。
    builder.add_node("explain_schedule", explain_schedule_node)
    builder.add_node("render_ready_response", render_ready_response_node)
    builder.add_node("render_infeasible_response", render_infeasible_response_node)
    builder.add_node("render_failed_response", render_failed_response_node)

    # ------------------------------------------------------------------
    # 8.7 Fixed edges — linear pipeline sections
    # 8.7 固定边 —— 线性流水线段
    # ------------------------------------------------------------------
    # P5-2: 控制器入口由配置门控。启用时 START 先走控制器循环；
    # 禁用时保持原 START -> validate_input 路径（零回归）。
    from cooking_plan_agent.config.settings import get_settings

    if get_settings().agent_controller_enabled:
        builder.add_edge(START, "agent_controller")
    else:
        builder.add_edge(START, "validate_input")

    # P5-2: 控制器循环 —— LLM 决策（think）→ 工具执行（act）→ 观察回填
    # （observe）→ 回到控制器。agent_max_steps / agent_mode=deterministic
    # 双重保证终止；error 短路 FAILED。
    builder.add_conditional_edges(
        "agent_controller",
        route_after_controller,
        {
            "run_tool": "run_tool",
            "validate_input": "validate_input",
            "render_failed_response": "render_failed_response",
        },
    )
    # back-edge：观察后回到控制器，形成 ReAct 循环。
    builder.add_edge("run_tool", "agent_controller")

    # ------------------------------------------------------------------
    # 8.6 P0-03 error short-circuit — every error-capable node routes to
    # FAILED the moment a WorkflowError is written. __continue__ carries
    # the happy path forward.
    # 8.6 P0-03 错误短路 —— 每个可能产生错误的节点一旦写入 WorkflowError
    # 就路由到 FAILED。__continue__ 携带正常路径继续推进。
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
    # 8.6 条件边
    # ------------------------------------------------------------------

    # detect_gaps: gaps exist -> infer_local; no gaps -> skip to validation
    # detect_gaps：有缺口 -> infer_local；无缺口 -> 跳到验证
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
    # infer_local：已解决 -> 验证；未解决的关键缺口 -> 确认；
    #   可研究（未来）-> research_missing
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
    # research_missing -> apply_research_evidence（P1-01）：证据被写回候选，
    # 然后路由到 IR 或确认。
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
    # validate_safety：政策解析错误 -> FAILED（P3-04）；
    # 不可修复的安全问题 -> INFEASIBLE；否则 -> 可行性检查
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
    # check_feasibility：可行 -> merge；不可行但可修复 -> confirm；
    #   不可行且不可修复 -> INFEASIBLE
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
    # 8.7 固定边 —— 预处理与排程流水线
    # ------------------------------------------------------------------
    # Linear chain: merge -> task graph -> CP-SAT solve -> verify
    # 线性链：merge -> task graph -> CP-SAT 求解 -> verify
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
    # solve_schedule：OPTIMAL/FEASIBLE -> verify；INFEASIBLE -> infeasible；
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
    # verify_schedule：通过 -> explain（P4-01）然后 READY；失败 -> 修复
    # 循环（P5-3）；错误 -> FAILED。
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
    # P4-01：解释是加法能力 —— 无论能否生成解释，已验证排程都会渲染为 READY。
    builder.add_edge("explain_schedule", "render_ready_response")

    # ------------------------------------------------------------------
    # 8.7 Terminal edges — every response node ends the graph
    # 8.7 终态边 —— 每个响应节点结束图
    # ------------------------------------------------------------------
    builder.add_edge("render_ready_response", END)
    # P5-4: 确认对话启用时，NEEDS_CONFIRMATION 不再是终态 —— 用户在
    # apply_confirmation 处 interrupt 挂起，answers 续接后重排；禁用或
    # 未注入 checkpointer（interrupt 必需）时保持原终态（零回归）。
    from cooking_plan_agent.config.settings import get_settings

    if get_settings().confirmation_dialog_enabled and checkpointer is not None:
        builder.add_edge("build_confirmation_response", "apply_confirmation")
        builder.add_conditional_edges(
            "apply_confirmation",
            route_after_confirmation,
            {
                "parse_recipes": "parse_recipes",
                "solve_schedule": "solve_schedule",
                "build_confirmation_response": "build_confirmation_response",
                "render_ready_response": "render_ready_response",
                "render_failed_response": "render_failed_response",
            },
        )
    else:
        builder.add_edge("build_confirmation_response", END)
    builder.add_edge("render_infeasible_response", END)
    builder.add_edge("render_failed_response", END)

    return builder.compile(checkpointer=checkpointer)
