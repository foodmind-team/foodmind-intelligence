# =============================================================================
# 纯路由函数模块（workflow/routing）
# -----------------------------------------------------------------------------
# LangGraph 条件边的纯路由函数。按手册 8.6：每个函数返回明确的节点名字面量。
# 无副作用、无服务调用 —— 仅做状态检查。
# =============================================================================

"""Pure routing functions for LangGraph conditional edges.

LangGraph 条件边的纯路由函数。

Per handbook 8.6: each function returns explicit node name literals.
No side effects, no service calls — only state inspection.

按手册 8.6：每个函数返回明确的节点名字面量。无副作用、无服务调用 —— 仅做状态检查。
"""

from typing import Literal

from cooking_plan_agent.workflow.state import PlanState

# ---------------------------------------------------------------------------
# 8.6 P0-03 error short-circuit — shared by every error-capable node
# 8.6 P0-03 错误短路 —— 由每个可能产生错误的节点共享
# ---------------------------------------------------------------------------


def route_on_workflow_error(
    state: PlanState,
) -> Literal["render_failed_response", "__continue__"]:
    """Short-circuit to FAILED the moment any WorkflowError is set.

    一旦设置任何 WorkflowError 就短路到 FAILED。

    Pure read-only routing: never mutates state, never clears or rewrites
    the original error, and never calls a service. Downstream nodes are
    skipped entirely once an error is present (P0-03).

    纯只读路由：绝不修改状态、绝不清除或重写原始错误、绝不调用服务。
    一旦出现错误，下游节点将被完全跳过（P0-03）。
    """
    if state.get("error") is not None:
        return "render_failed_response"
    return "__continue__"


# ---------------------------------------------------------------------------
# 8.6 Routing after gap detection
# 8.6 缺口检测之后的路由
# ---------------------------------------------------------------------------


def route_after_gap_detection(
    state: PlanState,
) -> Literal["infer_local", "validate_recipe_ir"]:
    """If gaps exist, try local inference before validation.

    若存在缺口，则在验证前先尝试本地推断。

    Otherwise, proceed directly to IR validation.
    The gap detection may return empty tuples — only non-empty gaps trigger inference.

    否则直接进入 IR 验证。缺口检测可能返回空元组 —— 仅非空缺口触发推断。
    """
    if state.get("gaps"):
        return "infer_local"
    return "validate_recipe_ir"


# ---------------------------------------------------------------------------
# 8.6 Routing after local inference
# 8.6 本地推断之后的路由
# ---------------------------------------------------------------------------


def route_after_local_inference(
    state: PlanState,
) -> Literal["research_missing", "build_confirmation_response", "validate_recipe_ir"]:
    """After local inference, remaining critical gaps route to:
    - web research (if enabled and gap is heat/duration related)
    - confirmation (if evidence insufficient or research disabled)
    - IR validation (if gaps resolved).

    本地推断之后，剩余的关键缺口路由到：
    - 联网研究（若启用且缺口与火候 / 时长相关）
    - 确认（若证据不足或研究已禁用）
    - IR 验证（若缺口已解决）。

    Only "critical" and "safety_critical" gaps block progress — minor gaps
    (e.g., garnish variation) are tolerated and passed through.

    仅 "critical" 与 "safety_critical" 缺口会阻塞推进 —— 次要缺口
    （如装饰变化）被容忍并放行。
    """
    gaps = state.get("gaps", ())
    critical_gaps = [g for g in gaps if g.gap_class in ("critical", "safety_critical")]

    if not critical_gaps:
        return "validate_recipe_ir"

    # Gap completion is LLM-only now (web search removed). Only non-safety
    # operational gaps (heat/duration/temperature) are researchable; safety-
    # critical and other gaps go straight to user confirmation.
    # 缺口补全现在仅走 LLM（已移除联网搜索）。只有非安全的操作性缺口
    # （火候 / 时长 / 温度）可研究；安全关键与其他缺口直接进入用户确认。
    non_safety_operational = [
        g
        for g in critical_gaps
        if g.gap_class != "safety_critical"
        and any(f in g.field_path.lower() for f in ("heat_level", "duration", "temperature"))
    ]
    return "research_missing" if non_safety_operational else "build_confirmation_response"


# ---------------------------------------------------------------------------
# 8.6 Routing after research evidence application (P1-01)
# 8.6 研究证据应用之后的路由（P1-01）
# ---------------------------------------------------------------------------


def route_after_research(
    state: PlanState,
) -> Literal["build_confirmation_response", "validate_recipe_ir"]:
    """After evidence application, reliable evidence proceeds to IR.

    证据应用之后，可靠的证据继续进入 IR。

    Routes to confirmation whenever the plan cannot proceed safely:
      - ``needs_confirmation`` set (disagreement over threshold, no sources,
        field-location failure, safety-critical temperature without a
        verifiable URL — P1-01 rules 5 & 6)
      - any critical / safety_critical gap remains unresolved after research

    只要计划无法安全推进，就路由到确认：
      - 设置了 ``needs_confirmation``（阈值之上的分歧、无来源、
        字段定位失败、无验证 URL 的安全关键温度 —— P1-01 规则 5 & 6）
      - 研究后仍存在任何未解决的关键 / 安全关键缺口

    Only fully-applied, reliable evidence continues to IR validation.

    只有完全应用、可靠的证据才继续进入 IR 验证。
    """
    if state.get("needs_confirmation"):
        return "build_confirmation_response"

    gaps = state.get("gaps", ())
    if any(g.gap_class in ("critical", "safety_critical") for g in gaps):
        return "build_confirmation_response"

    return "validate_recipe_ir"


# ---------------------------------------------------------------------------
# 8.6 Routing after safety validation
# 8.6 安全验证之后的路由
# ---------------------------------------------------------------------------


def route_after_safety(
    state: PlanState,
) -> Literal["check_feasibility", "render_infeasible_response", "render_failed_response"]:
    """Hard unrepairable safety findings -> INFEASIBLE.

    硬性不可修复的安全发现 -> INFEASIBLE。

    P3-04: a policy-resolution failure (unknown region, unknown version,
    not-yet-effective, or missing-source policy) set by validate_safety_node
    short-circuits to FAILED — a plan must never proceed — let alone reach
    READY — under an unverifiable policy.

    P3-04：validate_safety_node 设置的政策解析失败（未知区域、未知版本、
    尚未生效或缺少来源的政策）会短路到 FAILED —— 在不可验证的政策下，
    计划绝不能推进，更不能到达 READY。

    Otherwise, proceed to feasibility check.
    Safety findings that ARE repairable are injected as safety_tasks in merge_preparation.

    否则，进入可行性检查。可修复的安全发现会作为 safety_tasks 注入 merge_preparation。
    """
    if state.get("error") is not None:
        return "render_failed_response"
    safety_report = state.get("safety_report")
    if safety_report is not None and safety_report.has_unrepairable:
        return "render_infeasible_response"
    return "check_feasibility"


# ---------------------------------------------------------------------------
# 8.6 Routing after feasibility check
# 8.6 可行性检查之后的路由
# ---------------------------------------------------------------------------


def route_after_feasibility(
    state: PlanState,
) -> Literal["merge_preparation", "build_confirmation_response", "render_infeasible_response"]:
    """If infeasible: confirmation with repair options (if any) or INFEASIBLE.

    若不可行：带修复选项的确认（若有）或 INFEASIBLE。

    A None report means feasibility was not checked (e.g., safety short-circuited)
    — in that case, default to merge_preparation (downstream nodes handle it).

    None 报告意味着未做可行性检查（如安全已短路）—— 此时默认进入
    merge_preparation（由下游节点处理）。
    """
    report = state.get("feasibility_report")
    if report is None:
        return "merge_preparation"

    if not report.is_feasible:
        repair_options = state.get("repair_options", ())
        if repair_options:
            return "build_confirmation_response"
        return "render_infeasible_response"

    return "merge_preparation"


# ---------------------------------------------------------------------------
# 8.6 Routing after schedule solve
# 8.6 排程求解之后的路由
# ---------------------------------------------------------------------------


def route_after_solve(
    state: PlanState,
) -> Literal["verify_schedule", "render_infeasible_response", "render_failed_response"]:
    """Solver result determines next step.

    求解结果决定下一步。

    None result = solver errored without producing output -> FAILED.
    OPTIMAL/FEASIBLE proceed to independent verification.
    INFEASIBLE means the solver proved no solution exists.
    MODEL_INVALID/UNKNOWN -> FAILED (likely a model construction bug).

    None 结果 = 求解器出错且未产出结果 -> FAILED。
    OPTIMAL/FEASIBLE 继续进入独立验证。
    INFEASIBLE 表示求解器证明了无解。
    MODEL_INVALID/UNKNOWN -> FAILED（很可能是模型构建缺陷）。

    Any workflow error takes precedence and short-circuits to FAILED (P0-03).
    Statuses are compared via the SolverStatus enum, never raw strings (P1-04).

    任何工作流错误优先，并短路到 FAILED（P0-03）。
    状态通过 SolverStatus 枚举比较，绝不使用原始字符串（P1-04）。
    """
    from cooking_plan_agent.domain.enums import SolverStatus

    if state.get("error") is not None:
        return "render_failed_response"

    result = state.get("schedule_result")
    if result is None:
        return "render_failed_response"

    status = result.status
    if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        return "verify_schedule"
    if status == SolverStatus.INFEASIBLE:
        return "render_infeasible_response"
    # MODEL_INVALID, UNKNOWN -> FAILED
    # MODEL_INVALID、UNKNOWN -> FAILED
    return "render_failed_response"


# ---------------------------------------------------------------------------
# 8.6 Routing after verification
# 8.6 验证之后的路由
# ---------------------------------------------------------------------------


def route_after_verification(
    state: PlanState,
) -> Literal["explain_schedule", "repair_schedule", "render_failed_response"]:
    """Verification passes -> explain; fails -> repair loop; else FAILED. (P5-3)

    验证通过 -> 解释；失败 -> 修复循环；否则 FAILED。（P5-3）

    The verifier checks constraint satisfaction independently from the solver,
    catching optimiser bugs before they reach the user. The explain node is
    additive and never blocks READY.

    验证器独立于求解器检查约束满足情况，在优化器缺陷触达用户前捕获它们。
    解释节点是加法能力，绝不阻塞 READY。

    Any workflow error takes precedence and short-circuits to FAILED (P0-03).

    任何工作流错误优先，并短路到 FAILED（P0-03）。
    """
    if state.get("error") is not None:
        return "render_failed_response"

    report = state.get("verification_report")
    if report is not None and report.passed:
        return "explain_schedule"
    # P5-3: 验证失败先进入反思修复循环；gave_up 由 route_after_repair 落 FAILED。
    return "repair_schedule"


def route_after_repair(
    state: PlanState,
) -> Literal["solve_schedule", "render_failed_response"]:
    """Repair 后：retrying -> 重新求解；gave_up -> FAILED。（P5-3）"""
    if state.get("error") is not None:
        return "render_failed_response"
    history = state.get("repair_history", ())
    if history and history[-1].outcome == "gave_up":
        return "render_failed_response"
    return "solve_schedule"


# ---------------------------------------------------------------------------
# 8.6 P5-2 Routing after the ReAct controller
# 8.6 P5-2 ReAct 控制器之后的路由
# ---------------------------------------------------------------------------


def route_after_controller(
    state: PlanState,
) -> Literal["run_tool", "validate_input", "render_failed_response"]:
    """P5-2: 控制器决策路由。

    - 模式回退 / 控制器输出 final / 步数耗尽 → 确定性 DAG（validate_input）
    - 有待执行工具 → run_tool
    - error 短路 → FAILED
    """
    if state.get("error") is not None:
        return "render_failed_response"
    if state.get("agent_mode") == "deterministic":
        return "validate_input"
    if state.get("pending_decision", {}).get("type") == "tool_call":
        return "run_tool"
    # final / 无待执行决策 → 确定性 DAG（渐进式降级：LLM 只做软决策）。
    return "validate_input"


# ---------------------------------------------------------------------------
# 8.6 P5-4 Routing after confirmation answers applied
# 8.6 P5-4 确认答复应用之后的路由
# ---------------------------------------------------------------------------


def route_after_confirmation(
    state: PlanState,
) -> Literal[
    "parse_recipes",
    "solve_schedule",
    "build_confirmation_response",
    "render_ready_response",
    "render_failed_response",
]:
    """P5-4: 根据确认答复的效果路由。

    - 答复应用后仍有 critical gap / 新确认问题 → 再次确认
    - 答复改变了 request 内容 → 从 parse_recipes 重新推进
    - 答复仅调整约束（extend_time / reduce_servings）→ 直接 solve_schedule
    - 全部满足 → READY
    - error 短路 → FAILED（P0-03）
    """
    if state.get("error") is not None:
        return "render_failed_response"
    if state.get("needs_confirmation"):
        return "build_confirmation_response"
    route = state.get("confirmation_route")
    if route == "parse_recipes":
        return "parse_recipes"
    if route == "solve_schedule":
        return "solve_schedule"
    return "render_ready_response"
