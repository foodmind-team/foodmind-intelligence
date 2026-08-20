# =============================================================================
# 排程与解释节点（workflow/scheduling_nodes）
# -----------------------------------------------------------------------------
# 单个流水线阶段的节点实现：构建任务 DAG、CP-SAT 求解、独立验证，
# 以及 P4-01 的加法式排程解释。公共兼容面仍为 cooking_plan_agent.workflow.nodes。
# =============================================================================

"""Workflow node implementations for a single pipeline stage.

单个流水线阶段的节点实现。

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.

公共兼容面仍为 ``cooking_plan_agent.workflow.nodes``。本模块仅包含一个内聚阶段。
"""

import logging

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState

logger = logging.getLogger(__name__)


def _solver_timeout() -> float:
    """Return the configured CP-SAT solver timeout in seconds.

    返回配置的 CP-SAT 求解器超时（秒）。
    """
    from cooking_plan_agent.config.settings import get_settings

    return get_settings().solver_timeout_seconds


async def build_task_graph_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Build the task DAG from recipe, prep, and safety tasks.

    从菜谱、预处理与安全任务构建任务 DAG。

    Wired to existing preparation/task_graph.py.

    接线到现有 preparation/task_graph.py。

    Lazy-imports build_task_graph inside the function so the module
    can be imported even when preparation dependencies are missing.

    在函数内部懒加载 build_task_graph，使本模块即使在 preparation
    依赖缺失时也可被导入。
    """
    from cooking_plan_agent.preparation.task_graph import build_task_graph

    recipe_tasks = state.get("recipe_tasks", ())
    prep_tasks = state.get("prep_tasks", ())
    safety_tasks = state.get("safety_tasks", ())

    # Defensive: if merge_preparation returned nothing, skip building
    # 防御性处理：若 merge_preparation 未返回任何内容，则跳过构建
    if not recipe_tasks and not prep_tasks:
        return {}

    try:
        graph = build_task_graph(
            recipe_tasks=recipe_tasks,
            prep_tasks=prep_tasks,
            safety_tasks=safety_tasks,
        )
        return {"task_graph": graph}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Cycle detection or invalid dependencies -> workflow error. P2-03:
        # only the exception type is retained as diagnostic context.
        # 环检测或非法依赖 -> 工作流错误。P2-03：仅保留异常类型作为诊断上下文。
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.TASK_GRAPH_CYCLE.value,
                message="Task graph construction failed",
                correlation_id=state["request"].request_id,
                node_name="build_task_graph",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


async def solve_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Solve the CP-SAT scheduling problem.

    求解 CP-SAT 排程问题。

    Wired to existing scheduling/orchestrator.py.
    schedule() returns tuple[ScheduleResult, VerificationReport] — we store
    only the result; verification is done independently in verify_schedule_node.

    接线到现有 scheduling/orchestrator.py。schedule() 返回
    tuple[ScheduleResult, VerificationReport] —— 我们只存储结果；
    验证由 verify_schedule_node 独立完成。

    Error semantics (P1-04): ``SCHEDULE_INFEASIBLE`` means ONLY that the
    solver proved no solution exists for a VALID model. Everything else uses
    a distinct code:
      - MODEL_INVALID → SCHEDULE_MODEL_INVALID (model construction bug)
      - UNKNOWN       → SCHEDULE_UNKNOWN (solver hit its limit, undetermined)
      - missing task graph → INTERNAL_ERROR (invariant break, never INFEASIBLE)
      - ValueError/TypeError during solve → SCHEDULE_MODEL_INVALID
      - RuntimeError from the solver → INTERNAL_ERROR

    错误语义（P1-04）：``SCHEDULE_INFEASIBLE`` 仅表示求解器证明了一个
    VALID 模型无解。其他情况使用不同错误码：
      - MODEL_INVALID → SCHEDULE_MODEL_INVALID（模型构建缺陷）
      - UNKNOWN       → SCHEDULE_UNKNOWN（求解器达到上限，未定）
      - 缺少任务图   → INTERNAL_ERROR（不变量破坏，绝不是 INFEASIBLE）
      - 求解期间的 ValueError/TypeError → SCHEDULE_MODEL_INVALID
      - 求解器抛出的 RuntimeError → INTERNAL_ERROR
    """
    import asyncio

    from cooking_plan_agent.domain.enums import SolverStatus
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.orchestrator import ScheduleOrchestrator

    task_graph = state.get("task_graph")
    request = state["request"]
    if task_graph is None:
        # Missing DAG is an internal invariant failure, not a business
        # infeasibility — solve must never run without a task graph.
        # 缺少 DAG 是内部不变量失效，而非业务不可行 —— 没有任务图时绝不运行求解。
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="No task graph available for scheduling — internal invariant violated",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=request.kitchen_resources,
        requested_time_limit_minutes=request.time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        # CP-SAT solving is CPU-bound — run it in a worker thread so the
        # event loop stays responsive (P1-02). The verifier is synchronous
        # and stays inside the solve call; it is not moved to a thread.
        # P3-03: ScheduleOrchestrator runs the lexicographic phases
        # (makespan → holding → context switch); Phase 4 stays gated.
        # The depth is configurable for rollback (solver_optimization_level).
        # CP-SAT 求解是 CPU 密集型 —— 在工作线程中运行，使事件循环保持响应（P1-02）。
        # 验证器是同步的，留在 solve 调用内部；不将其移到线程。
        # P3-03：ScheduleOrchestrator 运行字典序阶段（makespan → holding →
        # context switch）；Phase 4 保持门控。深度可配置以便回退（solver_optimization_level）。
        from cooking_plan_agent.config.settings import get_settings

        overrides = state.get("solver_overrides", {})
        optimization_level = str(overrides.get("optimization_level") or get_settings().solver_optimization_level)

        orchestrator = ScheduleOrchestrator()
        result, _ = await asyncio.to_thread(
            orchestrator.solve,
            problem,
            optimization_level,
        )
    except (ValueError, TypeError) as exc:
        # Model-construction phase: bad variable shapes, contradictory
        # constraints → the model was never valid. P2-03: keep only the
        # exception type as diagnostic context.
        # 模型构建阶段：变量形状错误、约束矛盾 → 模型从未有效。P2-03：
        # 仅保留异常类型作为诊断上下文。
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="Scheduling model construction failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }
    except RuntimeError as exc:
        # Solver-internal failure (runtime) — not a business outcome.
        # 求解器内部故障（运行时）—— 不是业务结果。
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.INTERNAL_ERROR.value,
                message="Scheduling solver failed",
                correlation_id=request.request_id,
                node_name="solve_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }

    # Map solver status to a stable, distinct error code. Only INFEASIBLE is a
    # business outcome (routes to render_infeasible_response); MODEL_INVALID
    # and UNKNOWN are FAILED responses (P1-04).
    # 将求解器状态映射为稳定、可区分的错误码。仅 INFEASIBLE 是业务结果
    # （路由到 render_infeasible_response）；MODEL_INVALID 与 UNKNOWN 是
    # FAILED 响应（P1-04）。
    status = result.status
    if status == SolverStatus.MODEL_INVALID:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_MODEL_INVALID.value,
                message="The scheduling model is invalid — likely a data inconsistency",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }
    if status == SolverStatus.UNKNOWN:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_UNKNOWN.value,
                message="The solver could not determine feasibility within the time limit",
                correlation_id=request.request_id,
                node_name="solve_schedule",
            )
        }

    return {"schedule_result": result}


async def verify_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Independent verification of solver output.

    对求解器输出做独立验证。

    Wired to existing scheduling/verifier.py.
    verify() signature: verify(problem: SchedulingProblem, result: ScheduleResult)

    接线到现有 scheduling/verifier.py。
    verify() 签名：verify(problem: SchedulingProblem, result: ScheduleResult)

    Verification is done in a SEPARATE node (not inside solve_schedule) so that:
    - verification can be skipped/instrumented independently
    - the verifier catches bugs in the solver itself

    验证在独立节点中完成（而非 solve_schedule 内部），以便：
    - 可独立跳过 / 插桩验证
    - 验证器能捕获求解器本身的缺陷
    """
    from cooking_plan_agent.scheduling.models import SchedulingProblem
    from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

    schedule_result = state.get("schedule_result")
    if schedule_result is None:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No schedule result to verify",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    task_graph = state.get("task_graph")
    if task_graph is None:
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="No task graph available for verification",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
            )
        }

    problem = SchedulingProblem(
        tasks=task_graph.tasks,
        resources=state["request"].kitchen_resources,
        requested_time_limit_minutes=state["request"].time_limit_minutes,
        solver_timeout_seconds=_solver_timeout(),
    )

    try:
        verifier = ScheduleVerifier()
        report = verifier.verify(problem, schedule_result)
        return {"verification_report": report}
    except (ValueError, TypeError, RuntimeError) as exc:
        # Verification failure is an invariant break — the solver output
        # must never reach the client. P2-03: public text comes from the
        # catalog; keep only the exception type as diagnostic context.
        # 验证失败是不变量破坏 —— 求解器输出绝不能触达客户端。P2-03：
        # 公共文本来自目录；仅保留异常类型作为诊断上下文。
        return {
            "error": WorkflowError(
                error_code=DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value,
                message="Schedule verification failed",
                correlation_id=state["request"].request_id,
                node_name="verify_schedule",
                diagnostics={"exception_type": type(exc).__name__},
            )
        }


# ============================================================================
# P4-01: schedule explanation (between verify and READY render)
# P4-01：排程解释（位于 verify 与 READY 渲染之间）
# ============================================================================


def _build_schedule_summary(state: PlanState) -> dict[str, object]:
    """Build the compact, non-sensitive summary the explainer consumes (D3/D4).

    构建解释器消费的紧凑、非敏感摘要（D3/D4）。

    Only facts already present in the verified schedule are included:
    makespan minutes, per-dish completion minutes, and the maximum number of
    concurrently ACTIVE tasks (parallel groups). No recipe text, inventory,
    or user identity is ever included.

    仅包含已验证排程中已有的事实：总时长（分钟）、每道菜完成时间（分钟），
    以及最大并发 ACTIVE 任务数（并行组）。绝不包含菜谱文本、库存或用户身份。
    """
    from cooking_plan_agent.rendering.builder import build_dish_completion_summary

    schedule = state.get("schedule_result")
    makespan: int = (schedule.makespan_minutes or 0) if schedule is not None else 0

    dish_completions: list[dict[str, object]] = []
    if schedule is not None:
        task_graph = state.get("task_graph")
        tasks = task_graph.tasks if task_graph is not None else ()
        for entry in build_dish_completion_summary(schedule, tasks):
            # builder emits "dish_id"; the explainer consumes "dish".
            # builder 输出 "dish_id"；解释器消费 "dish"。
            raw_completion = entry.get("completion_minute")
            dish_completions.append(
                {
                    "dish": str(entry.get("dish_id") or "?"),
                    "completion_minute": int(raw_completion) if isinstance(raw_completion, int) else 0,
                }
            )

    return {
        "makespan_minutes": makespan,
        "dish_completions": dish_completions,
        "parallel_groups": _max_parallel_active(state),
    }


def _max_parallel_active(state: PlanState) -> int:
    """Maximum number of concurrently ACTIVE tasks across the timeline (D3).

    时间线上并发 ACTIVE 任务的最大数量（D3）。

    A simple sweep over (start, end) events of ACTIVE tasks gives the peak
    concurrency. Falls back to 0 when no schedule/timeline is available.

    对 ACTIVE 任务的（开始、结束）事件做简单扫描即可得到峰值并发。
    无排程 / 时间线时回退为 0。
    """
    from cooking_plan_agent.domain.enums import WorkMode
    from cooking_plan_agent.rendering.builder import build_timeline

    schedule = state.get("schedule_result")
    if schedule is None:
        return 0

    task_graph = state.get("task_graph")
    tasks = task_graph.tasks if task_graph is not None else ()
    events: list[tuple[int, int]] = []  # (minute, +1 start / -1 end)  （分钟，+1 开始 / -1 结束）
    for entry in build_timeline(schedule, tasks):
        if entry.get("work_mode") != WorkMode.ACTIVE.value:
            continue
        raw_start = entry.get("start_minute")
        raw_end = entry.get("end_minute")
        start = int(raw_start) if isinstance(raw_start, int) else 0
        end = int(raw_end) if isinstance(raw_end, int) else start
        events.append((start, 1))
        events.append((end, -1))
    # Half-open intervals [start, end): at a shared boundary the ending task
    # is already done before the starting task begins, so -1 sorts before +1.
    # 半开区间 [start, end)：在共享边界处，结束任务在开始任务之前已完成，
    # 因此 -1 排在 +1 之前。
    events.sort(key=lambda event: (event[0], event[1]))
    current = 0
    peak = 0
    for _minute, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _deterministic_explanation(summary: dict[str, object]) -> str:
    """Deterministic fallback: re-states only verified schedule facts (D3).

    确定性回退：仅重述已验证的排程事实（D3）。

    Used when the LLM explainer is absent or fails. The content is always
    derived from the summary — no new claims are introduced.

    当 LLM 解释器缺失或失败时使用。内容始终派生自摘要 —— 不引入新断言。
    """
    makespan = summary.get("makespan_minutes")
    parts = [f"Plan completes in approximately {makespan} minutes."]
    raw_completions = summary.get("dish_completions")
    completions = raw_completions if isinstance(raw_completions, list) else []
    if completions:
        parts.append(
            "Dishes finish at: "
            + ", ".join(
                f"{entry.get('dish', '?')} at {entry.get('completion_minute', '?')} min" for entry in completions
            )
        )
    return " ".join(parts)


async def explain_schedule_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Attach a short, additive explanation to a verified schedule (P4-01).

    为已验证的排程附加简短的加法式解释（P4-01）。

    Placed between verify_schedule and render_ready_response. The node NEVER
    writes a WorkflowError: an absent explainer, LLM timeout, malformed output
    or any exception degrades to a deterministic summary, so the verified
    READY response is never blocked (P2-02 fault matrix).

    位于 verify_schedule 与 render_ready_response 之间。本节点绝不写入
    WorkflowError：解释器缺失、LLM 超时、输出格式错误或任何异常都会降级为
    确定性摘要，因此已验证的 READY 响应绝不被阻塞（P2-02 故障矩阵）。

    Returns state fields:
      - explanation: prose or None (feature disabled).
      - explanation_source: "llm" | "deterministic" | "disabled".

    Returns（返回状态字段）：
      - explanation：散文或 None（功能禁用）。
      - explanation_source："llm" | "deterministic" | "disabled"。
    """
    from cooking_plan_agent.config.settings import get_settings

    if not get_settings().explanation_enabled:
        return {"explanation": None, "explanation_source": "disabled"}

    summary = _build_schedule_summary(state)
    explainer = runtime.context.explainer
    if explainer is not None:
        try:
            import asyncio

            text = await asyncio.wait_for(
                explainer.explain(summary),
                timeout=get_settings().research_timeout_seconds,
            )
            if isinstance(text, str) and text.strip():
                return {"explanation": text, "explanation_source": "llm"}
        except Exception:  # noqa: BLE001 — additive capability must never fail READY
            # 加法能力绝不能让 READY 失败
            logger.warning("Schedule explanation failed — using deterministic fallback")

    return {
        "explanation": _deterministic_explanation(summary),
        "explanation_source": "deterministic",
    }
