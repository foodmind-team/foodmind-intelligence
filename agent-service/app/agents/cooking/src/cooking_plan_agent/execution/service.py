# =============================================================================
# 执行状态机模块（execution/service）
# -----------------------------------------------------------------------------
# 本文件实现烹饪计划任务 DAG 的“纯执行状态迁移”逻辑，核心职责：
#   - build_execution_snapshot    ：基于依赖与资源占用，计算“可执行/进行中/已完成/被阻塞”四类任务视图
#   - transition_execution_state  ：校验并应用单个任务的状态迁移（PENDING → IN_PROGRESS → COMPLETED）
# 设计要点：调度表只是“估算”，运行时的推进由“任务依赖 + 当前占用资源”动态推导，
# 因此一个被动等待的焯水任务，不会阻塞厨师同时准备虾。
# =============================================================================

"""Pure execution-state transitions for the cooking-plan task DAG.

烹饪计划任务 DAG 的纯执行状态迁移。

The schedule is only an estimate.  Runtime progression is derived from task
dependencies and currently occupied resources, so a passive blanching task
does not prevent the cook from preparing shrimp at the same time.

调度表只是估算。运行时的推进由任务依赖与当前占用资源推导得出，
因此一个被动焯水任务不会阻止厨师同时准备虾。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PENDING = "PENDING"
IN_PROGRESS = "IN_PROGRESS"
COMPLETED = "COMPLETED"
_VALID_STATES = frozenset((PENDING, IN_PROGRESS, COMPLETED))
# ↑ 合法状态集合（不可变 frozenset），用于过滤非法状态值


class ExecutionStateError(ValueError):
    """执行状态错误：请求的状态迁移对当前计划状态非法。

    A requested runtime transition is invalid for the current plan state.

    相比普通 ValueError，额外携带稳定机器可读的错误码 code，便于上层按码处理。
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _flow_map(flow: object) -> dict[str, dict[str, Any]]:
    """把执行流（flow）归一化为 {task_id: task_dict} 映射。"""
    if not isinstance(flow, (list, tuple)):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in flow:
        if isinstance(item, dict) and isinstance(item.get("task_id"), str):
            result[item["task_id"]] = item
    return result


def _dependencies(task: dict[str, Any]) -> tuple[str, ...]:
    """提取任务的前驱依赖（depends_on 字段）。"""
    raw = task.get("depends_on", ())
    return tuple(item for item in raw if isinstance(item, str)) if isinstance(raw, (list, tuple)) else ()


def _resource_needs(task: dict[str, Any]) -> Counter[str]:
    """统计任务对各类型资源的需求数量（Counter）。"""
    raw = task.get("resource_needs")
    if isinstance(raw, (list, tuple)):
        result: Counter[str] = Counter()
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("resource_type"), str):
                result[item["resource_type"]] += int(item.get("quantity", 1))
        return result
    # Backward compatibility with READY responses created before
    # ``resource_needs`` was added.
    # 向后兼容：兼容在引入 ``resource_needs`` 字段之前生成的 READY 响应。
    resources = task.get("resources", ())
    return (
        Counter(item for item in resources if isinstance(item, str))
        if isinstance(resources, (list, tuple))
        else Counter()
    )


def _resource_capacities(raw_resources: object) -> Counter[str]:
    """统计各类型资源的可用容量（忽略不可用资源）。"""
    capacities: Counter[str] = Counter()
    if not isinstance(raw_resources, (list, tuple)):
        return capacities
    for resource in raw_resources:
        if not isinstance(resource, dict) or resource.get("available") is False:
            continue
        resource_type = resource.get("resource_type")
        if not isinstance(resource_type, str):
            continue
        capacity = resource.get("capacity") or 1
        try:
            capacities[resource_type] += int(capacity)
        except (TypeError, ValueError):
            capacities[resource_type] += 1
    return capacities


def build_execution_snapshot(
    flow: object,
    states: dict[str, str] | None,
    kitchen_resources: object = (),
) -> dict[str, object]:
    """构建执行快照：返回可执行 / 进行中 / 已完成 / 被阻塞四类任务视图。

    Return available, in-progress, completed and blocked task views.
    """
    task_map = _flow_map(flow)
    states = states or {}
    normalised = {task_id: states.get(task_id, PENDING) for task_id in task_map}
    # ↑ 规范化状态：未知任务默认 PENDING
    in_progress = {task_id for task_id, status in normalised.items() if status == IN_PROGRESS}
    completed = {task_id for task_id, status in normalised.items() if status == COMPLETED}
    active_in_progress = any(task_map[task_id].get("work_mode") == "ACTIVE" for task_id in in_progress)
    # ↑ 是否存在正在进行的“主动（需动手）”任务
    capacities = _resource_capacities(kitchen_resources)
    used: Counter[str] = Counter()
    for task_id in in_progress:
        used.update(_resource_needs(task_map[task_id]))
    # ↑ 累加所有进行中任务占用的资源量

    available: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for task_id, task in task_map.items():
        if normalised[task_id] != PENDING:
            continue
        # --- 1) 依赖检查：前驱未全部完成则阻塞 ---
        unmet = tuple(dep for dep in _dependencies(task) if dep not in completed)
        if unmet:
            blocked.append({"task_id": task_id, "blocked_by": unmet})
            continue
        # --- 2) 资源检查：所需资源超出剩余容量则阻塞 ---
        needs = _resource_needs(task)
        resource_blocked = tuple(
            resource
            for resource, quantity in needs.items()
            if capacities[resource] and used[resource] + quantity > capacities[resource]
        )
        if resource_blocked:
            blocked.append({"task_id": task_id, "blocked_by_resources": resource_blocked})
            continue
        # --- 3) 厨师（cook）检查：主动任务不能与进行中的主动任务并行 ---
        if task.get("work_mode") == "ACTIVE" and active_in_progress:
            blocked.append({"task_id": task_id, "blocked_by": ("active_cook",)})
            continue
        available.append(task)

    return {
        "available_tasks": tuple(available),
        "in_progress_task_ids": tuple(sorted(in_progress)),
        "completed_task_ids": tuple(sorted(completed)),
        "blocked_tasks": tuple(blocked),
        "is_complete": bool(task_map) and len(completed) == len(task_map),
    }


def transition_execution_state(
    flow: object,
    states: dict[str, str] | None,
    kitchen_resources: object,
    task_id: str,
    target_status: str,
) -> tuple[dict[str, str], dict[str, object]]:
    """校验并应用单个任务的状态迁移，返回新的 (状态字典, 执行快照)。

    Validate and apply one task transition, returning the new snapshot.
    """
    if target_status not in (IN_PROGRESS, COMPLETED):
        raise ExecutionStateError("INVALID_EXECUTION_STATUS", "Only IN_PROGRESS or COMPLETED is allowed.")
    task_map = _flow_map(flow)
    if task_id not in task_map:
        raise ExecutionStateError("UNKNOWN_COOKING_TASK", f"Cooking task '{task_id}' is not in this plan.")
    next_states = {key: value for key, value in (states or {}).items() if key in task_map and value in _VALID_STATES}
    # ↑ 只保留属于本计划且状态合法的条目，过滤脏数据
    current = next_states.get(task_id, PENDING)
    if current == COMPLETED:
        raise ExecutionStateError("TASK_ALREADY_COMPLETED", f"Cooking task '{task_id}' is already complete.")
    snapshot = build_execution_snapshot(flow, next_states, kitchen_resources)
    raw_available = snapshot.get("available_tasks", ())
    available_ids = (
        {item["task_id"] for item in raw_available if isinstance(item, dict) and isinstance(item.get("task_id"), str)}
        if isinstance(raw_available, (list, tuple))
        else set()
    )
    # ↑ 从快照中提取当前“可执行”任务的 ID 集合
    if current == PENDING and task_id not in available_ids:
        raise ExecutionStateError(
            "TASK_NOT_READY", f"Cooking task '{task_id}' is blocked by a dependency, cook, or resource."
        )
    if target_status == COMPLETED and current == PENDING and task_id not in available_ids:
        raise ExecutionStateError("TASK_NOT_READY", f"Cooking task '{task_id}' cannot be completed before it is ready.")
    next_states[task_id] = target_status
    return next_states, build_execution_snapshot(flow, next_states, kitchen_resources)
