"""Pure execution-state transitions for the cooking-plan task DAG.

The schedule is only an estimate.  Runtime progression is derived from task
dependencies and currently occupied resources, so a passive blanching task
does not prevent the cook from preparing shrimp at the same time.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

PENDING = "PENDING"
IN_PROGRESS = "IN_PROGRESS"
COMPLETED = "COMPLETED"
_VALID_STATES = frozenset((PENDING, IN_PROGRESS, COMPLETED))


class ExecutionStateError(ValueError):
    """A requested runtime transition is invalid for the current plan state."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _flow_map(flow: object) -> dict[str, dict[str, Any]]:
    if not isinstance(flow, (list, tuple)):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in flow:
        if isinstance(item, dict) and isinstance(item.get("task_id"), str):
            result[item["task_id"]] = item
    return result


def _dependencies(task: dict[str, Any]) -> tuple[str, ...]:
    raw = task.get("depends_on", ())
    return tuple(item for item in raw if isinstance(item, str)) if isinstance(raw, (list, tuple)) else ()


def _resource_needs(task: dict[str, Any]) -> Counter[str]:
    raw = task.get("resource_needs")
    if isinstance(raw, (list, tuple)):
        result: Counter[str] = Counter()
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("resource_type"), str):
                result[item["resource_type"]] += int(item.get("quantity", 1))
        return result
    # Backward compatibility with READY responses created before
    # ``resource_needs`` was added.
    resources = task.get("resources", ())
    return (
        Counter(item for item in resources if isinstance(item, str))
        if isinstance(resources, (list, tuple))
        else Counter()
    )


def _resource_capacities(raw_resources: object) -> Counter[str]:
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
    """Return available, in-progress, completed and blocked task views."""
    task_map = _flow_map(flow)
    states = states or {}
    normalised = {task_id: states.get(task_id, PENDING) for task_id in task_map}
    in_progress = {task_id for task_id, status in normalised.items() if status == IN_PROGRESS}
    completed = {task_id for task_id, status in normalised.items() if status == COMPLETED}
    active_in_progress = any(task_map[task_id].get("work_mode") == "ACTIVE" for task_id in in_progress)
    capacities = _resource_capacities(kitchen_resources)
    used: Counter[str] = Counter()
    for task_id in in_progress:
        used.update(_resource_needs(task_map[task_id]))

    available: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for task_id, task in task_map.items():
        if normalised[task_id] != PENDING:
            continue
        unmet = tuple(dep for dep in _dependencies(task) if dep not in completed)
        if unmet:
            blocked.append({"task_id": task_id, "blocked_by": unmet})
            continue
        needs = _resource_needs(task)
        resource_blocked = tuple(
            resource
            for resource, quantity in needs.items()
            if capacities[resource] and used[resource] + quantity > capacities[resource]
        )
        if resource_blocked:
            blocked.append({"task_id": task_id, "blocked_by_resources": resource_blocked})
            continue
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
    """Validate and apply one task transition, returning the new snapshot."""
    if target_status not in (IN_PROGRESS, COMPLETED):
        raise ExecutionStateError("INVALID_EXECUTION_STATUS", "Only IN_PROGRESS or COMPLETED is allowed.")
    task_map = _flow_map(flow)
    if task_id not in task_map:
        raise ExecutionStateError("UNKNOWN_COOKING_TASK", f"Cooking task '{task_id}' is not in this plan.")
    next_states = {key: value for key, value in (states or {}).items() if key in task_map and value in _VALID_STATES}
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
    if current == PENDING and task_id not in available_ids:
        raise ExecutionStateError(
            "TASK_NOT_READY", f"Cooking task '{task_id}' is blocked by a dependency, cook, or resource."
        )
    if target_status == COMPLETED and current == PENDING and task_id not in available_ids:
        raise ExecutionStateError("TASK_NOT_READY", f"Cooking task '{task_id}' cannot be completed before it is ready.")
    next_states[task_id] = target_status
    return next_states, build_execution_snapshot(flow, next_states, kitchen_resources)
