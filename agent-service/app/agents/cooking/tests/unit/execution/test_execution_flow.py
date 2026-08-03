"""Execution-flow runtime tests: dependencies, passive work and resources."""

import pytest

from cooking_plan_agent.execution import ExecutionStateError, build_execution_snapshot, transition_execution_state


FLOW = (
    {
        "task_id": "crab_prep",
        "instruction": "处理蟹脚",
        "depends_on": (),
        "work_mode": "ACTIVE",
        "resources": (),
    },
    {
        "task_id": "crab_heat",
        "instruction": "蟹脚焯水",
        "depends_on": ("crab_prep",),
        "work_mode": "PASSIVE",
        "resource_needs": ({"resource_type": "stove", "quantity": 1},),
    },
    {
        "task_id": "shrimp_prep",
        "instruction": "处理鲜虾",
        "depends_on": (),
        "work_mode": "ACTIVE",
        "resources": (),
    },
)

RESOURCES = ({"resource_type": "stove", "capacity": 1, "available": True},)


def _available(snapshot: dict[str, object]) -> set[str]:
    return {item["task_id"] for item in snapshot["available_tasks"]}  # type: ignore[index]


def test_crab_blanche_unlocks_after_prep_and_shrimp_remains_parallel():
    states, snapshot = transition_execution_state(FLOW, {}, RESOURCES, "crab_prep", "COMPLETED")
    assert {"crab_heat", "shrimp_prep"} <= _available(snapshot)

    states, snapshot = transition_execution_state(FLOW, states, RESOURCES, "crab_heat", "IN_PROGRESS")
    assert "shrimp_prep" in _available(snapshot)
    assert snapshot["in_progress_task_ids"] == ("crab_heat",)


def test_active_cook_cannot_start_two_hands_on_tasks_at_once():
    states, _ = transition_execution_state(FLOW, {}, RESOURCES, "crab_prep", "IN_PROGRESS")
    with pytest.raises(ExecutionStateError, match="blocked"):
        transition_execution_state(FLOW, states, RESOURCES, "shrimp_prep", "IN_PROGRESS")


def test_dependency_cannot_be_completed_early():
    with pytest.raises(ExecutionStateError, match="blocked"):
        transition_execution_state(FLOW, {}, RESOURCES, "crab_heat", "COMPLETED")


def test_snapshot_is_empty_only_after_every_task_is_complete():
    snapshot = build_execution_snapshot(
        FLOW,
        {"crab_prep": "COMPLETED", "crab_heat": "COMPLETED", "shrimp_prep": "COMPLETED"},
        RESOURCES,
    )
    assert snapshot["is_complete"] is True
    assert snapshot["available_tasks"] == ()
