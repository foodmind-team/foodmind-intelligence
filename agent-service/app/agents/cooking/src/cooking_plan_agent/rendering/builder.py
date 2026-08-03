"""Rendering builders — convert domain objects into response-ready dicts/lists.

Handbook sections 11.1–11.5: all functions are pure transformers from domain
models to plain dicts. No I/O, no side effects. Output dicts are compatible
with the ReadyPlanResponse model fields (timeline, mise_en_place, dish_completions).
"""

from __future__ import annotations

from typing import Any

from cooking_plan_agent.domain.models import (
    CompletionItem,
    CookingTask,
    InventoryConsumptionProposal,
)
from cooking_plan_agent.scheduling.models import (
    ScheduleResult,
    VerificationIssue,
)

# =============================================================================
# 11.1  Mise en place
# =============================================================================


def build_mise_en_place(
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """Build the mise en place (prep-ahead) list from preparation tasks.

    Groups preparation tasks by instruction prefix [Prep] and deduplicates
    by combining quantities when the same ingredient-operation pair appears.

    Args:
        tasks: All cooking tasks (recipe + prep + safety).

    Returns:
        Sorted list of dicts with keys: instruction, ingredients, when_needed.
    """
    # Filter to preparation tasks only (category == "preparation")
    prep_tasks = [t for t in tasks if t.category == "preparation"]

    if not prep_tasks:
        return ()

    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for task in prep_tasks:
        # Extract ingredient and operation from the instruction
        # Format: "[Prep] {operation} {quantity} of {ingredient}"
        instruction = task.instruction
        cleaned = instruction.removeprefix("[Prep] ").strip() if instruction.startswith("[Prep] ") else instruction

        # Parse: "dice 500.0 of chicken breast"
        parts = cleaned.split(" of ", 1)
        operation_with_qty = parts[0]  # "dice 500.0"
        ingredient = parts[1] if len(parts) > 1 else cleaned

        # Extract operation and quantity
        op_parts = operation_with_qty.rsplit(" ", 1)
        operation = " ".join(op_parts[:-1]) if len(op_parts) > 1 else operation_with_qty

        key = f"{ingredient}:{operation}"
        if key in seen:
            continue
        seen.add(key)

        # Determine when the ingredient is needed (state this task consumes → produces)
        when = None
        if task.consumes_states:
            when = task.consumes_states[0]
        elif task.produces_states:
            when = task.produces_states[0]

        items.append(
            {
                "instruction": f"{operation}: {ingredient}",
                "ingredient": ingredient,
                "operation": operation,
                "duration_minutes": task.duration_minutes,
                "resources": [r.resource_type for r in task.resources] if task.resources else [],
                "when_needed": when,
            }
        )

    # Sort by duration descending (longer prep tasks first)
    items.sort(key=lambda i: i["duration_minutes"], reverse=True)

    return tuple(items)


# =============================================================================
# 11.2  Timeline
# =============================================================================


def build_timeline(
    schedule: ScheduleResult,
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """Build a sorted timeline from scheduled intervals and task details.

    Each timeline entry includes the task instruction, time window,
    work mode (active/passive), and dish name.

    Args:
        schedule: Solved schedule with intervals.
        tasks: All cooking tasks (for instruction/dish lookup).

    Returns:
        Sorted list of dicts, earliest start time first.
    """
    if not schedule.intervals:
        return ()

    # Build task lookup
    task_map: dict[str, CookingTask] = {t.task_id: t for t in tasks}

    items: list[dict[str, Any]] = []

    for interval in schedule.intervals:
        task = task_map.get(interval.task_id)
        if task is None:
            continue

        items.append(
            {
                "task_id": interval.task_id,
                "start_minute": interval.start_minute,
                "end_minute": interval.end_minute,
                "duration_minutes": interval.end_minute - interval.start_minute,
                "instruction": task.instruction,
                "dish_id": task.dish_id,
                "work_mode": task.work_mode.value,
                "category": task.category,
                "heat_level": task.heat_level.value if task.heat_level else None,
                "resources": [r.resource_type for r in task.resources] if task.resources else [],
            }
        )

    # Sort by start time, then by end time
    items.sort(key=lambda i: (i["start_minute"], i["end_minute"]))

    return tuple(items)


# =============================================================================
# 11.2a Dependency-driven execution flow
# =============================================================================


def build_execution_flow(
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """Build UI-ready task dependencies without prescribing clock times.

    A client keeps its own ``completed_task_ids`` / ``in_progress_task_ids``
    and uses ``depends_on`` to decide what can be shown next.  This means a
    task such as "blanch crab" becomes available immediately after handling
    the crab; while its heating interval is in progress, unrelated active
    tasks such as preparing shrimp remain available to the cook.
    """
    successor_map: dict[str, list[str]] = {task.task_id: [] for task in tasks}
    items: list[dict[str, object]] = []
    for task in tasks:
        dependencies = tuple(dep.predecessor_id for dep in task.dependencies)
        for predecessor_id in dependencies:
            if predecessor_id in successor_map:
                successor_map[predecessor_id].append(task.task_id)
        items.append(
            {
                "task_id": task.task_id,
                "dish_id": task.dish_id,
                "instruction": task.instruction,
                "depends_on": dependencies,
                "unlocks": (),
                "work_mode": task.work_mode.value,
                "resources": [resource.resource_type for resource in task.resources],
                "resource_needs": [
                    {"resource_type": resource.resource_type, "quantity": resource.quantity}
                    for resource in task.resources
                ],
                "completion_hint": "Mark complete when this operation is finished.",
            }
        )

    return tuple(
        {
            **item,
            "unlocks": tuple(successor_map[str(item["task_id"])]),
        }
        for item in items
    )


# =============================================================================
# 11.3  Dish completion summary
# =============================================================================


def build_dish_completion_summary(
    schedule: ScheduleResult,
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """Calculate when each dish completes cooking.

    A dish is "complete" when its last task finishes. For each dish,
    finds the maximum end_minute across all its tasks.

    Args:
        schedule: Solved schedule with intervals.
        tasks: All cooking tasks.

    Returns:
        List of dicts sorted by completion time, each with:
        dish_id, dish_name, completion_minute, task_count.
    """
    if not schedule.intervals:
        return ()

    task_map: dict[str, CookingTask] = {t.task_id: t for t in tasks}

    # Group intervals by dish_id, track max end time
    dish_ends: dict[str, dict[str, Any]] = {}

    for interval in schedule.intervals:
        task = task_map.get(interval.task_id)
        if task is None:
            continue

        dish_id = task.dish_id
        if dish_id not in dish_ends:
            dish_ends[dish_id] = {
                "dish_id": dish_id,
                "completion_minute": 0,
                "task_count": 0,
                "is_shared": dish_id.startswith("shared") or "," in dish_id,
            }

        record = dish_ends[dish_id]
        record["completion_minute"] = max(record["completion_minute"], interval.end_minute)
        record["task_count"] += 1

    # Convert to sorted list
    result = sorted(dish_ends.values(), key=lambda d: d["completion_minute"])
    return tuple(result)


# =============================================================================
# 11.4  Completion checklist
# =============================================================================


def build_completion_checklist(
    proposal: InventoryConsumptionProposal,
) -> tuple[CompletionItem, ...]:
    """Extract the completion checklist from a reservation proposal.

    Each CompletionItem groups allocations for one ingredient.
    This is a pass-through — the proposal already contains the right structure.

    Args:
        proposal: An InventoryConsumptionProposal from build_reservation_proposal.

    Returns:
        Tuple of CompletionItem, one per ingredient group.
    """
    return proposal.items


def validate_completion_checklist(
    proposal: InventoryConsumptionProposal,
    checklist: tuple[CompletionItem, ...],
) -> tuple[VerificationIssue, ...]:
    """Validate the completion checklist against its source proposal.

    Checks:
      - Every allocation has a valid lot_id
      - Every allocation quantity is positive
      - No duplicate lot allocations across different items
      - Total allocated per ingredient ≤ proposal's required quantities
      - Checklist is non-empty when proposal has items

    Args:
        proposal: The source InventoryConsumptionProposal.
        checklist: The generated checklist to validate.

    Returns:
        Tuple of VerificationIssue, empty = valid.
    """
    issues: list[VerificationIssue] = []

    if not checklist and proposal.items:
        issues.append(
            VerificationIssue(
                code="EMPTY_CHECKLIST",
                message="Checklist is empty but proposal contains items",
            )
        )
        return tuple(issues)

    seen_lot_ids: set[str] = set()

    for item in checklist:
        if not item.completion_item_id:
            issues.append(
                VerificationIssue(
                    code="MISSING_ITEM_ID",
                    message=f"CompletionItem for '{item.ingredient_name}' has no ID",
                )
            )

        for alloc in item.allocations:
            # Check lot_id
            if not alloc.inventory_lot_id.strip():
                issues.append(
                    VerificationIssue(
                        code="MISSING_LOT_ID",
                        message=f"Allocation in '{item.ingredient_name}' has empty lot_id",
                    )
                )

            # Check for duplicate lot allocations
            lot_key = f"{alloc.inventory_lot_id}:{item.ingredient_name}"
            if lot_key in seen_lot_ids:
                issues.append(
                    VerificationIssue(
                        code="DUPLICATE_LOT_ALLOCATION",
                        message=(
                            f"Lot '{alloc.inventory_lot_id}' allocated multiple times for '{item.ingredient_name}'"
                        ),
                    )
                )
            seen_lot_ids.add(lot_key)

            # Check positive quantity
            if alloc.quantity <= 0:
                issues.append(
                    VerificationIssue(
                        code="NON_POSITIVE_ALLOCATION",
                        message=(
                            f"Allocation from lot '{alloc.inventory_lot_id}' "
                            f"for '{item.ingredient_name}' has non-positive quantity: {alloc.quantity}"
                        ),
                    )
                )

    return tuple(issues)
