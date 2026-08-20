# =============================================================================
# 渲染构建器模块（rendering/builder）
# -----------------------------------------------------------------------------
# 把领域对象转换为“响应就绪”的 dict / list（手册 11.1–11.5）。
# 所有函数都是从领域模型到普通 dict 的纯转换器，无 I/O、无副作用。
# 输出 dict 与 ReadyPlanResponse 模型字段（timeline、mise_en_place、dish_completions）兼容。
# 核心：
#   - build_mise_en_place            ：预处理提前清单（mise en place）
#   - build_timeline                 ：按时间排序的时间线
#   - build_execution_flow           ：依赖驱动的执行流（不预设钟点）
#   - build_dish_completion_summary  ：每道菜的完成时间汇总
#   - build_completion_checklist     ：从预留方案提取完成清单
#   - validate_completion_checklist  ：校验完成清单
# =============================================================================

"""Rendering builders — convert domain objects into response-ready dicts/lists.

渲染构建器 —— 把领域对象转换为响应就绪的 dict / list。

Handbook sections 11.1–11.5: all functions are pure transformers from domain
models to plain dicts. No I/O, no side effects. Output dicts are compatible
with the ReadyPlanResponse model fields (timeline, mise_en_place, dish_completions).

手册 11.1–11.5：所有函数都是从领域模型到普通 dict 的纯转换器。无 I/O、无副作用。
输出 dict 与 ReadyPlanResponse 模型字段（timeline、mise_en_place、dish_completions）兼容。
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
# 11.1  预处理提前（mise en place）
# =============================================================================


def build_mise_en_place(
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """从预处理任务构建 mise en place（提前准备）清单。

    Build the mise en place (prep-ahead) list from preparation tasks.

    Groups preparation tasks by instruction prefix [Prep] and deduplicates
    by combining quantities when the same ingredient-operation pair appears.

    按指令前缀 [Prep] 分组预处理任务，并在相同食材-操作对出现时通过合并数量去重。

    Args:
        tasks: All cooking tasks (recipe + prep + safety).
            tasks：所有烹饪任务（recipe + prep + safety）。

    Returns:
        Sorted list of dicts with keys: instruction, ingredients, when_needed.
        带 instruction、ingredients、when_needed 键的排序 dict 列表。
    """
    # Filter to preparation tasks only (category == "preparation")
    # 仅过滤 category == "preparation" 的预处理任务
    prep_tasks = [t for t in tasks if t.category == "preparation"]

    if not prep_tasks:
        return ()

    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for task in prep_tasks:
        # Extract ingredient and operation from the instruction
        # Format: "[Prep] {operation} {quantity} of {ingredient}"
        # 从指令提取食材与操作。格式："[Prep] {operation} {quantity} of {ingredient}"
        instruction = task.instruction
        cleaned = instruction.removeprefix("[Prep] ").strip() if instruction.startswith("[Prep] ") else instruction

        # Parse: "dice 500.0 of chicken breast"
        # 解析："dice 500.0 of chicken breast"
        parts = cleaned.split(" of ", 1)
        operation_with_qty = parts[0]  # "dice 500.0"
        ingredient = parts[1] if len(parts) > 1 else cleaned

        # Extract operation and quantity
        # 提取操作与数量
        op_parts = operation_with_qty.rsplit(" ", 1)
        operation = " ".join(op_parts[:-1]) if len(op_parts) > 1 else operation_with_qty

        key = f"{ingredient}:{operation}"
        if key in seen:
            continue
        seen.add(key)

        # Determine when the ingredient is needed (state this task consumes → produces)
        # 确定食材何时需要（该任务 consume → produce 的状态）
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
    # 按时长降序排序（较长预处理任务在前）
    items.sort(key=lambda i: i["duration_minutes"], reverse=True)

    return tuple(items)


# =============================================================================
# 11.2  Timeline
# 11.2  时间线
# =============================================================================


def build_timeline(
    schedule: ScheduleResult,
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """从调度区间与任务详情构建排序后的时间线。

    Build a sorted timeline from scheduled intervals and task details.

    Each timeline entry includes the task instruction, time window,
    work mode (active/passive), and dish name.

    每个时间线条目包含任务指令、时间窗口、工作模式（主动/被动）与菜名。

    Args:
        schedule: Solved schedule with intervals.
            schedule：含区间的已求解调度。
        tasks: All cooking tasks (for instruction/dish lookup).
            tasks：所有烹饪任务（用于指令/菜名查找）。

    Returns:
        Sorted list of dicts, earliest start time first.
        按最早开始时间排序的 dict 列表。
    """
    if not schedule.intervals:
        return ()

    # Build task lookup
    # 构建任务查找表
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
    # 按开始时间、再按结束时间排序
    items.sort(key=lambda i: (i["start_minute"], i["end_minute"]))

    return tuple(items)


# =============================================================================
# 11.2a Dependency-driven execution flow
# 11.2a 依赖驱动的执行流
# =============================================================================


def build_execution_flow(
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """构建 UI 就绪的任务依赖，而不预设钟点时间。

    Build UI-ready task dependencies without prescribing clock times.

    A client keeps its own ``completed_task_ids`` / ``in_progress_task_ids``
    and uses ``depends_on`` to decide what can be shown next.  This means a
    task such as "blanch crab" becomes available immediately after handling
    the crab; while its heating interval is in progress, unrelated active
    tasks such as preparing shrimp remain available to the cook.

    客户端维护自己的 completed_task_ids / in_progress_task_ids，并用 depends_on
    决定下一步显示什么。这意味着“焯蟹”这类任务在处理完蟹后立即可用；当其加热
    区间进行中时，不相关的主动任务（如准备虾）仍对厨师可用。
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
# 11.3  菜品完成汇总
# =============================================================================


def build_dish_completion_summary(
    schedule: ScheduleResult,
    tasks: tuple[CookingTask, ...],
) -> tuple[dict[str, object], ...]:
    """计算每道菜何时完成烹饪。

    Calculate when each dish completes cooking.

    A dish is "complete" when its last task finishes. For each dish,
    finds the maximum end_minute across all its tasks.

    一道菜在其最后一个任务完成时“完成”。对每道菜，找出其所有任务的最大 end_minute。

    Args:
        schedule: Solved schedule with intervals.
            schedule：含区间的已求解调度。
        tasks: All cooking tasks.
            tasks：所有烹饪任务。

    Returns:
        List of dicts sorted by completion time, each with:
        dish_id, dish_name, completion_minute, task_count.
        按完成时间排序的 dict 列表，每个含：dish_id、dish_name、completion_minute、task_count。
    """
    if not schedule.intervals:
        return ()

    task_map: dict[str, CookingTask] = {t.task_id: t for t in tasks}

    # Group intervals by dish_id, track max end time
    # 按 dish_id 分组区间，追踪最大结束时间
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
    # 转为排序列表
    result = sorted(dish_ends.values(), key=lambda d: d["completion_minute"])
    return tuple(result)


# =============================================================================
# 11.4  Completion checklist
# 11.4  完成清单
# =============================================================================


def build_completion_checklist(
    proposal: InventoryConsumptionProposal,
) -> tuple[CompletionItem, ...]:
    """从预留方案提取完成清单。

    Extract the completion checklist from a reservation proposal.

    Each CompletionItem groups allocations for one ingredient.
    This is a pass-through — the proposal already contains the right structure.

    每个 CompletionItem 为一种食材分组分配。这是透传 —— 方案已含正确结构。

    Args:
        proposal: An InventoryConsumptionProposal from build_reservation_proposal.
            proposal：来自 build_reservation_proposal 的 InventoryConsumptionProposal。

    Returns:
        Tuple of CompletionItem, one per ingredient group.
        每种食材组一个 CompletionItem 的元组。
    """
    return proposal.items


def validate_completion_checklist(
    proposal: InventoryConsumptionProposal,
    checklist: tuple[CompletionItem, ...],
) -> tuple[VerificationIssue, ...]:
    """对完成清单与其来源方案做校验。

    Validate the completion checklist against its source proposal.

    Checks:
      - Every allocation has a valid lot_id
      - Every allocation quantity is positive
      - No duplicate lot allocations across different items
      - Total allocated per ingredient ≤ proposal's required quantities
      - Checklist is non-empty when proposal has items

    检查：
      - 每项分配都有有效 lot_id
      - 每项分配数量为正
      - 不同条目间无重复批次分配
      - 每种食材的总分配 ≤ 方案所需数量
      - 方案有条目时清单非空

    Args:
        proposal: The source InventoryConsumptionProposal.
            proposal：来源 InventoryConsumptionProposal。
        checklist: The generated checklist to validate.
            checklist：要校验的生成清单。

    Returns:
        Tuple of VerificationIssue, empty = valid.
        VerificationIssue 元组，空 = 有效。
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
            # Check lot_id  检查 lot_id
            if not alloc.inventory_lot_id.strip():
                issues.append(
                    VerificationIssue(
                        code="MISSING_LOT_ID",
                        message=f"Allocation in '{item.ingredient_name}' has empty lot_id",
                    )
                )

            # Check for duplicate lot allocations  检查重复批次分配
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

            # Check positive quantity  检查正数量
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
