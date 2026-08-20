# =============================================================================
# 预处理合并节点（workflow/preparation_nodes）
# -----------------------------------------------------------------------------
# 单个流水线阶段的节点实现：把 RecipeStep 分解为 CookingTask，并经 prep_trie
# 合并共享预处理，再按安全报告锚定插入安全任务。公共兼容面仍为
# cooking_plan_agent.workflow.nodes。
# =============================================================================

"""Workflow node implementations for a single pipeline stage.

单个流水线阶段的节点实现。

The public compatibility surface remains ``cooking_plan_agent.workflow.nodes``.
This module contains one cohesive stage only.

公共兼容面仍为 ``cooking_plan_agent.workflow.nodes``。本模块仅包含一个内聚阶段。
"""

from langgraph.runtime import Runtime

from cooking_plan_agent.domain.errors import DomainErrorCode
from cooking_plan_agent.domain.models import (
    CookingTask,
    WorkflowError,
)
from cooking_plan_agent.workflow.context import WorkflowContext
from cooking_plan_agent.workflow.state import PlanState


def _wire_prep_consumption(
    recipe_tasks: list[CookingTask],
    demand_final_states: dict[str, str],
) -> list[CookingTask]:
    """Let each recipe's first task consume the final states of its demands.

    让每道菜的第一个任务消费其需求（食材预处理）的最终状态。

    ``demand_final_states`` keys are ``"recipe_id:index"``. The first
    decomposed task of every affected recipe consumes all of that recipe's
    prep outputs, so ``build_task_graph`` creates real prep → recipe edges.
    This is a pure transform — no mutation of frozen CookingTask objects.
    Returns a mutable list because the caller still wires safety-task
    reverse dependencies into it (P0-07).

    ``demand_final_states`` 的键为 ``"recipe_id:index"``。每个受影响菜谱的
    第一个分解任务消费该菜谱的所有预处理输出，使 ``build_task_graph``
    创建真实的 prep → recipe 边。这是纯变换 —— 不修改冻结的 CookingTask 对象。
    返回可变列表，因为调用方还要把安全任务的反向依赖接入其中（P0-07）。
    """
    if not demand_final_states:
        return list(recipe_tasks)

    states_by_recipe: dict[str, list[str]] = {}
    for demand_id, state in demand_final_states.items():
        recipe_id = demand_id.split(":", 1)[0]
        states_by_recipe.setdefault(recipe_id, []).append(state)

    seen: set[str] = set()
    updated: list[CookingTask] = []
    for task in recipe_tasks:
        if task.dish_id in states_by_recipe and task.dish_id not in seen:
            seen.add(task.dish_id)
            extra = tuple(dict.fromkeys(states_by_recipe[task.dish_id]))
            task = task.model_copy(update={"consumes_states": task.consumes_states + extra})
        updated.append(task)
    return updated


async def merge_preparation_node(
    state: PlanState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, object]:
    """Decompose recipe steps + merge shared preparation into CookingTasks.

    分解菜谱步骤 + 将共享预处理合并为 CookingTask。

    Iterates over validated RecipeIR steps, calls decompose_step for each,
    and collects the resulting CookingTasks. Preparation merging via prep_trie
    is deferred to MVP+1 (requires ingredient-demand-to-operation-chain bridge).

    遍历已验证的 RecipeIR 步骤，对每个步骤调用 decompose_step，并收集生成的
    CookingTask。通过 prep_trie 的预处理合并推迟到 MVP+1（需要食材需求到
    操作链的桥接）。

    Safety tasks are generated from the safety report when present.

    存在安全报告时，从安全报告生成安全任务。
    """
    from uuid import uuid4

    from cooking_plan_agent.domain.enums import WorkMode
    from cooking_plan_agent.domain.models import CookingTask, ResourceNeed, TaskDependency
    from cooking_plan_agent.preparation.decompose import decompose_step

    parsed_recipes = state.get("parsed_recipes", ())
    if not parsed_recipes:
        return {"recipe_tasks": (), "prep_tasks": (), "safety_tasks": ()}

    all_recipe_tasks: list[CookingTask] = []
    # Track the last task of each recipe to chain subsequent steps
    # 追踪每道菜的最后一个任务，以串联后续步骤
    recipe_last_task: dict[str, str] = {}
    # P0-07: per-recipe map step_number → (first_task_id, last_task_id) so
    # safety insertions can anchor between exact recipe steps.
    # P0-07：按菜谱建立 step_number → (first_task_id, last_task_id) 的映射，
    # 使安全插入能锚定在确切的菜谱步骤之间。
    step_task_anchors: dict[str, dict[int, tuple[str, str]]] = {}

    for recipe in parsed_recipes:
        last_task_id: str | None = None
        anchors: dict[int, tuple[str, str]] = {}
        for step in recipe.steps:
            tasks = decompose_step(recipe.recipe_id, step)
            if not tasks:
                continue

            # If this is not the first step in the recipe, add a dependency
            # from the previous step's last task to this step's first task
            # 若这不是菜谱的第一步，则添加从前一步最后一个任务到本步第一个任务的依赖
            if last_task_id is not None and tasks:
                first = tasks[0]
                dep = TaskDependency(predecessor_id=last_task_id)
                tasks = (first.model_copy(update={"dependencies": first.dependencies + (dep,)}),) + tasks[1:]

            all_recipe_tasks.extend(tasks)
            last_task_id = tasks[-1].task_id
            anchors[step.step_number] = (tasks[0].task_id, tasks[-1].task_id)

        step_task_anchors[recipe.recipe_id] = anchors
        if last_task_id is not None:
            recipe_last_task[recipe.recipe_id] = last_task_id

    # --- P2-01: shared preparation merging via prep_trie ---
    # --- P2-01：通过 prep_trie 合并共享预处理 ---
    # Merges identical prep operations across recipes (one wash for N dishes)
    # and wires each recipe's first task to consume the prep outputs. When
    # the feature is disabled, prep_tasks stays empty (per-recipe prep).
    # 合并跨菜谱的相同预处理操作（N 道菜共用一次清洗），并让每道菜的
    # 第一个任务消费预处理输出。功能禁用时，prep_tasks 保持为空（逐菜谱预处理）。
    prep_tasks: tuple[CookingTask, ...] = ()
    prep_observations: tuple[str, ...] = ()
    from cooking_plan_agent.config.settings import get_settings

    if get_settings().shared_prep_enabled:
        from cooking_plan_agent.normalisation.errors import InvalidQuantityError
        from cooking_plan_agent.preparation.prep_trie import build_shared_prep_tasks

        demands = tuple((recipe.recipe_id, demand) for recipe in parsed_recipes for demand in recipe.ingredients)
        try:
            shared = build_shared_prep_tasks(demands)
        except InvalidQuantityError as exc:
            # D1: conservation failure must never produce a half-built task
            # graph — terminate to FAILED via INTERNAL_ERROR. P2-03: keep
            # only the exception type as diagnostic context.
            # D1：守恒失败绝不能让半构建的任务图产出 —— 通过 INTERNAL_ERROR
            # 终止到 FAILED。P2-03：仅保留异常类型作为诊断上下文。
            return {
                "error": WorkflowError(
                    error_code=DomainErrorCode.INTERNAL_ERROR.value,
                    message="Preparation quantity conservation failed",
                    correlation_id=state["request"].request_id,
                    node_name="merge_preparation",
                    diagnostics={"exception_type": type(exc).__name__},
                )
            }
        prep_tasks = shared.tasks
        prep_observations = shared.observations
        if prep_tasks:
            # A menu plan should feel like a real mise en place: take all
            # ingredients out once, finish the shared washing/cutting work,
            # then begin cooking.  The previous state-only wiring allowed
            # unrelated recipes to start while other ingredients were still
            # being fetched, which causes repeated fridge trips and a
            # fragmented prep phase.
            # 菜单计划应该像真实的 mise en place：一次性取出所有食材，完成共享的
            # 清洗 / 切配工作，然后开始烹饪。此前的仅状态接线允许无关菜谱在
            # 其他食材仍被取出时就开始，导致反复开冰箱和碎片化的预处理阶段。
            gather_task = CookingTask(
                task_id="prep_gather_all_ingredients",
                dish_id="shared",
                instruction="[Mise en place] Gather all ingredients, seasonings, and required tools",
                duration_minutes=5,
                work_mode=WorkMode.ACTIVE,
                category="preparation",
            )
            staged_prep_tasks = tuple(
                task.model_copy(
                    update={"dependencies": task.dependencies + (TaskDependency(predecessor_id=gather_task.task_id),)}
                )
                for task in prep_tasks
            )
            ready_task = CookingTask(
                task_id="prep_mise_en_place_complete",
                dish_id="shared",
                instruction="[Mise en place] Confirm that washing, cutting, and seasoning portions are ready",
                duration_minutes=1,
                work_mode=WorkMode.ACTIVE,
                category="preparation",
                dependencies=tuple(TaskDependency(predecessor_id=task.task_id) for task in staged_prep_tasks),
            )
            first_task_ids = {first for anchors in step_task_anchors.values() for first, _last in anchors.values()}
            all_recipe_tasks = [
                task.model_copy(
                    update={"dependencies": task.dependencies + (TaskDependency(predecessor_id=ready_task.task_id),)}
                )
                if task.task_id in first_task_ids
                else task
                for task in all_recipe_tasks
            ]
            prep_tasks = (gather_task, *staged_prep_tasks, ready_task)
        all_recipe_tasks = _wire_prep_consumption(all_recipe_tasks, shared.demand_final_states)

    # --- Safety tasks (P0-07: anchored insertions from the safety report) ---
    # --- 安全任务（P0-07：来自安全报告的锚定插入） ---
    safety_task_list: list[CookingTask] = []
    safety_report = state.get("safety_report")
    if safety_report is not None:
        # 1. Structured insertions with exact step anchors.
        # 1. 带确切步骤锚点的结构化插入。
        for insertion in safety_report.insertions:
            anchors = step_task_anchors.get(insertion.recipe_id, {})
            after_pair = anchors.get(insertion.after_step_number) if insertion.after_step_number is not None else None
            before_pair = (
                anchors.get(insertion.before_step_number) if insertion.before_step_number is not None else None
            )
            _after_first, after_last = after_pair if after_pair is not None else (None, None)
            before_first, _before_last = before_pair if before_pair is not None else (None, None)

            task_id = f"safety_{insertion.insertion_id}_{uuid4().hex[:8]}"
            deps: list[TaskDependency] = []
            if after_last is not None:
                # raw task → sanitise task
                # 原始任务 → 消毒任务
                deps.append(TaskDependency(predecessor_id=after_last))
            resources = tuple(ResourceNeed(quantity=1, resource_type=r) for r in insertion.required_resources)
            task = CookingTask(
                task_id=task_id,
                dish_id=insertion.recipe_id,
                instruction=insertion.task_instruction,
                duration_minutes=insertion.duration_minutes,
                work_mode=WorkMode.ACTIVE,
                category="safety",
                dependencies=tuple(deps),
                resources=resources,
                safety_tags=(insertion.rule_id,),
            )
            safety_task_list.append(task)

            # 2. sanitise task → RTE task (reverse dependency on the RTE head)
            # 2. 消毒任务 → RTE 任务（对 RTE 头部的反向依赖）
            if before_first is not None:
                rte_task = next((t for t in all_recipe_tasks if t.task_id == before_first), None)
                if rte_task is not None:
                    rte_dep = TaskDependency(predecessor_id=task_id)
                    updated = rte_task.model_copy(update={"dependencies": rte_task.dependencies + (rte_dep,)})
                    for idx, t in enumerate(all_recipe_tasks):
                        if t.task_id == before_first:
                            all_recipe_tasks[idx] = updated
                            break

        # 3. Fallback: legacy bare task IDs (no anchors) — keep the old
        #    behaviour for rules that still emit IDs only.
        # 3. 回退：遗留的裸任务 ID（无锚点）—— 对仍只产出 ID 的规则保持旧行为。
        anchored_ids = {s.insertion_id for s in safety_report.insertions}
        for task_id in safety_report.required_safety_task_ids:
            if any(s.rule_id.lower() in task_id for s in safety_report.insertions):
                continue  # already materialised via insertion above  # 已通过上方插入物化
            task = CookingTask(
                task_id=task_id,
                dish_id="_safety",
                instruction=f"Safety task: {task_id}",
                duration_minutes=3,
                work_mode=WorkMode.ACTIVE,
                category="safety",
                safety_tags=(task_id,),
            )
            safety_task_list.append(task)
        _ = anchored_ids  # keep for clarity  # 仅为清晰保留

    return {
        "recipe_tasks": tuple(all_recipe_tasks),
        "prep_tasks": prep_tasks,
        "safety_tasks": tuple(safety_task_list),
        "prep_observations": prep_observations,
    }
