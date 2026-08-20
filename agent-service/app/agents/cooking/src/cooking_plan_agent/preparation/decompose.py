# =============================================================================
# 菜谱步骤分解模块（preparation/decompose）
# -----------------------------------------------------------------------------
# 实现手册 6.2：把 RecipeStep 分解为可调度的 CookingTask 序列（主动 / 被动拆分）。
# 每种菜谱模式（boil、bake、marinate、stir_fry、simmer 等）都有自己的分解辅助函数，
# 产出一组由 TaskDependency 串联的子任务。
# 核心：
#   - DecompositionPolicy：控制每个模式的主动 / 被动拆分时长（可配置）
#   - decompose_step      ：按 step.pattern 分派到对应分解函数
#   - _decompose_*        ：各模式的分解实现（fill→heat→check 等）
#   - format_food_state   ：食材状态表示（ingredient:state:scope，手册 6.3）
# 相关模块：prep_trie.py（前缀树合并食材预处理）、task_graph.py（任务 DAG 构建）
# =============================================================================

"""Recipe step decomposition — RecipeStep → tuple[CookingTask, ...]

菜谱步骤分解 —— RecipeStep → tuple[CookingTask, ...]

This module handles the active/passive split of recipe steps into
schedulable CookingTasks.  Each recipe pattern (boil, bake, marinate,
etc.) has its own decomposition helper that produces a sequence of
sub-tasks linked by TaskDependency.

本模块处理菜谱步骤的“主动 / 被动”拆分，生成可调度的 CookingTask。
每种菜谱模式（boil、bake、marinate 等）都有自己的分解辅助函数，
产出一组由 TaskDependency 串联的子任务。

See also
--------
prep_trie.py : Ingredient preparation merging via prefix tree
task_graph.py : Task DAG construction, cycle detection, critical path

另见：
prep_trie.py：通过前缀树合并食材预处理
task_graph.py：任务 DAG 构建、环检测、关键路径
"""

from collections.abc import Callable
from decimal import Decimal
from typing import ClassVar

from cooking_plan_agent.domain.enums import HeatLevel, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    RecipeStep,
    ResourceNeed,
    StrictModel,
    TaskDependency,
)

# ============================================================================
# 6.2  Decomposition policy — controls active/passive split per recipe pattern
# 6.2  分解策略 —— 控制每种菜谱模式的主动 / 被动拆分
# ============================================================================

# Default durations (minutes) for sub-tasks when the source step does not
# specify exact per-phase times.  These are conservative estimates suitable
# for an MVP — production systems should source them from a vetted catalogue.
# 当来源步骤未指定各阶段精确时间时，子任务的默认时长（分钟）。这些是适合 MVP 的
# 保守估计 —— 生产系统应从审核过的目录中获取。
_DEFAULT_ACTIVE_SETUP = 2  # Opening containers, filling water, turning on heat  开容器、装水、开火
_DEFAULT_ACTIVE_CHECK = 1  # Verifying boil, checking doneness  确认沸腾、检查熟度
_DEFAULT_ACTIVE_LOAD = 3  # Loading oven, arranging tray  装入烤箱、摆放烤盘
_DEFAULT_ACTIVE_UNLOAD = 3  # Unloading oven, plating  取出烤箱、装盘
_DEFAULT_ACTIVE_MARINATE = 5  # Mixing, rubbing, applying marinade  搅拌、揉搓、涂抹腌料


class DecompositionPolicy(StrictModel):
    """控制菜谱步骤如何拆分为主动 / 被动子任务。

    Controls how recipe steps are split into active/passive sub-tasks.

    Attributes
    ----------
    active_setup_minutes
        Default duration for setup sub-tasks (boil:fill, bake:load, etc.).
        准备子任务的默认时长（煮：装水、烤：装入等）。
    active_check_minutes
        Default duration for check/verification sub-tasks.
        检查 / 验证子任务的默认时长。
    active_load_minutes
        Default duration for oven/tray loading sub-tasks.
        烤箱 / 烤盘装载子任务的默认时长。
    active_unload_minutes
        Default duration for oven unload/plate sub-tasks.
        烤箱卸载 / 装盘子任务的默认时长。
    active_marinate_minutes
        Default duration for marinade application.
        涂抹腌料的默认时长。
    """

    active_setup_minutes: int = _DEFAULT_ACTIVE_SETUP
    active_check_minutes: int = _DEFAULT_ACTIVE_CHECK
    active_load_minutes: int = _DEFAULT_ACTIVE_LOAD
    active_unload_minutes: int = _DEFAULT_ACTIVE_UNLOAD
    active_marinate_minutes: int = _DEFAULT_ACTIVE_MARINATE

    # ------------------------------------------------------------------
    # Per-pattern decomposition dispatch table
    # 每种模式的分解分派表
    # ------------------------------------------------------------------
    # Each entry maps a RecipeStep.pattern to a method that returns a
    # tuple of CookingTask instances.  Unknown patterns fall through to
    # _decompose_simple (pass-through).
    # 每个条目把 RecipeStep.pattern 映射到一个返回 CookingTask 元组的方法。
    # 未知模式落到 _decompose_simple（透传）。
    _PATTERN_DISPATCH: ClassVar[dict[str, str]] = {
        "simple": "_decompose_simple",
        "boil": "_decompose_boil",
        "marinate": "_decompose_marinate",
        "bake": "_decompose_bake",
        "stir_fry": "_decompose_stir_fry",
        "simmer": "_decompose_simmer",
    }


# ============================================================================
# 6.2  decompose_step — RecipeStep → tuple[CookingTask, ...]
# 6.2  decompose_step —— RecipeStep → tuple[CookingTask, ...]
# ============================================================================


def _make_task_id(recipe_id: str, step_number: int, suffix: str = "") -> str:
    """从 recipe + step + 可选后缀构建确定性的任务 ID。"""
    base = f"{recipe_id}_s{step_number}"
    return f"{base}_{suffix}" if suffix else base


def _build_task(
    task_id: str,
    dish_id: str,
    instruction: str,
    duration_minutes: int,
    work_mode: WorkMode,
    category: str,
    heat_level: HeatLevel = HeatLevel.NONE,
    target_temperature_c: Decimal | None = None,
    dependencies: tuple[TaskDependency, ...] = (),
    resources: tuple[ResourceNeed, ...] = (),
    consumes_states: tuple[str, ...] = (),
    produces_states: tuple[str, ...] = (),
    batch_key: str | None = None,
    safety_tags: tuple[str, ...] = (),
) -> CookingTask:
    """用显式字段构造一个 CookingTask。

    Construct a CookingTask with all fields explicit.

    This factory exists so that decomposition helpers stay concise —
    they only supply what differs from defaults.

    这个工厂使分解辅助函数保持简洁 —— 它们只提供与默认值不同的字段。
    """
    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=instruction,
        duration_minutes=duration_minutes,
        work_mode=work_mode,
        category=category,
        heat_level=heat_level,
        target_temperature_c=target_temperature_c,
        dependencies=dependencies,
        resources=resources,
        consumes_states=consumes_states,
        produces_states=produces_states,
        batch_key=batch_key,
        safety_tags=safety_tags,
    )


def _chain_deps(tasks: tuple[CookingTask, ...]) -> tuple[CookingTask, ...]:
    """线性化依赖：每个任务依赖前一个任务。

    Linearise dependencies: each task depends on the previous one.

    Used for sub-tasks that must execute sequentially (fill → heat → check).

    用于必须顺序执行的子任务（装水 → 加热 → 检查）。
    """
    if len(tasks) <= 1:
        return tasks
    result: list[CookingTask] = []
    for i, t in enumerate(tasks):
        if i == 0:
            result.append(t)
        else:
            prev_id = tasks[i - 1].task_id
            dep = TaskDependency(predecessor_id=prev_id)
            # Rebuild with updated dependencies — CookingTask is frozen.
            # 用更新后的依赖重建 —— CookingTask 是 frozen。
            result.append(t.model_copy(update={"dependencies": t.dependencies + (dep,)}))
    return tuple(result)


def decompose_step(
    recipe_id: str,
    step: RecipeStep,
    policy: DecompositionPolicy | None = None,
) -> tuple[CookingTask, ...]:
    """把单个菜谱步骤分解为可调度的 CookingTask。

    Decompose a single recipe step into schedulable CookingTasks.

    The dispatch key is ``step.pattern``.  If the pattern is unrecognised,
    the step is treated as a simple active task (pass-through).

    分派键是 step.pattern。若模式未识别，该步骤按简单主动任务处理（透传）。

    If a recipe needs periodic attention, represent explicit check tasks.

    若菜谱需要周期性关注，用显式检查任务表示。

    Args:
        recipe_id: Stable identifier for the parent recipe.
            recipe_id：父菜谱的稳定标识。
        step: A raw recipe step with pattern, duration, and heat hints.
            step：带模式、时长与火力提示的原始菜谱步骤。
        policy: Optional decomposition policy controlling sub-task durations.
            When ``None``, default durations are used.
            policy：控制子任务时长的可选分解策略。None 时用默认时长。

    Returns:
        A tuple of one or more CookingTask instances, chained by
        ``TaskDependency`` where sub-tasks are sequential.
        一个或多个 CookingTask 实例组成的元组，子任务顺序执行处用 TaskDependency 串联。
    """
    if policy is None:
        policy = DecompositionPolicy()

    method_name = DecompositionPolicy._PATTERN_DISPATCH.get(step.pattern)
    if method_name is None or step.pattern == "simple":
        return _decompose_simple(recipe_id, step, policy)

    # Dispatch to the appropriate decomposition helper.
    # 分派到合适的分解辅助函数。
    dispatcher: dict[
        str,
        Callable[[str, RecipeStep, DecompositionPolicy], tuple[CookingTask, ...]],
    ] = {
        "boil": _decompose_boil,
        "marinate": _decompose_marinate,
        "bake": _decompose_bake,
        "stir_fry": _decompose_stir_fry,
        "simmer": _decompose_simmer,
    }
    handler = dispatcher.get(step.pattern, _decompose_simple)
    return handler(recipe_id, step, policy)


# ---------------------------------------------------------------------------
# Per-pattern decomposition helpers
# 各模式的分解辅助函数
# ---------------------------------------------------------------------------


def _decompose_simple(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """透传：简单步骤生成一个主动任务。"""
    duration = step.active_duration_minutes or 5
    task = _build_task(
        task_id=_make_task_id(recipe_id, step.step_number),
        dish_id=recipe_id,
        instruction=step.instruction,
        duration_minutes=duration,
        work_mode=WorkMode.ACTIVE,
        category=step.category,
        heat_level=step.heat_level,
        target_temperature_c=step.target_temperature_c,
    )
    return (task,)


def _decompose_boil(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """煮模式：装水（主动）→ 加热至沸腾（被动）→ 检查（主动）。"""
    passive_dur = step.passive_duration_minutes or 10
    sn = step.step_number
    fill = _build_task(
        task_id=_make_task_id(recipe_id, sn, "fill"),
        dish_id=recipe_id,
        instruction=f"[Fill] {step.instruction}",
        duration_minutes=policy.active_setup_minutes,
        work_mode=WorkMode.ACTIVE,
        category="setup",
        resources=(ResourceNeed(quantity=1, resource_type="sink"),),
    )
    heat = _build_task(
        task_id=_make_task_id(recipe_id, sn, "heat"),
        dish_id=recipe_id,
        instruction=f"[Heat] {step.instruction}",
        duration_minutes=passive_dur,
        work_mode=WorkMode.PASSIVE,
        category="heating",
        heat_level=HeatLevel.HIGH,
        target_temperature_c=step.target_temperature_c,
        resources=(ResourceNeed(quantity=1, resource_type="stove"),),
    )
    check = _build_task(
        task_id=_make_task_id(recipe_id, sn, "check"),
        dish_id=recipe_id,
        instruction=f"[Check boil] {step.instruction}",
        duration_minutes=policy.active_check_minutes,
        work_mode=WorkMode.ACTIVE,
        category="checking",
    )
    return _chain_deps((fill, heat, check))


def _decompose_marinate(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """腌制模式：涂抹（主动）→ 等待（被动）。"""
    passive_dur = step.passive_duration_minutes or 20
    sn = step.step_number
    apply_task = _build_task(
        task_id=_make_task_id(recipe_id, sn, "apply"),
        dish_id=recipe_id,
        instruction=f"[Apply marinade] {step.instruction}",
        duration_minutes=policy.active_marinate_minutes,
        work_mode=WorkMode.ACTIVE,
        category="mixing",
        resources=(ResourceNeed(quantity=1, resource_type="mixing_bowl"),),
    )
    wait_task = _build_task(
        task_id=_make_task_id(recipe_id, sn, "wait"),
        dish_id=recipe_id,
        instruction=f"[Marinate] {step.instruction}",
        duration_minutes=passive_dur,
        work_mode=WorkMode.PASSIVE,
        category="resting",
        safety_tags=("raw_meat",) if _is_raw_protein(step) else (),
        # ↑ 若涉及生蛋白质则打上 "raw_meat" 安全标签
    )
    return _chain_deps((apply_task, wait_task))


def _decompose_bake(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """烘焙模式：装载（主动）→ 烘焙（被动）→ 卸载（主动）。"""
    passive_dur = step.passive_duration_minutes or 25
    sn = step.step_number
    load = _build_task(
        task_id=_make_task_id(recipe_id, sn, "load"),
        dish_id=recipe_id,
        instruction=f"[Load oven] {step.instruction}",
        duration_minutes=policy.active_load_minutes,
        work_mode=WorkMode.ACTIVE,
        category="setup",
        heat_level=HeatLevel.NONE,
        resources=(ResourceNeed(quantity=1, resource_type="oven"),),
    )
    bake = _build_task(
        task_id=_make_task_id(recipe_id, sn, "bake"),
        dish_id=recipe_id,
        instruction=f"[Bake] {step.instruction}",
        duration_minutes=passive_dur,
        work_mode=WorkMode.PASSIVE,
        category="heating",
        heat_level=step.heat_level,
        target_temperature_c=step.target_temperature_c,
        resources=(ResourceNeed(quantity=1, resource_type="oven"),),
        batch_key=f"oven_{step.target_temperature_c or 'default'}C",
        # ↑ 相同烤箱温度的任务可合并（batch_key）
    )
    unload = _build_task(
        task_id=_make_task_id(recipe_id, sn, "unload"),
        dish_id=recipe_id,
        instruction=f"[Unload oven] {step.instruction}",
        duration_minutes=policy.active_unload_minutes,
        work_mode=WorkMode.ACTIVE,
        category="finishing",
        heat_level=HeatLevel.NONE,
        resources=(ResourceNeed(quantity=1, resource_type="oven"),),
    )
    return _chain_deps((load, bake, unload))


def _decompose_stir_fry(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """爆炒模式：一个主动任务占用灶台 + 锅 + 锅铲。"""
    duration = step.active_duration_minutes or 5
    task = _build_task(
        task_id=_make_task_id(recipe_id, step.step_number),
        dish_id=recipe_id,
        instruction=step.instruction,
        duration_minutes=duration,
        work_mode=WorkMode.ACTIVE,
        category="heating",
        heat_level=HeatLevel.HIGH,
        resources=(
            ResourceNeed(quantity=1, resource_type="stove"),
            ResourceNeed(quantity=1, resource_type="wok"),
            ResourceNeed(quantity=1, resource_type="spatula"),
        ),
    )
    return (task,)


def _decompose_simmer(recipe_id: str, step: RecipeStep, policy: DecompositionPolicy) -> tuple[CookingTask, ...]:
    """炖煮模式：被动区间由周期性的检查 / 搅拌任务分隔。

    Simmer pattern: passive intervals separated by periodic check/stir tasks.

    Example: 'Simmer and stir every 5 minutes' → passive intervals
    separated by short active check/stir tasks.  Do not use fractional
    attention capacity.

    例：'Simmer and stir every 5 minutes' → 被动区间由短主动检查 / 搅拌任务分隔。
    不使用分数注意力容量。
    """
    passive_dur = step.passive_duration_minutes or 30
    interval = step.interval_minutes or 5
    sn = step.step_number

    # How many full intervals fit? e.g. 30 min / 5 min interval = 6 intervals,
    # each with a wait period followed by a check.
    # But the last interval may not need a check if total time is exact.
    # 能容纳多少个完整区间？如 30 分钟 / 5 分钟区间 = 6 个区间，每个都是等待后检查。
    # 但若总时间恰好整除，最后一个区间可能不需要检查。
    num_intervals = max(1, passive_dur // interval)
    remainder = passive_dur % interval

    tasks: list[CookingTask] = []
    elapsed = 0
    for i in range(num_intervals):
        # Passive wait sub-task  被动等待子任务
        wait_dur = interval if i < num_intervals - 1 or remainder == 0 else interval + remainder
        tasks.append(
            _build_task(
                task_id=_make_task_id(recipe_id, sn, f"wait{i + 1}"),
                dish_id=recipe_id,
                instruction=f"[Simmer {elapsed + 1}-{elapsed + wait_dur} min] {step.instruction}",
                duration_minutes=wait_dur,
                work_mode=WorkMode.PASSIVE,
                category="heating",
                heat_level=HeatLevel.LOW,
                resources=(ResourceNeed(quantity=1, resource_type="stove"),),
            )
        )
        elapsed += wait_dur
        # Active check/stir sub-task  主动检查 / 搅拌子任务
        tasks.append(
            _build_task(
                task_id=_make_task_id(recipe_id, sn, f"stir{i + 1}"),
                dish_id=recipe_id,
                instruction=f"[Stir at {elapsed} min] {step.instruction}",
                duration_minutes=policy.active_check_minutes,
                work_mode=WorkMode.ACTIVE,
                category="checking",
                resources=(ResourceNeed(quantity=1, resource_type="spatula"),),
            )
        )

    return _chain_deps(tuple(tasks))


def _is_raw_protein(step: RecipeStep) -> bool:
    """启发式：该步骤是否可能涉及生肉 / 禽 / 海鲜？"""
    raw_keywords = ("chicken", "beef", "pork", "fish", "shrimp", "meat", "poultry", "seafood")
    lower = step.instruction.lower()
    return any(kw in lower for kw in raw_keywords)


# ============================================================================
# 6.3  Food-state representation
# 6.3  食材状态表示
# ============================================================================


def format_food_state(
    ingredient: str,
    state: str,
    scope: str = "shared",
) -> str:
    """用 ``ingredient:state:scope`` 记法格式化食材状态标识。

    Format a food-state identifier in ``ingredient:state:scope`` notation.

    Handbook 6.3 examples:
        chilli:raw:portion-a
        chilli:washed:shared
        chicken:marinated:dish-b

    手册 6.3 示例：
        chilli:raw:portion-a
        chilli:washed:shared
        chicken:marinated:dish-b

    Args:
        ingredient: Canonical ingredient name (e.g. ``"chilli"``).
            ingredient：规范食材名（如 "chilli"）。
        state: Processing state (e.g. ``"washed"``, ``"diced"``, ``"cooked"``).
            state：加工状态（如 "washed"、"diced"、"cooked"）。
        scope: Which dish/portion (e.g. ``"shared"``, ``"dish-a"``).
            scope：哪道菜 / 哪份（如 "shared"、"dish-a"）。

    Returns:
        A food-state string usable in ``CookingTask.consumes_states`` /
        ``produces_states``.
        可用于 CookingTask.consumes_states / produces_states 的食材状态字符串。
    """
    return f"{ingredient}:{state}:{scope}"
