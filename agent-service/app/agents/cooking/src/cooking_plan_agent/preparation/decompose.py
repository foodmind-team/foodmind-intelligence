"""Recipe step decomposition — RecipeStep → tuple[CookingTask, ...]

This module handles the active/passive split of recipe steps into
schedulable CookingTasks.  Each recipe pattern (boil, bake, marinate,
etc.) has its own decomposition helper that produces a sequence of
sub-tasks linked by TaskDependency.

See also
--------
prep_trie.py : Ingredient preparation merging via prefix tree 
task_graph.py : Task DAG construction, cycle detection, critical path
"""

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
# ============================================================================

# Default durations (minutes) for sub-tasks when the source step does not
# specify exact per-phase times.  These are conservative estimates suitable
# for an MVP — production systems should source them from a vetted catalogue.
_DEFAULT_ACTIVE_SETUP = 2   # Opening containers, filling water, turning on heat
_DEFAULT_ACTIVE_CHECK = 1   # Verifying boil, checking doneness
_DEFAULT_ACTIVE_LOAD = 3    # Loading oven, arranging tray
_DEFAULT_ACTIVE_UNLOAD = 3  # Unloading oven, plating
_DEFAULT_ACTIVE_MARINATE = 5  # Mixing, rubbing, applying marinade


class DecompositionPolicy(StrictModel):
    """Controls how recipe steps are split into active/passive sub-tasks.

    Attributes
    ----------
    active_setup_minutes
        Default duration for setup sub-tasks (boil:fill, bake:load, etc.).
    active_check_minutes
        Default duration for check/verification sub-tasks.
    active_load_minutes
        Default duration for oven/tray loading sub-tasks.
    active_unload_minutes
        Default duration for oven unload/plate sub-tasks.
    active_marinate_minutes
        Default duration for marinade application.
    """

    active_setup_minutes: int = _DEFAULT_ACTIVE_SETUP
    active_check_minutes: int = _DEFAULT_ACTIVE_CHECK
    active_load_minutes: int = _DEFAULT_ACTIVE_LOAD
    active_unload_minutes: int = _DEFAULT_ACTIVE_UNLOAD
    active_marinate_minutes: int = _DEFAULT_ACTIVE_MARINATE

    # ------------------------------------------------------------------
    # Per-pattern decomposition dispatch table
    # ------------------------------------------------------------------
    # Each entry maps a RecipeStep.pattern to a method that returns a
    # tuple of CookingTask instances.  Unknown patterns fall through to
    # _decompose_simple (pass-through).
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
# ============================================================================


def _make_task_id(recipe_id: str, step_number: int, suffix: str = "") -> str:
    """Build a deterministic task ID from recipe + step + optional suffix."""
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
    """Construct a CookingTask with all fields explicit.

    This factory exists so that decomposition helpers stay concise —
    they only supply what differs from defaults.
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
    """Linearise dependencies: each task depends on the previous one.

    Used for sub-tasks that must execute sequentially (fill → heat → check).
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
            result.append(
                t.model_copy(update={"dependencies": t.dependencies + (dep,)})
            )
    return tuple(result)


def decompose_step(
    recipe_id: str,
    step: RecipeStep,
    policy: DecompositionPolicy | None = None,
) -> tuple[CookingTask, ...]:
    """Decompose a single recipe step into schedulable CookingTasks.

    The dispatch key is ``step.pattern``.  If the pattern is unrecognised,
    the step is treated as a simple active task (pass-through).

    If a recipe needs periodic attention, represent explicit check tasks.

    Args:
        recipe_id: Stable identifier for the parent recipe.
        step: A raw recipe step with pattern, duration, and heat hints.
        policy: Optional decomposition policy controlling sub-task durations.
            When ``None``, default durations are used.

    Returns:
        A tuple of one or more CookingTask instances, chained by
        ``TaskDependency`` where sub-tasks are sequential.
    """
    if policy is None:
        policy = DecompositionPolicy()

    method_name = DecompositionPolicy._PATTERN_DISPATCH.get(step.pattern)
    if method_name is None or step.pattern == "simple":
        return _decompose_simple(recipe_id, step, policy)

    # Dispatch to the appropriate decomposition helper.
    dispatcher: dict = {
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
# ---------------------------------------------------------------------------


def _decompose_simple(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Pass-through: one active task from a simple step."""
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


def _decompose_boil(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Boil pattern: fill (active) → heat to boil (passive) → check (active)."""
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


def _decompose_marinate(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Marinate pattern: apply (active) → wait (passive)."""
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
    )
    return _chain_deps((apply_task, wait_task))


def _decompose_bake(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Bake pattern: load (active) → bake (passive) → unload (active)."""
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


def _decompose_stir_fry(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Stir-fry pattern: one active task occupying stove + pan + utensil."""
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


def _decompose_simmer(
    recipe_id: str, step: RecipeStep, policy: DecompositionPolicy
) -> tuple[CookingTask, ...]:
    """Simmer pattern: passive intervals separated by periodic check/stir tasks.

    Example: 'Simmer and stir every 5 minutes' → passive intervals
    separated by short active check/stir tasks.  Do not use fractional
    attention capacity.
    """
    passive_dur = step.passive_duration_minutes or 30
    interval = step.interval_minutes or 5
    sn = step.step_number

    # How many full intervals fit? e.g. 30 min / 5 min interval = 6 intervals,
    # each with a wait period followed by a check.
    # But the last interval may not need a check if total time is exact.
    num_intervals = max(1, passive_dur // interval)
    remainder = passive_dur % interval

    tasks: list[CookingTask] = []
    elapsed = 0
    for i in range(num_intervals):
        # Passive wait sub-task
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
        # Active check/stir sub-task
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
    """Heuristic: does this step likely involve raw meat/poultry/seafood?"""
    raw_keywords = ("chicken", "beef", "pork", "fish", "shrimp", "meat", "poultry", "seafood")
    lower = step.instruction.lower()
    return any(kw in lower for kw in raw_keywords)


# ============================================================================
# 6.3  Food-state representation
# ============================================================================


def format_food_state(
    ingredient: str,
    state: str,
    scope: str = "shared",
) -> str:
    """Format a food-state identifier in ``ingredient:state:scope`` notation.

    Handbook 6.3 examples:
        chilli:raw:portion-a
        chilli:washed:shared
        chicken:marinated:dish-b

    Args:
        ingredient: Canonical ingredient name (e.g. ``"chilli"``).
        state: Processing state (e.g. ``"washed"``, ``"diced"``, ``"cooked"``).
        scope: Which dish/portion (e.g. ``"shared"``, ``"dish-a"``).

    Returns:
        A food-state string usable in ``CookingTask.consumes_states`` /
        ``produces_states``.
    """
    return f"{ingredient}:{state}:{scope}"
