"""P2-01: shared preparation merging enters the main workflow.

Verifies:
- convert_trie_to_tasks no longer emits dangling predecessor IDs (food
  states must never be used as task predecessor IDs).
- demand_id -> final food-state extraction.
- build_shared_prep_tasks merges identical prep chains, branches on
  different cuts, conserves quantity, and isolates raw protein from RTE.
- merge_preparation_node returns non-empty prep_tasks when the feature is
  enabled and falls back to per-recipe preparation when disabled.
"""

from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import (
    GeneratePlanRequest,
    IngredientDemand,
    RecipeInput,
    RecipeIR,
    RecipeStep,
)
from cooking_plan_agent.preparation.prep_trie import (
    PreparationOperation,
    PrepTrieNode,
    build_shared_prep_tasks,
    convert_trie_to_tasks,
    demand_final_states_for_ingredient,
    insert_operation_chain,
)
from cooking_plan_agent.workflow.nodes import merge_preparation_node


def _demand(
    name: str = "brown onion",
    quantity: str = "100",
    unit: str = "g",
    spec: str | None = "diced",
    input_state: str = "raw",
) -> IngredientDemand:
    return IngredientDemand(
        canonical_name=name,
        raw_name=name,
        quantity=Decimal(quantity),
        unit=unit,
        preparation_spec=spec,
        input_state=input_state,
        confidence=Decimal("0.9"),
    )


# =============================================================================
# convert_trie_to_tasks — dangling-dependency regression (P2-01 step 1)
# =============================================================================


def _branch_trie() -> PrepTrieNode:
    """wash(300) -> cut:julienned(100) + cut:diced(200)."""
    root = PrepTrieNode(operation_key="root")
    wash = PreparationOperation(operation="wash", quantity=Decimal(300), unit="g")
    julienne = PreparationOperation(
        operation="julienne",
        specification="julienned",
        quantity=Decimal(100),
        unit="g",
    )
    dice = PreparationOperation(
        operation="dice",
        specification="diced",
        quantity=Decimal(200),
        unit="g",
    )
    insert_operation_chain(root, (wash, julienne), "d1")
    insert_operation_chain(root, (wash, dice), "d2")
    return root


def test_convert_tasks_have_no_dangling_state_dependencies() -> None:
    """Every TaskDependency predecessor must be a real task_id.

    Regression: convert_trie_to_tasks used to write food-state strings
    (e.g. 'chilli:washed:shared') as TaskDependency.predecessor_id, which
    produced dangling edges in the task graph.
    """
    tasks = convert_trie_to_tasks(_branch_trie(), "chilli", ())
    task_ids = {t.task_id for t in tasks}
    for task in tasks:
        for dep in task.dependencies:
            assert dep.predecessor_id in task_ids, f"dangling predecessor {dep.predecessor_id!r} on {task.task_id!r}"
    # Parent-child ordering must come from food states, not fake task deps.
    for task in tasks:
        assert task.dependencies == (), "prep tasks must not carry fake task deps"


def test_convert_cut_tasks_consume_parent_state() -> None:
    """Cut tasks consume the wash food state so the graph wires the edge."""
    tasks = convert_trie_to_tasks(_branch_trie(), "chilli", ())
    wash_task = next(t for t in tasks if t.task_id.startswith("prep_chilli_wash"))
    wash_state = wash_task.produces_states[0]
    cut_tasks = [t for t in tasks if "cut" in t.task_id]
    assert cut_tasks, "expected cut branches"
    for ct in cut_tasks:
        assert wash_state in ct.consumes_states


# =============================================================================
# demand_final_states_for_ingredient
# =============================================================================


def test_demand_final_states_map_to_terminal_states() -> None:
    """d1 (julienne) -> cut_julienned state; d2 (dice) -> cut_diced state."""
    states = demand_final_states_for_ingredient(_branch_trie(), "chilli")
    assert states == {
        "d1": "chilli:cut_julienned:shared",
        "d2": "chilli:cut_diced:shared",
    }


def test_single_operation_chain_final_state() -> None:
    root = PrepTrieNode(operation_key="root")
    chain = (PreparationOperation(operation="wash", quantity=Decimal(50), unit="g"),)
    insert_operation_chain(root, chain, "d1")
    assert demand_final_states_for_ingredient(root, "item") == {"d1": "item:wash:shared"}


# =============================================================================
# build_shared_prep_tasks
# =============================================================================


def test_identical_spec_merges_single_wash_and_cut() -> None:
    """Two recipes both need 100g diced onion -> one wash + one cut."""
    demands = (("r1", _demand(quantity="100")), ("r2", _demand(quantity="100")))
    result = build_shared_prep_tasks(demands)

    assert len(result.tasks) == 2, f"expected wash+cut, got {len(result.tasks)} tasks"
    wash_tasks = [t for t in result.tasks if t.task_id.startswith("prep_brown onion_wash")]
    cut_tasks = [t for t in result.tasks if "cut" in t.task_id]
    assert len(wash_tasks) == 1
    assert len(cut_tasks) == 1
    # 200g total washed once.
    assert "200" in wash_tasks[0].instruction
    # Both demands resolve to the same final state.
    assert set(result.demand_final_states.values()) == {"brown onion:cut_diced:shared"}


@pytest.mark.parametrize(
    "name",
    ("盐", "生抽", "蚝油", "火锅底料", "火锅料", "白芝麻", "料酒", "蛋清", "鸡翅中", "cornstarch"),
)
def test_condiments_do_not_get_inferred_wash_tasks(name: str) -> None:
    """Pantry ingredients are ready to use, even when represented as raw."""
    result = build_shared_prep_tasks((("r1", _demand(name=name, spec=None)),))
    assert result.tasks == ()
    assert result.demand_final_states == {}


def test_different_fresh_ingredients_share_one_batched_wash() -> None:
    """A single basin wash produces the states required by each dish."""
    result = build_shared_prep_tasks(
        (
            ("r1", _demand(name="cabbage", spec=None)),
            ("r2", _demand(name="carrot", spec=None)),
            ("r3", _demand(name="onion", spec=None)),
        )
    )

    assert len(result.tasks) == 1
    batch = result.tasks[0]
    assert batch.task_id == "prep_batch_wash_fresh_ingredients"
    assert set(batch.produces_states) == {
        "cabbage:wash:shared",
        "carrot:wash:shared",
        "onion:wash:shared",
    }
    assert batch.duration_minutes < 15
    assert batch.instruction == "[Prep] Rinse and drain together: cabbage, carrot, onion"


def test_branching_split_500_into_three_cuts() -> None:
    """500g onion -> washed once, then split into 100/200/200 cut branches."""
    demands = (
        ("r1", _demand(quantity="100", spec="diced")),
        ("r2", _demand(quantity="200", spec="sliced")),
        ("r3", _demand(quantity="200", spec="julienned")),
    )
    result = build_shared_prep_tasks(demands)

    wash_tasks = [t for t in result.tasks if t.task_id.startswith("prep_brown onion_wash")]
    cut_tasks = [t for t in result.tasks if "cut" in t.task_id]
    assert len(wash_tasks) == 1, "onion must be washed exactly once"
    assert len(cut_tasks) == 3, f"expected 3 cut branches, got {len(cut_tasks)}"
    # Conservation: wash quantity equals the sum of all branch quantities.
    assert "500" in wash_tasks[0].instruction
    assert len(result.demand_final_states) == 3


def test_different_cut_specs_produce_distinct_states() -> None:
    """Each demand's final state matches its own cut branch."""
    demands = (
        ("r1", _demand(quantity="100", spec="diced")),
        ("r2", _demand(quantity="200", spec="sliced")),
    )
    result = build_shared_prep_tasks(demands)
    assert result.demand_final_states["r1:0"] == "brown onion:cut_diced:shared"
    assert result.demand_final_states["r2:0"] == "brown onion:cut_sliced:shared"


def test_raw_protein_and_rte_never_merge() -> None:
    """Raw chicken and cooked (RTE) chicken must never share prep operations."""
    raw = _demand(name="chicken breast", quantity="200", spec="diced", input_state="raw")
    rte = _demand(name="chicken breast", quantity="300", spec="diced", input_state="cooked")
    result = build_shared_prep_tasks((("r1", raw), ("r2", rte)))

    cut_tasks = [t for t in result.tasks if "cut" in t.task_id]
    assert len(cut_tasks) == 2, "raw and RTE chicken cuts must be isolated"
    quantities = "|".join(t.instruction for t in cut_tasks)
    assert "200" in quantities and "300" in quantities
    assert "500" not in quantities, "quantities must not be merged into one shared task"


def test_raw_protein_is_not_inferred_as_a_washable_ingredient() -> None:
    """Raw chicken must never enter a generic vegetable washing batch."""
    chicken = _demand(name="chicken breast", quantity="200", spec="diced")
    lettuce = _demand(name="lettuce", quantity="150", spec="diced")
    result = build_shared_prep_tasks((("r1", chicken), ("r2", lettuce)))

    wash_tasks = [t for t in result.tasks if "wash" in t.task_id]
    assert len(wash_tasks) == 1
    assert "lettuce:wash:shared" in wash_tasks[0].produces_states
    assert "chicken breast:wash:shared" not in wash_tasks[0].produces_states


def test_quantity_conservation_total_matches_sum() -> None:
    """Total prep quantity per ingredient equals the sum of demand quantities."""
    demands = (
        ("r1", _demand(quantity="100", spec="diced")),
        ("r2", _demand(quantity="200", spec="sliced")),
        ("r3", _demand(quantity="200", spec="julienned")),
    )
    result = build_shared_prep_tasks(demands)
    wash_tasks = [t for t in result.tasks if t.task_id.startswith("prep_brown onion_wash")]
    assert len(wash_tasks) == 1
    # Instruction encodes the aggregated quantity; 500 = 100+200+200.
    assert "500" in wash_tasks[0].instruction


def test_empty_demands_returns_empty() -> None:
    result = build_shared_prep_tasks(())
    assert result.tasks == ()
    assert result.demand_final_states == {}


# =============================================================================
# merge_preparation_node integration
# =============================================================================


class _FakeRuntime:
    def __init__(self, context: object = None) -> None:
        self.context = context


def _recipe(recipe_id: str, name: str, spec: str, quantity: str) -> RecipeIR:
    step = RecipeStep(
        step_number=1,
        instruction="Stir-fry for 5 minutes",
        pattern="stir_fry",
        active_duration_minutes=5,
    )
    return RecipeIR(
        recipe_id=recipe_id,
        dish_name=name,
        original_servings=Decimal(2),
        target_servings=Decimal(2),
        source_language="en",
        ingredients=(_demand(name=name, quantity=quantity, spec=spec),),
        steps=(step,),
    )


def _state(recipes: tuple[RecipeIR, ...]) -> dict[str, object]:
    request = GeneratePlanRequest(
        request_id="req-shared-prep",
        user_id="u",
        recipes=tuple(RecipeInput(recipe_id=r.recipe_id, text="Cook.", target_servings=Decimal(2)) for r in recipes),
    )
    return {"request": request, "parsed_recipes": recipes}


@pytest.mark.asyncio
async def test_shared_prep_produces_non_empty_prep_tasks(monkeypatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "shared_prep_enabled", True)
    recipes = (
        _recipe("r1", "brown onion", "diced", "100"),
        _recipe("r2", "brown onion", "sliced", "200"),
    )
    result = await merge_preparation_node(_state(recipes), _FakeRuntime())

    assert "prep_tasks" in result
    assert result["prep_tasks"], "P2-01: prep_tasks must no longer be fixed empty"
    assert "prep_observations" in result
    assert result["prep_observations"], "expected merge observations"
    prep_ids = {task.task_id for task in result["prep_tasks"]}
    assert {"prep_gather_all_ingredients", "prep_mise_en_place_complete"} <= prep_ids
    instructions = {task.task_id: task.instruction for task in result["prep_tasks"]}
    assert instructions["prep_gather_all_ingredients"] == (
        "[Mise en place] Gather all ingredients, seasonings, and required tools"
    )
    assert instructions["prep_mise_en_place_complete"] == (
        "[Mise en place] Confirm that washing, cutting, and seasoning portions are ready"
    )


@pytest.mark.asyncio
async def test_flag_off_falls_back_to_empty_prep_tasks(monkeypatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "shared_prep_enabled", False)
    recipes = (
        _recipe("r1", "brown onion", "diced", "100"),
        _recipe("r2", "brown onion", "sliced", "200"),
    )
    result = await merge_preparation_node(_state(recipes), _FakeRuntime())

    assert result["prep_tasks"] == ()


@pytest.mark.asyncio
async def test_recipe_first_task_consumes_prep_final_state(monkeypatch) -> None:
    from cooking_plan_agent.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "shared_prep_enabled", True)
    recipes = (_recipe("r1", "brown onion", "diced", "100"),)
    result = await merge_preparation_node(_state(recipes), _FakeRuntime())

    recipe_tasks = result["recipe_tasks"]
    assert recipe_tasks, "expected decomposed recipe tasks"
    first = recipe_tasks[0]
    assert any("brown onion:cut_diced" in s for s in first.consumes_states), (
        "recipe first task must consume the prep final state"
    )
    assert "prep_mise_en_place_complete" in {dep.predecessor_id for dep in first.dependencies}
