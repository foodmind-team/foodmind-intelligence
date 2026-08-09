"""Preparation trie — shared ingredient-prep merging
This module handles:
- PreparationOperation: a single atomic step in an ingredient's prep chain
- PrepTrieNode: prefix-tree node for merging identical prep operations across dishes
- Trie conversion: turn the merged trie back into schedulable CookingTasks
- Quantity-conservation verification
- Inferring prep chains from IngredientDemand records.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from cooking_plan_agent.domain.enums import WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    IngredientDemand,
    ResourceNeed,
    StrictModel,
)
from cooking_plan_agent.normalisation.errors import InvalidQuantityError

# Lazy imports to avoid circular dependency at type-check time,
# but at runtime these are always available because decompose.py
# is loaded first (prep_trie imports from it).
from cooking_plan_agent.preparation.decompose import _build_task, format_food_state

# ============================================================================
# 6.4  PreparationOperation — operation-chain model
# ============================================================================


class PreparationOperation(StrictModel):
    """A single operation in an ingredient's preparation chain.

    Example: each ingredient demand maps to an ordered chain of
    operations (wash → peel → deseed → julienne, etc.).  The operation
    vocabulary is controlled — store user-facing phrases separately if needed.

    Attributes
    ----------
    operation
        Controlled vocabulary key (e.g. ``"wash"``, ``"dice"``, ``"portion"``).
    specification
        Optional user-facing detail (e.g. ``"fine dice"``, ``"thick slice"``).
        Not used for trie merging — only ``operation`` keys determine shareability.
    quantity
        The ingredient quantity at this operation stage.
    unit
        Unit of measure (e.g. ``"g"``, ``"piece"``).
    resource_needs
        Equipment required for this operation.
    """

    operation: str
    """Controlled vocabulary key (e.g. 'wash', 'dice', 'portion')."""

    specification: str | None = None
    """Optional user-facing detail (e.g. 'fine dice').  Not used for merging."""

    quantity: Decimal
    """Ingredient quantity at this stage."""

    unit: str
    """Unit of measure (e.g. 'g', 'piece')."""

    resource_needs: tuple[ResourceNeed, ...] = ()
    """Equipment required for this operation."""


# ============================================================================
# 6.5  PrepTrieNode — prefix tree for shared preparation merging
# ============================================================================


@dataclass
class PrepTrieNode:
    """A node in the preparation prefix tree (trie).

    Each node represents one operation in the ingredient-prep chain.
    Identical operations on the same ingredient are merged by aggregating
    quantities.  Branches occur when subsequent operations diverge
    (e.g. one ingredient is julienned, another is diced).

    Example: reuse an existing child only when the operation is
    semantically shareable.  Not every identical word is shareable —
    raw-protein and ready-to-eat vegetable washing may require separate
    resources or sanitation boundaries.

    Attributes
    ----------
    operation_key
        Stable key derived from the operation name (e.g. ``"wash"``,
        ``"cut:julienne"``).
    total_quantity
        Aggregated quantity across all demands sharing this node.
    child_nodes
        Child operations (next step in each preparation chain).
    demand_ids
        IDs of demands that traverse this node (for traceability).
    """

    operation_key: str
    """Stable operation key (e.g. 'wash', 'cut:julienne')."""

    total_quantity: Decimal = Decimal(0)
    """Aggregated quantity from all demands at this node."""

    child_nodes: dict[str, "PrepTrieNode"] = field(default_factory=dict)
    """Child operations keyed by operation_key."""

    demand_ids: set[str] = field(default_factory=set)
    """Demand IDs that pass through this node."""


# ============================================================================
# 6.5  Trie insertion algorithm
# ============================================================================


def _operation_stable_key(op: PreparationOperation, ingredient_name: str) -> str:
    """Derive a stable key for trie merging.

    Simple operations use their name directly (e.g. ``"wash"``).
    Cutting operations combine category + specification for branching
    (e.g. ``"cut:julienne"`` vs ``"cut:dice"``).

    The key MUST be deterministic — two operations that are semantically
    shareable must produce the same key, and semantically different
    operations must produce different keys.
    """
    cutting_ops = {"julienne", "slice", "dice", "mince", "chop"}
    if op.operation in cutting_ops:
        spec = op.specification or op.operation
        return f"cut:{spec}"
    return op.operation


def _can_share_operation(
    op_a: PreparationOperation,
    op_b: PreparationOperation,
) -> bool:
    """Check whether two operations at the same position in different
    chains can be merged (shared preparation).

    Handbook 6.5: not every identical word is shareable.  Raw-protein
    and RTE-vegetable washing must NOT be merged.

    In MVP, we use a simple rule: operations must have the same
    ``operation`` key AND the same specification (if any).
    """
    if op_a.operation != op_b.operation:
        return False
    return op_a.specification == op_b.specification


def insert_operation_chain(
    root: PrepTrieNode,
    chain: tuple[PreparationOperation, ...],
    demand_id: str,
) -> None:
    """Insert a preparation operation chain into the prefix tree.

    insertion algorithm:
    1. Start at the ingredient's root node.
    2. For each operation, derive a stable key.
    3. Reuse an existing child only when the operation is semantically shareable.
    4. Add the demand quantity to the traversed node.
    5. Record the demand ID.
    6. Branch when specifications diverge.

    Each operation's quantity is accumulated on the node that represents
    that operation (the child, not the parent).  The root node carries
    no ingredient quantity — it only accumulates demand_ids for traceability.

    Args:
        root: The trie root node (typically ``operation_key="root"``).
        chain: Ordered preparation operations for one ingredient demand.
        demand_id: Unique identifier for the demand (for traceability).
    """
    current = root
    for op in chain:
        key = _operation_stable_key(op, "")

        if key in current.child_nodes:
            # Existing child — merge quantities and demand IDs.
            child = current.child_nodes[key]
        else:
            # New branch — create a child node.
            child = PrepTrieNode(operation_key=key)
            current.child_nodes[key] = child

        child.total_quantity += op.quantity
        child.demand_ids.add(demand_id)

        current = child

    # Record demand ID on root for traceability.
    root.demand_ids.add(demand_id)


# ============================================================================
# 6.6  Convert trie back to tasks
# ============================================================================


def convert_trie_to_tasks(
    root: PrepTrieNode,
    ingredient_name: str,
    dish_ids: tuple[str, ...],
    setup_minutes: Decimal = Decimal(5),
    minutes_per_base_unit: Decimal = Decimal("0.01"),
) -> tuple[CookingTask, ...]:
    """Convert a preparation prefix tree into a list of CookingTasks.

    1. Create a task with the node's aggregated quantity.
    2. Calculate duration using the batch time policy.
    3. Add parent-task dependency.
    4. Produce one shared state for children.
    5. At a branch, create explicit portion/split information.

    duration = setup_minutes + minutes_per_base_unit × Q

    Args:
        root: The populated preparation trie root.
        ingredient_name: Canonical ingredient name for food-state labels.
        dish_ids: All dish IDs that contribute to this trie (for batch_key).
        setup_minutes: Fixed setup time per batch (τ_setup).
        minutes_per_base_unit: Time per unit quantity (ρ).

    Returns:
        A flat tuple of CookingTask instances in DFS pre-order.
    """
    tasks: list[CookingTask] = []

    def _dfs(
        node: PrepTrieNode,
        parent_state: str | None,
        seq: list[int],
    ) -> str:
        """DFS traversal: build tasks and return the state this node produces."""
        if not node.operation_key or node.operation_key == "root":
            # Root node: recurse into children.
            for child in node.child_nodes.values():
                _dfs(child, parent_state, seq)
            return parent_state or ""

        # Build food-state for this node.
        state_label = node.operation_key.replace(":", "_")
        state = format_food_state(ingredient_name, state_label)
        seq.append(1)
        task_id = f"prep_{ingredient_name}_{node.operation_key}_{sum(seq)}"

        # Duration via batch formula: τ_setup + ρ × Q
        duration = int(setup_minutes + minutes_per_base_unit * node.total_quantity)
        duration = max(duration, 1)  # Minimum 1 minute

        # P2-01: parent-child ordering is expressed ONLY through food-state
        # consume/produce pairs — the graph builder (build_task_graph) turns
        # produces_states → consumes_states into real edges. Writing the
        # parent food-state as a TaskDependency.predecessor_id would create a
        # dangling edge (food states are not task IDs).
        consumes: tuple[str, ...] = (parent_state,) if parent_state else ()

        task = _build_task(
            task_id=task_id,
            dish_id=",".join(dish_ids) if dish_ids else "shared",
            instruction=(f"[Prep] {node.operation_key} {node.total_quantity} of {ingredient_name}"),
            duration_minutes=duration,
            work_mode=WorkMode.ACTIVE,
            category="preparation",
            dependencies=(),
            consumes_states=consumes,
            produces_states=(state,),
            batch_key=f"prep_{ingredient_name}",
        )
        tasks.append(task)

        # Recurse into children with this node's state as parent.
        for child in node.child_nodes.values():
            _dfs(child, state, seq)

        return state

    _dfs(root, None, [])
    return tuple(tasks)


# ============================================================================
# 6.7  Quantity-conservation verification
# ============================================================================


def verify_quantity_conservation(
    root: PrepTrieNode,
) -> None:
    """Verify the quantity-conservation invariant across the trie.

    Q_parent = Σ Q_child + Q_consumed_at_parent

    In the MVP trie, quantities are managed as: each operation carries
    the total quantity that passes through its node.  The invariant is
    that child nodes' quantities sum to the parent's quantity (because
    we have no 'consumed at parent' leftover in MVP).

    Raises:
        InvalidQuantityError: If conservation is violated at any node.
    """
    _verify_node(root)


def _verify_node(node: PrepTrieNode) -> None:
    """Recursively check quantity conservation."""
    if not node.child_nodes:
        return

    children_sum = sum(child.total_quantity for child in node.child_nodes.values())
    # Root node is exempt — it accumulates totals from all chains.
    if node.operation_key != "root" and children_sum != node.total_quantity:
        raise InvalidQuantityError(
            f"Quantity conservation violated at node {node.operation_key!r}: "
            f"parent={node.total_quantity}, children_sum={children_sum}"
        )

    for child in node.child_nodes.values():
        _verify_node(child)


# ============================================================================
# Infer preparation chains from IngredientDemand
# ============================================================================

# Pantry seasonings and cooking media are ready to use; treating their generic
# raw input state as produce and scheduling a sink wash is both nonsensical
# and makes a multi-dish schedule much longer. This is intentionally a
# conservative keyword allow-list: items not matched here retain the existing
# wash-first behaviour until a proper ingredient taxonomy is available.
_NON_WASHABLE_INGREDIENT_KEYWORDS = (
    "salt",
    "sugar",
    "soy sauce",
    "vinegar",
    "oil",
    "starch",
    "flour",
    "pepper",
    "seasoning",
    "sauce",
    "stock cube",
    "盐",
    "糖",
    "生抽",
    "老抽",
    "蚝油",
    "醋",
    "油",
    "淀粉",
    "胡椒",
    "鸡精",
    "鸡粉",
    "味精",
    "火锅底料",
    "火锅料",
    "豆瓣酱",
    "辣椒面",
    "白芝麻",
    "料酒",
    "米酒",
    "酒糟",
    "五香粉",
    "egg",
    "chicken",
    "beef",
    "pork",
    "fish",
    "shrimp",
    "prawn",
    "lamb",
    "turkey",
    "duck",
    "sausage",
    "蛋清",
    "鸡翅",
    "鲜虾",
    "基围虾",
    "蟹",
    "排骨",
    "腊肠",
)


def _is_washable_ingredient(demand: IngredientDemand) -> bool:
    """Return whether a raw ingredient should receive an inferred wash step."""
    name = demand.canonical_name.lower()
    return not any(keyword in name for keyword in _NON_WASHABLE_INGREDIENT_KEYWORDS)


def _infer_prep_chain(demand: IngredientDemand) -> tuple[PreparationOperation, ...]:
    """Infer a basic preparation chain from an IngredientDemand.

    Derives operations from ``preparation_spec`` and ``input_state``.
    For MVP, uses a simple keyword-based mapping.  Production systems
    should use a catalogue of vetted prep chains.

    Example:
        demand with preparation_spec="diced" →
        (wash → peel(if applicable) → dice → portion)
    """
    ops: list[PreparationOperation] = []

    # Start with wash only for raw, washable ingredients. Condiments and
    # cooking media are supplied ready to use even though the MVP model marks
    # otherwise-unclassified inputs as raw.
    if demand.input_state == "raw" and _is_washable_ingredient(demand):
        ops.append(
            PreparationOperation(
                operation="wash",
                quantity=demand.quantity,
                unit=demand.unit,
                resource_needs=(ResourceNeed(quantity=1, resource_type="sink"),),
            )
        )

    # Infer cutting operation from preparation_spec.
    if demand.preparation_spec:
        spec_lower = demand.preparation_spec.lower()
        # Map common specs to cutting operations.
        cut_map = {
            "diced": "dice",
            "dice": "dice",
            "julienned": "julienne",
            "julienne": "julienne",
            "sliced": "slice",
            "slice": "slice",
            "minced": "mince",
            "mince": "mince",
            "chopped": "chop",
            "chop": "chop",
        }
        for keyword, op_name in cut_map.items():
            if keyword in spec_lower:
                ops.append(
                    PreparationOperation(
                        operation=op_name,
                        specification=demand.preparation_spec,
                        quantity=demand.quantity,
                        unit=demand.unit,
                        resource_needs=(
                            ResourceNeed(quantity=1, resource_type="cutting_board"),
                            ResourceNeed(quantity=1, resource_type="knife"),
                        ),
                    )
                )
                break

    return tuple(ops)


# ============================================================================
# P2-01  Shared-preparation merging for the main workflow
# ============================================================================

# Raw proteins must NEVER share preparation operations with ready-to-eat
# items, even when the operation word is identical (P2-01 D2). Matching is
# keyword-based on the canonical ingredient name; production systems should
# source this from a vetted ingredient catalogue.
_RAW_PROTEIN_KEYWORDS = (
    "chicken",
    "beef",
    "pork",
    "fish",
    "shrimp",
    "prawn",
    "lamb",
    "turkey",
    "duck",
    "salmon",
    "tuna",
    "ham",
    "sausage",
    "bacon",
    "mince",
    "meat",
    "poultry",
    "seafood",
)


def safety_class_for_ingredient(canonical_name: str, input_state: str) -> str:
    """Categorise an ingredient for prep-sharing safety isolation (P2-01 D2).

    Returns ``"raw_protein"`` when the ingredient is an uncooked animal
    protein, otherwise ``"rte"`` (ready-to-eat / general). Raw-protein and
    RTE demands of the same ingredient build into separate tries so their
    operations never merge.
    """
    name = canonical_name.lower()
    if input_state == "raw" and any(kw in name for kw in _RAW_PROTEIN_KEYWORDS):
        return "raw_protein"
    return "rte"


def demand_final_states_for_ingredient(
    root: PrepTrieNode,
    ingredient_name: str,
) -> dict[str, str]:
    """Map each demand_id to the food state its terminal prep node produces.

    A demand's final state is the state of the deepest (leaf) node it passes
    through. Leaf nodes carry the demand_ids of the chains that end there.
    The state label follows ``convert_trie_to_tasks`` exactly (``:`` in the
    operation key is replaced with ``_``).

    Args:
        root: The populated trie for one ingredient (and safety class).
        ingredient_name: Canonical ingredient name used in state labels.

    Returns:
        ``{demand_id: final_food_state}``.
    """
    final: dict[str, str] = {}

    def _dfs(node: PrepTrieNode) -> None:
        if not node.operation_key or node.operation_key == "root":
            for child in node.child_nodes.values():
                _dfs(child)
            return
        if node.child_nodes:
            for child in node.child_nodes.values():
                _dfs(child)
            return
        state = format_food_state(ingredient_name, node.operation_key.replace(":", "_"))
        for demand_id in node.demand_ids:
            final[demand_id] = state

    _dfs(root)
    return final


@dataclass(frozen=True)
class SharedPrepResult:
    """Output of shared-preparation merging (P2-01).

    Attributes
    ----------
    tasks
        Merged preparation tasks across all ingredient demands.
    demand_final_states
        ``demand_id -> final food state`` so the workflow can wire the
        consuming recipe task (``recipe_id:index`` keys).
    observations
        Human-readable summaries of merge/branch/isolate decisions for
        observability (logged and stored in workflow state).
    """

    tasks: tuple[CookingTask, ...]
    demand_final_states: dict[str, str]
    observations: tuple[str, ...]


def _batch_wash_tasks(tasks: list[CookingTask]) -> list[CookingTask]:
    """Combine independent fresh-ingredient washes into one basin operation.

    All wash candidates entering this function are non-protein ingredients:
    raw proteins and pantry items are excluded by the preparation inference.
    A single task produces every original food state, so downstream recipe
    dependencies remain intact while the user performs one concentrated wash
    and drain operation instead of repeatedly returning to the sink.
    """
    wash_tasks = [
        task
        for task in tasks
        if task.task_id.startswith("prep_") and "_wash_" in task.task_id and not task.consumes_states
    ]
    if len(wash_tasks) < 2:
        return tasks

    names = [task.task_id.removeprefix("prep_").rsplit("_wash_", 1)[0] for task in wash_tasks]
    produced_states = tuple(state for task in wash_tasks for state in task.produces_states)
    # One large basin pass: five minutes of setup/rinsing plus roughly one
    # minute for every three additional ingredient groups, capped at 15.
    duration = min(15, 5 + (len(wash_tasks) - 1 + 2) // 3)
    batch_task = wash_tasks[0].model_copy(
        update={
            "task_id": "prep_batch_wash_fresh_ingredients",
            "dish_id": "shared",
            "instruction": f"[Prep] Rinse and drain together: {', '.join(names)}",
            "duration_minutes": duration,
            "resources": (ResourceNeed(quantity=1, resource_type="sink"),),
            "produces_states": produced_states,
            "batch_key": "prep_batch_wash_fresh_ingredients",
        }
    )
    wash_ids = {task.task_id for task in wash_tasks}
    return [task for task in tasks if task.task_id not in wash_ids] + [batch_task]


def build_shared_prep_tasks(
    demands: tuple[tuple[str, IngredientDemand], ...],
) -> SharedPrepResult:
    """Merge identical preparation chains across recipes via prefix tries.

    Workflow step (P2-01):
    1. Group demands by (canonical ingredient, safety class) — raw proteins
       never merge with RTE items (D2), and different ingredients never merge.
    2. Build one trie per group; identical operation prefixes aggregate
       quantities; divergent cuts branch.
    3. Verify quantity conservation (D1) — a violation raises
       ``InvalidQuantityError`` and the workflow must NOT build a task graph.
    4. Convert each trie to prep tasks; produce per-demand final food states.

    Args:
        demands: ``(recipe_id, demand)`` pairs. Demand IDs are derived as
            ``"{recipe_id}:{index}"`` where index is the demand's ordinal
            within its recipe.

    Returns:
        SharedPrepResult with merged tasks, final-state mapping, and
        observations.

    Raises:
        InvalidQuantityError: If merged quantity does not equal the sum of
            the input demand quantities.
    """
    # 1. Group by (canonical_name, safety_class).
    groups: dict[tuple[str, str], list[tuple[str, str, IngredientDemand]]] = {}
    recipe_counts: dict[str, int] = {}
    for recipe_id, demand in demands:
        idx = recipe_counts.get(recipe_id, 0)
        recipe_counts[recipe_id] = idx + 1
        demand_id = f"{recipe_id}:{idx}"
        key = (
            demand.canonical_name,
            safety_class_for_ingredient(demand.canonical_name, demand.input_state),
        )
        groups.setdefault(key, []).append((demand_id, recipe_id, demand))

    tasks: list[CookingTask] = []
    demand_final_states: dict[str, str] = {}
    observations: list[str] = []

    # Track raw/RTE co-occurrence for the same ingredient (isolation report).
    class_by_ingredient: dict[str, set[str]] = {}
    for (name, sclass), _group in groups.items():
        class_by_ingredient.setdefault(name, set()).add(sclass)

    for (name, sclass), group in groups.items():
        root = PrepTrieNode(operation_key="root")
        group_total = Decimal(0)
        dish_ids: list[str] = []
        for demand_id, recipe_id, demand in group:
            chain = _infer_prep_chain(demand)
            if not chain:
                continue
            group_total += demand.quantity
            if recipe_id not in dish_ids:
                dish_ids.append(recipe_id)
            insert_operation_chain(root, chain, demand_id)

        if not root.child_nodes:
            continue  # No preparable operations for this demand group.

        # 3. Quantity conservation (D1): aggregated == sum of demands.
        verify_quantity_conservation(root)
        # The root node only tracks demand_ids (insert_operation_chain never
        # aggregates on it); the first-level nodes carry the per-ingredient
        # totals, which must equal the sum of every demand quantity.
        trie_total = sum(child.total_quantity for child in root.child_nodes.values())
        if trie_total != group_total:
            raise InvalidQuantityError(
                f"Quantity conservation violated for {name!r}: trie total={trie_total}, demand sum={group_total}"
            )

        # 4. Convert trie → tasks and record final states + observations.
        tasks.extend(convert_trie_to_tasks(root, name, tuple(dish_ids)))
        demand_final_states.update(demand_final_states_for_ingredient(root, name))

        if len(group) > 1:
            observations.append(f"merged {len(group)} demand(s) for {name} [{sclass}]")
        if any("cut:" in k for k in root.child_nodes):
            observations.append(f"branching prep for {name} [{sclass}]")
        if len(class_by_ingredient.get(name, set())) > 1:
            observations.append(f"isolated raw/RTE prep for {name}")

    batched_tasks = _batch_wash_tasks(tasks)
    if len(batched_tasks) < len(tasks):
        observations.append(f"batched {len(tasks) - len(batched_tasks) + 1} fresh-ingredient washes")

    return SharedPrepResult(
        tasks=tuple(batched_tasks),
        demand_final_states=demand_final_states,
        observations=tuple(observations),
    )
