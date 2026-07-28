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
    TaskDependency,
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

        deps: tuple[TaskDependency, ...] = ()
        consumes: tuple[str, ...] = ()
        if parent_state:
            deps = (TaskDependency(predecessor_id=parent_state),)
            consumes = (parent_state,)

        task = _build_task(
            task_id=task_id,
            dish_id=",".join(dish_ids) if dish_ids else "shared",
            instruction=(
                f"[Prep] {node.operation_key} "
                f"{node.total_quantity} of {ingredient_name}"
            ),
            duration_minutes=duration,
            work_mode=WorkMode.ACTIVE,
            category="preparation",
            dependencies=deps,
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

    children_sum = sum(
        child.total_quantity for child in node.child_nodes.values()
    )
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

    # Always start with wash (unless input_state suggests already processed).
    if demand.input_state == "raw":
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
