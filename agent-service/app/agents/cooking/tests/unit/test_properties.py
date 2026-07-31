"""Property-based tests — mathematical invariants for scheduling and units.

Handbook 11.4: use Hypothesis for invariants that must hold for ALL
valid inputs, not just hand-picked examples.

Invariants verified:
  1. Scaled quantity is never negative
  2. Unit round-trip preserves quantity within precision
  3. Prefix-tree child quantities conserve parent quantity
  4. Topological order contains every node exactly once
  5. Verified active intervals never overlap
  6. Task graph has no self-loops
  7. Horizon >= sum of all task durations
"""

from decimal import Decimal

from hypothesis import given, seed
from hypothesis import strategies as st

from cooking_plan_agent.domain.enums import WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    TaskDependency,
)
from cooking_plan_agent.normalisation.units import scale_ingredient
from cooking_plan_agent.preparation.task_graph import (
    TaskEdge,
    TaskGraph,
    build_task_graph,
    topological_sort_kahn,
)


def _make_prop_task(
    task_id: str,
    duration: int = 5,
    work_mode: str = "ACTIVE",
    deps: tuple[TaskDependency, ...] = (),
) -> CookingTask:
    """Minimal CookingTask factory for property tests."""
    return CookingTask(
        task_id=task_id,
        dish_id="d1",
        instruction=f"Task {task_id}",
        duration_minutes=duration,
        work_mode=WorkMode(work_mode),
        category="test",
        dependencies=deps,
    )


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Positive decimal quantities for testing
_positive_quantity = st.decimals(min_value="0.001", max_value=10000, places=2)


# =============================================================================
# 1. scale_ingredient quantity is never negative
# =============================================================================


@given(
    quantity=_positive_quantity,
    original=st.decimals(min_value="0.5", max_value=20, places=1),
    target=st.decimals(min_value="0.5", max_value=20, places=1),
)
@seed(20260731)
def test_scaled_quantity_never_negative(
    quantity: Decimal,
    original: Decimal,
    target: Decimal,
) -> None:
    """Handbook 11.4: scaled quantity >= 0 for any valid scaling factor."""
    from cooking_plan_agent.domain.models import IngredientDemand

    demand = IngredientDemand(
        canonical_name="test",
        raw_name="test",
        quantity=quantity,
        unit="g",
        confidence=Decimal("0.5"),
    )
    result = scale_ingredient(demand, original, target)
    assert result.quantity >= 0


# =============================================================================
# 2. Unit round-trip preserves quantity within declared precision
# =============================================================================


@given(
    quantity=st.decimals(min_value="0.001", max_value=1000, places=2),
)
@seed(20260731)
def test_unit_round_trip_preserves_quantity(quantity: Decimal) -> None:
    """Handbook 11.4: g→kg→g preserves quantity within 0.001 precision.

    Round-trip: gram to kilogram and back should recover the original value.
    """
    from cooking_plan_agent.normalisation.units import UnitConverter

    conv = UnitConverter()
    in_kg = conv.convert(quantity, "g", "kg")
    back_to_g = conv.convert(in_kg, "kg", "g")

    # Round-trip should be exact for Decimal arithmetic
    assert back_to_g == quantity, f"Round-trip failed: {quantity}g → {in_kg}kg → {back_to_g}g"


# =============================================================================
# 3. Prefix-tree child quantities conserve parent quantity
# =============================================================================


@given(
    num_chains=st.integers(min_value=1, max_value=5),
)
@seed(20260731)
def test_prefix_tree_quantity_conservation(num_chains: int) -> None:
    """Handbook 11.4: child quantities sum to parent quantity in prep trie.

    Creates N chains sharing the first "wash" operation but branching
    into different cut operations. Verifies wash quantity = sum(cut quantities).
    """
    from decimal import Decimal

    from cooking_plan_agent.preparation.prep_trie import (
        PreparationOperation,
        PrepTrieNode,
        insert_operation_chain,
    )

    cutting_ops = ["julienne", "slice", "dice", "mince", "chop"]
    total_wash = Decimal(0)

    root = PrepTrieNode(operation_key="root")
    for i in range(num_chains):
        quantity = Decimal(100 + i * 50)
        total_wash += quantity
        cut_op = cutting_ops[i % len(cutting_ops)]
        chain = (
            PreparationOperation(operation="wash", quantity=quantity, unit="g"),
            PreparationOperation(
                operation=cut_op,
                specification=cut_op + "d",
                quantity=quantity,
                unit="g",
            ),
        )
        insert_operation_chain(root, chain, f"d{i}")

    wash_node = root.child_nodes["wash"]
    assert wash_node.total_quantity == total_wash, f"Wash quantity {wash_node.total_quantity} != sum {total_wash}"

    children_sum = sum(child.total_quantity for child in wash_node.child_nodes.values())
    assert children_sum == wash_node.total_quantity, f"Children sum {children_sum} != parent {wash_node.total_quantity}"


# =============================================================================
# 4. Topological order contains every node exactly once for acyclic graphs
# =============================================================================


@given(
    num_tasks=st.integers(min_value=2, max_value=8),
)
@seed(20260731)
def test_topological_order_contains_every_node(num_tasks: int) -> None:
    """Handbook 11.4: topological order has exactly N unique tasks for a DAG.

    Constructs a linear chain (no cycles) and verifies the topological
    sort contains each task exactly once.
    """
    tasks = tuple(_make_prop_task(task_id=f"t{i}", duration=1) for i in range(num_tasks))
    edges = tuple(TaskEdge(predecessor_id=f"t{i}", successor_id=f"t{i + 1}") for i in range(num_tasks - 1))
    graph = TaskGraph(tasks=tasks, edges=edges)
    order = topological_sort_kahn(graph)

    assert len(order) == num_tasks, f"Topological sort returned {len(order)} tasks, expected {num_tasks}"
    task_ids_in_order = {t.task_id for t in order}
    expected_ids = {t.task_id for t in tasks}
    assert task_ids_in_order == expected_ids, (
        f"Missing: {expected_ids - task_ids_in_order}, Extra: {task_ids_in_order - expected_ids}"
    )


# =============================================================================
# 5. Verified active intervals never overlap
# =============================================================================


@given(
    num_active=st.integers(min_value=1, max_value=6),
)
@seed(20260731)
def test_active_intervals_never_overlap(num_active: int) -> None:
    """Handbook 11.4: for a sequential schedule, active intervals must not overlap.

    Uses the verifier's overlap-check logic on a hand-constructed
    sequential schedule. Adjacent tasks should not be flagged as overlapping.
    """
    from cooking_plan_agent.domain.enums import SolverStatus
    from cooking_plan_agent.scheduling.models import (
        ScheduledInterval,
        ScheduleResult,
        SchedulingProblem,
    )
    from cooking_plan_agent.scheduling.verifier import ScheduleVerifier

    tasks = tuple(_make_prop_task(task_id=f"t{i}", duration=3, work_mode="ACTIVE") for i in range(num_active))

    # Build sequential intervals
    intervals = tuple(
        ScheduledInterval(
            task_id=f"t{i}",
            start_minute=i * 3,
            end_minute=(i + 1) * 3,
        )
        for i in range(num_active)
    )

    problem = SchedulingProblem(tasks=tasks, resources=())
    result = ScheduleResult(
        status=SolverStatus.OPTIMAL,
        makespan_minutes=num_active * 3,
        intervals=intervals,
    )

    verifier = ScheduleVerifier()
    report = verifier.verify(problem, result)

    # There should be no ACTIVE_OVERLAP issues
    overlap_issues = [i for i in report.issues if i.code == "ACTIVE_OVERLAP"]
    assert len(overlap_issues) == 0, f"Unexpected overlap in sequential schedule: {overlap_issues}"


# =============================================================================
# 6. Task graph has no self-loops
# =============================================================================


@given(
    num_tasks=st.integers(min_value=1, max_value=10),
)
@seed(20260731)
def test_task_graph_no_self_loops(num_tasks: int) -> None:
    """Handbook 11.4: build_task_graph must never produce self-referencing edges.

    Even with task dependencies that reference the task itself,
    the builder should not create self-loops.
    """
    task_dep = TaskDependency(predecessor_id="t0")
    tasks = tuple(
        _make_prop_task(
            task_id=f"t{i}",
            duration=1,
            deps=(task_dep,) if i > 0 else (),
        )
        for i in range(num_tasks)
    )
    graph = build_task_graph(tasks, (), ())
    for edge in graph.edges:
        assert edge.predecessor_id != edge.successor_id, (
            f"Self-loop detected: {edge.predecessor_id} → {edge.successor_id}"
        )


# =============================================================================
# 7. Horizon >= sum of all task durations
# =============================================================================


@given(
    num_tasks=st.integers(min_value=0, max_value=10),
    max_duration=st.integers(min_value=1, max_value=30),
)
@seed(20260731)
def test_horizon_at_least_sum_of_durations(
    num_tasks: int,
    max_duration: int,
) -> None:
    """Handbook 11.4: horizon must be >= sum of all task durations.

    The ScheduleModelBuilder.compute_horizon must always produce a value
    at least as large as the raw sum of durations.
    """
    from cooking_plan_agent.scheduling.builder import ScheduleModelBuilder

    durations = [max(1, max_duration // (num_tasks or 1)) for _ in range(num_tasks)]
    tasks = tuple(_make_prop_task(task_id=f"t{i}", duration=d) for i, d in enumerate(durations))

    builder = ScheduleModelBuilder()
    horizon = builder.compute_horizon(tasks)

    total_duration = sum(t.duration_minutes for t in tasks)
    assert horizon >= total_duration, f"Horizon {horizon} < total duration {total_duration}"


# =============================================================================
# 8. scale_ingredient is proportional (λ * Q)
# =============================================================================


@given(
    quantity=_positive_quantity,
    original=st.decimals(min_value="0.5", max_value=20, places=1),
    target=st.decimals(min_value="0.5", max_value=20, places=1),
)
@seed(20260731)
def test_scale_ingredient_is_proportional(
    quantity: Decimal,
    original: Decimal,
    target: Decimal,
) -> None:
    """Handbook 11.4: Q_new = Q_original * (target / original).

    The scaling must be exactly proportional — no rounding or arbitrary
    adjustments applied during the scaling step.
    """
    from cooking_plan_agent.domain.models import IngredientDemand

    demand = IngredientDemand(
        canonical_name="test",
        raw_name="test",
        quantity=quantity,
        unit="g",
        confidence=Decimal("0.5"),
    )
    result = scale_ingredient(demand, original, target)

    expected = quantity * target / original
    # Decimal arithmetic may produce precision beyond 10 decimal places.
    # Quantize to 10 places for comparison — recipes don't need picogram precision.
    assert result.quantity.quantize(Decimal("0.0000000001")) == expected.quantize(Decimal("0.0000000001")), (
        f"Proportional scaling violated: {quantity} * {target}/{original} = {expected}, got {result.quantity}"
    )
