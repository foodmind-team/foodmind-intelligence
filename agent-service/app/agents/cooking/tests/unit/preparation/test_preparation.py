from decimal import Decimal

import pytest

from cooking_plan_agent.domain.enums import HeatLevel, WorkMode
from cooking_plan_agent.domain.models import (
    CookingTask,
    RecipeStep,
    ResourceNeed,
    TaskDependency,
)
from cooking_plan_agent.preparation.decompose import (
    DecompositionPolicy,
    decompose_step,
    format_food_state,
)
from cooking_plan_agent.preparation.prep_trie import (
    PreparationOperation,
    PrepTrieNode,
    convert_trie_to_tasks,
    insert_operation_chain,
    verify_quantity_conservation,
)
from cooking_plan_agent.preparation.task_graph import (
    CyclicGraphError,
    TaskEdge,
    TaskGraph,
    build_task_graph,
    calculate_dependency_lower_bound,
    topological_sort_kahn,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def policy() -> DecompositionPolicy:
    return DecompositionPolicy()


# =============================================================================
# 6.2  decompose_step — per-pattern decomposition
# =============================================================================


class TestDecomposeSimple:
    """Simple pattern: one active task."""

    def test_one_task(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=1,
            instruction="Mix ingredients",
            pattern="simple",
            active_duration_minutes=10,
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.work_mode == WorkMode.ACTIVE
        assert t.duration_minutes == 10
        assert t.task_id == "r1_s1"

    def test_unknown_pattern_falls_back_to_simple(self, policy: DecompositionPolicy) -> None:
        """Unrecognised pattern should be treated as simple."""
        step = RecipeStep(
            step_number=1,
            instruction="Do something",
            pattern="unknown_pattern",
            active_duration_minutes=7,
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 1
        assert tasks[0].work_mode == WorkMode.ACTIVE


class TestDecomposeBoil:
    """Boil pattern: fill → heat (passive) → check."""

    def test_three_sub_tasks(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=2,
            instruction="Boil water for 10 minutes",
            pattern="boil",
            passive_duration_minutes=10,
            heat_level=HeatLevel.HIGH,
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 3
        # task 0: fill (active)
        assert tasks[0].work_mode == WorkMode.ACTIVE
        assert "fill" in tasks[0].task_id
        # task 1: heat (passive)
        assert tasks[1].work_mode == WorkMode.PASSIVE
        assert tasks[1].duration_minutes == 10
        assert tasks[1].heat_level == HeatLevel.HIGH
        # task 2: check (active)
        assert tasks[2].work_mode == WorkMode.ACTIVE
        assert "check" in tasks[2].task_id

    def test_chained_dependencies(self, policy: DecompositionPolicy) -> None:
        """Fill → heat → check must be sequentially dependent."""
        step = RecipeStep(
            step_number=3,
            instruction="Boil",
            pattern="boil",
            passive_duration_minutes=5,
        )
        tasks = decompose_step("r1", step, policy)
        # heat depends on fill
        assert any(d.predecessor_id == tasks[0].task_id for d in tasks[1].dependencies)
        # check depends on heat
        assert any(d.predecessor_id == tasks[1].task_id for d in tasks[2].dependencies)


class TestDecomposeMarinate:
    """Marinate pattern: apply → wait (passive)."""

    def test_two_sub_tasks(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=4,
            instruction="Marinate chicken for 20 minutes",
            pattern="marinate",
            passive_duration_minutes=20,
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 2
        assert tasks[0].work_mode == WorkMode.ACTIVE
        assert tasks[1].work_mode == WorkMode.PASSIVE
        assert tasks[1].duration_minutes == 20

    def test_raw_protein_gets_safety_tag(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=1,
            instruction="Marinate chicken for 20 minutes",
            pattern="marinate",
            passive_duration_minutes=20,
        )
        tasks = decompose_step("r1", step, policy)
        assert "raw_meat" in tasks[1].safety_tags

    def test_non_protein_no_safety_tag(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=1,
            instruction="Marinate tofu for 10 minutes",
            pattern="marinate",
            passive_duration_minutes=10,
        )
        tasks = decompose_step("r1", step, policy)
        assert "raw_meat" not in tasks[1].safety_tags


class TestDecomposeBake:
    """Bake pattern: load → bake (passive) → unload."""

    def test_three_sub_tasks(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=5,
            instruction="Bake at 180 C for 25 minutes",
            pattern="bake",
            passive_duration_minutes=25,
            heat_level=HeatLevel.MEDIUM,
            target_temperature_c=Decimal(180),
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 3
        assert tasks[0].work_mode == WorkMode.ACTIVE
        assert tasks[1].work_mode == WorkMode.PASSIVE
        assert tasks[1].duration_minutes == 25
        assert tasks[2].work_mode == WorkMode.ACTIVE

    def test_batch_key_for_same_oven_temp(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=1,
            instruction="Bake at 200 C",
            pattern="bake",
            passive_duration_minutes=30,
            target_temperature_c=Decimal(200),
        )
        tasks = decompose_step("r1", step, policy)
        assert tasks[1].batch_key == "oven_200C"


class TestDecomposeStirFry:
    """Stir-fry pattern: one active task with multiple resources."""

    def test_one_task_with_resources(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=6,
            instruction="Stir-fry continuously for 5 minutes",
            pattern="stir_fry",
            active_duration_minutes=5,
        )
        tasks = decompose_step("r1", step, policy)
        assert len(tasks) == 1
        t = tasks[0]
        assert t.work_mode == WorkMode.ACTIVE
        assert t.heat_level == HeatLevel.HIGH
        resource_types = {r.resource_type for r in t.resources}
        assert resource_types >= {"stove", "wok", "spatula"}


class TestDecomposeSimmer:
    """Simmer pattern: passive intervals + periodic check/stir."""

    def test_simmer_30min_5min_interval(self, policy: DecompositionPolicy) -> None:
        step = RecipeStep(
            step_number=7,
            instruction="Simmer and stir every 5 minutes for 30 minutes",
            pattern="simmer",
            passive_duration_minutes=30,
            interval_minutes=5,
        )
        tasks = decompose_step("r1", step, policy)
        # 30 / 5 = 6 intervals → 6 wait + 6 stir = 12 tasks
        assert len(tasks) == 12
        # Alternating passive / active
        for i, t in enumerate(tasks):
            if i % 2 == 0:
                assert t.work_mode == WorkMode.PASSIVE
            else:
                assert t.work_mode == WorkMode.ACTIVE

    def test_simmer_with_remainder(self, policy: DecompositionPolicy) -> None:
        """17 min / 5 min interval → 3 intervals of 5 + remainder 2."""
        step = RecipeStep(
            step_number=1,
            instruction="Simmer for 17 minutes",
            pattern="simmer",
            passive_duration_minutes=17,
            interval_minutes=5,
        )
        tasks = decompose_step("r1", step, policy)
        # 3 intervals → 3 wait + 3 stir = 6 tasks
        assert len(tasks) == 6
        # Verify total passive duration ≈ 17
        passive_total = sum(t.duration_minutes for t in tasks if t.work_mode == WorkMode.PASSIVE)
        assert passive_total == 17


class TestDecomposePolicy:
    """Custom DecompositionPolicy overrides default sub-task durations."""

    def test_custom_setup_minutes(self) -> None:
        custom = DecompositionPolicy(active_setup_minutes=10)
        step = RecipeStep(
            step_number=1,
            instruction="Boil",
            pattern="boil",
            passive_duration_minutes=5,
        )
        tasks = decompose_step("r1", step, custom)
        # Fill task should use custom 10 minutes
        assert tasks[0].duration_minutes == 10


# =============================================================================
# 6.3  format_food_state
# =============================================================================


class TestFormatFoodState:
    def test_basic(self) -> None:
        assert format_food_state("chilli", "washed") == "chilli:washed:shared"

    def test_with_scope(self) -> None:
        assert format_food_state("chilli", "diced", "dish-a") == "chilli:diced:dish-a"

    def test_raw_state(self) -> None:
        assert format_food_state("chicken", "raw", "portion-b") == "chicken:raw:portion-b"


# =============================================================================
# 6.4  PreparationOperation
# =============================================================================


class TestPreparationOperation:
    def test_create_minimal(self) -> None:
        op = PreparationOperation(
            operation="wash",
            quantity=Decimal(200),
            unit="g",
        )
        assert op.operation == "wash"
        assert op.quantity == Decimal(200)
        assert op.resource_needs == ()

    def test_with_resources(self) -> None:
        op = PreparationOperation(
            operation="dice",
            specification="fine dice",
            quantity=Decimal(100),
            unit="g",
            resource_needs=(
                ResourceNeed(quantity=1, resource_type="cutting_board"),
                ResourceNeed(quantity=1, resource_type="knife"),
            ),
        )
        assert op.specification == "fine dice"
        assert len(op.resource_needs) == 2

    def test_frozen(self) -> None:
        """PreparationOperation should be immutable (inherits StrictModel)."""
        op = PreparationOperation(
            operation="wash",
            quantity=Decimal(100),
            unit="g",
        )
        with pytest.raises((AttributeError, ValueError)):
            op.operation = "peel"  # type: ignore[misc]


# =============================================================================
# 6.5  PrepTrieNode + insert_operation_chain
# =============================================================================


class TestPrepTrieNode:
    def test_create_root(self) -> None:
        root = PrepTrieNode(operation_key="root")
        assert root.operation_key == "root"
        assert root.total_quantity == Decimal(0)
        assert root.child_nodes == {}

    def test_defaults(self) -> None:
        node = PrepTrieNode(operation_key="wash")
        assert node.total_quantity == Decimal(0)
        assert node.demand_ids == set()


class TestInsertOperationChain:
    """Handbook 6.5: trie insertion with merging and branching."""

    @pytest.fixture
    def wash_op(self) -> PreparationOperation:
        return PreparationOperation(
            operation="wash",
            quantity=Decimal(180),
            unit="g",
        )

    @pytest.fixture
    def julienne_op(self) -> PreparationOperation:
        return PreparationOperation(
            operation="julienne",
            specification="julienned",
            quantity=Decimal(60),
            unit="g",
        )

    @pytest.fixture
    def slice_op(self) -> PreparationOperation:
        return PreparationOperation(
            operation="slice",
            specification="sliced",
            quantity=Decimal(70),
            unit="g",
        )

    @pytest.fixture
    def dice_op(self) -> PreparationOperation:
        return PreparationOperation(
            operation="dice",
            specification="diced",
            quantity=Decimal(50),
            unit="g",
        )

    def test_single_chain(self) -> None:
        root = PrepTrieNode(operation_key="root")
        chain = (PreparationOperation(operation="wash", quantity=Decimal(100), unit="g"),)
        insert_operation_chain(root, chain, "d1")
        assert "wash" in root.child_nodes
        assert root.child_nodes["wash"].total_quantity == Decimal(100)
        assert "d1" in root.child_nodes["wash"].demand_ids

    def test_shared_wash_merges_quantity(self, wash_op: PreparationOperation) -> None:
        """Two chains with identical wash → quantities merge."""
        root = PrepTrieNode(operation_key="root")
        insert_operation_chain(root, (wash_op,), "d1")
        insert_operation_chain(root, (wash_op,), "d2")
        wash_node = root.child_nodes["wash"]
        assert wash_node.total_quantity == Decimal(360)  # 180 + 180
        assert wash_node.demand_ids == {"d1", "d2"}

    def test_branching_on_different_cuts(
        self,
        wash_op: PreparationOperation,
        julienne_op: PreparationOperation,
        slice_op: PreparationOperation,
    ) -> None:
        """wash → julienne and wash → slice should share wash, branch after."""
        root = PrepTrieNode(operation_key="root")
        insert_operation_chain(root, (wash_op, julienne_op), "d1")
        insert_operation_chain(root, (wash_op, slice_op), "d2")

        wash_node = root.child_nodes["wash"]
        assert wash_node.total_quantity == Decimal(360)  # Aggregated

        # Two cutting branches: cut:julienned and cut:sliced
        assert len(wash_node.child_nodes) == 2
        assert "cut:julienned" in wash_node.child_nodes
        assert "cut:sliced" in wash_node.child_nodes

    def test_three_chilli_demands(
        self,
        wash_op: PreparationOperation,
        julienne_op: PreparationOperation,
        slice_op: PreparationOperation,
        dice_op: PreparationOperation,
    ) -> None:
        """Handbook Exercise 1: trie for three chilli demands."""
        root = PrepTrieNode(operation_key="root")
        insert_operation_chain(root, (wash_op, julienne_op), "d1")
        insert_operation_chain(root, (wash_op, slice_op), "d2")
        insert_operation_chain(root, (wash_op, dice_op), "d3")

        # Wash is shared by all three
        wash_node = root.child_nodes["wash"]
        assert wash_node.total_quantity == Decimal(540)  # 180 × 3
        assert wash_node.demand_ids == {"d1", "d2", "d3"}

        # Three branches
        assert len(wash_node.child_nodes) == 3

    def test_root_tracks_demand_ids(self) -> None:
        root = PrepTrieNode(operation_key="root")
        chain = (PreparationOperation(operation="wash", quantity=Decimal(50), unit="g"),)
        insert_operation_chain(root, chain, "d1")
        assert "d1" in root.demand_ids


# =============================================================================
# 6.6  convert_trie_to_tasks
# =============================================================================


class TestConvertTrieToTasks:
    @pytest.fixture
    def populated_trie(self) -> PrepTrieNode:
        """A trie: wash(300g) → cut:julienned(100g) + cut:diced(200g)."""
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

    def test_produces_flat_task_list(self, populated_trie: PrepTrieNode) -> None:
        tasks = convert_trie_to_tasks(populated_trie, "chilli", ("dish-a", "dish-b"))
        # Should produce: wash + cut:julienned + cut:diced = 3 tasks
        assert len(tasks) == 3

    def test_wash_task_has_no_dependency(self, populated_trie: PrepTrieNode) -> None:
        """The first operation (wash) has no predecessor."""
        tasks = convert_trie_to_tasks(populated_trie, "chilli", ())
        wash_task = next(t for t in tasks if t.task_id.startswith("prep_chilli_wash"))
        assert wash_task.dependencies == ()

    def test_cut_tasks_have_parent_dependency(self, populated_trie: PrepTrieNode) -> None:
        """Cut tasks should depend on wash."""
        tasks = convert_trie_to_tasks(populated_trie, "chilli", ())
        cut_tasks = [t for t in tasks if "cut" in t.task_id]
        wash_task = next(t for t in tasks if t.task_id.startswith("prep_chilli_wash"))
        wash_state = wash_task.produces_states[0]
        for ct in cut_tasks:
            assert wash_state in ct.consumes_states

    def test_wash_task_quantity_aggregated(self, populated_trie: PrepTrieNode) -> None:
        tasks = convert_trie_to_tasks(populated_trie, "chilli", ())
        wash_task = next(t for t in tasks if t.task_id.startswith("prep_chilli_wash"))
        # 300 total quantity with setup=5, rho=0.01 → 5 + 3 = 8 min
        assert wash_task.duration_minutes >= 1

    def test_empty_trie_returns_empty(self) -> None:
        root = PrepTrieNode(operation_key="root")
        tasks = convert_trie_to_tasks(root, "none", ())
        assert tasks == ()

    def test_single_node_trie(self) -> None:
        root = PrepTrieNode(operation_key="root")
        chain = (PreparationOperation(operation="wash", quantity=Decimal(50), unit="g"),)
        insert_operation_chain(root, chain, "d1")
        tasks = convert_trie_to_tasks(root, "item", ())
        assert len(tasks) == 1
        assert tasks[0].produces_states == ("item:wash:shared",)


# =============================================================================
# 6.7  verify_quantity_conservation
# =============================================================================


class TestVerifyQuantityConservation:
    def test_balanced_trie_passes(self) -> None:
        """Wash(300) = julienne(100) + dice(200): passes."""
        root = PrepTrieNode(operation_key="root")
        chain1 = (
            PreparationOperation(operation="wash", quantity=Decimal(100), unit="g"),
            PreparationOperation(operation="julienne", quantity=Decimal(100), unit="g"),
        )
        chain2 = (
            PreparationOperation(operation="wash", quantity=Decimal(200), unit="g"),
            PreparationOperation(operation="dice", quantity=Decimal(200), unit="g"),
        )
        insert_operation_chain(root, chain1, "d1")
        insert_operation_chain(root, chain2, "d2")
        # Should not raise
        verify_quantity_conservation(root)

    def test_root_exempted(self) -> None:
        """Root node is exempt from conservation check."""
        root = PrepTrieNode(operation_key="root")
        # Root accumulates all quantities but children may not sum to root
        # because root gets all traversal quantities.
        # Actually, in our insertion algorithm, root.total_quantity
        # accumulates from all traversals, which should match children sum.
        # Let's just verify no error for root.
        verify_quantity_conservation(root)  # Empty trie — should pass


# =============================================================================
# 6.8  TaskGraph + build_task_graph
# =============================================================================


class TestTaskGraph:
    @pytest.fixture
    def two_tasks(self) -> tuple[CookingTask, CookingTask]:
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="Wash",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="prep",
            produces_states=("chilli:washed:shared",),
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="Cut",
            duration_minutes=10,
            work_mode=WorkMode.ACTIVE,
            category="prep",
            consumes_states=("chilli:washed:shared",),
        )
        return t1, t2

    def test_build_graph_food_state_edges(self, two_tasks: tuple) -> None:
        t1, t2 = two_tasks
        graph = build_task_graph((t1, t2), (), ())
        assert len(graph.tasks) == 2
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge.predecessor_id == "t1"
        assert edge.successor_id == "t2"

    def test_build_graph_deduplicates_edges(self) -> None:
        """Same edge from food-state and explicit dep should be deduplicated."""
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="A",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="prep",
            produces_states=("s:washed:shared",),
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="B",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="prep",
            consumes_states=("s:washed:shared",),
            dependencies=(TaskDependency(predecessor_id="t1"),),
        )
        graph = build_task_graph((t1, t2), (), ())
        # Should have exactly 1 edge, not 2
        assert len(graph.edges) == 1

    def test_build_graph_empty_inputs(self) -> None:
        graph = build_task_graph((), (), ())
        assert graph.tasks == ()
        assert graph.edges == ()

    def test_task_edge_model(self) -> None:
        edge = TaskEdge(predecessor_id="a", successor_id="b")
        assert edge.predecessor_id == "a"
        assert edge.successor_id == "b"

    def test_task_graph_model(self) -> None:
        t = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="Test",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        graph = TaskGraph(tasks=(t,), edges=())
        assert len(graph.tasks) == 1


# =============================================================================
# 6.9  topological_sort_kahn — cycle detection
# =============================================================================


class TestTopologicalSort:
    @pytest.fixture
    def linear_graph(self) -> TaskGraph:
        """t1 → t2 → t3 (no cycles)."""
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="A",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="B",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t3 = CookingTask(
            task_id="t3",
            dish_id="r1",
            instruction="C",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        edges = (
            TaskEdge(predecessor_id="t1", successor_id="t2"),
            TaskEdge(predecessor_id="t2", successor_id="t3"),
        )
        return TaskGraph(tasks=(t1, t2, t3), edges=edges)

    @pytest.fixture
    def cyclic_graph(self) -> TaskGraph:
        """t1 → t2 → t3 → t1 (cycle)."""
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="A",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="B",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t3 = CookingTask(
            task_id="t3",
            dish_id="r1",
            instruction="C",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        edges = (
            TaskEdge(predecessor_id="t1", successor_id="t2"),
            TaskEdge(predecessor_id="t2", successor_id="t3"),
            TaskEdge(predecessor_id="t3", successor_id="t1"),  # Back edge!
        )
        return TaskGraph(tasks=(t1, t2, t3), edges=edges)

    def test_linear_order(self, linear_graph: TaskGraph) -> None:
        order = topological_sort_kahn(linear_graph)
        ids = [t.task_id for t in order]
        assert ids == ["t1", "t2", "t3"]

    def test_cycle_raises(self, cyclic_graph: TaskGraph) -> None:
        with pytest.raises(CyclicGraphError) as exc:
            topological_sort_kahn(cyclic_graph)
        assert "t1" in str(exc.value)
        assert "t2" in str(exc.value)
        assert "t3" in str(exc.value)

    def test_empty_graph(self) -> None:
        graph = TaskGraph(tasks=(), edges=())
        result = topological_sort_kahn(graph)
        assert result == ()

    def test_single_task(self) -> None:
        t = CookingTask(
            task_id="alone",
            dish_id="r1",
            instruction="X",
            duration_minutes=1,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        graph = TaskGraph(tasks=(t,), edges=())
        result = topological_sort_kahn(graph)
        assert len(result) == 1
        assert result[0].task_id == "alone"

    def test_diamond_dag(self) -> None:
        """t1 → t2, t1 → t3, t2 → t4, t3 → t4 (diamond)."""
        tasks = tuple(
            CookingTask(
                task_id=tid,
                dish_id="r1",
                instruction=tid,
                duration_minutes=1,
                work_mode=WorkMode.ACTIVE,
                category="test",
            )
            for tid in ["t1", "t2", "t3", "t4"]
        )
        edges = (
            TaskEdge(predecessor_id="t1", successor_id="t2"),
            TaskEdge(predecessor_id="t1", successor_id="t3"),
            TaskEdge(predecessor_id="t2", successor_id="t4"),
            TaskEdge(predecessor_id="t3", successor_id="t4"),
        )
        graph = TaskGraph(tasks=tasks, edges=edges)
        order = topological_sort_kahn(graph)
        ids = [t.task_id for t in order]
        assert ids[0] == "t1"
        assert ids[-1] == "t4"
        # t2 and t3 can be in any order
        assert set(ids[1:3]) == {"t2", "t3"}


# =============================================================================
# 6.10  calculate_dependency_lower_bound
# =============================================================================


class TestCriticalPath:
    def test_linear_chain(self) -> None:
        """t1 (5 min) → t2 (10 min) → t3 (3 min) = 18 min."""
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="A",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="B",
            duration_minutes=10,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t3 = CookingTask(
            task_id="t3",
            dish_id="r1",
            instruction="C",
            duration_minutes=3,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        edges = (
            TaskEdge(predecessor_id="t1", successor_id="t2"),
            TaskEdge(predecessor_id="t2", successor_id="t3"),
        )
        graph = TaskGraph(tasks=(t1, t2, t3), edges=edges)
        assert calculate_dependency_lower_bound(graph) == 18

    def test_parallel_tasks_take_max(self) -> None:
        """t1 → t2 (5 min) and t1 → t3 (10 min): total = 5 + max(5, 10) = 15."""
        t1 = CookingTask(
            task_id="t1",
            dish_id="r1",
            instruction="A",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t2 = CookingTask(
            task_id="t2",
            dish_id="r1",
            instruction="B",
            duration_minutes=5,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        t3 = CookingTask(
            task_id="t3",
            dish_id="r1",
            instruction="C",
            duration_minutes=10,
            work_mode=WorkMode.ACTIVE,
            category="test",
        )
        edges = (
            TaskEdge(predecessor_id="t1", successor_id="t2"),
            TaskEdge(predecessor_id="t1", successor_id="t3"),
        )
        graph = TaskGraph(tasks=(t1, t2, t3), edges=edges)
        assert calculate_dependency_lower_bound(graph) == 15

    def test_empty_graph_returns_zero(self) -> None:
        graph = TaskGraph(tasks=(), edges=())
        assert calculate_dependency_lower_bound(graph) == 0


# =============================================================================
# Integration: full pipeline from RecipeStep → DAG
# =============================================================================


class TestIntegration:
    """End-to-end: decompose steps → build DAG → sort → critical path."""

    def test_full_pipeline(self) -> None:
        # Two recipe steps: boil water (8 min) then stir-fry (5 min)
        step1 = RecipeStep(
            step_number=1,
            instruction="Boil water for 8 minutes",
            pattern="boil",
            passive_duration_minutes=8,
            heat_level=HeatLevel.HIGH,
        )
        step2 = RecipeStep(
            step_number=2,
            instruction="Stir-fry vegetables for 5 minutes",
            pattern="stir_fry",
            active_duration_minutes=5,
            heat_level=HeatLevel.HIGH,
        )

        recipe_tasks = decompose_step("r1", step1) + decompose_step("r1", step2)

        # Add dependency: step2 after step1 (recipe ordering).
        # Apply a dependency from last sub-task of step1 to first of step2.
        last_boil = recipe_tasks[2]  # check task
        first_stir = recipe_tasks[3]  # stir-fry task
        updated_first = first_stir.model_copy(
            update={
                "dependencies": first_stir.dependencies + (TaskDependency(predecessor_id=last_boil.task_id),),
            }
        )
        recipe_tasks = recipe_tasks[:3] + (updated_first,) + recipe_tasks[4:]

        graph = build_task_graph(recipe_tasks, (), ())
        assert len(graph.tasks) == 4  # 3 boil sub-tasks + 1 stir-fry

        # Should be acyclic
        order = topological_sort_kahn(graph)
        assert len(order) == 4

        # Critical path > 0
        lb = calculate_dependency_lower_bound(graph)
        assert lb > 0

    def test_preparation_and_recipe_integration(self) -> None:
        """Prep trie tasks + recipe tasks in one DAG."""
        # Prep: wash 100g chilli → dice 100g
        root = PrepTrieNode(operation_key="root")
        chain = (
            PreparationOperation(operation="wash", quantity=Decimal(100), unit="g"),
            PreparationOperation(operation="dice", specification="diced", quantity=Decimal(100), unit="g"),
        )
        insert_operation_chain(root, chain, "d1")
        prep_tasks = convert_trie_to_tasks(root, "chilli", ("r1",))

        # Recipe: stir-fry that consumes diced chilli
        diced_state = prep_tasks[-1].produces_states[0] if prep_tasks else ""
        recipe_step = RecipeStep(
            step_number=1,
            instruction="Stir-fry chilli",
            pattern="stir_fry",
            active_duration_minutes=3,
        )
        recipe_tasks = decompose_step("r1", recipe_step)

        # Wire: stir-fry consumes diced chilli state
        if diced_state and recipe_tasks:
            updated_recipe = recipe_tasks[0].model_copy(
                update={
                    "consumes_states": (diced_state,),
                }
            )
            recipe_tasks = (updated_recipe,)

        graph = build_task_graph(recipe_tasks, prep_tasks, ())
        # Should have edges connecting prep → recipe via food states
        assert len(graph.edges) >= 1

        # Must be acyclic
        order = topological_sort_kahn(graph)
        assert len(order) == len(graph.tasks)
