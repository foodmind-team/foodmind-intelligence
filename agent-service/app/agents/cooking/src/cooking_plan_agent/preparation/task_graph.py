"""Task graph construction, cycle detection, and critical-path analysis

This module handles:
- TaskEdge / TaskGraph: provider-independent DAG representation 
- build_task_graph: merge recipe + prep + safety tasks into one graph 
- topological_sort_kahn: Kahn's algorithm with cycle detection 
- calculate_dependency_lower_bound: dependency-only makespan lower bound 
"""

from collections import deque

from cooking_plan_agent.domain.models import CookingTask, StrictModel

# ============================================================================
# 6.8  Task DAG construction
# ============================================================================


class TaskEdge(StrictModel):
    """A directed edge in the task graph.

    Represents a precedence constraint: predecessor must finish before
    successor can start.
    """

    predecessor_id: str
    """The task that must complete first."""

    successor_id: str
    """The task that depends on the predecessor."""


class TaskGraph(StrictModel):
    """A directed acyclic graph of cooking tasks.

    Handbook 6.8: provider-independent representation that the scheduler
    consumes.  Edges encode real dependencies only — never the author's
    sentence order when steps are independent.
    """

    tasks: tuple[CookingTask, ...]
    """All tasks in the graph."""

    edges: tuple[TaskEdge, ...]
    """Precedence edges between tasks."""


def build_task_graph(
    recipe_tasks: tuple[CookingTask, ...],
    prep_tasks: tuple[CookingTask, ...],
    safety_tasks: tuple[CookingTask, ...],
) -> TaskGraph:
    """Build a TaskGraph from decomposed recipe, preparation, and safety tasks.

    Example: adds edges from:
    - Original explicit recipe ordering that is semantically required.
    - Food-state producer → consumer.
    - Preparation parent → child.
    - Safety-required predecessor/successor.
    - Minimum marinating, resting, heating, or cooling lags.

    Does NOT add edges merely to preserve the author's sentence order
    when the steps are independent.

    Args:
        recipe_tasks: Tasks decomposed from recipe steps (6.2).
        prep_tasks: Tasks derived from preparation trie (6.6).
        safety_tasks: Tasks enforcing safety rules (e.g. clean after raw meat).

    Returns:
        A TaskGraph with all tasks and computed edges.
    """
    all_tasks = recipe_tasks + prep_tasks + safety_tasks
    edges: list[TaskEdge] = []

    # Build a state→producer index for food-state producer→consumer edges.
    state_producer: dict[str, str] = {}
    for t in all_tasks:
        for s in t.produces_states:
            state_producer[s] = t.task_id

    # Build edges from multiple sources.
    for t in all_tasks:
        # --- Food-state producer → consumer edges ---
        for consumed in t.consumes_states:
            if consumed in state_producer:
                edges.append(
                    TaskEdge(
                        predecessor_id=state_producer[consumed],
                        successor_id=t.task_id,
                    )
                )

        # --- Explicit TaskDependency edges ---
        for dep in t.dependencies:
            edges.append(
                TaskEdge(
                    predecessor_id=dep.predecessor_id,
                    successor_id=t.task_id,
                )
            )

    # Deduplicate edges (same predecessor→successor pair may come from
    # multiple sources).
    seen: set[tuple[str, str]] = set()
    unique_edges: list[TaskEdge] = []
    for e in edges:
        pair = (e.predecessor_id, e.successor_id)
        if pair not in seen:
            seen.add(pair)
            unique_edges.append(e)

    return TaskGraph(tasks=all_tasks, edges=tuple(unique_edges))


# ============================================================================
# 6.9  Cycle detection — Kahn's topological sort
# ============================================================================


class CyclicGraphError(ValueError):
    """Raised when a cycle is detected in the task graph."""

    def __init__(self, task_ids: frozenset[str]) -> None:
        self.task_ids = task_ids
        super().__init__(
            f"Cyclic dependency detected among tasks: "
            f"{sorted(task_ids)}"
        )


def topological_sort_kahn(graph: TaskGraph) -> tuple[CookingTask, ...]:
    """Return tasks in topological order using Kahn's algorithm.

    Example:
    1. Count the indegree of every task.
    2. Put all zero-indegree tasks in a deterministic queue.
    3. Remove one task, append to order, decrement successor indegrees.
    4. Add newly zero-indegree successors to the queue.
    5. If output count < task count, a cycle exists → raise CyclicGraphError.

    Args:
        graph: The task graph to topologically sort.

    Returns:
        Tasks in a valid topological order (deterministic for the same graph).

    Raises:
        CyclicGraphError: If a cycle is detected.  Do not send cyclic
            graphs to OR-Tools (Handbook 6.9).
    """
    task_map: dict[str, CookingTask] = {t.task_id: t for t in graph.tasks}

    # 1. Count indegree for every task.
    indegree: dict[str, int] = {t.task_id: 0 for t in graph.tasks}
    successors: dict[str, list[str]] = {t.task_id: [] for t in graph.tasks}

    for edge in graph.edges:
        if edge.predecessor_id in indegree and edge.successor_id in indegree:
            indegree[edge.successor_id] += 1
            successors[edge.predecessor_id].append(edge.successor_id)

    # 2. Queue all zero-indegree tasks (sorted for determinism).
    queue: deque[str] = deque(
        sorted(tid for tid, deg in indegree.items() if deg == 0)
    )

    # 3-4. Process queue.
    result: list[CookingTask] = []
    while queue:
        tid = queue.popleft()
        result.append(task_map[tid])
        for succ in successors.get(tid, []):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                # Insert in sorted position for determinism.
                queue.append(succ)
                # Re-sort the queue on each append to maintain determinism
                # when multiple tasks become ready simultaneously.
                # Using list conversion for clarity — small n in practice.
                sorted_queue = sorted(queue)
                queue = deque(sorted_queue)

    # 5. Cycle detection.
    if len(result) < len(graph.tasks):
        unfinished = frozenset(
            tid for tid, deg in indegree.items() if deg > 0
        )
        raise CyclicGraphError(unfinished)

    return tuple(result)


# ============================================================================
# 6.10  Critical path baseline
# ============================================================================


def calculate_dependency_lower_bound(graph: TaskGraph) -> int:
    """Calculate the dependency-only earliest finish time (makespan lower bound).

    Example: for each task in topological order:
        earliest_start(task) = max(earliest_end(predecessors))
        earliest_end(task)   = earliest_start(task) + duration(task)

    This lower bound ignores resource constraints — real makespan can only
    be equal or longer.  Useful for debugging and as a sanity check before
    CP-SAT scheduling.

    Args:
        graph: A valid acyclic task graph.

    Returns:
        The earliest possible finish time in minutes.
    """
    topo_order = topological_sort_kahn(graph)

    earliest_end: dict[str, int] = {}

    for task in topo_order:
        # Aggregate predecessor end times via TaskDependency (explicit deps
        # within the task model) and via graph edges.
        pred_ids: set[str] = set()
        for dep in task.dependencies:
            pred_ids.add(dep.predecessor_id)
        for edge in graph.edges:
            if edge.successor_id == task.task_id:
                pred_ids.add(edge.predecessor_id)

        pred_end_times = [
            earliest_end.get(pid, 0) for pid in pred_ids
        ]
        start = max(pred_end_times) if pred_end_times else 0
        end = start + task.duration_minutes
        earliest_end[task.task_id] = end

    return max(earliest_end.values()) if earliest_end else 0
