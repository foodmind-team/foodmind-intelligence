# =============================================================================
# 任务图构建、环检测与关键路径分析模块（preparation/task_graph）
# -----------------------------------------------------------------------------
# 处理：
#   - TaskEdge / TaskGraph      ：与调度器无关的 DAG 表示
#   - build_task_graph          ：把 recipe + prep + safety 任务合并为一个图
#   - topological_sort_kahn     ：Kahn 拓扑排序 + 环检测
#   - calculate_dependency_lower_bound：仅依赖的 makespan 下界
# 关键：图只编码“真实依赖”，绝不因步骤在原文中的顺序而添加无意义的边（手册 6.8）。
# =============================================================================

"""Task graph construction, cycle detection, and critical-path analysis
任务图构建、环检测与关键路径分析

This module handles:
- TaskEdge / TaskGraph: provider-independent DAG representation
- build_task_graph: merge recipe + prep + safety tasks into one graph
- topological_sort_kahn: Kahn's algorithm with cycle detection
- calculate_dependency_lower_bound: dependency-only makespan lower bound

本模块处理：
- TaskEdge / TaskGraph：与调度器无关的 DAG 表示
- build_task_graph：把 recipe + prep + safety 任务合并为一个图
- topological_sort_kahn：Kahn 算法 + 环检测
- calculate_dependency_lower_bound：仅依赖的 makespan 下界
"""

from collections import deque

from cooking_plan_agent.domain.models import CookingTask, StrictModel, TaskDependency

# ============================================================================
# 6.8  Task DAG construction
# 6.8  任务 DAG 构建
# ============================================================================


class TaskEdge(StrictModel):
    """任务图中的一条有向边。

    A directed edge in the task graph.

    Represents a precedence constraint: predecessor must finish before
    successor can start.

    表示一个先后约束：前驱必须完成后继才能开始。
    """

    predecessor_id: str
    """The task that must complete first.
    必须先完成的任务。"""

    successor_id: str
    """The task that depends on the predecessor.
    依赖前驱的任务。"""


class TaskGraph(StrictModel):
    """烹饪任务的有向无环图。

    A directed acyclic graph of cooking tasks.

    Handbook 6.8: provider-independent representation that the scheduler
    consumes.  Edges encode real dependencies only — never the author's
    sentence order when steps are independent.

    手册 6.8：调度器消费的、与提供方无关的表示。边只编码真实依赖 ——
    当步骤相互独立时，绝不因作者的行文顺序添加边。
    """

    tasks: tuple[CookingTask, ...]
    """All tasks in the graph.
    图中的所有任务。"""

    edges: tuple[TaskEdge, ...]
    """Precedence edges between tasks.
    任务之间的先后边。"""


def build_task_graph(
    recipe_tasks: tuple[CookingTask, ...],
    prep_tasks: tuple[CookingTask, ...],
    safety_tasks: tuple[CookingTask, ...],
) -> TaskGraph:
    """从分解后的 recipe、preparation、safety 任务构建 TaskGraph。

    Build a TaskGraph from decomposed recipe, preparation, and safety tasks.

    Example: adds edges from:
    - Original explicit recipe ordering that is semantically required.
    - Food-state producer → consumer.
    - Preparation parent → child.
    - Safety-required predecessor/successor.
    - Minimum marinating, resting, heating, or cooling lags.

    添加边的来源：
    - 语义上要求的原始显式菜谱顺序
    - 食材状态 producer → consumer
    - 预处理 parent → child
    - 安全要求的前驱 / 后继
    - 最小腌制、静置、加热或冷却间隔

    Does NOT add edges merely to preserve the author's sentence order
    when the steps are independent.

    当步骤相互独立时，不因保持作者的行文顺序而添加边。

    Args:
        recipe_tasks: Tasks decomposed from recipe steps (6.2).
            recipe_tasks：从菜谱步骤分解出的任务（6.2）。
        prep_tasks: Tasks derived from preparation trie (6.6).
            prep_tasks：从预处理前缀树派生的任务（6.6）。
        safety_tasks: Tasks enforcing safety rules (e.g. clean after raw meat).
            safety_tasks：执行安全规则的任务（如处理生肉后清洗）。

    Returns:
        A TaskGraph with all tasks and computed edges.
        含所有任务与计算所得边的 TaskGraph。
    """
    all_tasks = recipe_tasks + prep_tasks + safety_tasks
    edges: list[TaskEdge] = []

    # Build a state→producer index for food-state producer→consumer edges.
    # 为食材状态 producer→consumer 边构建 state→producer 索引
    state_producer: dict[str, str] = {}
    for t in all_tasks:
        for s in t.produces_states:
            state_producer[s] = t.task_id

    # Build edges from multiple sources.
    # 从多个来源构建边
    for t in all_tasks:
        # --- Food-state producer → consumer edges ---
        # --- 食材状态 producer → consumer 边 ---
        for consumed in t.consumes_states:
            if consumed in state_producer:
                edges.append(
                    TaskEdge(
                        predecessor_id=state_producer[consumed],
                        successor_id=t.task_id,
                    )
                )

        # --- Explicit TaskDependency edges ---
        # --- 显式 TaskDependency 边 ---
        for dep in t.dependencies:
            edges.append(
                TaskEdge(
                    predecessor_id=dep.predecessor_id,
                    successor_id=t.task_id,
                )
            )

    # Deduplicate edges (same predecessor→successor pair may come from
    # multiple sources).
    # 去重边（相同 predecessor→successor 对可能来自多个来源）
    seen: set[tuple[str, str]] = set()
    unique_edges: list[TaskEdge] = []
    for e in edges:
        pair = (e.predecessor_id, e.successor_id)
        if pair not in seen:
            seen.add(pair)
            unique_edges.append(e)

    # The scheduler operates on CookingTask.dependencies, while this DAG also
    # expresses prerequisites through food-state producer → consumer edges.
    # Materialise those derived edges back onto the successor tasks so the
    # solver, verifier and completion-time logic all enforce the same graph.
    # Without this, a shared prep task can be scheduled after the recipe step
    # that consumes it even though TaskGraph.edges correctly shows the edge.
    # 调度器操作 CookingTask.dependencies，而本 DAG 也通过食材状态 producer → consumer
    # 边表达前置条件。把这些派生边物化回后继任务，使求解器、校验器与完成时间逻辑
    # 都执行同一个图。否则，一个共享预处理任务可能被排到消费它的菜谱步骤之后，
    # 尽管 TaskGraph.edges 正确显示了这条边。
    deps_by_successor: dict[str, list[str]] = {}
    for edge in unique_edges:
        deps_by_successor.setdefault(edge.successor_id, []).append(edge.predecessor_id)

    materialised_tasks: list[CookingTask] = []
    for task in all_tasks:
        added = deps_by_successor.get(task.task_id, [])
        existing = {dep.predecessor_id for dep in task.dependencies}
        new_dependencies = tuple(
            TaskDependency(predecessor_id=predecessor_id) for predecessor_id in added if predecessor_id not in existing
        )
        materialised_tasks.append(
            task.model_copy(update={"dependencies": task.dependencies + new_dependencies}) if new_dependencies else task
        )

    return TaskGraph(tasks=tuple(materialised_tasks), edges=tuple(unique_edges))


# ============================================================================
# 6.9  Cycle detection — Kahn's topological sort
# 6.9  环检测 —— Kahn 拓扑排序
# ============================================================================


class CycleReport(StrictModel):
    """检测到任务图中有环时的结构化报告。

    Structured report when a cycle is detected in the task graph.

    Handbook 6.9: Do not send cyclic graphs to OR-Tools. This report
    provides actionable detail for debugging.

    手册 6.9：不要向 OR-Tools 发送有环图。本报告为调试提供可操作细节。
    """

    task_ids: tuple[str, ...]
    """Tasks involved in the cycle (all tasks with remaining indegree > 0).
    环中涉及的任务（所有剩余入度 > 0 的任务）。"""

    task_count: int
    """How many tasks are in the cycle.
    环中有多少个任务。"""

    edges: tuple[TaskEdge, ...] = ()
    """Edges that form or contribute to the cycle, if identifiable.
    形成或促成该环的边（若可识别）。"""


class CyclicGraphError(ValueError):
    """检测到任务图中有环时抛出。"""

    def __init__(self, report: CycleReport) -> None:
        self.report = report
        super().__init__(f"Cyclic dependency detected among {report.task_count} tasks: {sorted(report.task_ids)}")


def topological_sort_kahn(graph: TaskGraph) -> tuple[CookingTask, ...]:
    """用 Kahn 算法按拓扑顺序返回任务。

    Return tasks in topological order using Kahn's algorithm.

    Example:
    1. Count the indegree of every task.
    2. Put all zero-indegree tasks in a deterministic queue.
    3. Remove one task, append to order, decrement successor indegrees.
    4. Add newly zero-indegree successors to the queue.
    5. If output count < task count, a cycle exists → raise CyclicGraphError.

    算法：
    1. 统计每个任务的入度。
    2. 把所有零入度任务放入确定性队列。
    3. 移除一个任务，加入顺序，递减后继入度。
    4. 把新变为零入度的后继加入队列。
    5. 若输出数量 < 任务数，存在环 → 抛 CyclicGraphError。

    Args:
        graph: The task graph to topologically sort.
            graph：要拓扑排序的任务图。

    Returns:
        Tasks in a valid topological order (deterministic for the same graph).
        处于合法拓扑顺序的任务（对同一图是确定性的）。

    Raises:
        CyclicGraphError: If a cycle is detected.  Do not send cyclic
            graphs to OR-Tools (Handbook 6.9).
        CyclicGraphError：若检测到环。不要向 OR-Tools 发送有环图（手册 6.9）。
    """
    task_map: dict[str, CookingTask] = {t.task_id: t for t in graph.tasks}

    # 1. Count indegree for every task.
    # 1. 统计每个任务的入度
    indegree: dict[str, int] = {t.task_id: 0 for t in graph.tasks}
    successors: dict[str, list[str]] = {t.task_id: [] for t in graph.tasks}

    for edge in graph.edges:
        if edge.predecessor_id in indegree and edge.successor_id in indegree:
            indegree[edge.successor_id] += 1
            successors[edge.predecessor_id].append(edge.successor_id)

    # 2. Queue all zero-indegree tasks (sorted for determinism).
    # 2. 入队所有零入度任务（排序以保证确定性）
    queue: deque[str] = deque(sorted(tid for tid, deg in indegree.items() if deg == 0))

    # 3-4. Process queue.
    # 3-4. 处理队列
    result: list[CookingTask] = []
    while queue:
        tid = queue.popleft()
        result.append(task_map[tid])
        for succ in successors.get(tid, []):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                # Insert in sorted position for determinism.
                # 按排序位置插入以保持确定性
                queue.append(succ)
                # Re-sort the queue on each append to maintain determinism
                # when multiple tasks become ready simultaneously.
                # Using list conversion for clarity — small n in practice.
                # 每次追加后重排队列，使多个任务同时就绪时保持确定性。
                # 用列表转换以求清晰 —— 实际 n 很小。
                sorted_queue = sorted(queue)
                queue = deque(sorted_queue)

    # 5. Cycle detection.
    # 5. 环检测
    if len(result) < len(graph.tasks):
        unfinished = frozenset(tid for tid, deg in indegree.items() if deg > 0)
        cycle_edges: list[TaskEdge] = []
        for edge in graph.edges:
            if edge.predecessor_id in unfinished and edge.successor_id in unfinished:
                cycle_edges.append(edge)
        report = CycleReport(
            task_ids=tuple(sorted(unfinished)),
            task_count=len(unfinished),
            edges=tuple(cycle_edges),
        )
        raise CyclicGraphError(report)

    return tuple(result)


# ============================================================================
# 6.10  Critical path baseline
# 6.10  关键路径基线
# ============================================================================


def calculate_dependency_lower_bound(graph: TaskGraph) -> int:
    """计算仅依赖的最早完成时间（makespan 下界）。

    Calculate the dependency-only earliest finish time (makespan lower bound).

    Example: for each task in topological order:
        earliest_start(task) = max(earliest_end(predecessors))
        earliest_end(task)   = earliest_start(task) + duration(task)

    算法：按拓扑顺序对每个任务：
        earliest_start(task) = max(earliest_end(predecessors))
        earliest_end(task)   = earliest_start(task) + duration(task)

    This lower bound ignores resource constraints — real makespan can only
    be equal or longer.  Useful for debugging and as a sanity check before
    CP-SAT scheduling.

    该下界忽略资源约束 —— 真实 makespan 只会相等或更长。用于调试，以及 CP-SAT
    调度前的合理性检查。

    Args:
        graph: A valid acyclic task graph.
            graph：合法的无环任务图。

    Returns:
        The earliest possible finish time in minutes.
        最早可能的完成时间（分钟）。
    """
    topo_order = topological_sort_kahn(graph)

    earliest_end: dict[str, int] = {}

    for task in topo_order:
        # Aggregate predecessor end times via TaskDependency (explicit deps
        # within the task model) and via graph edges.
        # 通过 TaskDependency（任务模型内的显式依赖）与图边聚合前驱结束时间
        pred_ids: set[str] = set()
        for dep in task.dependencies:
            pred_ids.add(dep.predecessor_id)
        for edge in graph.edges:
            if edge.successor_id == task.task_id:
                pred_ids.add(edge.predecessor_id)

        pred_end_times = [earliest_end.get(pid, 0) for pid in pred_ids]
        start = max(pred_end_times) if pred_end_times else 0
        end = start + task.duration_minutes
        earliest_end[task.task_id] = end

    return max(earliest_end.values()) if earliest_end else 0
