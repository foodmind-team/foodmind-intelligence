# =============================================================================
# 食材预处理前缀树模块（preparation/prep_trie）
# -----------------------------------------------------------------------------
# 处理跨菜谱的“相同食材预处理”合并。核心内容：
#   - PreparationOperation       ：食材预处理链中的单个原子操作
#   - PrepTrieNode               ：用于跨菜谱合并相同预处理操作的前缀树节点
#   - 前缀树转换                 ：把合并后的前缀树转回可调度的 CookingTask
#   - 数量守恒校验               ：验证父节点数量 == 子节点数量之和
#   - 从 IngredientDemand 推断预处理链
# 安全要点（P2-01）：
#   - 生蛋白质绝不与即食（RTE）食材共享预处理操作（D2），即使操作词相同；
#   - 数量守恒违反（D1）会抛 InvalidQuantityError，工作流绝不构建任务图。
# =============================================================================

"""Preparation trie — shared ingredient-prep merging
预处理前缀树 —— 共享食材预处理合并

This module handles:
- PreparationOperation: a single atomic step in an ingredient's prep chain
- PrepTrieNode: prefix-tree node for merging identical prep operations across dishes
- Trie conversion: turn the merged trie back into schedulable CookingTasks
- Quantity-conservation verification
- Inferring prep chains from IngredientDemand records.

本模块处理：
- PreparationOperation：食材预处理链中的单个原子步骤
- PrepTrieNode：跨菜谱合并相同预处理操作的前缀树节点
- 前缀树转换：把合并后的前缀树转回可调度的 CookingTask
- 数量守恒校验
- 从 IngredientDemand 记录推断预处理链
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
# 延迟导入以避免类型检查时的循环依赖；但运行时这些始终可用，因为 decompose.py 先被加载。
from cooking_plan_agent.preparation.decompose import _build_task, format_food_state

# ============================================================================
# 6.4  PreparationOperation — operation-chain model
# 6.4  PreparationOperation —— 操作链模型
# ============================================================================


class PreparationOperation(StrictModel):
    """食材预处理链中的单个操作。

    A single operation in an ingredient's preparation chain.

    Example: each ingredient demand maps to an ordered chain of
    operations (wash → peel → deseed → julienne, etc.).  The operation
    vocabulary is controlled — store user-facing phrases separately if needed.

    例：每个食材需求映射到有序操作链（洗 → 去皮 → 去籽 → 切丝等）。操作词汇受控 ——
    如需面向用户的短语，另行存储。

    Attributes
    ----------
    operation
        Controlled vocabulary key (e.g. ``"wash"``, ``"dice"``, ``"portion"``).
        受控词汇键（如 "wash"、"dice"、"portion"）。
    specification
        Optional user-facing detail (e.g. ``"fine dice"``, ``"thick slice"``).
        Not used for trie merging — only ``operation`` keys determine shareability.
        可选的面向用户细节（如 "fine dice"、"thick slice"）。不用于前缀树合并 ——
        只有 operation 键决定可共享性。
    quantity
        The ingredient quantity at this operation stage.
        该操作阶段的食材数量。
    unit
        Unit of measure (e.g. ``"g"``, ``"piece"``).
        计量单位（如 "g"、"piece"）。
    resource_needs
        Equipment required for this operation.
        该操作所需设备。
    """

    operation: str
    """Controlled vocabulary key (e.g. 'wash', 'dice', 'portion').
    受控词汇键（如 'wash'、'dice'、'portion'）。"""

    specification: str | None = None
    """Optional user-facing detail (e.g. 'fine dice').  Not used for merging.
    可选的面向用户细节（如 'fine dice'）。不用于合并。"""

    quantity: Decimal
    """Ingredient quantity at this stage.
    该阶段的食材数量。"""

    unit: str
    """Unit of measure (e.g. 'g', 'piece').
    计量单位（如 'g'、'piece'）。"""

    resource_needs: tuple[ResourceNeed, ...] = ()
    """Equipment required for this operation.
    该操作所需设备。"""


# ============================================================================
# 6.5  PrepTrieNode — prefix tree for shared preparation merging
# 6.5  PrepTrieNode —— 用于共享预处理合并的前缀树
# ============================================================================


@dataclass
class PrepTrieNode:
    """预处理前缀树（trie）中的一个节点。

    A node in the preparation prefix tree (trie).

    Each node represents one operation in the ingredient-prep chain.
    Identical operations on the same ingredient are merged by aggregating
    quantities.  Branches occur when subsequent operations diverge
    (e.g. one ingredient is julienned, another is diced).

    每个节点代表食材预处理链中的一个操作。相同食材上的相同操作通过聚合数量合并。
    当后续操作分叉时（如一个食材切丝、另一个切丁）产生分支。

    Example: reuse an existing child only when the operation is
    semantically shareable.  Not every identical word is shareable —
    raw-protein and ready-to-eat vegetable washing may require separate
    resources or sanitation boundaries.

    例：仅当操作语义上可共享时才复用已有子节点。并非每个相同词都可共享 ——
    生蛋白质与即食蔬菜的清洗可能需要分开的资源或卫生边界。

    Attributes
    ----------
    operation_key
        Stable key derived from the operation name (e.g. ``"wash"``,
        ``"cut:julienne"``).
        由操作名派生的稳定键（如 "wash"、"cut:julienne"）。
    total_quantity
        Aggregated quantity across all demands sharing this node.
        共享此节点的所有需求的聚合数量。
    child_nodes
        Child operations (next step in each preparation chain).
        子操作（每条预处理链的下一步）。
    demand_ids
        IDs of demands that traverse this node (for traceability).
        经过此节点的需求 ID（用于追溯）。
    """

    operation_key: str
    """Stable operation key (e.g. 'wash', 'cut:julienne').
    稳定操作键（如 'wash'、'cut:julienne'）。"""

    total_quantity: Decimal = Decimal(0)
    """Aggregated quantity from all demands at this node.
    此节点所有需求的聚合数量。"""

    child_nodes: dict[str, "PrepTrieNode"] = field(default_factory=dict)
    """Child operations keyed by operation_key.
    以 operation_key 为键的子操作。"""

    demand_ids: set[str] = field(default_factory=set)
    """Demand IDs that pass through this node.
    经过此节点的需求 ID。"""


# ============================================================================
# 6.5  Trie insertion algorithm
# 6.5  前缀树插入算法
# ============================================================================


def _operation_stable_key(op: PreparationOperation, ingredient_name: str) -> str:
    """为前缀树合并派生稳定键。

    Derive a stable key for trie merging.

    Simple operations use their name directly (e.g. ``"wash"``).
    Cutting operations combine category + specification for branching
    (e.g. ``"cut:julienne"`` vs ``"cut:dice"``).

    简单操作直接用其名称（如 "wash"）。切割操作结合类别 + 规格以产生分支
    （如 "cut:julienne" 与 "cut:dice"）。

    The key MUST be deterministic — two operations that are semantically
    shareable must produce the same key, and semantically different
    operations must produce different keys.

    键必须确定 —— 语义上可共享的两个操作必须产生相同键，语义不同的操作必须产生不同键。
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
    """检查不同链中同一位置的两个操作是否能合并（共享预处理）。

    Check whether two operations at the same position in different
    chains can be merged (shared preparation).

    Handbook 6.5: not every identical word is shareable.  Raw-protein
    and RTE-vegetable washing must NOT be merged.

    手册 6.5：并非每个相同词都可共享。生蛋白质与即食蔬菜的清洗绝不能合并。

    In MVP, we use a simple rule: operations must have the same
    ``operation`` key AND the same specification (if any).

    MVP 中我们用简单规则：操作必须有相同的 operation 键且相同的 specification（若有）。
    """
    if op_a.operation != op_b.operation:
        return False
    return op_a.specification == op_b.specification


def insert_operation_chain(
    root: PrepTrieNode,
    chain: tuple[PreparationOperation, ...],
    demand_id: str,
) -> None:
    """把一条预处理操作链插入前缀树。

    Insert a preparation operation chain into the prefix tree.

    insertion algorithm:
    1. Start at the ingredient's root node.
    2. For each operation, derive a stable key.
    3. Reuse an existing child only when the operation is semantically shareable.
    4. Add the demand quantity to the traversed node.
    5. Record the demand ID.
    6. Branch when specifications diverge.

    插入算法：
    1. 从食材的根节点开始。
    2. 对每个操作派生稳定键。
    3. 仅当操作语义上可共享时才复用已有子节点。
    4. 把需求数量加到所经过的节点。
    5. 记录需求 ID。
    6. 规格分叉时产生分支。

    Each operation's quantity is accumulated on the node that represents
    that operation (the child, not the parent).  The root node carries
    no ingredient quantity — it only accumulates demand_ids for traceability.

    每个操作的数量累加在表示该操作的节点上（子节点，而非父节点）。
    根节点不携带食材数量 —— 只累加 demand_ids 用于追溯。

    Args:
        root: The trie root node (typically ``operation_key="root"``).
            root：前缀树根节点（通常 operation_key="root"）。
        chain: Ordered preparation operations for one ingredient demand.
            chain：一个食材需求的有序预处理操作。
        demand_id: Unique identifier for the demand (for traceability).
            demand_id：需求的唯一标识（用于追溯）。
    """
    current = root
    for op in chain:
        key = _operation_stable_key(op, "")

        if key in current.child_nodes:
            # Existing child — merge quantities and demand IDs.
            # 已有子节点 —— 合并数量与需求 ID
            child = current.child_nodes[key]
        else:
            # New branch — create a child node.
            # 新分支 —— 创建子节点
            child = PrepTrieNode(operation_key=key)
            current.child_nodes[key] = child

        child.total_quantity += op.quantity
        child.demand_ids.add(demand_id)

        current = child

    # Record demand ID on root for traceability.
    # 在根节点记录需求 ID 以便追溯。
    root.demand_ids.add(demand_id)


# ============================================================================
# 6.6  Convert trie back to tasks
# 6.6  把前缀树转回任务
# ============================================================================


def convert_trie_to_tasks(
    root: PrepTrieNode,
    ingredient_name: str,
    dish_ids: tuple[str, ...],
    setup_minutes: Decimal = Decimal(5),
    minutes_per_base_unit: Decimal = Decimal("0.01"),
) -> tuple[CookingTask, ...]:
    """把预处理前缀树转换为 CookingTask 列表。

    Convert a preparation prefix tree into a list of CookingTasks.

    1. Create a task with the node's aggregated quantity.
    2. Calculate duration using the batch time policy.
    3. Add parent-task dependency.
    4. Produce one shared state for children.
    5. At a branch, create explicit portion/split information.

    1. 用节点的聚合数量创建任务。
    2. 用批量时间策略计算时长。
    3. 添加父任务依赖。
    4. 为子节点产出一个共享状态。
    5. 在分支处创建显式的份量 / 拆分信息。

    duration = setup_minutes + minutes_per_base_unit × Q

    Args:
        root: The populated preparation trie root.
            root：已填充的预处理前缀树根。
        ingredient_name: Canonical ingredient name for food-state labels.
            ingredient_name：用于食材状态标签的规范食材名。
        dish_ids: All dish IDs that contribute to this trie (for batch_key).
            dish_ids：贡献给该前缀树的所有菜 ID（用于 batch_key）。
        setup_minutes: Fixed setup time per batch (τ_setup).
            setup_minutes：每批的固定准备时间（τ_setup）。
        minutes_per_base_unit: Time per unit quantity (ρ).
            minutes_per_base_unit：每单位数量的时间（ρ）。

    Returns:
        A flat tuple of CookingTask instances in DFS pre-order.
        以 DFS 前序遍历的 CookingTask 实例扁平元组。
    """
    tasks: list[CookingTask] = []

    def _dfs(
        node: PrepTrieNode,
        parent_state: str | None,
        seq: list[int],
    ) -> str:
        """DFS 遍历：构建任务并返回该节点产出的状态。"""
        if not node.operation_key or node.operation_key == "root":
            # Root node: recurse into children.
            # 根节点：递归进入子节点
            for child in node.child_nodes.values():
                _dfs(child, parent_state, seq)
            return parent_state or ""

        # Build food-state for this node.
        # 为该节点构建食材状态
        state_label = node.operation_key.replace(":", "_")
        state = format_food_state(ingredient_name, state_label)
        seq.append(1)
        task_id = f"prep_{ingredient_name}_{node.operation_key}_{sum(seq)}"

        # Duration via batch formula: τ_setup + ρ × Q
        # 用批量公式计算时长：τ_setup + ρ × Q
        duration = int(setup_minutes + minutes_per_base_unit * node.total_quantity)
        duration = max(duration, 1)  # Minimum 1 minute  最小 1 分钟

        # P2-01: parent-child ordering is expressed ONLY through food-state
        # consume/produce pairs — the graph builder (build_task_graph) turns
        # produces_states → consumes_states into real edges. Writing the
        # parent food-state as a TaskDependency.predecessor_id would create a
        # dangling edge (food states are not task IDs).
        # P2-01：父子顺序只通过食材状态的 consume/produce 对表达 —— 图构建器
        # （build_task_graph）把 produces_states → consumes_states 转为真实边。
        # 把父食材状态写成 TaskDependency.predecessor_id 会产生悬空边（食材状态不是任务 ID）。
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
        # 以该节点的状态作为父状态递归进入子节点
        for child in node.child_nodes.values():
            _dfs(child, state, seq)

        return state

    _dfs(root, None, [])
    return tuple(tasks)


# ============================================================================
# 6.7  Quantity-conservation verification
# 6.7  数量守恒校验
# ============================================================================


def verify_quantity_conservation(
    root: PrepTrieNode,
) -> None:
    """跨前缀树验证数量守恒不变量。

    Verify the quantity-conservation invariant across the trie.

    Q_parent = Σ Q_child + Q_consumed_at_parent

    In the MVP trie, quantities are managed as: each operation carries
    the total quantity that passes through its node.  The invariant is
    that child nodes' quantities sum to the parent's quantity (because
    we have no 'consumed at parent' leftover in MVP).

    MVP 前缀树中数量管理为：每个操作携带经过其节点的总量。不变量是子节点数量之和
    等于父节点数量（因为 MVP 中没有“父节点消耗”的剩余）。

    Raises:
        InvalidQuantityError: If conservation is violated at any node.
        InvalidQuantityError：任一节点守恒被违反时抛出。
    """
    _verify_node(root)


def _verify_node(node: PrepTrieNode) -> None:
    """递归检查数量守恒。"""
    if not node.child_nodes:
        return

    children_sum = sum(child.total_quantity for child in node.child_nodes.values())
    # Root node is exempt — it accumulates totals from all chains.
    # 根节点豁免 —— 它累加所有链的总量。
    if node.operation_key != "root" and children_sum != node.total_quantity:
        raise InvalidQuantityError(
            f"Quantity conservation violated at node {node.operation_key!r}: "
            f"parent={node.total_quantity}, children_sum={children_sum}"
        )

    for child in node.child_nodes.values():
        _verify_node(child)


# ============================================================================
# Infer preparation chains from IngredientDemand
# 从 IngredientDemand 推断预处理链
# ============================================================================

# Pantry seasonings and cooking media are ready to use; treating their generic
# raw input state as produce and scheduling a sink wash is both nonsensical
# and makes a multi-dish schedule much longer. This is intentionally a
# conservative keyword allow-list: items not matched here retain the existing
# wash-first behaviour until a proper ingredient taxonomy is available.
# 调味品与烹饪介质开袋即用；把它们通用的 raw 输入状态当作生鲜并调度水槽清洗，
# 既荒谬又会大大拉长多菜调度。这是刻意的保守关键词白名单：此处未匹配的项保持
# 现有“先清洗”行为，直到有合适的食材分类体系。
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
    """返回一个 raw 食材是否应得到一个推断的清洗步骤。"""
    name = demand.canonical_name.lower()
    return not any(keyword in name for keyword in _NON_WASHABLE_INGREDIENT_KEYWORDS)


def _infer_prep_chain(demand: IngredientDemand) -> tuple[PreparationOperation, ...]:
    """从 IngredientDemand 推断一条基础预处理链。

    Infer a basic preparation chain from an IngredientDemand.

    Derives operations from ``preparation_spec`` and ``input_state``.
    For MVP, uses a simple keyword-based mapping.  Production systems
    should use a catalogue of vetted prep chains.

    从 preparation_spec 与 input_state 派生操作。MVP 用简单的基于关键词映射。
    生产系统应使用审核过的预处理链目录。

    Example:
        demand with preparation_spec="diced" →
        (wash → peel(if applicable) → dice → portion)

    例：preparation_spec="diced" 的需求 →（洗 → 去皮（如适用）→ 切丁 → 分份）
    """
    ops: list[PreparationOperation] = []

    # Start with wash only for raw, washable ingredients. Condiments and
    # cooking media are supplied ready to use even though the MVP model marks
    # otherwise-unclassified inputs as raw.
    # 仅对 raw、可清洗食材从“洗”开始。调味品与烹饪介质即使 MVP 模型把
    # 未分类输入标为 raw，也按即用处理。
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
    # 从 preparation_spec 推断切割操作
    if demand.preparation_spec:
        spec_lower = demand.preparation_spec.lower()
        # Map common specs to cutting operations.
        # 把常见规格映射到切割操作
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
# P2-01  主工作流的共享预处理合并
# ============================================================================

# Raw proteins must NEVER share preparation operations with ready-to-eat
# items, even when the operation word is identical (P2-01 D2). Matching is
# keyword-based on the canonical ingredient name; production systems should
# source this from a vetted ingredient catalogue.
# 生蛋白质绝不能与即食物品共享预处理操作，即使操作词相同（P2-01 D2）。
# 匹配基于规范食材名的关键词；生产系统应从审核过的食材目录获取。
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
    """为预处理共享安全隔离对食材分类（P2-01 D2）。

    Categorise an ingredient for prep-sharing safety isolation (P2-01 D2).

    Returns ``"raw_protein"`` when the ingredient is an uncooked animal
    protein, otherwise ``"rte"`` (ready-to-eat / general). Raw-protein and
    RTE demands of the same ingredient build into separate tries so their
    operations never merge.

    当食材是未烹煮的动物蛋白时返回 "raw_protein"，否则返回 "rte"（即食 / 通用）。
    同一食材的 raw-protein 与 RTE 需求构建进不同前缀树，使它们的操作绝不合并。
    """
    name = canonical_name.lower()
    if input_state == "raw" and any(kw in name for kw in _RAW_PROTEIN_KEYWORDS):
        return "raw_protein"
    return "rte"


def demand_final_states_for_ingredient(
    root: PrepTrieNode,
    ingredient_name: str,
) -> dict[str, str]:
    """把每个 demand_id 映射到其终端预处理节点产出的食材状态。

    Map each demand_id to the food state its terminal prep node produces.

    A demand's final state is the state of the deepest (leaf) node it passes
    through. Leaf nodes carry the demand_ids of the chains that end there.
    The state label follows ``convert_trie_to_tasks`` exactly (``:`` in the
    operation key is replaced with ``_``).

    需求的最终状态是它经过的最深（叶）节点的状态。叶节点携带终结于此的链的 demand_ids。
    状态标签与 convert_trie_to_tasks 完全一致（操作键中的 ``:`` 替换为 ``_``）。

    Args:
        root: The populated trie for one ingredient (and safety class).
            root：一个食材（及安全类别）的已填充前缀树。
        ingredient_name: Canonical ingredient name used in state labels.
            ingredient_name：状态标签中使用的规范食材名。

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
    """共享预处理合并的输出（P2-01）。

    Output of shared-preparation merging (P2-01).

    Attributes
    ----------
    tasks
        Merged preparation tasks across all ingredient demands.
        所有食材需求合并后的预处理任务。
    demand_final_states
        ``demand_id -> final food state`` so the workflow can wire the
        consuming recipe task (``recipe_id:index`` keys).
        ``demand_id -> 最终食材状态``，使工作流能接线消费菜谱任务（recipe_id:index 键）。
    observations
        Human-readable summaries of merge/branch/isolate decisions for
        observability (logged and stored in workflow state).
        合并 / 分支 / 隔离决策的可读摘要，用于可观测性（记录并存入工作流状态）。
    """

    tasks: tuple[CookingTask, ...]
    demand_final_states: dict[str, str]
    observations: tuple[str, ...]


def _batch_wash_tasks(tasks: list[CookingTask]) -> list[CookingTask]:
    """把独立的生鲜食材清洗合并为一次水槽操作。

    Combine independent fresh-ingredient washes into one basin operation.

    All wash candidates entering this function are non-protein ingredients:
    raw proteins and pantry items are excluded by the preparation inference.
    A single task produces every original food state, so downstream recipe
    dependencies remain intact while the user performs one concentrated wash
    and drain operation instead of repeatedly returning to the sink.

    进入此函数的所有清洗候选都是非蛋白质食材：生蛋白质与调味品被预处理推断排除。
    单个任务产出所有原始食材状态，使下游菜谱依赖保持完整，而用户只需做一次集中
    清洗沥干操作，不必反复回到水槽。
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
    # 一次大水槽操作：5 分钟准备 / 冲洗，另每三个额外食材组约 1 分钟，上限 15 分钟。
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
    """通过前缀树跨菜谱合并相同预处理链。

    Merge identical preparation chains across recipes via prefix tries.

    Workflow step (P2-01):
    1. Group demands by (canonical ingredient, safety class) — raw proteins
       never merge with RTE items (D2), and different ingredients never merge.
    2. Build one trie per group; identical operation prefixes aggregate
       quantities; divergent cuts branch.
    3. Verify quantity conservation (D1) — a violation raises
       ``InvalidQuantityError`` and the workflow must NOT build a task graph.
    4. Convert each trie to prep tasks; produce per-demand final food states.

    工作流步骤（P2-01）：
    1. 按（规范食材、安全类别）分组 —— 生蛋白质绝不与 RTE 项合并（D2），
       不同食材绝不合并。
    2. 每组构建一个前缀树；相同操作前缀聚合数量；不同切割分叉。
    3. 校验数量守恒（D1）—— 违反则抛 InvalidQuantityError，工作流绝不构建任务图。
    4. 把每个前缀树转为预处理任务；产出每个需求的最终食材状态。

    Args:
        demands: ``(recipe_id, demand)`` pairs. Demand IDs are derived as
            ``"{recipe_id}:{index}"`` where index is the demand's ordinal
            within its recipe.
            demands：(recipe_id, demand) 对。需求 ID 派生为 "{recipe_id}:{index}"，
            其中 index 是需求在其菜谱中的序号。

    Returns:
        SharedPrepResult with merged tasks, final-state mapping, and
        observations.
        含合并任务、最终状态映射与观察摘要的 SharedPrepResult。

    Raises:
        InvalidQuantityError: If merged quantity does not equal the sum of
            the input demand quantities.
        InvalidQuantityError：若合并数量不等于输入需求数量之和。
    """
    # 1. Group by (canonical_name, safety_class).
    # 1. 按（canonical_name, safety_class）分组
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
    # 追踪同一食材的 raw/RTE 共现（隔离报告）
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
            continue  # No preparable operations for this demand group.  该需求组无可预处理操作

        # 3. Quantity conservation (D1): aggregated == sum of demands.
        # 3. 数量守恒（D1）：聚合 == 需求之和
        verify_quantity_conservation(root)
        # The root node only tracks demand_ids (insert_operation_chain never
        # aggregates on it); the first-level nodes carry the per-ingredient
        # totals, which must equal the sum of every demand quantity.
        # 根节点只追踪 demand_ids（insert_operation_chain 从不在其上聚合）；
        # 第一层节点携带每个食材的总量，必须等于每个需求数量之和。
        trie_total = sum(child.total_quantity for child in root.child_nodes.values())
        if trie_total != group_total:
            raise InvalidQuantityError(
                f"Quantity conservation violated for {name!r}: trie total={trie_total}, demand sum={group_total}"
            )

        # 4. Convert trie → tasks and record final states + observations.
        # 4. 前缀树 → 任务，并记录最终状态 + 观察摘要
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
