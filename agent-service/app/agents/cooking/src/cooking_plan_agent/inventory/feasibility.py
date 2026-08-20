# =============================================================================
# 库存与资源可行性检查模块（inventory/feasibility）
# -----------------------------------------------------------------------------
# 实现手册 5.10–5.16 的“库存 / 资源可行性检查”，核心职责：
#   - is_lot_usable              ：判断库存批次是否可用（数量 + 过期）
#   - allocate_fefo              ：按 FEFO（先到期先出）为单个食材分配批次
#   - check_all_inventory        ：聚合所有食材需求并逐一做 FEFO 分配，产出可行性报告
#   - resource_is_compatible     ：判断厨房资源是否满足某项资源需求
#   - check_required_resources   ：检查所有任务所需的资源类型是否齐备
#   - build_reservation_proposal ：由可行性报告生成库存消耗方案（供 READY 响应）
# 设计：纯领域函数，只操作不可变 Pydantic 模型，无 I/O，无共享可变状态，
#       每个函数均可独立测试。
# =============================================================================

"""Inventory and resource feasibility checks — handbook 5.10–5.16.

库存与资源可行性检查 —— 手册 5.10–5.16。

Pure domain functions: operate on immutable Pydantic models, no I/O.
Every function is independently testable — no shared mutable state.

纯领域函数：只操作不可变 Pydantic 模型，无 I/O。每个函数独立可测 —— 无共享可变状态。
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

from cooking_plan_agent.domain.models import (
    CompletionItem,
    CookingTask,
    FeasibilityReport,
    IngredientDemand,
    IngredientFeasibility,
    InventoryConsumptionProposal,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    LotAllocation,
    ResourceNeed,
)
from cooking_plan_agent.normalisation.names import (
    normalise_ingredient_name,
    normalise_resource_type,
)

# =============================================================================
# 5.10  Lot usability
# 5.10  批次可用性
# =============================================================================


def is_lot_usable(
    lot: InventoryLotSnapshot,
    cooking_date: date | None = None,
) -> bool:
    """判断某个库存批次在给定烹饪日期是否可用。

    Check whether an inventory lot is usable on the given cooking date.

    A lot is usable when:
      - available_quantity > 0 (on_hand minus reserved)
      - If cooking_date is provided and lot.expiry_date is set:
        cooking_date <= expiry_date

    批次可用需满足：
      - 可用数量 > 0（on_hand 减 reserved）
      - 若提供了 cooking_date 且批次设置了 expiry_date：cooking_date <= expiry_date

    Args:
        lot: An inventory lot snapshot.
            lot：库存批次快照。
        cooking_date: The target cooking date. If None, expiry is NOT checked
            (e.g. inventory pre-check before a date is confirmed).
            cooking_date：目标烹饪日期。为 None 时不检查过期（如日期确认前的库存预检）。

    Returns:
        True if the lot can be drawn from.
        若该批次可被取用则返回 True。
    """
    if available_lot_quantity(lot) <= 0:
        return False

    if cooking_date is not None and lot.expiry_date is not None:
        if cooking_date > lot.expiry_date:
            return False

    return True


def available_lot_quantity(lot: InventoryLotSnapshot) -> Decimal:
    """返回批次的可用数量：on_hand - reserved。

    Return the usable quantity for a lot: on_hand - reserved.

    InventoryLotSnapshot guarantees reserved <= on_hand at construction,
    so the result is always >= 0.

    InventoryLotSnapshot 在构造时保证 reserved <= on_hand，因此结果恒 >= 0。

    Args:
        lot: An inventory lot snapshot.
            lot：库存批次快照。

    Returns:
        Unreserved quantity available for allocation.
        可供分配（未预留）的数量。
    """
    return lot.on_hand - lot.reserved


# =============================================================================
# 5.11  FEFO allocation
# 5.11  FEFO 分配
# =============================================================================


def allocate_fefo(
    requirement: IngredientDemand,
    lots: tuple[InventoryLotSnapshot, ...],
    cooking_date: date | None = None,
) -> IngredientFeasibility:
    """按 FEFO 策略为单个食材需求分配库存批次。

    Allocate inventory lots to fulfil one ingredient requirement via FEFO.

    FEFO = First Expiry First Out:
      1. Filter lots matching the ingredient (case-insensitive name match).
      2. Filter to usable lots only (see is_lot_usable).
      3. Sort by expiry_date ascending (None = no expiry → last).
      4. Allocate from earliest expiry forward until requirement is met.
      5. Track shortage and produce proposed LotAllocation objects.

    FEFO = 先到期先出：
      1. 过滤出名称匹配该食材的批次（大小写不敏感）。
      2. 再过滤出可用批次（见 is_lot_usable）。
      3. 按 expiry_date 升序排序（None 表示无过期 → 排最后）。
      4. 从最早到期的批次向前分配，直到满足需求。
      5. 追踪缺口并产出拟分配 LotAllocation 对象。

    Args:
        requirement: An IngredientDemand to fulfil.
            requirement：待满足的食材需求。
        lots: All available inventory lot snapshots.
            lots：所有可用库存批次快照。
        cooking_date: The target cooking date for expiry checking.
            cooking_date：用于过期检查的目标烹饪日期。

    Returns:
        IngredientFeasibility with shortage (0 if fully satisfiable) and
        proposed allocations.
        含 shortage（完全满足时为 0）与拟分配的 IngredientFeasibility。
    """
    required_name = normalise_ingredient_name(requirement.canonical_name)

    # Step 1–2: filter matching + usable lots
    # 第 1–2 步：过滤出名称匹配 + 可用的批次
    matching = [
        lot
        for lot in lots
        if normalise_ingredient_name(lot.canonical_name) == required_name and is_lot_usable(lot, cooking_date)
    ]

    # Step 3: sort by expiry (earliest first, None last)
    # 第 3 步：按过期排序（最早在前，无过期最后）
    matching.sort(
        key=lambda lot: (
            # Items WITHOUT expiry_date sort AFTER items with one
            # 无 expiry_date 的批次排在“有 expiry_date 的批次”之后
            (0, lot.expiry_date) if lot.expiry_date is not None else (1, date.max)
        )
    )

    # Step 4: allocate greedily
    # 第 4 步：贪心分配
    allocated: list[LotAllocation] = []
    remaining = requirement.quantity

    for lot in matching:
        available = available_lot_quantity(lot)
        if available <= 0:
            continue

        take = min(remaining, available)
        allocated.append(
            LotAllocation(
                inventory_lot_id=lot.lot_id,
                quantity=take,
                unit=lot.unit,
            )
        )
        remaining -= take
        if remaining <= 0:
            break

    shortage = max(Decimal(0), remaining)

    return IngredientFeasibility(
        ingredient_name=requirement.canonical_name,
        required=requirement.quantity,
        available=requirement.quantity - shortage,
        shortage=shortage,
        unit=requirement.unit,
        proposed_allocations=tuple(allocated),
    )


# =============================================================================
# 5.12  Aggregate ingredient check
# 5.12  聚合食材检查
# =============================================================================


def _aggregate_demands(
    demands: tuple[IngredientDemand, ...],
) -> dict[str, IngredientDemand]:
    """按 canonical_name 聚合食材需求。

    Aggregate ingredient demands by canonical_name.

    Sums quantities for the same ingredient across recipes.
    The returned demands use the first occurrence's unit — cross-recipe
    unit mismatches must be resolved by upstream canonicalisation.

    跨菜谱对同一食材的数量求和。返回的需求使用首次出现的单位 ——
    跨菜谱单位不一致必须由上游规范化解决。
    """
    aggregated: dict[str, IngredientDemand] = {}
    for d in demands:
        key = normalise_ingredient_name(d.canonical_name)
        if key in aggregated:
            existing = aggregated[key]
            # Quantity summing — unit must match (upstream responsibility)
            # 数量求和 —— 单位必须一致（由上游负责）
            new_quantity = existing.quantity + d.quantity
            aggregated[key] = existing.model_copy(update={"quantity": new_quantity})
        else:
            aggregated[key] = d
    return aggregated


def check_all_inventory(
    requirements: tuple[IngredientDemand, ...],
    lots: tuple[InventoryLotSnapshot, ...],
    cooking_date: date | None = None,
) -> FeasibilityReport:
    """检查所有食材需求的库存是否充足。

    Check inventory sufficiency for all ingredient requirements.

    Aggregates duplicate ingredients, then runs FEFO allocation for each.
    Returns a FeasibilityReport with per-ingredient shortage details.

    先聚合重复食材，再对每个食材执行 FEFO 分配，返回含每个食材缺口明细的 FeasibilityReport。

    Args:
        requirements: All ingredient demands across all recipes.
            requirements：所有菜谱的全部食材需求。
        lots: Available inventory lot snapshots.
            lots：可用库存批次快照。
        cooking_date: The target cooking date.
            cooking_date：目标烹饪日期。

    Returns:
        FeasibilityReport with is_feasible = True only when every ingredient
        is fully satisfiable.
        仅当每个食材都可被完全满足时 is_feasible 才为 True。
    """
    aggregated = _aggregate_demands(requirements)

    results: list[IngredientFeasibility] = []
    shortages: list[IngredientFeasibility] = []
    for demand in aggregated.values():
        result = allocate_fefo(demand, lots, cooking_date)
        # 保留每个食材的完整分配结果（含满足的食材），供 READY 消耗清单使用；
        # ingredient_shortages 仍只收录有缺口的条目，确认/修复语义不变。
        results.append(result)
        if result.shortage > 0:
            shortages.append(result)

    is_feasible = len(shortages) == 0

    return FeasibilityReport(
        report_id=f"inv_{uuid4().hex[:12]}",
        ingredient_shortages=tuple(shortages),
        missing_resources=(),  # Inventory check only — resources checked separately
        # ↑ 仅做库存检查 —— 资源在别处单独检查
        is_feasible=is_feasible,
        ingredient_results=tuple(results),
    )


# =============================================================================
# 5.13–5.14  Resource compatibility
# 5.13–5.14  资源兼容性
# =============================================================================


def resource_is_compatible(
    need: ResourceNeed,
    resource: KitchenResourceSnapshot,
) -> bool:
    """判断某个 KitchenResourceSnapshot 是否满足某项 ResourceNeed。

    Check whether a KitchenResourceSnapshot satisfies a ResourceNeed.

    Compatibility requires:
      - resource_type matches (exact, case-insensitive)
      - resource is available
      - resource has all required_capabilities (⊆ check)
      - resource capacity >= need.minimum_capacity (if both are set)

    兼容需满足：
      - resource_type 匹配（精确、大小写不敏感）
      - 资源可用
      - 资源具备所有 required_capabilities（子集 ⊆ 检查）
      - resource capacity >= need.minimum_capacity（当两者均设置时）

    Args:
        need: A resource requirement from a CookingTask.
            need：来自 CookingTask 的资源需求。
        resource: An available kitchen resource snapshot.
            resource：可用的厨房资源快照。

    Returns:
        True if the resource satisfies the need.
        若该资源满足需求则返回 True。
    """
    # Type match
    # 类型匹配
    if normalise_resource_type(resource.resource_type) != normalise_resource_type(need.resource_type):
        return False

    # Availability
    # 可用性
    if not resource.available:
        return False

    # Capabilities: all required capabilities must be present
    # 能力：所有必需能力都必须具备
    if need.required_capabilities:
        resource_caps = {c.lower() for c in resource.capabilities}
        needed_caps = {c.lower() for c in need.required_capabilities}
        if not needed_caps.issubset(resource_caps):
            return False

    # Capacity check
    # 容量检查
    if need.minimum_capacity is not None and resource.capacity is not None:
        # Units must match for capacity comparison
        # 容量比较前单位必须一致
        if need.capacity_unit and resource.capacity_unit:
            if need.capacity_unit.lower() != resource.capacity_unit.lower():
                return False
        if resource.capacity < need.minimum_capacity:
            return False

    return True


def find_compatible_resources(
    need: ResourceNeed,
    resources: tuple[KitchenResourceSnapshot, ...],
) -> tuple[str, ...]:
    """找出所有与给定 ResourceNeed 兼容的资源 ID。

    Find all resource IDs compatible with the given ResourceNeed.

    Args:
        need: A resource requirement from a CookingTask.
            need：来自 CookingTask 的资源需求。
        resources: All available kitchen resource snapshots.
            resources：所有可用厨房资源快照。

    Returns:
        Tuple of resource_id strings (may be empty if no compatible resource).
        资源 ID 字符串元组（若无兼容资源则为空）。
    """
    return tuple(r.resource_id for r in resources if resource_is_compatible(need, r))


def check_required_resources(
    tasks: tuple[CookingTask, ...],
    resources: tuple[KitchenResourceSnapshot, ...],
) -> tuple[str, ...]:
    """检查所有任务所需的厨房资源是否齐备。

    Check all tasks against available kitchen resources.

    Returns the set of resource types that are required by at least one
    task but have NO compatible resource available.

    返回“被至少一个任务需要、但没有任何兼容资源可用”的资源类型集合。

    Args:
        tasks: All cooking tasks to check.
            tasks：要检查的所有烹饪任务。
        resources: Available kitchen resource snapshots.
            resources：可用厨房资源快照。

    Returns:
        Tuple of missing resource_type strings. Empty = all needs satisfied.
        缺失资源类型字符串元组。空 = 所有需求均被满足。
    """
    missing: set[str] = set()

    for task in tasks:
        for need in task.resources:
            compatible = find_compatible_resources(need, resources)
            if not compatible:
                # Include the required capability in the description
                # 在描述中附带所需能力
                desc = need.resource_type
                if need.required_capabilities:
                    desc += f":{','.join(need.required_capabilities)}"
                missing.add(desc)

    return tuple(sorted(missing))


# =============================================================================
# 5.16  Reservation proposal
# 5.16  预留方案
# =============================================================================


def build_reservation_proposal(
    report: FeasibilityReport,
) -> InventoryConsumptionProposal:
    """由可行性报告构建库存消耗方案（InventoryConsumptionProposal）。

    Build an InventoryConsumptionProposal from a feasibility report.

    Converts each ingredient's FEFO allocations into a CompletionItem
    grouped by ingredient. Sources the FULL allocation results
    (``ingredient_results`` — every ingredient, satisfied or not) so a
    READY plan carries the consumption plan for all ingredients, not
    only the short ones; falls back to ``ingredient_shortages`` for
    callers that still construct legacy reports. The snapshot version is
    derived from the number of allocations (simple non-crypto version
    for MVP).

    将每个食材的 FEFO 分配转换为按食材分组的 CompletionItem。优先取完整的分配结果
    （``ingredient_results`` —— 每个食材，无论满足与否），使 READY 计划携带所有食材的
    消耗方案，而非仅有缺口的食材；对仍构造旧报告的调用方回退到 ``ingredient_shortages``。
    快照版本由分配数量派生（MVP 用的简单非加密版本）。

    Args:
        report: A FeasibilityReport with ingredient allocations.
            report：含食材分配的 FeasibilityReport。

    Returns:
        InventoryConsumptionProposal ready for inclusion in a READY response.
        可直接放入 READY 响应的 InventoryConsumptionProposal。
    """
    items: list[CompletionItem] = []

    # 优先使用完整分配结果；兼容仅含短缺条目的旧报告。
    sources = report.ingredient_results or report.ingredient_shortages
    for result in sources:
        if not result.proposed_allocations:
            continue

        items.append(
            CompletionItem(
                completion_item_id=f"comp_{result.ingredient_name}_{uuid4().hex[:8]}",
                ingredient_name=result.ingredient_name,
                recipe_ids=(),  # MVP: recipe attribution deferred to rendering layer
                # ↑ MVP：菜谱归因延后到渲染层处理
                allocations=result.proposed_allocations,
            )
        )

    # Simple snapshot version: count of total allocations
    # 简单快照版本：以分配总数计
    total_allocations = sum(len(item.allocations) for item in items)
    snapshot_version = f"v1_{total_allocations}_{uuid4().hex[:8]}"

    return InventoryConsumptionProposal(
        inventory_snapshot_version=snapshot_version,
        items=tuple(items),
    )
