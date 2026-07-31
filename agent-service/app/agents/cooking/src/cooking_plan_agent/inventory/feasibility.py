"""Inventory and resource feasibility checks — handbook 5.10–5.16.

Pure domain functions: operate on immutable Pydantic models, no I/O.
Every function is independently testable — no shared mutable state.
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

# =============================================================================
# 5.10  Lot usability
# =============================================================================


def is_lot_usable(
    lot: InventoryLotSnapshot,
    cooking_date: date | None = None,
) -> bool:
    """Check whether an inventory lot is usable on the given cooking date.

    A lot is usable when:
      - available_quantity > 0 (on_hand minus reserved)
      - If cooking_date is provided and lot.expiry_date is set:
        cooking_date <= expiry_date

    Args:
        lot: An inventory lot snapshot.
        cooking_date: The target cooking date. If None, expiry is NOT checked
            (e.g. inventory pre-check before a date is confirmed).

    Returns:
        True if the lot can be drawn from.
    """
    if available_lot_quantity(lot) <= 0:
        return False

    if cooking_date is not None and lot.expiry_date is not None:
        if cooking_date > lot.expiry_date:
            return False

    return True


def available_lot_quantity(lot: InventoryLotSnapshot) -> Decimal:
    """Return the usable quantity for a lot: on_hand - reserved.

    InventoryLotSnapshot guarantees reserved <= on_hand at construction,
    so the result is always >= 0.

    Args:
        lot: An inventory lot snapshot.

    Returns:
        Unreserved quantity available for allocation.
    """
    return lot.on_hand - lot.reserved


# =============================================================================
# 5.11  FEFO allocation
# =============================================================================


def allocate_fefo(
    requirement: IngredientDemand,
    lots: tuple[InventoryLotSnapshot, ...],
    cooking_date: date | None = None,
) -> IngredientFeasibility:
    """Allocate inventory lots to fulfil one ingredient requirement via FEFO.

    FEFO = First Expiry First Out:
      1. Filter lots matching the ingredient (case-insensitive name match).
      2. Filter to usable lots only (see is_lot_usable).
      3. Sort by expiry_date ascending (None = no expiry → last).
      4. Allocate from earliest expiry forward until requirement is met.
      5. Track shortage and produce proposed LotAllocation objects.

    Args:
        requirement: An IngredientDemand to fulfil.
        lots: All available inventory lot snapshots.
        cooking_date: The target cooking date for expiry checking.

    Returns:
        IngredientFeasibility with shortage (0 if fully satisfiable) and
        proposed allocations.
    """
    required_name = requirement.canonical_name.lower().strip()

    # Step 1–2: filter matching + usable lots
    matching = [
        lot for lot in lots
        if lot.canonical_name.lower().strip() == required_name
        and is_lot_usable(lot, cooking_date)
    ]

    # Step 3: sort by expiry (earliest first, None last)
    matching.sort(
        key=lambda lot: (
            # Items WITHOUT expiry_date sort AFTER items with one
            (0, lot.expiry_date) if lot.expiry_date is not None else (1, date.max)
        )
    )

    # Step 4: allocate greedily
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
# =============================================================================


def _aggregate_demands(
    demands: tuple[IngredientDemand, ...],
) -> dict[str, IngredientDemand]:
    """Aggregate ingredient demands by canonical_name.

    Sums quantities for the same ingredient across recipes.
    The returned demands use the first occurrence's unit — cross-recipe
    unit mismatches must be resolved by upstream canonicalisation.
    """
    aggregated: dict[str, IngredientDemand] = {}
    for d in demands:
        key = d.canonical_name.lower().strip()
        if key in aggregated:
            existing = aggregated[key]
            # Quantity summing — unit must match (upstream responsibility)
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
    """Check inventory sufficiency for all ingredient requirements.

    Aggregates duplicate ingredients, then runs FEFO allocation for each.
    Returns a FeasibilityReport with per-ingredient shortage details.

    Args:
        requirements: All ingredient demands across all recipes.
        lots: Available inventory lot snapshots.
        cooking_date: The target cooking date.

    Returns:
        FeasibilityReport with is_feasible = True only when every ingredient
        is fully satisfiable.
    """
    aggregated = _aggregate_demands(requirements)

    shortages: list[IngredientFeasibility] = []
    for demand in aggregated.values():
        result = allocate_fefo(demand, lots, cooking_date)
        if result.shortage > 0:
            shortages.append(result)

    is_feasible = len(shortages) == 0

    return FeasibilityReport(
        report_id=f"inv_{uuid4().hex[:12]}",
        ingredient_shortages=tuple(shortages),
        missing_resources=(),  # Inventory check only — resources checked separately
        is_feasible=is_feasible,
    )


# =============================================================================
# 5.13–5.14  Resource compatibility
# =============================================================================


def resource_is_compatible(
    need: ResourceNeed,
    resource: KitchenResourceSnapshot,
) -> bool:
    """Check whether a KitchenResourceSnapshot satisfies a ResourceNeed.

    Compatibility requires:
      - resource_type matches (exact, case-insensitive)
      - resource is available
      - resource has all required_capabilities (⊆ check)
      - resource capacity >= need.minimum_capacity (if both are set)

    Args:
        need: A resource requirement from a CookingTask.
        resource: An available kitchen resource snapshot.

    Returns:
        True if the resource satisfies the need.
    """
    # Type match
    if resource.resource_type.lower() != need.resource_type.lower():
        return False

    # Availability
    if not resource.available:
        return False

    # Capabilities: all required capabilities must be present
    if need.required_capabilities:
        resource_caps = {c.lower() for c in resource.capabilities}
        needed_caps = {c.lower() for c in need.required_capabilities}
        if not needed_caps.issubset(resource_caps):
            return False

    # Capacity check
    if need.minimum_capacity is not None and resource.capacity is not None:
        # Units must match for capacity comparison
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
    """Find all resource IDs compatible with the given ResourceNeed.

    Args:
        need: A resource requirement from a CookingTask.
        resources: All available kitchen resource snapshots.

    Returns:
        Tuple of resource_id strings (may be empty if no compatible resource).
    """
    return tuple(
        r.resource_id
        for r in resources
        if resource_is_compatible(need, r)
    )


def check_required_resources(
    tasks: tuple[CookingTask, ...],
    resources: tuple[KitchenResourceSnapshot, ...],
) -> tuple[str, ...]:
    """Check all tasks against available kitchen resources.

    Returns the set of resource types that are required by at least one
    task but have NO compatible resource available.

    Args:
        tasks: All cooking tasks to check.
        resources: Available kitchen resource snapshots.

    Returns:
        Tuple of missing resource_type strings. Empty = all needs satisfied.
    """
    missing: set[str] = set()

    for task in tasks:
        for need in task.resources:
            compatible = find_compatible_resources(need, resources)
            if not compatible:
                # Include the required capability in the description
                desc = need.resource_type
                if need.required_capabilities:
                    desc += f":{','.join(need.required_capabilities)}"
                missing.add(desc)

    return tuple(sorted(missing))


# =============================================================================
# 5.16  Reservation proposal
# =============================================================================


def build_reservation_proposal(
    report: FeasibilityReport,
) -> InventoryConsumptionProposal:
    """Build an InventoryConsumptionProposal from a feasibility report.

    Converts each IngredientFeasibility.proposed_allocations into a
    CompletionItem grouped by ingredient. The snapshot version is derived
    from the number of allocations (simple non-crypto version for MVP).

    Args:
        report: A FeasibilityReport with ingredient_shortages that carry
            proposed_allocations.

    Returns:
        InventoryConsumptionProposal ready for inclusion in a READY response.
    """
    items: list[CompletionItem] = []

    for shortage in report.ingredient_shortages:
        if not shortage.proposed_allocations:
            continue

        items.append(
            CompletionItem(
                completion_item_id=f"comp_{shortage.ingredient_name}_{uuid4().hex[:8]}",
                ingredient_name=shortage.ingredient_name,
                recipe_ids=(),  # MVP: recipe attribution deferred to rendering layer
                allocations=shortage.proposed_allocations,
            )
        )

    # Simple snapshot version: count of total allocations
    total_allocations = sum(len(item.allocations) for item in items)
    snapshot_version = f"v1_{total_allocations}_{uuid4().hex[:8]}"

    return InventoryConsumptionProposal(
        inventory_snapshot_version=snapshot_version,
        items=tuple(items),
    )
