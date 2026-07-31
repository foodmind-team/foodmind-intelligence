"""Unit tests for inventory and resource feasibility — handbook 5.10–5.16."""

from datetime import date
from decimal import Decimal

import pytest

from cooking_plan_agent.domain.models import (
    CookingTask,
    FeasibilityReport,
    IngredientDemand,
    IngredientFeasibility,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    LotAllocation,
    ResourceNeed,
)
from cooking_plan_agent.inventory.feasibility import (
    _aggregate_demands,
    allocate_fefo,
    available_lot_quantity,
    build_reservation_proposal,
    check_all_inventory,
    check_required_resources,
    find_compatible_resources,
    is_lot_usable,
    resource_is_compatible,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def chicken_lot() -> InventoryLotSnapshot:
    """A fresh chicken breast lot: 500 g, no expiration."""
    return InventoryLotSnapshot(
        lot_id="lot-001",
        item_id="item-chicken",
        canonical_name="chicken breast",
        on_hand=Decimal(500),
        reserved=Decimal(0),
        unit="g",
    )


@pytest.fixture
def chicken_lot_expiring() -> InventoryLotSnapshot:
    """Chicken lot expiring tomorrow: 300 g."""
    return InventoryLotSnapshot(
        lot_id="lot-002",
        item_id="item-chicken",
        canonical_name="chicken breast",
        on_hand=Decimal(300),
        reserved=Decimal(0),
        unit="g",
        expiry_date=date(2026, 8, 1),
    )


@pytest.fixture
def chicken_lot_expired() -> InventoryLotSnapshot:
    """Chicken lot already expired."""
    return InventoryLotSnapshot(
        lot_id="lot-003",
        item_id="item-chicken",
        canonical_name="chicken breast",
        on_hand=Decimal(200),
        reserved=Decimal(0),
        unit="g",
        expiry_date=date(2025, 1, 1),
    )


@pytest.fixture
def tomato_lot() -> InventoryLotSnapshot:
    """Tomato lot: 6 pieces."""
    return InventoryLotSnapshot(
        lot_id="lot-004",
        item_id="item-tomato",
        canonical_name="tomato",
        on_hand=Decimal(6),
        reserved=Decimal(0),
        unit="piece",
    )


@pytest.fixture
def chicken_demand() -> IngredientDemand:
    return IngredientDemand(
        canonical_name="chicken breast",
        raw_name="chicken breast",
        quantity=Decimal(400),
        unit="g",
        confidence=Decimal("0.95"),
    )


@pytest.fixture
def tomato_demand() -> IngredientDemand:
    return IngredientDemand(
        canonical_name="tomato",
        raw_name="tomato",
        quantity=Decimal(4),
        unit="piece",
        confidence=Decimal("0.9"),
    )


@pytest.fixture
def stove_resource() -> KitchenResourceSnapshot:
    return KitchenResourceSnapshot(
        resource_id="res-stove-1",
        resource_type="stove",
        capacity=Decimal(4),
        capacity_unit="burners",
        available=True,
    )


@pytest.fixture
def oven_resource() -> KitchenResourceSnapshot:
    return KitchenResourceSnapshot(
        resource_id="res-oven-1",
        resource_type="oven",
        capacity=Decimal(1),
        available=True,
    )


@pytest.fixture
def induction_resource() -> KitchenResourceSnapshot:
    return KitchenResourceSnapshot(
        resource_id="res-stove-induction",
        resource_type="stove",
        capacity=Decimal(2),
        capacity_unit="burners",
        capabilities=("induction",),
        available=True,
    )


_TODAY = date(2026, 7, 31)
_TOMORROW = date(2026, 8, 1)


# ======================================================================
# is_lot_usable
# ======================================================================


class TestIsLotUsable:
    """5.10 Lot usability checks."""

    def test_usable_with_positive_available(self, chicken_lot):
        assert is_lot_usable(chicken_lot, _TODAY) is True

    def test_unusable_when_fully_reserved(self, chicken_lot):
        lot = chicken_lot.model_copy(update={"reserved": chicken_lot.on_hand})
        assert is_lot_usable(lot, _TODAY) is False

    def test_usable_when_expiry_after_cooking_date(self, chicken_lot_expiring):
        assert is_lot_usable(chicken_lot_expiring, _TOMORROW) is True

    def test_unusable_when_expired(self, chicken_lot_expired):
        assert is_lot_usable(chicken_lot_expired, _TODAY) is False

    def test_usable_when_no_cooking_date_provided(self, chicken_lot_expired):
        # No cooking_date → skip expiry check → lot is usable
        assert is_lot_usable(chicken_lot_expired) is True

    def test_usable_when_no_expiry_date(self, chicken_lot):
        assert is_lot_usable(chicken_lot, _TODAY) is True


# ======================================================================
# available_lot_quantity
# ======================================================================


class TestAvailableLotQuantity:
    """5.10 Available quantity calculation."""

    def test_full_available_when_not_reserved(self, chicken_lot):
        assert available_lot_quantity(chicken_lot) == Decimal(500)

    def test_partial_available_when_partially_reserved(self, chicken_lot):
        lot = chicken_lot.model_copy(update={"reserved": Decimal(300)})
        assert available_lot_quantity(lot) == Decimal(200)

    def test_zero_when_fully_reserved(self, chicken_lot):
        lot = chicken_lot.model_copy(update={"reserved": Decimal(500)})
        assert available_lot_quantity(lot) == Decimal(0)


# ======================================================================
# allocate_fefo
# ======================================================================


class TestAllocateFefo:
    """5.11 FEFO allocation algorithm."""

    def test_fully_satisfied_from_single_lot(self, chicken_demand, chicken_lot):
        result = allocate_fefo(chicken_demand, (chicken_lot,), _TODAY)
        assert result.shortage == 0
        assert len(result.proposed_allocations) == 1
        alloc = result.proposed_allocations[0]
        assert alloc.inventory_lot_id == "lot-001"
        assert alloc.quantity == Decimal(400)

    def test_fefo_prefers_earliest_expiry(
        self, chicken_demand, chicken_lot, chicken_lot_expiring
    ):
        """Expiring lot (Aug 1) should be drawn first, then the non-expiring one."""
        # Demand 400: expiring has 300 → taken first, remainder from fresh lot
        result = allocate_fefo(
            chicken_demand, (chicken_lot, chicken_lot_expiring), _TODAY
        )
        assert result.shortage == 0
        assert result.available == Decimal(400)
        allocs = result.proposed_allocations
        # First allocation should be from the expiring lot
        assert allocs[0].inventory_lot_id == "lot-002"
        assert allocs[0].quantity == Decimal(300)
        assert allocs[1].inventory_lot_id == "lot-001"
        assert allocs[1].quantity == Decimal(100)

    def test_partial_shortage(self, chicken_demand):
        """Only 200g available out of 400g demand."""
        lot = InventoryLotSnapshot(
            lot_id="lot-small",
            item_id="item-chicken",
            canonical_name="chicken breast",
            on_hand=Decimal(200),
            reserved=Decimal(0),
            unit="g",
        )
        result = allocate_fefo(chicken_demand, (lot,), _TODAY)
        assert result.shortage == Decimal(200)
        assert result.available == Decimal(200)
        assert len(result.proposed_allocations) == 1

    def test_excludes_expired_lots(self, chicken_demand, chicken_lot_expired):
        """Expired lot should be excluded, resulting in full shortage."""
        result = allocate_fefo(chicken_demand, (chicken_lot_expired,), _TODAY)
        # Expired lot excluded → no usable lots → full shortage
        assert result.shortage == Decimal(400)
        assert result.available == Decimal(0)
        assert len(result.proposed_allocations) == 0

    def test_name_match_case_insensitive(self, chicken_demand):
        """'CHICKEN BREAST' should match 'chicken breast'."""
        lot = InventoryLotSnapshot(
            lot_id="lot-upper",
            item_id="item-chicken",
            canonical_name="CHICKEN BREAST",
            on_hand=Decimal(300),
            reserved=Decimal(0),
            unit="g",
        )
        result = allocate_fefo(chicken_demand, (lot,), _TODAY)
        assert result.shortage == Decimal(100)
        assert len(result.proposed_allocations) == 1

    def test_name_match_whitespace_insensitive(self, chicken_demand):
        """' chicken breast ' should match 'chicken breast'."""
        lot = InventoryLotSnapshot(
            lot_id="lot-spaces",
            item_id="item-chicken",
            canonical_name="  chicken breast  ",
            on_hand=Decimal(300),
            reserved=Decimal(0),
            unit="g",
        )
        result = allocate_fefo(chicken_demand, (lot,), _TODAY)
        assert result.shortage == Decimal(100)

    def test_exact_satisfaction(self, chicken_demand, chicken_lot):
        """Demand exactly matches available."""
        demand = chicken_demand.model_copy(update={"quantity": Decimal(500)})
        result = allocate_fefo(demand, (chicken_lot,), _TODAY)
        assert result.shortage == 0
        assert result.available == Decimal(500)

    def test_no_matching_lots(self, chicken_demand, tomato_lot):
        """No lots matching the ingredient → full shortage."""
        result = allocate_fefo(chicken_demand, (tomato_lot,), _TODAY)
        assert result.shortage == Decimal(400)
        assert result.available == Decimal(0)

    def test_empty_lots(self, chicken_demand):
        result = allocate_fefo(chicken_demand, (), _TODAY)
        assert result.shortage == Decimal(400)
        assert result.available == Decimal(0)


# ======================================================================
# check_all_inventory
# ======================================================================


class TestCheckAllInventory:
    """5.12 Aggregate inventory check."""

    def test_all_satisfied(
        self, chicken_demand, tomato_demand, chicken_lot, tomato_lot
    ):
        report = check_all_inventory(
            (chicken_demand, tomato_demand),
            (chicken_lot, tomato_lot),
        )
        assert report.is_feasible is True
        assert len(report.ingredient_shortages) == 0

    def test_partial_shortage(
        self, chicken_demand, tomato_demand, tomato_lot
    ):
        """Chicken has no matching lot → one shortage."""
        # chicken_demand: 400g, but only tomato lot available
        report = check_all_inventory(
            (chicken_demand, tomato_demand),
            (tomato_lot,),
        )
        assert report.is_feasible is False
        assert len(report.ingredient_shortages) == 1
        assert report.ingredient_shortages[0].ingredient_name == "chicken breast"

    def test_aggregates_duplicate_ingredients(self, chicken_demand, chicken_lot):
        """Two chicken demands should be aggregated before allocation."""
        demand_small = chicken_demand.model_copy(update={"quantity": Decimal(100)})
        # Total chicken demand: 500g, lot has 500g → should be satisfied
        report = check_all_inventory(
            (chicken_demand, demand_small),
            (chicken_lot,),
        )
        assert report.is_feasible is True

    def test_empty_demands(self):
        report = check_all_inventory((), ())
        assert report.is_feasible is True

    def test_empty_lots(self, chicken_demand):
        report = check_all_inventory((chicken_demand,), ())
        assert report.is_feasible is False

    def test_expired_lots_excluded(self, chicken_demand, chicken_lot_expired):
        report = check_all_inventory(
            (chicken_demand,), (chicken_lot_expired,), _TODAY
        )
        assert report.is_feasible is False

    def test_expired_lots_included_without_date(self, chicken_demand, chicken_lot_expired):
        """Without cooking_date, expired lots are treated as usable."""
        report = check_all_inventory(
            (chicken_demand,), (chicken_lot_expired,), None
        )
        assert report.is_feasible is False  # 200 < 400


# ======================================================================
# resource_is_compatible
# ======================================================================


class TestResourceIsCompatible:
    """5.13–5.14 Resource compatibility checks."""

    def test_exact_match(self, stove_resource):
        need = ResourceNeed(resource_type="stove", quantity=1)
        assert resource_is_compatible(need, stove_resource) is True

    def test_case_insensitive_type_match(self, stove_resource):
        need = ResourceNeed(resource_type="STOVE", quantity=1)
        assert resource_is_compatible(need, stove_resource) is True

    def test_type_mismatch(self, oven_resource):
        need = ResourceNeed(resource_type="stove", quantity=1)
        assert resource_is_compatible(need, oven_resource) is False

    def test_unavailable_resource(self, stove_resource):
        unavailable = stove_resource.model_copy(update={"available": False})
        need = ResourceNeed(resource_type="stove", quantity=1)
        assert resource_is_compatible(need, unavailable) is False

    def test_capability_match(self, induction_resource):
        need = ResourceNeed(
            resource_type="stove",
            quantity=1,
            required_capabilities=("induction",),
        )
        assert resource_is_compatible(need, induction_resource) is True

    def test_capability_mismatch(self, stove_resource):
        """Stove without induction capability should not match."""
        need = ResourceNeed(
            resource_type="stove",
            quantity=1,
            required_capabilities=("induction",),
        )
        assert resource_is_compatible(need, stove_resource) is False

    def test_capacity_sufficient(self, induction_resource):
        need = ResourceNeed(resource_type="stove", quantity=1, minimum_capacity=Decimal(1))
        assert resource_is_compatible(need, induction_resource) is True

    def test_capacity_insufficient(self, induction_resource):
        need = ResourceNeed(resource_type="stove", quantity=1, minimum_capacity=Decimal(10))
        assert resource_is_compatible(need, induction_resource) is False

    def test_capacity_mixed_units(self, stove_resource):
        """Different capacity units → incompatible (can't compare)."""
        need = ResourceNeed(
            resource_type="stove",
            quantity=1,
            minimum_capacity=Decimal(2),
            capacity_unit="L",
        )
        assert resource_is_compatible(need, stove_resource) is False

    def test_no_capacity_requirement(self, stove_resource):
        """Need without minimum_capacity should match any capacity."""
        need = ResourceNeed(resource_type="stove", quantity=1)
        assert resource_is_compatible(need, stove_resource) is True

    def test_no_resource_capacity(self):
        """Resource without capacity should match needs without minimum_capacity."""
        res = KitchenResourceSnapshot(
            resource_id="res-knife",
            resource_type="knife",
            available=True,
        )
        need = ResourceNeed(resource_type="knife", quantity=1)
        assert resource_is_compatible(need, res) is True


# ======================================================================
# find_compatible_resources
# ======================================================================


class TestFindCompatibleResources:
    """5.14 Find all compatible resources for a need."""

    def test_finds_single_match(self, stove_resource, oven_resource):
        need = ResourceNeed(resource_type="stove", quantity=1)
        result = find_compatible_resources(need, (stove_resource, oven_resource))
        assert result == ("res-stove-1",)

    def test_finds_multiple_matches(self, stove_resource, induction_resource):
        need = ResourceNeed(resource_type="stove", quantity=1)
        result = find_compatible_resources(
            need, (stove_resource, induction_resource)
        )
        assert set(result) == {"res-stove-1", "res-stove-induction"}

    def test_no_match(self, oven_resource):
        need = ResourceNeed(resource_type="stove", quantity=1)
        result = find_compatible_resources(need, (oven_resource,))
        assert result == ()

    def test_capability_filter(self, stove_resource, induction_resource):
        need = ResourceNeed(
            resource_type="stove",
            quantity=1,
            required_capabilities=("induction",),
        )
        result = find_compatible_resources(
            need, (stove_resource, induction_resource)
        )
        assert result == ("res-stove-induction",)


# ======================================================================
# check_required_resources
# ======================================================================


class TestCheckRequiredResources:
    """5.15 Check all task resource needs."""

    def test_all_satisfied(self, stove_resource, oven_resource):
        task = CookingTask(
            task_id="task-1",
            dish_id="dish-1",
            instruction="Stir-fry chicken",
            duration_minutes=5,
            work_mode="ACTIVE",
            category="heating",
            resources=(
                ResourceNeed(resource_type="stove", quantity=1),
            ),
        )
        missing = check_required_resources(
            (task,), (stove_resource, oven_resource)
        )
        assert missing == ()

    def test_missing_resource(self, oven_resource):
        task = CookingTask(
            task_id="task-1",
            dish_id="dish-1",
            instruction="Stir-fry chicken",
            duration_minutes=5,
            work_mode="ACTIVE",
            category="heating",
            resources=(
                ResourceNeed(resource_type="stove", quantity=1),
            ),
        )
        missing = check_required_resources((task,), (oven_resource,))
        assert "stove" in missing

    def test_missing_capability(self, stove_resource):
        task = CookingTask(
            task_id="task-1",
            dish_id="dish-1",
            instruction="Induction stir-fry",
            duration_minutes=5,
            work_mode="ACTIVE",
            category="heating",
            resources=(
                ResourceNeed(
                    resource_type="stove",
                    quantity=1,
                    required_capabilities=("induction",),
                ),
            ),
        )
        missing = check_required_resources((task,), (stove_resource,))
        assert "stove:induction" in missing

    def test_multiple_tasks_same_resource(self, stove_resource):
        """Two tasks needing the same resource type — only one stove."""
        t1 = CookingTask(
            task_id="task-1",
            dish_id="dish-1",
            instruction="Boil water",
            duration_minutes=10,
            work_mode="PASSIVE",
            category="heating",
            resources=(ResourceNeed(resource_type="stove", quantity=1),),
        )
        t2 = CookingTask(
            task_id="task-2",
            dish_id="dish-2",
            instruction="Fry eggs",
            duration_minutes=5,
            work_mode="ACTIVE",
            category="heating",
            resources=(ResourceNeed(resource_type="stove", quantity=1),),
        )
        # Both tasks need "stove", resource exists → no missing
        missing = check_required_resources((t1, t2), (stove_resource,))
        assert missing == ()
        # Note: capacity-checking (whether one stove can handle both)
        # is the solver's job — this function only checks existence

    def test_empty_tasks(self):
        missing = check_required_resources((), ())
        assert missing == ()


# ======================================================================
# build_reservation_proposal
# ======================================================================


class TestBuildReservationProposal:
    """5.16 Reservation proposal construction."""

    def test_from_feasibility_report(self):
        report = FeasibilityReport(
            report_id="test-1",
            ingredient_shortages=(
                IngredientFeasibility(
                    ingredient_name="chicken breast",
                    required=Decimal(400),
                    available=Decimal(400),
                    shortage=Decimal(0),
                    unit="g",
                    proposed_allocations=(
                        LotAllocation(
                            inventory_lot_id="lot-001",
                            quantity=Decimal(400),
                            unit="g",
                        ),
                    ),
                ),
            ),
            missing_resources=(),
            is_feasible=True,
        )
        proposal = build_reservation_proposal(report)
        assert len(proposal.items) == 1
        item = proposal.items[0]
        assert item.ingredient_name == "chicken breast"
        assert item.allocations[0].inventory_lot_id == "lot-001"
        assert item.allocations[0].quantity == Decimal(400)

    def test_no_shortages(self):
        report = FeasibilityReport(
            report_id="test-2",
            is_feasible=True,
        )
        proposal = build_reservation_proposal(report)
        assert len(proposal.items) == 0

    def test_snapshot_version_is_generated(self):
        report = FeasibilityReport(
            report_id="test-3",
            is_feasible=True,
        )
        proposal = build_reservation_proposal(report)
        assert proposal.inventory_snapshot_version.startswith("v1_")


# ======================================================================
# _aggregate_demands (internal)
# ======================================================================


class TestAggregateDemands:
    """Internal demand aggregation before FEFO allocation."""

    def test_aggregates_same_ingredient(self, chicken_demand):
        d2 = chicken_demand.model_copy(update={"quantity": Decimal(100)})
        result = _aggregate_demands((chicken_demand, d2))
        assert len(result) == 1
        agg = result["chicken breast"]
        assert agg.quantity == Decimal(500)

    def test_keeps_different_ingredients_separate(self, chicken_demand, tomato_demand):
        result = _aggregate_demands((chicken_demand, tomato_demand))
        assert len(result) == 2
        assert "chicken breast" in result
        assert "tomato" in result

    def test_case_insensitive_aggregation(self, chicken_demand):
        d2 = chicken_demand.model_copy(
            update={
                "canonical_name": "CHICKEN BREAST",
                "quantity": Decimal(50),
            }
        )
        result = _aggregate_demands((chicken_demand, d2))
        assert len(result) == 1
