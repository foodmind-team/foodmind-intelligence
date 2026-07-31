"""Shared test fixtures — domain object factories for all test categories.

Handbook 11.2: create fakes for every external dependency.
"""

from decimal import Decimal

from cooking_plan_agent.domain.enums import HeatLevel
from cooking_plan_agent.domain.models import (
    CookingTask,
    IngredientDemand,
    InventoryLotSnapshot,
    KitchenResourceSnapshot,
    RecipeStep,
    ResourceNeed,
    TaskDependency,
)

# =============================================================================
# IngredientDemand fixtures
# =============================================================================


def chicken_breast_demand(quantity: Decimal | None = None) -> IngredientDemand:
    """Standard chicken breast ingredient for reuse across tests."""
    return IngredientDemand(
        canonical_name="chicken breast",
        raw_name="chicken breast",
        quantity=quantity or Decimal(500),
        unit="g",
        confidence=Decimal("0.95"),
    )


def onion_demand(quantity: Decimal | None = None) -> IngredientDemand:
    """Standard onion ingredient for reuse across tests."""
    return IngredientDemand(
        canonical_name="brown onion",
        raw_name="onion",
        quantity=quantity or Decimal(2),
        unit="piece",
        confidence=Decimal("0.9"),
    )


# =============================================================================
# RecipeStep fixtures
# =============================================================================


def boil_step(
    step_number: int = 1,
    passive_duration: int = 10,
    instruction: str = "Boil water",
) -> RecipeStep:
    """Standard boil recipe step."""
    return RecipeStep(
        step_number=step_number,
        instruction=instruction,
        pattern="boil",
        passive_duration_minutes=passive_duration,
        heat_level=HeatLevel.HIGH,
    )


def stir_fry_step(
    step_number: int = 1,
    active_duration: int = 5,
) -> RecipeStep:
    """Standard stir-fry recipe step."""
    return RecipeStep(
        step_number=step_number,
        instruction="Stir-fry continuously",
        pattern="stir_fry",
        active_duration_minutes=active_duration,
        heat_level=HeatLevel.HIGH,
    )


def bake_step(
    step_number: int = 1,
    passive_duration: int = 25,
    temperature_c: int = 180,
) -> RecipeStep:
    """Standard bake recipe step."""
    return RecipeStep(
        step_number=step_number,
        instruction=f"Bake at {temperature_c}C for {passive_duration} minutes",
        pattern="bake",
        passive_duration_minutes=passive_duration,
        heat_level=HeatLevel.MEDIUM,
        target_temperature_c=Decimal(temperature_c),
    )


def marinate_step(
    step_number: int = 1,
    passive_duration: int = 20,
    instruction: str = "Marinate chicken",
    is_raw_protein: bool = True,
) -> RecipeStep:
    """Standard marinate recipe step."""
    return RecipeStep(
        step_number=step_number,
        instruction=instruction,
        pattern="marinate",
        passive_duration_minutes=passive_duration,
    )


# =============================================================================
# Kitchen resource fixtures
# =============================================================================


def stove_resource(count: int = 4, resource_id: str = "stove:main") -> KitchenResourceSnapshot:
    """Standard stove resource."""
    return KitchenResourceSnapshot(
        resource_id=resource_id,
        resource_type="stove",
        capacity=Decimal(count),
        capacity_unit="burners",
    )


def oven_resource(resource_id: str = "oven:main") -> KitchenResourceSnapshot:
    """Standard oven resource."""
    return KitchenResourceSnapshot(
        resource_id=resource_id,
        resource_type="oven",
        capacity=Decimal(1),
    )


def sink_resource() -> KitchenResourceSnapshot:
    """Standard sink resource."""
    return KitchenResourceSnapshot(
        resource_id="sink:main",
        resource_type="sink",
        capacity=Decimal(2),
    )


# =============================================================================
# CookingTask fixtures
# =============================================================================


def make_task(
    task_id: str,
    dish_id: str = "d1",
    duration: int = 5,
    work_mode: str = "ACTIVE",
    category: str = "general",
    deps: tuple[TaskDependency, ...] = (),
    resources: tuple[ResourceNeed, ...] = (),
    consumes_states: tuple[str, ...] = (),
    produces_states: tuple[str, ...] = (),
    safety_tags: tuple[str, ...] = (),
) -> CookingTask:
    """General-purpose CookingTask factory."""
    from cooking_plan_agent.domain.enums import WorkMode as WorkModeEnum

    return CookingTask(
        task_id=task_id,
        dish_id=dish_id,
        instruction=f"Task {task_id}",
        duration_minutes=duration,
        work_mode=WorkModeEnum(work_mode),
        category=category,
        dependencies=deps,
        resources=resources,
        consumes_states=consumes_states,
        produces_states=produces_states,
        safety_tags=safety_tags,
    )


# =============================================================================
# Inventory snapshots
# =============================================================================


def inventory_lot(
    lot_id: str = "lot-001",
    item_id: str = "item-chicken",
    canonical_name: str = "chicken breast",
    on_hand: Decimal | None = None,
    reserved: Decimal | None = None,
    unit: str = "g",
) -> InventoryLotSnapshot:
    """Standard inventory lot snapshot."""
    return InventoryLotSnapshot(
        lot_id=lot_id,
        item_id=item_id,
        canonical_name=canonical_name,
        on_hand=on_hand or Decimal(1000),
        reserved=reserved or Decimal(0),
        unit=unit,
    )
