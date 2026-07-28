from datetime import date  # Date type for expiry_date (no time component needed)
from decimal import Decimal  # Exact decimal arithmetic — never float for inventory
from typing import Annotated  # Typed annotation composition (e.g. PositiveDecimal)

from pydantic import (  # Pydantic v2 building blocks
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from cooking_plan_agent.domain.enums import (  # Domain enums used in CookingTask
    HeatLevel,
    WorkMode,
)

# ---------------------------------------------------------------------------
# 3.3  Strict base model — all domain models inherit these constraints
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Foundation for every domain model. Enforces immutability, boundary
    strictness, and whitespace hygiene at the Pydantic layer."""

    model_config = ConfigDict(
        extra="forbid",               # 3.1 Reject unknown fields at API/LLM boundaries
        frozen=True,                  # 3.1 Prefer immutable models for snapshots & solver inputs
        str_strip_whitespace=True,    # 3.1 Normalize string inputs before validation
    )


# ---------------------------------------------------------------------------
# 3.4  Reusable annotated types
# ---------------------------------------------------------------------------

# 3.1 Use Decimal for quantities; never float for inventory arithmetic
PositiveDecimal = Annotated[Decimal, Field(gt=0)]

# 3.1 Store confidence and evidence for inferred facts
Confidence = Annotated[Decimal, Field(ge=0, le=1)]


# ---------------------------------------------------------------------------
# 3.4  Ingredient and recipe models
# ---------------------------------------------------------------------------


class EvidenceRef(StrictModel):
    """Provenance reference for an inferred fact (URL, document, retrieval timestamp)."""

    source_type: str                     # e.g. "web_search", "user_input", "LLM_guess"
    title: str | None = None             # Human-readable label for the source
    url: str | None = None               # Optional stable link to the evidence
    retrieved_at: str | None = None      # ISO-8601 timestamp of retrieval


class IngredientDemand(StrictModel):
    """A single ingredient entry with raw/canonical separation and confidence."""

    canonical_name: str                  # Normalized unique name (e.g. "chicken breast")
    raw_name: str                        # 3.1 Original text from the recipe source
    quantity: PositiveDecimal            # Parsed numeric quantity (> 0)
    unit: str                            # e.g. "g", "ml", "piece"
    preparation_spec: str | None = None  # e.g. "diced", "minced" — optional prep note
    input_state: str = "raw"             # State before use (default "raw")
    output_state: str | None = None      # State after processing (e.g. "cooked")
    allergen_tags: tuple[str, ...] = ()  # e.g. ("gluten", "dairy")
    confidence: Confidence               # LLM extraction confidence [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Chain of sources supporting this ingredient


class Assumption(StrictModel):
    """An inference or guess made by the LLM during recipe parsing.

    Captures uncertain decisions (e.g. 'assuming 200 C for baking') so that
    downstream consumers can surface them for user confirmation."""

    text: str                               # Human-readable assumption description
    confidence: Confidence                  # LLM confidence [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Supporting sources, if any


class RecipeStep(StrictModel):
    """A single recipe step before decomposition into schedulable CookingTasks."""

    step_number: int = Field(ge=1)                    # 1-based position in recipe
    instruction: str                                   # Raw instruction text
    category: str = "general"                          # e.g. "cutting", "heating", "mixing", "resting"
    pattern: str = "simple"                            # Decomposition hint: "simple" | "boil" | "marinate" | "bake" | "stir_fry" | "simmer"
    active_duration_minutes: int | None = None         # Hands-on time (if specified)
    passive_duration_minutes: int | None = None        # Wait/monitor time (if specified, e.g. boil 10 min)
    heat_level: HeatLevel = HeatLevel.NONE             # Stove intensity
    target_temperature_c: Decimal | None = None        # Target temperature in Celsius
    interval_minutes: int | None = None                # For periodic check/stir tasks
    resources_hint: tuple[str, ...] = ()               # Suggested equipment (e.g. "stove", "oven")


class RecipeIR(StrictModel):
    """Intermediate representation of a parsed recipe, before scheduling.

    'IR' stands for Intermediate Representation. This model separates raw
    source properties from what the scheduler needs."""

    recipe_id: str                          # Stable identifier across pipeline stages
    dish_name: str                          # Human-readable dish name
    original_servings: PositiveDecimal      # Servings as stated in the original recipe
    target_servings: PositiveDecimal        # Desired servings for this plan
    source_language: str                    # ISO language code of the source text
    ingredients: tuple[IngredientDemand, ...]  # Extracted ingredient list
    steps: tuple[RecipeStep, ...]              # Ordered cooking steps
    assumptions: tuple[Assumption, ...] = ()   # LLM assumptions made during parsing

    @model_validator(mode="after")
    def require_content(self) -> "RecipeIR":
        """3.4 Validate that the recipe has at least one ingredient and one step.
        No side effects — pure validation only."""
        if not self.ingredients:
            raise ValueError("recipe must contain at least one ingredient")
        if not self.steps:
            raise ValueError("recipe must contain at least one step")
        return self


# ---------------------------------------------------------------------------
# 3.5  Task model — schedulable unit decomposed from recipe steps
# ---------------------------------------------------------------------------


class ResourceNeed(StrictModel):
    """A resource required by a cooking task (e.g. stove burner, oven, mixing bowl)."""

    resource_type: str                       # Equipment category (e.g. "stove", "oven")
    quantity: int = Field(ge=1)              # How many units of this resource are needed
    minimum_capacity: Decimal | None = None  # Minimum usable capacity (e.g. 2.0 L for a pot)
    capacity_unit: str | None = None         # Unit for capacity (e.g. "L", "kg")
    required_capabilities: tuple[str, ...] = ()  # Specific features needed (e.g. "induction")


class TaskDependency(StrictModel):
    """A precedence constraint between two tasks."""

    predecessor_id: str                                   # The task that must finish first
    minimum_lag_minutes: int = Field(ge=0, default=0)     # Minimum gap after predecessor ends
    maximum_lag_minutes: int | None = Field(ge=0, default=None)  # Maximum gap (None = no upper bound)


class CookingTask(StrictModel):
    """A single schedulable unit — one recipe step decomposed into timing and resources.

    3.5 Do not represent 'boil for ten minutes' as one active task;
    the decomposition service splits it into start / passive-wait / finish."""

    task_id: str                                       # Unique task identifier
    dish_id: str                                       # Parent recipe this task belongs to
    instruction: str                                   # Human-readable cooking instruction
    duration_minutes: int = Field(ge=1)                # Active time this task occupies resources
    work_mode: WorkMode                                # ACTIVE (hands-on) or PASSIVE (monitoring)
    category: str                                      # e.g. "cutting", "heating", "mixing", "resting"
    heat_level: HeatLevel = HeatLevel.NONE             # Stove intensity; NONE for cold tasks
    target_temperature_c: Decimal | None = None        # Target temp in Celsius (if heating)
    dependencies: tuple[TaskDependency, ...] = ()      # Predecessor constraints
    resources: tuple[ResourceNeed, ...] = ()           # Equipment needed
    consumes_states: tuple[str, ...] = ()              # Ingredient states consumed (e.g. "diced_onion")
    produces_states: tuple[str, ...] = ()              # States this task produces (e.g. "caramelized_onion")
    batch_key: str | None = None                       # Shared key for batchable tasks (e.g. same oven temp)
    safety_tags: tuple[str, ...] = ()                  # Labels for safety rule enforcement (e.g. "raw_meat")


# ---------------------------------------------------------------------------
# 3.6  Inventory snapshot models — immutable point-in-time views
# ---------------------------------------------------------------------------


class InventoryLotSnapshot(StrictModel):
    """3.6 Immutable snapshot of an inventory lot. Spring Boot handles
    concurrent reservation decisions against this snapshot."""

    lot_id: str                                   # Unique lot identifier
    item_id: str                                  # Stock item reference
    canonical_name: str                           # Normalized item name
    on_hand: Decimal = Field(ge=0)                # Total quantity currently in stock
    reserved: Decimal = Field(ge=0)               # Quantity already reserved for other plans
    unit: str                                     # Unit of measure (e.g. "g", "ml")
    expiry_date: date | None = None               # Expiration date, if applicable

    @model_validator(mode="after")
    def reservation_cannot_exceed_stock(self) -> "InventoryLotSnapshot":
        """3.9 Reject reserved > on_hand at the model boundary."""
        if self.reserved > self.on_hand:
            raise ValueError("reserved quantity exceeds on-hand quantity")
        return self


class KitchenResourceSnapshot(StrictModel):
    """3.6 Immutable snapshot of a kitchen resource (appliance, tool, station)."""

    resource_id: str                          # Unique resource identifier
    resource_type: str                        # Category (e.g. "stove", "oven", "sink")
    capacity: Decimal | None = None           # Maximum capacity (e.g. 4 burners)
    capacity_unit: str | None = None          # Unit for capacity (e.g. "burners", "L")
    capabilities: tuple[str, ...] = ()        # Features (e.g. "induction", "convection")
    available: bool = True                    # Whether the resource is operational


# ---------------------------------------------------------------------------
# 3.7  Response contracts — plan output, not database mutation
# ---------------------------------------------------------------------------


class LotAllocation(StrictModel):
    """A proposed deduction from a specific inventory lot.

    3.7 This is a plan, not a database mutation. Spring Boot persists
    and the client confirms after cooking."""

    inventory_lot_id: str         # Which lot to draw from
    quantity: PositiveDecimal     # How much to deduct
    unit: str                     # Unit of the deduction


class CompletionItem(StrictModel):
    """Groups allocations that fulfill one ingredient across recipes."""

    completion_item_id: str                 # Unique ID for this completion group
    ingredient_name: str                    # Canonical ingredient name
    recipe_ids: tuple[str, ...]             # Which recipes contribute to this ingredient
    allocations: tuple[LotAllocation, ...]  # Specific lot deductions


class InventoryConsumptionProposal(StrictModel):
    """Top-level consumption plan included in a READY response.

    3.7 Carries a snapshot version so Spring Boot can detect stale proposals."""

    inventory_snapshot_version: str           # Version of the inventory snapshot this was computed from
    items: tuple[CompletionItem, ...]         # Per-ingredient completion groups
