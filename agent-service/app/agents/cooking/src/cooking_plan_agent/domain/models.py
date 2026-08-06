from datetime import date, datetime  # Date type for expiry_date (no time component needed)
from decimal import Decimal  # Exact decimal arithmetic — never float for inventory
from enum import StrEnum  # String enum base class (P4-02 response types)
from typing import (  # Typed annotation composition (e.g. PositiveDecimal)
    Annotated,
)

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
        extra="forbid",  # 3.1 Reject unknown fields at API/LLM boundaries
        frozen=True,  # 3.1 Prefer immutable models for snapshots & solver inputs
        str_strip_whitespace=True,  # 3.1 Normalize string inputs before validation
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

    source_type: str  # e.g. "web_search", "user_input", "LLM_guess"
    title: str | None = None  # Human-readable label for the source
    url: str | None = None  # Optional stable link to the evidence
    retrieved_at: str | None = None  # ISO-8601 timestamp of retrieval


class IngredientDemand(StrictModel):
    """A single ingredient entry with raw/canonical separation and confidence."""

    canonical_name: str  # Normalized unique name (e.g. "chicken breast")
    raw_name: str  # 3.1 Original text from the recipe source
    quantity: PositiveDecimal  # Parsed numeric quantity (> 0)
    unit: str  # e.g. "g", "ml", "piece"
    preparation_spec: str | None = None  # e.g. "diced", "minced" — optional prep note
    input_state: str = "raw"  # State before use (default "raw")
    output_state: str | None = None  # State after processing (e.g. "cooked")
    allergen_tags: tuple[str, ...] = ()  # e.g. ("gluten", "dairy")
    confidence: Confidence  # LLM extraction confidence [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Chain of sources supporting this ingredient


class Assumption(StrictModel):
    """An inference or guess made by the LLM during recipe parsing.

    Captures uncertain decisions (e.g. 'assuming 200 C for baking') so that
    downstream consumers can surface them for user confirmation."""

    text: str  # Human-readable assumption description
    confidence: Confidence  # LLM confidence [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Supporting sources, if any


class RecipeStep(StrictModel):
    """A single recipe step before decomposition into schedulable CookingTasks."""

    step_number: int = Field(ge=1)  # 1-based position in recipe
    instruction: str  # Raw instruction text
    category: str = "general"  # e.g. "cutting", "heating", "mixing", "resting"
    pattern: str = "simple"  # Decomposition hint: "simple" | "boil" | "marinate" | "bake" | "stir_fry" | "simmer"
    active_duration_minutes: int | None = None  # Hands-on time (if specified)
    passive_duration_minutes: int | None = None  # Wait/monitor time (if specified, e.g. boil 10 min)
    heat_level: HeatLevel = HeatLevel.NONE  # Stove intensity
    target_temperature_c: Decimal | None = None  # Target temperature in Celsius
    interval_minutes: int | None = None  # For periodic check/stir tasks
    resources_hint: tuple[str, ...] = ()  # Suggested equipment (e.g. "stove", "oven")


class RecipeIR(StrictModel):
    """Intermediate representation of a parsed recipe, before scheduling.

    'IR' stands for Intermediate Representation. This model separates raw
    source properties from what the scheduler needs."""

    recipe_id: str  # Stable identifier across pipeline stages
    dish_name: str  # Human-readable dish name
    original_servings: PositiveDecimal  # Servings as stated in the original recipe
    target_servings: PositiveDecimal  # Desired servings for this plan
    source_language: str  # ISO language code of the source text
    ingredients: tuple[IngredientDemand, ...]  # Extracted ingredient list
    steps: tuple[RecipeStep, ...]  # Ordered cooking steps
    assumptions: tuple[Assumption, ...] = ()  # LLM assumptions made during parsing

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

    resource_type: str  # Equipment category (e.g. "stove", "oven")
    quantity: int = Field(ge=1)  # How many units of this resource are needed
    minimum_capacity: Decimal | None = None  # Minimum usable capacity (e.g. 2.0 L for a pot)
    capacity_unit: str | None = None  # Unit for capacity (e.g. "L", "kg")
    required_capabilities: tuple[str, ...] = ()  # Specific features needed (e.g. "induction")


class TaskDependency(StrictModel):
    """A precedence constraint between two tasks."""

    predecessor_id: str  # The task that must finish first
    minimum_lag_minutes: int = Field(ge=0, default=0)  # Minimum gap after predecessor ends
    maximum_lag_minutes: int | None = Field(ge=0, default=None)  # Maximum gap (None = no upper bound)


class CookingTask(StrictModel):
    """A single schedulable unit — one recipe step decomposed into timing and resources.

    3.5 Do not represent 'boil for ten minutes' as one active task;
    the decomposition service splits it into start / passive-wait / finish."""

    task_id: str  # Unique task identifier
    dish_id: str  # Parent recipe this task belongs to
    instruction: str  # Human-readable cooking instruction
    duration_minutes: int = Field(ge=1)  # Active time this task occupies resources
    work_mode: WorkMode  # ACTIVE (hands-on) or PASSIVE (monitoring)
    category: str  # e.g. "cutting", "heating", "mixing", "resting"
    heat_level: HeatLevel = HeatLevel.NONE  # Stove intensity; NONE for cold tasks
    target_temperature_c: Decimal | None = None  # Target temp in Celsius (if heating)
    dependencies: tuple[TaskDependency, ...] = ()  # Predecessor constraints
    resources: tuple[ResourceNeed, ...] = ()  # Equipment needed
    consumes_states: tuple[str, ...] = ()  # Ingredient states consumed (e.g. "diced_onion")
    produces_states: tuple[str, ...] = ()  # States this task produces (e.g. "caramelized_onion")
    batch_key: str | None = None  # Shared key for batchable tasks (e.g. same oven temp)
    safety_tags: tuple[str, ...] = ()  # Labels for safety rule enforcement (e.g. "raw_meat")


# ---------------------------------------------------------------------------
# 3.6  Inventory snapshot models — immutable point-in-time views
# ---------------------------------------------------------------------------


class InventoryLotSnapshot(StrictModel):
    """3.6 Immutable snapshot of an inventory lot. Spring Boot handles
    concurrent reservation decisions against this snapshot."""

    lot_id: str  # Unique lot identifier
    item_id: str  # Stock item reference
    canonical_name: str  # Normalized item name
    on_hand: Decimal = Field(ge=0)  # Total quantity currently in stock
    reserved: Decimal = Field(ge=0)  # Quantity already reserved for other plans
    unit: str  # Unit of measure (e.g. "g", "ml")
    expiry_date: date | None = None  # Expiration date, if applicable

    @model_validator(mode="after")
    def reservation_cannot_exceed_stock(self) -> "InventoryLotSnapshot":
        """3.9 Reject reserved > on_hand at the model boundary."""
        if self.reserved > self.on_hand:
            raise ValueError("reserved quantity exceeds on-hand quantity")
        return self


class KitchenResourceSnapshot(StrictModel):
    """3.6 Immutable snapshot of a kitchen resource (appliance, tool, station)."""

    resource_id: str  # Unique resource identifier
    resource_type: str  # Category (e.g. "stove", "oven", "sink")
    capacity: Decimal | None = None  # Maximum capacity (e.g. 4 burners)
    capacity_unit: str | None = None  # Unit for capacity (e.g. "burners", "L")
    capabilities: tuple[str, ...] = ()  # Features (e.g. "induction", "convection")
    available: bool = True  # Whether the resource is operational


# ---------------------------------------------------------------------------
# 3.7  Response contracts — plan output, not database mutation
# ---------------------------------------------------------------------------


class LotAllocation(StrictModel):
    """A proposed deduction from a specific inventory lot.

    3.7 This is a plan, not a database mutation. Spring Boot persists
    and the client confirms after cooking."""

    inventory_lot_id: str  # Which lot to draw from
    quantity: PositiveDecimal  # How much to deduct
    unit: str  # Unit of the deduction


class CompletionItem(StrictModel):
    """Groups allocations that fulfill one ingredient across recipes."""

    completion_item_id: str  # Unique ID for this completion group
    ingredient_name: str  # Canonical ingredient name
    recipe_ids: tuple[str, ...]  # Which recipes contribute to this ingredient
    allocations: tuple[LotAllocation, ...]  # Specific lot deductions


class InventoryConsumptionProposal(StrictModel):
    """Top-level consumption plan included in a READY response.

    3.7 Carries a snapshot version so Spring Boot can detect stale proposals."""

    inventory_snapshot_version: str  # Version of the inventory snapshot this was computed from
    items: tuple[CompletionItem, ...]  # Per-ingredient completion groups


# ---------------------------------------------------------------------------
# 3.8  Evidence models — structured web research I/O
# ---------------------------------------------------------------------------


class EvidenceQuery(StrictModel):
    """A structured question for web research. Contains only the gap info,
    never private user data."""

    query_text: str
    gap_type: str
    recipe_context: str
    target_fields: tuple[str, ...] = ()


class EvidenceResult(StrictModel):
    """One cited piece of evidence from web research."""

    source_title: str
    source_url: str
    snippet: str
    confidence: Confidence
    extracted_fact: str
    fact_type: str
    fact_value: str


class SearchDocument(StrictModel):
    """Provider-neutral search result document.

    Normalised from any concrete provider (Brave, SerpAPI, Fake).
    No provider-specific fields — the rest of the code only sees this shape."""

    title: str
    url: str
    snippet: str
    # Raw content fetched from the page (may be empty for snippet-only providers)
    raw_content: str = ""
    # Domain extracted from URL for allow-list matching
    domain: str = ""


class CookingEvidence(StrictModel):
    """Evidence extracted from a single search document (handbook 10.6).

    Narrow schema: only cooking-relevant fields. Rejects unexpected fields.
    Excerpts limited to the shortest text needed for traceability."""

    operation: str  # e.g. "stir-fry", "bake", "boil"
    heat_level: HeatLevel | None = None  # Stove intensity if stated
    duration_min_minutes: int | None = None  # Lower bound of duration range
    duration_max_minutes: int | None = None  # Upper bound of duration range
    explicit_temperature_c: Decimal | None = None  # Target temperature in Celsius
    source_url: str  # Source page URL
    source_title: str  # Source page title
    source_excerpt: str  # Shortest text for traceability (not full page)


class ReconciledEvidence(StrictModel):
    """Consensus output from multi-source reconciliation (handbook 10.7).

    Reports both the reconciled value AND whether sources disagreed enough
    to warrant user confirmation."""

    heat_level: HeatLevel | None = None
    duration_min_minutes: int | None = None
    duration_max_minutes: int | None = None
    explicit_temperature_c: Decimal | None = None
    # How many independent sources contributed to each reconciled value
    source_count: int = 0
    # If True, disagreement exceeded threshold — surface for user confirmation
    needs_confirmation: bool = False
    # Raw evidence items that fed into the reconciliation
    evidence_items: tuple["CookingEvidence", ...] = ()


# ---------------------------------------------------------------------------
# 3.9  LLM extraction intermediate models
# ---------------------------------------------------------------------------


class ExtractedIngredient(StrictModel):
    """Raw ingredient as extracted by LLM, before canonicalisation."""

    raw_text: str
    name: str
    quantity: Decimal | None = None
    unit: str | None = None
    preparation: str | None = None
    extraction_source: str = "EXPLICIT"
    confidence: Confidence = Decimal("1.0")


class ExtractedStep(StrictModel):
    """Raw step as extracted by LLM, before decomposition."""

    step_number: int = Field(ge=1)
    instruction: str
    category: str = "general"
    active_duration_minutes: int | None = None
    passive_duration_minutes: int | None = None
    heat_level: HeatLevel = HeatLevel.NONE
    target_temperature_c: Decimal | None = None
    resources_hint: tuple[str, ...] = ()
    extraction_source: str = "EXPLICIT"
    confidence: Confidence = Decimal("1.0")


class ExtractedRecipeCandidate(StrictModel):
    """LLM extraction output — optional fields allowed, raw spans retained."""

    recipe_id: str
    dish_name: str
    original_servings: PositiveDecimal
    source_language: str
    ingredients: tuple[ExtractedIngredient, ...]
    steps: tuple[ExtractedStep, ...]
    extraction_source: str = "LLM"


# ---------------------------------------------------------------------------
# 3.10  Recipe gap detection
# ---------------------------------------------------------------------------


class RecipeGap(StrictModel):
    """A detected gap in a recipe candidate."""

    gap_id: str
    recipe_id: str
    field_path: str
    current_value: str | None = None
    gap_class: str  # "critical" | "safety_critical" | "resource_critical" | "optimisation" | "cosmetic"
    description: str
    confidence: Confidence
    evidence: tuple[EvidenceRef, ...] = ()


# ---------------------------------------------------------------------------
# 3.11  Safety rule engine models
# ---------------------------------------------------------------------------


class SafetyInsertion(StrictModel):
    """A structured safety-task insertion anchored between recipe steps (P0-07).

    Produced by safety rules (e.g. cross-contamination) instead of bare
    task IDs. Carries the exact step anchors so merge_preparation can build
    the ``raw task → sanitise task → RTE task`` dependency chain:

      - after_step_number: the LAST step that must finish before the safety
        task starts (e.g. raw protein handling).
      - before_step_number: the FIRST step that must start after the safety
        task ends (e.g. ready-to-eat assembly/plating).

    Duration and resources come from policy configuration — never the old
    fixed 1-minute placeholder.
    """

    insertion_id: str
    rule_id: str
    recipe_id: str
    after_step_number: int | None = None
    before_step_number: int | None = None
    task_instruction: str
    duration_minutes: int = Field(ge=1)
    required_resources: tuple[str, ...] = ()


class SafetyFinding(StrictModel):
    """Output of a single safety rule evaluation."""

    rule_id: str
    severity: str  # "hard_unrepairable" | "hard_repairable" | "warning"
    description: str
    affected_task_ids: tuple[str, ...] = ()
    affected_ingredient_names: tuple[str, ...] = ()
    recommended_action: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    # P0-07: structured insertion template when the finding is repairable by
    # injecting a safety task between two recipe steps.
    insertion: SafetyInsertion | None = None


class SafetyReport(StrictModel):
    """Aggregated safety evaluation for the entire plan."""

    report_id: str
    findings: tuple[SafetyFinding, ...] = ()
    is_safe: bool
    has_unrepairable: bool
    required_safety_task_ids: tuple[str, ...] = ()
    # P0-07: structured insertions anchored between recipe steps.
    insertions: tuple[SafetyInsertion, ...] = ()
    # P3-04: the regional policy pack that produced this report (None when a
    # legacy engine without a bound policy ran — never blocks evaluation).
    safety_policy: "SafetyPolicyRecord | None" = None


class SafetyContext(StrictModel):
    """Input context for safety rule evaluation."""

    recipes: tuple["RecipeIR", ...]
    dietary_restrictions: tuple[str, ...] = ()
    user_allergens: tuple[str, ...] = ()
    inventory_lots: tuple["InventoryLotSnapshot", ...] = ()
    cooking_date: date | None = None


# ---------------------------------------------------------------------------
# 3.11b  Regional safety policy records (P3-04)
# ---------------------------------------------------------------------------


class PolicySourceRef(StrictModel):
    """Serialisable reference to an official safety-policy source (D7)."""

    source_id: str
    title: str
    url: str


class SafetyPolicyRecord(StrictModel):
    """Policy provenance attached to plans (P3-04).

    Recorded on READY/CONFIRMATION responses and retained in state so every
    plan carries the region, version, and official sources that produced its
    safety constraints — the basis for threshold traceability and audit of
    historical checkpoints (old versions remain registered for that purpose).
    """

    region: str
    version: str
    effective_at: date
    sources: tuple[PolicySourceRef, ...] = ()


# ---------------------------------------------------------------------------
# 3.12  Feasibility check and repair models
# ---------------------------------------------------------------------------


class IngredientFeasibility(StrictModel):
    """Feasibility result for one ingredient."""

    ingredient_name: str
    required: Decimal
    available: Decimal
    shortage: Decimal
    unit: str
    proposed_allocations: tuple["LotAllocation", ...] = ()


class FeasibilityReport(StrictModel):
    """Aggregated feasibility across all dimensions."""

    report_id: str
    ingredient_shortages: tuple["IngredientFeasibility", ...] = ()
    missing_resources: tuple[str, ...] = ()
    is_feasible: bool
    # 完整库存分配结果（含完全满足的食材），供 READY 响应的消耗清单使用。
    # ingredient_shortages 只保留 shortage > 0 的条目（确认/修复语义不变）；
    # 本字段保留每个食材的 required/available/shortage/proposed_allocations，
    # 避免满足的食材的 FEFO 分配在渲染层丢失（P4 缺陷修复）。
    ingredient_results: tuple["IngredientFeasibility", ...] = ()


class RepairOption(StrictModel):
    """A validated choice the user can select to resolve infeasibility."""

    option_id: str
    option_type: str  # "substitute_ingredient" | "reduce_servings" | "alternative_equipment" | "replace_dish" | "extend_time" | "purchase"
    description: str
    changes: tuple[str, ...]
    effects: tuple[str, ...]
    revalidation_status: str = "validated"


class ApprovedDecision(StrictModel):
    """A structured, client-submittable decision (P0-06).

    Unlike a bare option_id string, an ApprovedDecision carries:
      - option_id:  which presented option was chosen
      - option_type: one of the five supported decision kinds
      - payload:    machine-readable values (servings, minutes, ingredient
                    substitution, resource alternative, dish to replace)
      - plan_revision: version of the confirmation response the client is
                    answering — used to reject stale confirmations

    The confirmation response returns these verbatim; the client resubmits
    them in the next request's approved_decisions field.
    """

    option_id: str
    option_type: str
    payload: dict[str, object] = {}
    plan_revision: str | None = None


class WorkflowError(StrictModel):
    """Structured error for workflow-level failures.

    P2-03: the client-facing text is resolved from the centralised public
    message catalog (domain.errors) — ``message`` is an internal diagnostic
    and must never leak provider payloads, secrets or recipe text. A node
    may explicitly override the public text with ``public_message`` (still
    free of sensitive detail); otherwise the catalog row decides.
    """

    error_code: str
    # Internal diagnostic message for logs/support. Not rendered verbatim to
    # the client — the catalog row for error_code provides the public text.
    message: str
    correlation_id: str
    node_name: str | None = None
    recoverable: bool = False
    # Optional explicit override of the catalog's public message. Must stay
    # stable and free of sensitive detail; None falls back to the catalog.
    public_message: str | None = None
    # Controlled diagnostic context (e.g. exception_type) for internal logs
    # only. Must not contain secrets or raw provider payloads.
    diagnostics: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# 3.13  API request / response contracts
# ---------------------------------------------------------------------------


class RecipeInput(StrictModel):
    """Typed input for one recipe in a GeneratePlanRequest (P0-03).

    Replaces the loose ``tuple[dict, ...]`` so structural constraints,
    positive servings, and string bounds are enforced at the Pydantic
    boundary instead of inside workflow nodes.
    """

    recipe_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=1_000_000)
    target_servings: PositiveDecimal


class GeneratePlanRequest(StrictModel):
    """Internal request from Spring Boot."""

    request_id: str
    user_id: str
    recipes: tuple[RecipeInput, ...]  # Typed recipe inputs (P0-03)
    dietary_restrictions: tuple[str, ...] = ()
    user_allergens: tuple[str, ...] = ()
    time_limit_minutes: int | None = None
    # --- Time semantics (P0-05) ---
    # cooking_date: the calendar day the plan is executed on. Drives the
    # safety engine's expired-lot check and FEFO inventory allocation.
    cooking_date: date | None = None
    # serving_at: absolute serving time WITH timezone. Only when both date
    # and timezone are known is this converted to an absolute instant; the
    # legacy `serving_time` (HH:MM string) is kept for back-compat but is
    # never treated as an absolute wall-clock by itself.
    serving_at: datetime | None = None
    serving_time: str | None = None
    inventory_lots: tuple["InventoryLotSnapshot", ...] = ()
    kitchen_resources: tuple["KitchenResourceSnapshot", ...] = ()
    approved_decisions: tuple["ApprovedDecision", ...] = ()
    schema_version: str = "1.0"
    # Revision of the confirmation response these decisions answer (P0-06).
    # Used to reject stale confirmations when the plan has changed.
    plan_revision: str | None = None
    # Structured candidates injected by the compat layer. When non-empty,
    # parse_recipes_node uses them directly and never calls the LLM
    # extractor (P0-02 rule 4). Kept optional so native requests are
    # unaffected.
    preparsed_candidates: tuple["ExtractedRecipeCandidate", ...] = ()
    # P3-04: explicit regional food-safety policy selection (ISO alpha-2,
    # e.g. "US"/"SG"). When unset, the deployment default
    # (Settings.safety_policy_region) applies. An unknown region is rejected —
    # never silently falls back (D6).
    region: str | None = None


class ReadyPlanResponse(StrictModel):
    """READY response: verified plan with schedule."""

    plan_id: str
    status: str = "READY"
    solver_status: str
    makespan_minutes: int
    timeline: tuple[dict[str, object], ...]
    # Dependency-driven task graph for execution UIs.  Unlike ``timeline``,
    # this never asks the user to start a task at a fixed minute.
    execution_flow: tuple[dict[str, object], ...] = ()
    completion_checklist: tuple["CompletionItem", ...]
    mise_en_place: tuple[dict[str, object], ...]
    dish_completions: tuple[dict[str, object], ...]
    # P3-04: policy provenance (region/version/sources) that produced the plan.
    safety_policy: "SafetyPolicyRecord | None" = None
    # P4-01: optional additive schedule explanation ("why this timing/order").
    # explanation_source ∈ {"llm", "deterministic", "disabled"}. The
    # explanation never alters the verified schedule — it is display-only.
    explanation: str | None = None
    explanation_source: str | None = None


# ---------------------------------------------------------------------------
# 3.13a  Structured confirmation questions (P4-02)
# ---------------------------------------------------------------------------


class QuestionResponseType(StrEnum):
    """How a ConfirmationQuestion expects to be answered (P4-02)."""

    CHOICE = "CHOICE"  # The client selects exactly one QuestionOption value
    TEXT = "TEXT"  # The client supplies a bounded free-text value


class QuestionOption(StrictModel):
    """A single selectable answer for a CHOICE confirmation question (P4-02).

    ``value`` is the stable token the client echoes back inside a
    QuestionAnswer. For repair-option questions it is the presented
    ApprovedDecision's ``option_id``, so the mapping back to the decision
    is lossless — the server never rewrites or re-derives the payload
    from prose (D9).
    """

    value: str
    label: str
    suggested: bool = False


class ConfirmationQuestion(StrictModel):
    """A field-level, client-renderable confirmation question (P4-02).

    Replaces the fixed legacy ``questions`` strings with a structured
    form: each question carries a stable ``question_id`` (derived from
    stable domain keys — recipe_id + field_path — never from array
    position, D6), the domain field it targets, the prompt, the expected
    response type, and — for CHOICE — the exact allowed option values.
    """

    question_id: str
    field_path: str
    prompt: str
    response_type: QuestionResponseType
    options: tuple[QuestionOption, ...] = ()
    required: bool = True
    suggested_value: str | None = None


class QuestionAnswer(StrictModel):
    """A client-submitted answer to a ConfirmationQuestion (P4-02).

    ``value`` is validated against the presented question: for CHOICE it
    must hit one of the option values; for TEXT it must be non-empty and
    within the configured length bound. Unknown question_ids and
    duplicate answers are rejected.
    """

    question_id: str
    value: str


class ConfirmationPlanResponse(StrictModel):
    """NEEDS_CONFIRMATION response.

    ``decisions`` carries the structured, client-submittable approved
    decisions (P0-06). The client resubmits these verbatim in the next
    request's ``approved_decisions`` field.

    P4-02: ``confirmation_questions`` carries the field-level structured
    form the client renders and answers; answers map losslessly back to
    ``ApprovedDecision`` (repair-option questions) before re-entry.
    ``questions`` remains a legacy dual-emit of plain strings for older
    clients — deprecated since P4-02, removed when contract v2 lands.
    """

    plan_id: str
    status: str = "NEEDS_CONFIRMATION"
    assumptions: tuple["Assumption", ...] = ()
    repair_options: tuple["RepairOption", ...] = ()
    # P4-02: legacy plain-string questions (dual-emit, deprecated).
    questions: tuple[str, ...] = ()
    # P4-02: field-level structured confirmation form.
    confirmation_questions: tuple["ConfirmationQuestion", ...] = ()
    decisions: tuple["ApprovedDecision", ...] = ()
    plan_revision: str | None = None
    # P3-04: policy provenance (region/version/sources) that produced the plan.
    safety_policy: "SafetyPolicyRecord | None" = None


class InfeasiblePlanResponse(StrictModel):
    """INFEASIBLE response."""

    plan_id: str
    status: str = "INFEASIBLE"
    reasons: tuple[str, ...]
    safe_alternatives: tuple[str, ...] = ()


class FailedPlanResponse(StrictModel):
    """FAILED response."""

    status: str = "FAILED"
    error_code: str
    correlation_id: str
    message: str


class ErrorEnvelope(StrictModel):
    """Unified protocol-error envelope (P3-05).

    Every managed endpoint returns this shape for protocol/HTTP-level
    failures — Pydantic validation (422), auth (401/403), not-found (404),
    idempotency conflict (409), backpressure (429/503), and unexpected
    internal errors (500). Legal business outcomes (READY / NEEDS_
    CONFIRMATION / INFEASIBLE / FAILED) keep their own response models and
    are never disguised as protocol errors.

    ``retryable`` is decided by the error catalog (domain/errors.py), never
    inferred from the message text, so clients can programmatically decide
    whether to retry. ``details`` carries only field-level, safe
    information — never raw input, stack traces, or provider payloads.
    """

    status: int
    """HTTP status code (4xx/5xx) of the failing response."""

    error_code: str
    """Stable machine-readable code from the error catalog."""

    message: str
    """Short, human-readable, non-sensitive description."""

    correlation_id: str
    """Same value echoed in the X-Request-ID response header."""

    details: dict[str, object] | list[dict[str, object]] | None = None
    """Field-level diagnostics only (validation loc/type, retry hint)."""

    retryable: bool = False
    """Whether clients may retry; decided by the error catalog."""


# ---------------------------------------------------------------------------
# Union type for polymorphic response
# ---------------------------------------------------------------------------

PlanResponse = ReadyPlanResponse | ConfirmationPlanResponse | InfeasiblePlanResponse | FailedPlanResponse

# ---------------------------------------------------------------------------
# Resolve forward references via model_rebuild()
# ---------------------------------------------------------------------------

EvidenceQuery.model_rebuild()
SafetyContext.model_rebuild()
IngredientFeasibility.model_rebuild()
GeneratePlanRequest.model_rebuild()
ReadyPlanResponse.model_rebuild()
ConfirmationPlanResponse.model_rebuild()
ReconciledEvidence.model_rebuild()
