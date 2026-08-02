"""Spring Boot v1 contract models — strict mirrors of the Java DTOs.

Contract version: ``cooking-agent-v1``.

The Java caller (``AgentCookingRequest`` / ``AgentCookingResponse``) is
deserialised with ``fail-on-unknown-properties=true`` and validated by
``CookingPlanResultValidator``.  Therefore every field name, nullability
and constraint in this module mirrors the Java record EXACTLY:

  - camelCase field names (Jackson default, no naming strategy)
  - ``extra="forbid"`` so unknown fields fail fast (MALFORMED_JSON on Java side)
  - ``sequenceNo``/``stepNo`` must be contiguous starting from 1
  - warning codes are restricted to the allow-list in ``CompatWarningResponse``

These models are used ONLY by the compat router and its adapter; the core
domain services never see Java DTO details (P0-02 design decision).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from cooking_plan_agent.domain.models import StrictModel

# Stable contract identifier shared with the Spring Boot caller.
CONTRACT_VERSION = "cooking-agent-v1"

# Warning codes allowed by CookingPlanResultValidator.WarningCode (allow-list).
CompatWarningCode = Literal[
    "CHECK_ALLERGEN_LABELS",
    "MAY_REQUIRE_EXTRA_TIME",
    "BUDGET_ESTIMATE_ONLY",
    "PANTRY_ITEM_UNVERIFIED",
    "COOK_THOROUGHLY",
]

PositiveInt = Annotated[int, Field(gt=0)]
PositiveDecimal = Annotated[Decimal, Field(gt=0)]


# ===========================================================================
# Request side — AgentCookingRequest + nested snapshots
# ===========================================================================


class CompatInputIngredient(StrictModel):
    """One ingredient the user already has (request.ingredients[])."""

    ingredientName: str
    quantity: PositiveDecimal
    unit: str
    source: str = "MANUAL"


class CompatConstraints(StrictModel):
    """request.constraints — merged dietary + allergen rules."""

    requiredDietaryTagCodes: tuple[str, ...] = ()
    avoidAllergenCodes: tuple[str, ...] = ()


class CompatRequestSnapshot(StrictModel):
    """request — the cooking-public-v1 request snapshot."""

    contractVersion: str
    ingredients: tuple[CompatInputIngredient, ...] = ()
    servings: PositiveInt = 2
    maxMinutes: PositiveInt | None = None
    maxBudget: PositiveDecimal | None = None
    currency: str | None = None
    constraints: CompatConstraints = CompatConstraints()


class CompatPreferences(StrictModel):
    """preferences — merged dietary + allergen codes."""

    requiredDietaryTagCodes: tuple[str, ...] = ()
    avoidAllergenCodes: tuple[str, ...] = ()


class CompatIngredientSnapshot(StrictModel):
    """One ingredient inside a candidate's snapshot."""

    sequenceNo: int = Field(ge=1)
    ingredientName: str
    quantity: Decimal | None = None
    unit: str | None = None
    optional: bool = False


class CompatStepSnapshot(StrictModel):
    """One step inside a candidate's snapshot."""

    stepNo: int = Field(ge=1)
    instruction: str


class CompatCandidateSnapshot(StrictModel):
    """snapshot — the serialised RecipeCandidate (JdbcCookingPlanRepository).

    Contains every structured field the candidate carries; the compat
    adapter maps it to an internal ExtractedRecipeCandidate without
    invoking the LLM (P0-02: no LLM re-parsing for compat requests).
    """

    recipeId: str
    name: str
    description: str = ""
    defaultServings: PositiveInt
    totalMinutes: PositiveInt | None = None
    estimatedCost: PositiveDecimal | None = None
    currency: str | None = None
    dietaryTagCodes: tuple[str, ...] = ()
    allergenCodes: tuple[str, ...] = ()
    ingredients: tuple[CompatIngredientSnapshot, ...] = ()
    steps: tuple[CompatStepSnapshot, ...] = ()


class CompatCandidateRequest(StrictModel):
    """candidates[] — a controlled candidate the agent may select."""

    recipeId: UUID
    snapshot: CompatCandidateSnapshot


class CompatCookingRequest(StrictModel):
    """AgentCookingRequest — the full request body Spring Boot sends."""

    contractVersion: str
    requestId: UUID
    planId: UUID
    traceId: str
    deadlineAt: datetime | None = None
    request: CompatRequestSnapshot
    preferences: CompatPreferences = CompatPreferences()
    candidates: tuple[CompatCandidateRequest, ...] = ()


# ===========================================================================
# Response side — AgentCookingResponse + nested DTOs
# ===========================================================================


class CompatIngredientResponse(StrictModel):
    """ingredients[] — validated by CookingPlanResultValidator.validateIngredient."""

    sequenceNo: int = Field(ge=1)
    ingredientName: str
    quantity: Decimal | None = None
    unit: str | None = None
    availability: Literal["AVAILABLE", "TO_BUY"]


class CompatStepResponse(StrictModel):
    """steps[] — stepNo must be contiguous from 1, no safety claims."""

    stepNo: int = Field(ge=1)
    instruction: str


class CompatWarningResponse(StrictModel):
    """warnings[] — warningCode must hit the validator allow-list."""

    sequenceNo: int = Field(ge=1)
    warningCode: CompatWarningCode
    message: str


class CompatCookingResponse(StrictModel):
    """AgentCookingResponse — must NOT carry fields beyond this record.

    ``status`` is ``"SUCCEEDED"`` on success; any other value is mapped
    by the Java adapter to AGENT_UNAVAILABLE (safe terminal state).
    """

    contractVersion: str
    requestId: UUID
    planId: UUID
    traceId: str
    agentTraceId: str
    status: str
    sourceRecipeId: UUID | None = None
    servings: int = 0
    totalMinutes: int | None = None
    estimatedCost: Decimal | None = None
    currency: str | None = None
    ingredients: tuple[CompatIngredientResponse, ...] = ()
    steps: tuple[CompatStepResponse, ...] = ()
    warnings: tuple[CompatWarningResponse, ...] = ()
