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

# 模块概览（中文）：这是 Spring Boot v1 契约模型的“严格镜像”。
# 用途：与 Java 调用方（AgentCookingRequest/AgentCookingResponse）保持字段名、
#       空值约束、数值约束完全一致，避免反序列化/校验不一致导致契约漂移。
# 关键约束（对应英文 docstring）：
#   - 字段名使用 camelCase（Jackson 默认，无命名策略）
#   - extra="forbid"：未知字段快速失败（Java 侧映射为 MALFORMED_JSON）
#   - sequenceNo/stepNo 必须从 1 开始连续递增
#   - warningCode 仅允许 CompatWarningResponse 白名单内的值
# 边界：这些模型只被 compat 路由及其适配器使用，核心领域服务不接触 Java DTO 细节。

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from cooking_plan_agent.domain.models import StrictModel

# Stable contract identifier shared with the Spring Boot caller.
# 与 Spring Boot 调用方共享的稳定契约标识符
CONTRACT_VERSION = "cooking-agent-v1"

# Warning codes allowed by CookingPlanResultValidator.WarningCode (allow-list).
# 校验器允许的警告码白名单（对应 Java 侧 CookingPlanResultValidator.WarningCode）
CompatWarningCode = Literal[
    "CHECK_ALLERGEN_LABELS",  # 检查过敏原标签
    "MAY_REQUIRE_EXTRA_TIME",  # 可能需要额外时间
    "BUDGET_ESTIMATE_ONLY",  # 预算仅为估算
    "PANTRY_ITEM_UNVERIFIED",  # 库存食材未核实
    "COOK_THOROUGHLY",  # 需彻底烹饪
]

# 正整数 / 正小数 的可复用注解类型（复用 domain 里已有的约束语义）
PositiveInt = Annotated[int, Field(gt=0)]
PositiveDecimal = Annotated[Decimal, Field(gt=0)]


# ===========================================================================
# Request side — AgentCookingRequest + nested snapshots
# ===========================================================================


class CompatInputIngredient(StrictModel):
    """One ingredient the user already has (request.ingredients[])."""

    # 用户已拥有的单条食材（对应 request.ingredients[]）

    ingredientName: str  # 食材名称（必填）
    quantity: PositiveDecimal  # 数量（必填，>0）
    unit: str  # 单位（必填）
    source: str = "MANUAL"  # 来源，默认 MANUAL（手动录入）


class CompatConstraints(StrictModel):
    """request.constraints — merged dietary + allergen rules."""

    # 请求中的约束：合并后的饮食要求 + 过敏原规避规则

    requiredDietaryTagCodes: tuple[str, ...] = ()  # 必须满足的饮食标签码（默认空）
    avoidAllergenCodes: tuple[str, ...] = ()  # 需要规避的过敏原码（默认空）


class CompatRequestSnapshot(StrictModel):
    """request — the cooking-public-v1 request snapshot."""

    # 请求快照（cooking-public-v1 版本）

    contractVersion: str  # 契约版本号（必填）
    ingredients: tuple[CompatInputIngredient, ...] = ()  # 用户已有食材列表（默认空）
    servings: PositiveInt = 2  # 用餐人数，默认 2
    maxMinutes: PositiveInt | None = None  # 最长烹饪分钟数（可选）
    maxBudget: PositiveDecimal | None = None  # 最大预算（可选）
    currency: str | None = None  # 货币代码（可选）
    constraints: CompatConstraints = CompatConstraints()  # 约束（默认空约束）


class CompatPreferences(StrictModel):
    """preferences — merged dietary + allergen codes."""

    # 偏好：合并后的饮食标签码 + 过敏原码

    requiredDietaryTagCodes: tuple[str, ...] = ()  # 必须满足的饮食标签码（默认空）
    avoidAllergenCodes: tuple[str, ...] = ()  # 需要规避的过敏原码（默认空）


class CompatIngredientSnapshot(StrictModel):
    """One ingredient inside a candidate's snapshot."""

    # 候选方案快照中的单条食材

    sequenceNo: int = Field(ge=1)  # 序号，从 1 开始（连续递增）
    ingredientName: str  # 食材名称（必填）
    quantity: Decimal | None = None  # 数量（可选，这里不用 PositiveDecimal 以兼容历史数据）
    unit: str | None = None  # 单位（可选）
    optional: bool = False  # 是否可选（默认 False）


class CompatStepSnapshot(StrictModel):
    """One step inside a candidate's snapshot."""

    # 候选方案快照中的单个步骤

    stepNo: int = Field(ge=1)  # 步骤号，从 1 开始（连续递增）
    instruction: str  # 步骤说明（必填）


class CompatCandidateSnapshot(StrictModel):
    """snapshot — the serialised RecipeCandidate (JdbcCookingPlanRepository).

    Contains every structured field the candidate carries; the compat
    adapter maps it to an internal ExtractedRecipeCandidate without
    invoking the LLM (P0-02: no LLM re-parsing for compat requests).
    """

    # 候选方案快照：序列化后的 RecipeCandidate（来自 JdbcCookingPlanRepository）。
    # 适配器会直接把它映射为内部 ExtractedRecipeCandidate，而不调用 LLM 重新解析。

    recipeId: str  # 菜谱 ID（必填）
    name: str  # 菜名（必填）
    description: str = ""  # 描述（默认空字符串）
    defaultServings: PositiveInt  # 默认用餐人数（必填，>0）
    totalMinutes: PositiveInt | None = None  # 总分钟数（可选）
    estimatedCost: PositiveDecimal | None = None  # 估算成本（可选）
    currency: str | None = None  # 货币（可选）
    dietaryTagCodes: tuple[str, ...] = ()  # 饮食标签码（默认空）
    allergenCodes: tuple[str, ...] = ()  # 过敏原码（默认空）
    ingredients: tuple[CompatIngredientSnapshot, ...] = ()  # 食材列表（默认空）
    steps: tuple[CompatStepSnapshot, ...] = ()  # 步骤列表（默认空）


class CompatCandidateRequest(StrictModel):
    """candidates[] — a controlled candidate the agent may select."""

    # 候选请求：代理可选择的“受控候选方案”

    recipeId: UUID  # 菜谱 ID（UUID 类型）
    snapshot: CompatCandidateSnapshot  # 对应的候选快照


class CompatCookingRequest(StrictModel):
    """AgentCookingRequest — the full request body Spring Boot sends."""

    # 完整请求体（对应 Java 的 AgentCookingRequest）

    contractVersion: str  # 契约版本号（必填）
    requestId: UUID  # 请求 ID（必填）
    planId: UUID  # 计划 ID（必填）
    traceId: str  # 链路追踪 ID（必填）
    deadlineAt: datetime | None = None  # 截止时间（可选）
    request: CompatRequestSnapshot  # 请求快照
    preferences: CompatPreferences = CompatPreferences()  # 偏好（默认空）
    candidates: tuple[CompatCandidateRequest, ...] = ()  # 候选方案列表（默认空）


# ===========================================================================
# Response side — AgentCookingResponse + nested DTOs
# ===========================================================================


class CompatIngredientResponse(StrictModel):
    """ingredients[] — validated by CookingPlanResultValidator.validateIngredient."""

    # 响应中的单条食材（由 CookingPlanResultValidator.validateIngredient 校验）

    sequenceNo: int = Field(ge=1)  # 序号，从 1 开始（连续递增）
    ingredientName: str  # 食材名称（必填）
    quantity: Decimal | None = None  # 数量（可选）
    unit: str | None = None  # 单位（可选）
    availability: Literal["AVAILABLE", "TO_BUY"]  # 可得性：已有 / 需购买


class CompatStepResponse(StrictModel):
    """steps[] — stepNo must be contiguous from 1, no safety claims."""

    # 响应中的单个步骤：stepNo 必须从 1 连续递增，且不得包含安全声明

    stepNo: int = Field(ge=1)  # 步骤号，从 1 开始（连续递增）
    instruction: str  # 步骤说明（必填）


class CompatWarningResponse(StrictModel):
    """warnings[] — warningCode must hit the validator allow-list."""

    # 响应中的单条警告：warningCode 必须在校验器白名单内

    sequenceNo: int = Field(ge=1)  # 序号，从 1 开始（连续递增）
    warningCode: CompatWarningCode  # 警告码（限定为白名单 Literal）
    message: str  # 警告说明（必填）


class CompatCookingResponse(StrictModel):
    """AgentCookingResponse — must NOT carry fields beyond this record.

    ``status`` is ``"SUCCEEDED"`` on success; any other value is mapped
    by the Java adapter to AGENT_UNAVAILABLE (safe terminal state).
    """

    # 完整响应体（对应 Java 的 AgentCookingResponse）：不得携带本记录之外的字段。
    # 成功时 status = "SUCCEEDED"；其它值会被 Java 适配器映射为 AGENT_UNAVAILABLE（安全终态）。

    contractVersion: str  # 契约版本号（必填）
    requestId: UUID  # 请求 ID（必填）
    planId: UUID  # 计划 ID（必填）
    traceId: str  # 链路追踪 ID（必填）
    agentTraceId: str  # Agent 侧链路追踪 ID（必填）
    status: str  # 状态（成功为 "SUCCEEDED"）
    sourceRecipeId: UUID | None = None  # 来源菜谱 ID（可选）
    servings: int = 0  # 用餐人数（默认 0）
    totalMinutes: int | None = None  # 总分钟数（可选）
    estimatedCost: Decimal | None = None  # 估算成本（可选）
    currency: str | None = None  # 货币（可选）
    ingredients: tuple[CompatIngredientResponse, ...] = ()  # 食材列表（默认空）
    steps: tuple[CompatStepResponse, ...] = ()  # 步骤列表（默认空）
    warnings: tuple[CompatWarningResponse, ...] = ()  # 警告列表（默认空）
