# =============================================================================
# 领域模型定义模块（domain/models）
# -----------------------------------------------------------------------------
# 本文件集中定义烹饪计划 Agent 的 Pydantic 领域模型，涵盖：
#   - 严格基础模型（StrictModel）：不可变、禁止未知字段、字符串去空白
#   - 食材 / 菜谱中间表示（RecipeIR）及其解析结果
#   - 可调度任务（CookingTask）及其资源 / 依赖约束
#   - 库存 / 厨房资源快照（用于并发预留判定与 FEFO 分配）
#   - 各类请求 / 响应契约（READY / NEEDS_CONFIRMATION / INFEASIBLE / FAILED）
# 所有模型均继承 StrictModel，遵循“结构化输出、边界校验、不可变快照”原则。
# =============================================================================

from datetime import date, datetime  # Date type for expiry_date (no time component needed)
# ↑ date 用于“过期日期”（无需时间分量）；datetime 用于带时区的“开餐时刻”

from decimal import Decimal  # Exact decimal arithmetic — never float for inventory
# ↑ 精确十进制运算 —— 库存数量绝不用 float（避免浮点精度误差）

from enum import StrEnum  # String enum base class (P4-02 response types)
# ↑ 字符串枚举基类（P4-02 响应类型）

from typing import (  # Typed annotation composition (e.g. PositiveDecimal)
    Annotated,
)
# ↑ 类型注解组合（例如 PositiveDecimal 即用 Annotated 附加 Field 约束）

from pydantic import (  # Pydantic v2 building blocks
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
# ↑ Pydantic v2 构建组件

from cooking_plan_agent.domain.enums import (  # Domain enums used in CookingTask
    HeatLevel,
    WorkMode,
)
# ↑ CookingTask 用到的领域枚举

# ---------------------------------------------------------------------------
# 3.3  Strict base model — all domain models inherit these constraints
#      严格基础模型 —— 所有领域模型都继承这些约束
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """严格基础模型：所有领域模型的共同基类。

    Foundation for every domain model. Enforces immutability, boundary
    strictness, and whitespace hygiene at the Pydantic layer.

    在 Pydantic 层强制以下三条：不可变性、边界严格性（拒绝未知字段）、
    字符串去首尾空白。
    """

    model_config = ConfigDict(
        extra="forbid",  # 3.1 Reject unknown fields at API/LLM boundaries
        # ↑ 在 API / LLM 边界拒绝未知字段（防止脏数据渗入）
        frozen=True,  # 3.1 Prefer immutable models for snapshots & solver inputs
        # ↑ 模型不可变（快照与求解器输入需防篡改）
        str_strip_whitespace=True,  # 3.1 Normalize string inputs before validation
        # ↑ 校验前先对字符串去首尾空白（规范化输入）
    )


# ---------------------------------------------------------------------------
# 3.4  Reusable annotated types
#      可复用的注解类型
# ---------------------------------------------------------------------------

# 3.1 Use Decimal for quantities; never float for inventory arithmetic
# 3.1 数量一律用 Decimal；库存算术绝不用 float
PositiveDecimal = Annotated[Decimal, Field(gt=0)]
# ↑ 正数 Decimal：必须 > 0（用于数量、份数等）

# 3.1 Store confidence and evidence for inferred facts
# 3.1 为推断事实记录置信度与证据
Confidence = Annotated[Decimal, Field(ge=0, le=1)]
# ↑ 置信度 Decimal：取值区间 [0, 1]


# ---------------------------------------------------------------------------
# 3.4  Ingredient and recipe models
#      食材与菜谱模型
# ---------------------------------------------------------------------------


class EvidenceRef(StrictModel):
    """证据引用：某条推断事实的溯源引用。

    Provenance reference for an inferred fact (URL, document, retrieval timestamp).

    记录推断事实的来源（URL、文档、检索时间戳），用于审计与解释。
    """

    source_type: str  # e.g. "web_search", "user_input", "LLM_guess"
    # ↑ 来源类型，例如 "web_search" / "user_input" / "LLM_guess"
    title: str | None = None  # Human-readable label for the source
    # ↑ 来源的可读标签
    url: str | None = None  # Optional stable link to the evidence
    # ↑ 证据的可选稳定链接
    retrieved_at: str | None = None  # ISO-8601 timestamp of retrieval
    # ↑ 检索时间（ISO-8601 时间戳）


class IngredientDemand(StrictModel):
    """食材需求：单条食材项，区分“原始文本 / 规范化名称”，并携带置信度。

    A single ingredient entry with raw/canonical separation and confidence.
    """

    canonical_name: str  # Normalized unique name (e.g. "chicken breast")
    # ↑ 规范化后的唯一名称（如 "chicken breast"）
    raw_name: str  # 3.1 Original text from the recipe source
    # ↑ 菜谱来源中的原始文本
    quantity: PositiveDecimal  # Parsed numeric quantity (> 0)
    # ↑ 解析出的数值数量（> 0）
    unit: str  # e.g. "g", "ml", "piece"
    # ↑ 单位，如 "g" / "ml" / "piece"
    preparation_spec: str | None = None  # e.g. "diced", "minced" — optional prep note
    # ↑ 预处理说明，如 "diced" / "minced"（可选）
    input_state: str = "raw"  # State before use (default "raw")
    # ↑ 使用前的状态（默认 "raw"）
    output_state: str | None = None  # State after processing (e.g. "cooked")
    # ↑ 加工后的状态（如 "cooked"）
    allergen_tags: tuple[str, ...] = ()  # e.g. ("gluten", "dairy")
    # ↑ 过敏原标签，如 ("gluten", "dairy")
    confidence: Confidence  # LLM extraction confidence [0, 1]
    # ↑ LLM 提取置信度 [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Chain of sources supporting this ingredient
    # ↑ 支撑该食材的来源链


class Assumption(StrictModel):
    """假设：LLM 在菜谱解析过程中做出的推断或猜测。

    An inference or guess made by the LLM during recipe parsing.

    Captures uncertain decisions (e.g. 'assuming 200 C for baking') so that
    downstream consumers can surface them for user confirmation.

    捕获不确定的决策（例如“烘焙按 200°C 假设”），供下游呈现给用户确认。
    """

    text: str  # Human-readable assumption description
    # ↑ 假设的可读描述
    confidence: Confidence  # LLM confidence [0, 1]
    # ↑ LLM 置信度 [0, 1]
    evidence: tuple[EvidenceRef, ...] = ()  # Supporting sources, if any
    # ↑ 支撑来源（若有）


class RecipeStep(StrictModel):
    """菜谱步骤：在分解为可调度的 CookingTask 之前的单条步骤。

    A single recipe step before decomposition into schedulable CookingTasks.
    """

    step_number: int = Field(ge=1)  # 1-based position in recipe
    # ↑ 菜谱中的序号（从 1 开始）
    instruction: str  # Raw instruction text
    # ↑ 原始操作说明文本
    category: str = "general"  # e.g. "cutting", "heating", "mixing", "resting"
    # ↑ 分类，如 "cutting" / "heating" / "mixing" / "resting"
    pattern: str = "simple"  # Decomposition hint: "simple" | "boil" | "marinate" | "bake" | "stir_fry" | "simmer"
    # ↑ 分解提示："simple" | "boil" | "marinate" | "bake" | "stir_fry" | "simmer"
    active_duration_minutes: int | None = None  # Hands-on time (if specified)
    # ↑ 主动操作时长（若指定）
    passive_duration_minutes: int | None = None  # Wait/monitor time (if specified, e.g. boil 10 min)
    # ↑ 被动等待 / 监控时长（若指定，如“煮 10 分钟”）
    heat_level: HeatLevel = HeatLevel.NONE  # Stove intensity
    # ↑ 炉灶火力档位
    target_temperature_c: Decimal | None = None  # Target temperature in Celsius
    # ↑ 目标温度（摄氏度）
    interval_minutes: int | None = None  # For periodic check/stir tasks
    # ↑ 周期检查 / 翻炒任务的间隔分钟数
    resources_hint: tuple[str, ...] = ()  # Suggested equipment (e.g. "stove", "oven")
    # ↑ 建议设备（如 "stove" / "oven"）


class RecipeIR(StrictModel):
    """菜谱中间表示（IR = Intermediate Representation）。

    Intermediate representation of a parsed recipe, before scheduling.

    在调度之前，将原始来源属性与调度器所需属性分离的中间表示。
    """

    recipe_id: str  # Stable identifier across pipeline stages
    # ↑ 跨流水线阶段稳定的标识
    dish_name: str  # Human-readable dish name
    # ↑ 菜名（可读）
    original_servings: PositiveDecimal  # Servings as stated in the original recipe
    # ↑ 原菜谱标注的份数
    target_servings: PositiveDecimal  # Desired servings for this plan
    # ↑ 本次计划的目标份数
    source_language: str  # ISO language code of the source text
    # ↑ 来源文本的 ISO 语言代码
    ingredients: tuple[IngredientDemand, ...]  # Extracted ingredient list
    # ↑ 提取出的食材列表
    steps: tuple[RecipeStep, ...]  # Ordered cooking steps
    # ↑ 有序的烹饪步骤
    assumptions: tuple[Assumption, ...] = ()  # LLM assumptions made during parsing
    # ↑ 解析过程中 LLM 做出的假设

    @model_validator(mode="after")
    def require_content(self) -> "RecipeIR":
        """3.4 Validate that the recipe has at least one ingredient and one step.
        No side effects — pure validation only.

        校验菜谱至少包含一个食材和一个步骤。无副作用 —— 纯校验。
        """
        if not self.ingredients:
            raise ValueError("recipe must contain at least one ingredient")
        if not self.steps:
            raise ValueError("recipe must contain at least one step")
        return self


# ---------------------------------------------------------------------------
# 3.5  Task model — schedulable unit decomposed from recipe steps
#      任务模型 —— 由菜谱步骤分解出的可调度单元
# ---------------------------------------------------------------------------


class ResourceNeed(StrictModel):
    """资源需求：烹饪任务所需的一种资源。

    A resource required by a cooking task (e.g. stove burner, oven, mixing bowl).

    例如灶台炉头、烤箱、搅拌碗等。
    """

    resource_type: str  # Equipment category (e.g. "stove", "oven")
    # ↑ 设备类别（如 "stove" / "oven"）
    quantity: int = Field(ge=1)  # How many units of this resource are needed
    # ↑ 需要该资源的数量
    minimum_capacity: Decimal | None = None  # Minimum usable capacity (e.g. 2.0 L for a pot)
    # ↑ 最小可用容量（如锅需 2.0 L）
    capacity_unit: str | None = None  # Unit for capacity (e.g. "L", "kg")
    # ↑ 容量单位（如 "L" / "kg"）
    required_capabilities: tuple[str, ...] = ()  # Specific features needed (e.g. "induction")
    # ↑ 所需特定功能（如 "induction" 电磁感应）


class TaskDependency(StrictModel):
    """任务依赖：两个任务之间的先后约束。

    A precedence constraint between two tasks.
    """

    predecessor_id: str  # The task that must finish first
    # ↑ 必须先完成的前驱任务
    minimum_lag_minutes: int = Field(ge=0, default=0)  # Minimum gap after predecessor ends
    # ↑ 前驱结束后的最小间隔分钟数
    maximum_lag_minutes: int | None = Field(ge=0, default=None)  # Maximum gap (None = no upper bound)
    # ↑ 最大间隔分钟数（None 表示无上限）


class CookingTask(StrictModel):
    """烹饪任务：可调度的最小单元。

    A single schedulable unit — one recipe step decomposed into timing and resources.

    3.5 Do not represent 'boil for ten minutes' as one active task;
    the decomposition service splits it into start / passive-wait / finish.

    不要把“煮十分钟”表示为单一活动任务；分解服务会把它拆成
    「开始 / 被动等待 / 结束」。
    """

    task_id: str  # Unique task identifier
    # ↑ 任务唯一标识
    dish_id: str  # Parent recipe this task belongs to
    # ↑ 所属父菜谱标识
    instruction: str  # Human-readable cooking instruction
    # ↑ 可读的烹饪操作说明
    duration_minutes: int = Field(ge=1)  # Active time this task occupies resources
    # ↑ 该任务占用资源的主动时长
    work_mode: WorkMode  # ACTIVE (hands-on) or PASSIVE (monitoring)
    # ↑ ACTIVE（需动手）或 PASSIVE（仅监控）
    category: str  # e.g. "cutting", "heating", "mixing", "resting"
    # ↑ 分类，如 "cutting" / "heating" / "mixing" / "resting"
    heat_level: HeatLevel = HeatLevel.NONE  # Stove intensity; NONE for cold tasks
    # ↑ 炉灶火力；冷加工任务为 NONE
    target_temperature_c: Decimal | None = None  # Target temp in Celsius (if heating)
    # ↑ 目标温度（摄氏度，加热类任务才有）
    dependencies: tuple[TaskDependency, ...] = ()  # Predecessor constraints
    # ↑ 前驱约束
    resources: tuple[ResourceNeed, ...] = ()  # Equipment needed
    # ↑ 所需设备
    consumes_states: tuple[str, ...] = ()  # Ingredient states consumed (e.g. "diced_onion")
    # ↑ 消耗的食材状态（如 "diced_onion"）
    produces_states: tuple[str, ...] = ()  # States this task produces (e.g. "caramelized_onion")
    # ↑ 该任务产出的状态（如 "caramelized_onion"）
    batch_key: str | None = None  # Shared key for batchable tasks (e.g. same oven temp)
    # ↑ 可合并批次任务的共享键（如同一烤箱温度）
    safety_tags: tuple[str, ...] = ()  # Labels for safety rule enforcement (e.g. "raw_meat")
    # ↑ 安全规则执行标签（如 "raw_meat"）


# ---------------------------------------------------------------------------
# 3.6  Inventory snapshot models — immutable point-in-time views
#      库存快照模型 —— 不可变的时间点视图
# ---------------------------------------------------------------------------


class InventoryLotSnapshot(StrictModel):
    """库存批次快照：某个库存批次的不可变快照。

    3.6 Immutable snapshot of an inventory lot. Spring Boot handles
    concurrent reservation decisions against this snapshot.

    Spring Boot 基于该快照处理并发预留决策。
    """

    lot_id: str  # Unique lot identifier
    # ↑ 批次唯一标识
    item_id: str  # Stock item reference
    # ↑ 库存条目引用
    canonical_name: str  # Normalized item name
    # ↑ 规范化条目名称
    on_hand: Decimal = Field(ge=0)  # Total quantity currently in stock
    # ↑ 当前在库总数量
    reserved: Decimal = Field(ge=0)  # Quantity already reserved for other plans
    # ↑ 已被其他计划预留的数量
    unit: str  # Unit of measure (e.g. "g", "ml")
    # ↑ 计量单位（如 "g" / "ml"）
    expiry_date: date | None = None  # Expiration date, if applicable
    # ↑ 过期日期（若适用）

    @model_validator(mode="after")
    def reservation_cannot_exceed_stock(self) -> "InventoryLotSnapshot":
        """3.9 Reject reserved > on_hand at the model boundary.

        在模型边界拒绝 reserved > on_hand（预留量超过在库量）。
        """
        if self.reserved > self.on_hand:
            raise ValueError("reserved quantity exceeds on-hand quantity")
        return self


class KitchenResourceSnapshot(StrictModel):
    """厨房资源快照：厨房资源（电器 / 工具 / 工位）的不可变快照。

    3.6 Immutable snapshot of a kitchen resource (appliance, tool, station).
    """

    resource_id: str  # Unique resource identifier
    # ↑ 资源唯一标识
    resource_type: str  # Category (e.g. "stove", "oven", "sink")
    # ↑ 类别（如 "stove" / "oven" / "sink"）
    capacity: Decimal | None = None  # Maximum capacity (e.g. 4 burners)
    # ↑ 最大容量（如 4 个炉头）
    capacity_unit: str | None = None  # Unit for capacity (e.g. "burners", "L")
    # ↑ 容量单位（如 "burners" / "L"）
    capabilities: tuple[str, ...] = ()  # Features (e.g. "induction", "convection")
    # ↑ 功能特性（如 "induction" / "convection"）
    available: bool = True  # Whether the resource is operational
    # ↑ 资源是否可用


# ---------------------------------------------------------------------------
# 3.7  Response contracts — plan output, not database mutation
#      响应契约 —— 计划输出，而非数据库变更
# ---------------------------------------------------------------------------


class LotAllocation(StrictModel):
    """批次分配：对某个库存批次的拟扣减。

    A proposed deduction from a specific inventory lot.

    3.7 This is a plan, not a database mutation. Spring Boot persists
    and the client confirms after cooking.

    这是计划而非数据库变更：由 Spring Boot 持久化，客户端烹饪后确认。
    """

    inventory_lot_id: str  # Which lot to draw from
    # ↑ 从哪个批次扣减
    quantity: PositiveDecimal  # How much to deduct
    # ↑ 扣减数量
    unit: str  # Unit of the deduction
    # ↑ 扣减单位


class CompletionItem(StrictModel):
    """完成项：将跨菜谱满足同一食材的分配分组。

    Groups allocations that fulfill one ingredient across recipes.
    """

    completion_item_id: str  # Unique ID for this completion group
    # ↑ 该完成分组的唯一标识
    ingredient_name: str  # Canonical ingredient name
    # ↑ 规范化食材名称
    recipe_ids: tuple[str, ...]  # Which recipes contribute to this ingredient
    # ↑ 哪些菜谱贡献了该食材
    allocations: tuple[LotAllocation, ...]  # Specific lot deductions
    # ↑ 具体的批次扣减


class InventoryConsumptionProposal(StrictModel):
    """库存消耗方案：READY 响应中包含的顶层消耗计划。

    Top-level consumption plan included in a READY response.

    3.7 Carries a snapshot version so Spring Boot can detect stale proposals.

    携带快照版本，使 Spring Boot 能检测出“过期方案”。
    """

    inventory_snapshot_version: str  # Version of the inventory snapshot this was computed from
    # ↑ 计算该方案所依据的库存快照版本
    items: tuple[CompletionItem, ...]  # Per-ingredient completion groups
    # ↑ 每个食材的完成分组


# ---------------------------------------------------------------------------
# 3.8  Evidence models — structured web research I/O
#      证据模型 —— 结构化联网研究的输入 / 输出
# ---------------------------------------------------------------------------


class EvidenceQuery(StrictModel):
    """证据查询：面向联网研究的结构化问题。

    A structured question for web research. Contains only the gap info,
    never private user data.

    仅包含缺口信息，绝不含用户隐私数据。
    """

    query_text: str
    gap_type: str
    recipe_context: str
    target_fields: tuple[str, ...] = ()


class EvidenceResult(StrictModel):
    """证据结果：联网研究中的一条带引用证据。

    One cited piece of evidence from web research.
    """

    source_title: str
    source_url: str
    snippet: str
    confidence: Confidence
    extracted_fact: str
    fact_type: str
    fact_value: str


class CookingEvidence(StrictModel):
    """烹饪证据：从单个搜索文档提取的证据（手册 10.6）。

    Evidence extracted from a single search document (handbook 10.6).

    Narrow schema: only cooking-relevant fields. Rejects unexpected fields.
    Excerpts limited to the shortest text needed for traceability.

    窄化模式：仅保留与烹饪相关的字段，拒绝未知字段；摘录仅保留追踪所需的最短文本。
    """

    operation: str  # e.g. "stir-fry", "bake", "boil"
    # ↑ 烹饪操作，如 "stir-fry" / "bake" / "boil"
    heat_level: HeatLevel | None = None  # Stove intensity if stated
    # ↑ 炉灶火力（若来源有说明）
    duration_min_minutes: int | None = None  # Lower bound of duration range
    # ↑ 时长区间下界
    duration_max_minutes: int | None = None  # Upper bound of duration range
    # ↑ 时长区间上界
    explicit_temperature_c: Decimal | None = None  # Target temperature in Celsius
    # ↑ 明确给出的目标温度（摄氏度）
    source_url: str  # Source page URL
    # ↑ 来源页面 URL
    source_title: str  # Source page title
    # ↑ 来源页面标题
    source_excerpt: str  # Shortest text for traceability (not full page)
    # ↑ 用于追踪的最短摘录（非整页）


class ReconciledEvidence(StrictModel):
    """调和证据：多来源调和后的共识输出（手册 10.7）。

    Consensus output from multi-source reconciliation (handbook 10.7).

    Reports both the reconciled value AND whether sources disagreed enough
    to warrant user confirmation.

    同时报告“调和后的取值”与“来源分歧是否大到需要用户确认”。
    """

    heat_level: HeatLevel | None = None
    duration_min_minutes: int | None = None
    duration_max_minutes: int | None = None
    explicit_temperature_c: Decimal | None = None
    # How many independent sources contributed to each reconciled value
    # 每个调和值由多少个独立来源贡献
    source_count: int = 0
    # If True, disagreement exceeded threshold — surface for user confirmation
    # 若为 True，分歧超过阈值 —— 需呈现给用户确认
    needs_confirmation: bool = False
    # Raw evidence items that fed into the reconciliation
    # 参与调和的原始证据条目
    evidence_items: tuple["CookingEvidence", ...] = ()


# ---------------------------------------------------------------------------
# 3.9  LLM extraction intermediate models
#      LLM 提取中间模型
# ---------------------------------------------------------------------------


class ExtractedIngredient(StrictModel):
    """提取食材：LLM 原始提取的食材（规范化之前）。

    Raw ingredient as extracted by LLM, before canonicalisation.
    """

    raw_text: str
    name: str
    quantity: Decimal | None = None
    unit: str | None = None
    preparation: str | None = None
    extraction_source: str = "EXPLICIT"
    confidence: Confidence = Decimal("1.0")


class ExtractedStep(StrictModel):
    """提取步骤：LLM 原始提取的步骤（分解之前）。

    Raw step as extracted by LLM, before decomposition.
    """

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
    """提取菜谱候选：LLM 提取输出。

    LLM extraction output — optional fields allowed, raw spans retained.

    允许可选字段缺失，并保留原始文本片段。
    """

    recipe_id: str
    dish_name: str
    original_servings: PositiveDecimal
    source_language: str
    ingredients: tuple[ExtractedIngredient, ...]
    steps: tuple[ExtractedStep, ...]
    extraction_source: str = "LLM"
    inferred_fields: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 3.10  Recipe gap detection
#       菜谱缺口检测
# ---------------------------------------------------------------------------


class RecipeGap(StrictModel):
    """菜谱缺口：菜谱候选中检测到的缺口。

    A detected gap in a recipe candidate.
    """

    gap_id: str
    recipe_id: str
    field_path: str
    current_value: str | None = None
    gap_class: str  # "critical" | "safety_critical" | "resource_critical" | "optimisation" | "cosmetic"
    # ↑ 缺口级别："critical" | "safety_critical" | "resource_critical" | "optimisation" | "cosmetic"
    description: str
    confidence: Confidence
    evidence: tuple[EvidenceRef, ...] = ()


# ---------------------------------------------------------------------------
# 3.11  Safety rule engine models
#       安全规则引擎模型
# ---------------------------------------------------------------------------


class SafetyInsertion(StrictModel):
    """安全插入：锚定在菜谱步骤之间的结构化安全任务（P0-07）。

    A structured safety-task insertion anchored between recipe steps (P0-07).

    Produced by safety rules (e.g. cross-contamination) instead of bare
    task IDs. Carries the exact step anchors so merge_preparation can build
    the ``raw task → sanitise task → RTE task`` dependency chain:

      - after_step_number: the LAST step that must finish before the safety
        task starts (e.g. raw protein handling).
      - before_step_number: the FIRST step that must start after the safety
        task ends (e.g. ready-to-eat assembly/plating).

    Duration and resources come from policy configuration — never the old
    fixed 1-minute placeholder.

    由安全规则（如交叉污染）产生，而非裸任务 ID。携带精确的步骤锚点，
    使 merge_preparation 能构建「生食任务 → 消毒任务 → 即食任务」的依赖链：
      - after_step_number：安全任务开始前必须完成的最后一个步骤（如处理生蛋白）
      - before_step_number：安全任务结束后才能开始的第一个步骤（如即食拼盘 / 装盘）
    时长与资源来自策略配置 —— 不再使用旧的固定 1 分钟占位值。
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
    """安全发现：单条安全规则评估的输出。

    Output of a single safety rule evaluation.
    """

    rule_id: str
    severity: str  # "hard_unrepairable" | "hard_repairable" | "warning"
    # ↑ 严重级别："hard_unrepairable" | "hard_repairable" | "warning"
    description: str
    affected_task_ids: tuple[str, ...] = ()
    affected_ingredient_names: tuple[str, ...] = ()
    recommended_action: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    # P0-07: structured insertion template when the finding is repairable by
    # injecting a safety task between two recipe steps.
    # P0-07：当可通过在两个菜谱步骤之间注入安全任务来修复时，使用的结构化插入模板。
    insertion: SafetyInsertion | None = None


class SafetyReport(StrictModel):
    """安全报告：整个计划的聚合安全评估。

    Aggregated safety evaluation for the entire plan.
    """

    report_id: str
    findings: tuple[SafetyFinding, ...] = ()
    is_safe: bool
    has_unrepairable: bool
    required_safety_task_ids: tuple[str, ...] = ()
    # P0-07: structured insertions anchored between recipe steps.
    # P0-07：锚定在菜谱步骤之间的结构化插入。
    insertions: tuple[SafetyInsertion, ...] = ()
    # P3-04: the regional policy pack that produced this report (None when a
    # legacy engine without a bound policy ran — never blocks evaluation).
    # P3-04：生成该报告的区域策略包（当运行的是未绑定策略的旧引擎时为 None —— 永不阻断评估）。
    safety_policy: "SafetyPolicyRecord | None" = None


class SafetyContext(StrictModel):
    """安全上下文：安全规则评估的输入上下文。

    Input context for safety rule evaluation.
    """

    recipes: tuple["RecipeIR", ...]
    dietary_restrictions: tuple[str, ...] = ()
    user_allergens: tuple[str, ...] = ()
    inventory_lots: tuple["InventoryLotSnapshot", ...] = ()
    cooking_date: date | None = None


# ---------------------------------------------------------------------------
# 3.11b  Regional safety policy records (P3-04)
#        区域安全策略记录（P3-04）
# ---------------------------------------------------------------------------


class PolicySourceRef(StrictModel):
    """策略来源引用：官方安全策略来源的可序列化引用（D7）。

    Serialisable reference to an official safety-policy source (D7).
    """

    source_id: str
    title: str
    url: str


class SafetyPolicyRecord(StrictModel):
    """安全策略记录：附加到计划上的策略溯源（P3-04）。

    Policy provenance attached to plans (P3-04).

    Recorded on READY/CONFIRMATION responses and retained in state so every
    plan carries the region, version, and official sources that produced its
    safety constraints — the basis for threshold traceability and audit of
    historical checkpoints (old versions remain registered for that purpose).

    记录在 READY / CONFIRMATION 响应中并保留在状态里，使每个计划都携带
    产生其安全约束的地区、版本与官方来源 —— 这是阈值可追溯与历史检查点审计的基础
    （旧版本为此仍保留注册）。
    """

    region: str
    version: str
    effective_at: date
    sources: tuple[PolicySourceRef, ...] = ()


# ---------------------------------------------------------------------------
# 3.12  Feasibility check and repair models
#       可行性检查与修复模型
# ---------------------------------------------------------------------------


class IngredientFeasibility(StrictModel):
    """食材可行性：单个食材的可行性结果。

    Feasibility result for one ingredient.
    """

    ingredient_name: str
    required: Decimal
    available: Decimal
    shortage: Decimal
    unit: str
    proposed_allocations: tuple["LotAllocation", ...] = ()


class FeasibilityReport(StrictModel):
    """可行性报告：跨所有维度的聚合可行性。

    Aggregated feasibility across all dimensions.
    """

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
    """修复选项：用户可选的、经校验的不可行修复选择。

    A validated choice the user can select to resolve infeasibility.
    """

    option_id: Annotated[str, Field(min_length=1, max_length=128)]
    option_type: str  # "substitute_ingredient" | "reduce_servings" | "alternative_equipment" | "replace_dish" | "extend_time" | "purchase"
    # ↑ 选项类型："substitute_ingredient" | "reduce_servings" | "alternative_equipment" | "replace_dish" | "extend_time" | "purchase"
    description: str
    changes: tuple[str, ...]
    effects: tuple[str, ...]
    # Machine-readable values used to build ApprovedDecision payloads. UI
    # prose is deliberately never parsed back into business data.
    # 用于构建 ApprovedDecision 负载的机器可读值。UI 文案刻意绝不被反向解析为业务数据。
    payload: dict[str, object] = {}
    revalidation_status: str = "validated"


class ApprovedDecision(StrictModel):
    """已批准决策：客户端可提交的结构化决策（P0-06）。

    A structured, client-submittable decision (P0-06).

    Unlike a bare option_id string, an ApprovedDecision carries:
      - option_id:  which presented option was chosen
      - option_type: one of the five supported decision kinds
      - payload:    machine-readable values (servings, minutes, ingredient
                    substitution, resource alternative, dish to replace)
      - plan_revision: version of the confirmation response the client is
                    answering — used to reject stale confirmations

    The confirmation response returns these verbatim; the client resubmits
    them in the next request's approved_decisions field.

    与裸的 option_id 字符串不同，ApprovedDecision 携带：
      - option_id：选择了哪个呈现的选项
      - option_type：五种受支持决策类型之一
      - payload：机器可读值（份数、分钟数、食材替换、资源替代、要替换的菜）
      - plan_revision：客户端所应答的确认响应版本 —— 用于拒绝过期确认
    确认响应原样返回这些字段；客户端在下一次请求的 approved_decisions 字段中原样重提。
    """

    option_id: str
    option_type: str
    payload: dict[str, object] = {}
    plan_revision: str | None = None


class WorkflowError(StrictModel):
    """工作流错误：工作流级失败的结构化错误。

    Structured error for workflow-level failures.

    P2-03: the client-facing text is resolved from the centralised public
    message catalog (domain.errors) — ``message`` is an internal diagnostic
    and must never leak provider payloads, secrets or recipe text. A node
    may explicitly override the public text with ``public_message`` (still
    free of sensitive detail); otherwise the catalog row decides.

    P2-03：面向客户端的文案从集中的公共消息目录（domain.errors）解析 ——
    ``message`` 是内部诊断信息，绝不能泄露提供商负载、密钥或菜谱文本。
    节点可用 ``public_message`` 显式覆盖公共文案（仍须无敏感细节）；否则由目录行决定。
    """

    error_code: str
    # Internal diagnostic message for logs/support. Not rendered verbatim to
    # the client — the catalog row for error_code provides the public text.
    # 内部诊断消息（供日志 / 支持使用）。不会原样渲染给客户端 —— 由 error_code 对应的目录行提供公共文案。
    message: str
    correlation_id: str
    node_name: str | None = None
    recoverable: bool = False
    # Optional explicit override of the catalog's public message. Must stay
    # stable and free of sensitive detail; None falls back to the catalog.
    # 对目录公共文案的可选显式覆盖。必须稳定且不含敏感细节；None 则回退到目录。
    public_message: str | None = None
    # Controlled diagnostic context (e.g. exception_type) for internal logs
    # only. Must not contain secrets or raw provider payloads.
    # 仅供内部日志使用的受控诊断上下文（如 exception_type）。不得包含密钥或原始提供商负载。
    diagnostics: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# 3.13  API request / response contracts
#       API 请求 / 响应契约
# ---------------------------------------------------------------------------


class RecipeInput(StrictModel):
    """菜谱输入：GeneratePlanRequest 中单个菜谱的类型化输入（P0-03）。

    Typed input for one recipe in a GeneratePlanRequest (P0-03).

    Replaces the loose ``tuple[dict, ...]`` so structural constraints,
    positive servings, and string bounds are enforced at the Pydantic
    boundary instead of inside workflow nodes.

    取代松散的 ``tuple[dict, ...]``，从而在 Pydantic 边界处（而非工作流节点内部）
    强制结构约束、正数份数与字符串长度边界。
    """

    recipe_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=1_000_000)
    target_servings: PositiveDecimal


class PreprocessRecipesRequest(StrictModel):
    """预处理请求：发给 preprocess 端点的请求。

    Request to the preprocess endpoint.

    The Spring Boot backend sends raw recipe text BEFORE generate() and
    receives fully-populated ``ExtractedRecipeCandidate`` snapshots back
    (LLM structuring/completion plus deterministic fallback done once).
    Those candidates are then passed back on the generate request as
    ``preparsed_candidates`` so the workflow never re-parses or re-asks
    about gaps.

    Spring Boot 后端在 generate() 之前发送原始菜谱文本，并收到完全填充的
    ``ExtractedRecipeCandidate`` 快照（LLM 结构化 / 补全 + 确定性兜底只做一次）。
    这些候选随后作为 ``preparsed_candidates`` 传回 generate 请求，
    使工作流不再重复解析、也不再就缺口反复询问。
    """

    request_id: str = Field(min_length=1, max_length=128)
    recipes: tuple[RecipeInput, ...]


class PreprocessRecipesResponse(StrictModel):
    """预处理响应：preprocess 端点的响应。

    Response from the preprocess endpoint.

    ``recipes`` carries one populated candidate per input recipe — missing
    operational values are completed by the LLM or deterministic fallback.

    ``recipes`` 为每个输入菜谱携带一个已填充的候选 —— 缺失的操作值由 LLM 或确定性兜底补齐。
    """

    recipes: tuple[ExtractedRecipeCandidate, ...]


class GeneratePlanRequest(StrictModel):
    """生成计划请求：来自 Spring Boot 的内部请求。

    Internal request from Spring Boot.
    """

    request_id: str
    user_id: str
    recipes: tuple[RecipeInput, ...]  # Typed recipe inputs (P0-03)
    # ↑ 类型化菜谱输入（P0-03）
    dietary_restrictions: tuple[str, ...] = ()
    user_allergens: tuple[str, ...] = ()
    time_limit_minutes: int | None = None
    # --- Time semantics (P0-05) ---
    # --- 时间语义（P0-05）---
    # cooking_date: the calendar day the plan is executed on. Drives the
    # safety engine's expired-lot check and FEFO inventory allocation.
    # cooking_date：计划执行的日历日。驱动安全引擎的过期批次检查与 FEFO 库存分配。
    cooking_date: date | None = None
    # serving_at: absolute serving time WITH timezone. Only when both date
    # and timezone are known is this converted to an absolute instant; the
    # legacy `serving_time` (HH:MM string) is kept for back-compat but is
    # never treated as an absolute wall-clock by itself.
    # serving_at：带时区的绝对开餐时刻。仅当日期与时区均已知时才转换为绝对时刻；
    # 旧字段 `serving_time`（HH:MM 字符串）为向后兼容保留，但绝不单独视为绝对墙上时钟。
    serving_at: datetime | None = None
    serving_time: str | None = None
    inventory_lots: tuple["InventoryLotSnapshot", ...] = ()
    kitchen_resources: tuple["KitchenResourceSnapshot", ...] = ()
    approved_decisions: tuple["ApprovedDecision", ...] = ()
    schema_version: str = "1.0"
    # Revision of the confirmation response these decisions answer (P0-06).
    # Used to reject stale confirmations when the plan has changed.
    # 这些决策所应答的确认响应版本（P0-06）。当计划已变更时用于拒绝过期确认。
    plan_revision: str | None = None
    # Structured candidates injected by the compat layer. When non-empty,
    # parse_recipes_node uses them directly and never calls the LLM
    # extractor (P0-02 rule 4). Kept optional so native requests are
    # unaffected.
    # 由兼容层注入的结构化候选。非空时，parse_recipes_node 直接使用它们，
    # 绝不调用 LLM 提取器（P0-02 规则 4）。保持可选，原生请求不受影响。
    preparsed_candidates: tuple["ExtractedRecipeCandidate", ...] = ()
    # P3-04: explicit regional food-safety policy selection (ISO alpha-2,
    # e.g. "US"/"SG"). When unset, the deployment default
    # (Settings.safety_policy_region) applies. An unknown region is rejected —
    # never silently falls back (D6).
    # P3-04：显式选择区域食品安全策略（ISO alpha-2，如 "US"/"SG"）。
    # 未设置时，使用部署默认值（Settings.safety_policy_region）。
    # 未知区域会被拒绝 —— 绝不静默回退（D6）。
    region: str | None = None


class ReadyPlanResponse(StrictModel):
    """就绪计划响应：带有调度表的已验证计划。

    READY response: verified plan with schedule.
    """

    plan_id: str
    status: str = "READY"
    solver_status: str
    makespan_minutes: int
    timeline: tuple[dict[str, object], ...]
    # Dependency-driven task graph for execution UIs.  Unlike ``timeline``,
    # this never asks the user to start a task at a fixed minute.
    # 供执行 UI 使用的、由依赖驱动的任务图。与 ``timeline`` 不同，
    # 它绝不会要求用户在某个固定分钟开始任务。
    execution_flow: tuple[dict[str, object], ...] = ()
    completion_checklist: tuple["CompletionItem", ...]
    mise_en_place: tuple[dict[str, object], ...]
    dish_completions: tuple[dict[str, object], ...]
    # P3-04: policy provenance (region/version/sources) that produced the plan.
    # P3-04：生成该计划的策略溯源（地区 / 版本 / 来源）。
    safety_policy: "SafetyPolicyRecord | None" = None
    # P4-01: optional additive schedule explanation ("why this timing/order").
    # explanation_source ∈ {"llm", "deterministic", "disabled"}. The
    # explanation never alters the verified schedule — it is display-only.
    # P4-01：可选的附加调度解释（“为何这个时间 / 顺序”）。
    # explanation_source ∈ {"llm", "deterministic", "disabled"}。
    # 该解释绝不改变已验证的调度 —— 仅用于展示。
    explanation: str | None = None
    explanation_source: str | None = None


# ---------------------------------------------------------------------------
# 3.13a  Structured confirmation questions (P4-02)
#        结构化确认问题（P4-02）
# ---------------------------------------------------------------------------


class QuestionResponseType(StrEnum):
    """How a ConfirmationQuestion expects to be answered (P4-02).

    ConfirmationQuestion 期望被如何回答（P4-02）。
    """

    CHOICE = "CHOICE"  # The client selects exactly one QuestionOption value
    # ↑ 客户端恰好选择一个 QuestionOption 的 value
    TEXT = "TEXT"  # The client supplies a bounded free-text value
    # ↑ 客户端提供一个有界自由文本值


class QuestionOption(StrictModel):
    """问题选项：CHOICE 确认问题的单个可选项（P4-02）。

    A single selectable answer for a CHOICE confirmation question (P4-02).

    ``value`` is the stable token the client echoes back inside a
    QuestionAnswer. For repair-option questions it is the presented
    ApprovedDecision's ``option_id``, so the mapping back to the decision
    is lossless — the server never rewrites or re-derives the payload
    from prose (D9).

    ``value`` 是客户端在 QuestionAnswer 中回显的稳定 token。对修复选项类问题，
    它是所呈现 ApprovedDecision 的 ``option_id``，因此回映射到决策是无损的 ——
    服务端绝不从文案重写或重新推导负载（D9）。
    """

    value: str
    label: str
    suggested: bool = False


class ConfirmationQuestion(StrictModel):
    """确认问题：字段级、可被客户端渲染的确认问题（P4-02）。

    A field-level, client-renderable confirmation question (P4-02).

    Replaces the fixed legacy ``questions`` strings with a structured
    form: each question carries a stable ``question_id`` (derived from
    stable domain keys — recipe_id + field_path — never from array
    position, D6), the domain field it targets, the prompt, the expected
    response type, and — for CHOICE — the exact allowed option values.

    用结构化形式取代固定的旧 ``questions`` 字符串：每个问题携带稳定的 ``question_id``
    （由稳定领域键派生 —— recipe_id + field_path —— 绝不来自数组下标，D6）、
    它所指向的领域字段、提示语、期望响应类型，以及（对 CHOICE 而言）精确允许的选项值。
    """

    question_id: str
    field_path: str
    prompt: str
    response_type: QuestionResponseType
    options: tuple[QuestionOption, ...] = ()
    required: bool = True
    suggested_value: str | None = None


class QuestionAnswer(StrictModel):
    """问题答案：客户端对 ConfirmationQuestion 提交的答案（P4-02）。

    A client-submitted answer to a ConfirmationQuestion (P4-02).

    ``value`` is validated against the presented question: for CHOICE it
    must hit one of the option values; for TEXT it must be non-empty and
    within the configured length bound. Unknown question_ids and
    duplicate answers are rejected.

    ``value`` 会针对所呈现的问题进行校验：CHOICE 必须命中某个选项值；
    TEXT 必须非空且在配置的长度边界内。未知的 question_id 与重复答案会被拒绝。
    """

    question_id: str
    value: str


class ConfirmationPlanResponse(StrictModel):
    """待确认计划响应：NEEDS_CONFIRMATION 响应。

    NEEDS_CONFIRMATION response.

    ``decisions`` carries the structured, client-submittable approved
    decisions (P0-06). The client resubmits these verbatim in the next
    request's ``approved_decisions`` field.

    P4-02: ``confirmation_questions`` carries the field-level structured
    form the client renders and answers; answers map losslessly back to
    ``ApprovedDecision`` (repair-option questions) before re-entry.
    ``questions`` remains a legacy dual-emit of plain strings for older
    clients — deprecated since P4-02, removed when contract v2 lands.

    ``decisions`` 携带结构化、可提交的已批准决策（P0-06）。客户端在下次请求的
    ``approved_decisions`` 字段中原样重提。

    P4-02：``confirmation_questions`` 携带客户端渲染并作答的字段级结构化表单；
    答案在重新进入前无损映射回 ``ApprovedDecision``（修复选项类问题）。
    ``questions`` 仍是为旧客户端双发（dual-emit）的纯字符串旧字段 —— 自 P4-02 起弃用，
    契约 v2 落地时移除。
    """

    plan_id: str
    status: str = "NEEDS_CONFIRMATION"
    assumptions: tuple["Assumption", ...] = ()
    repair_options: tuple["RepairOption", ...] = ()
    # P4-02: legacy plain-string questions (dual-emit, deprecated).
    # P4-02：旧版纯字符串问题（双发，已弃用）。
    questions: tuple[str, ...] = ()
    # P4-02: field-level structured confirmation form.
    # P4-02：字段级结构化确认表单。
    confirmation_questions: tuple["ConfirmationQuestion", ...] = ()
    decisions: tuple["ApprovedDecision", ...] = ()
    plan_revision: str | None = None
    # P3-04: policy provenance (region/version/sources) that produced the plan.
    # P3-04：生成该计划的策略溯源（地区 / 版本 / 来源）。
    safety_policy: "SafetyPolicyRecord | None" = None


class InfeasiblePlanResponse(StrictModel):
    """不可行计划响应：INFEASIBLE 响应。

    INFEASIBLE response.
    """

    plan_id: str
    status: str = "INFEASIBLE"
    reasons: tuple[str, ...]
    safe_alternatives: tuple[str, ...] = ()


class FailedPlanResponse(StrictModel):
    """失败计划响应：FAILED 响应。

    FAILED response.
    """

    status: str = "FAILED"
    error_code: str
    correlation_id: str
    message: str


class ConfirmationAnswersRequest(StrictModel):
    """P5-4: 确认续答请求体 —— 用户对 ConfirmationQuestion 的答复集。

    延续 GeneratePlanRequest 的契约风格：新增字段全可选、不破坏既有
    请求。user_id 未提供时不启用长期偏好记忆（零回归）。
    """

    plan_id: str
    answers: tuple["QuestionAnswer", ...]
    user_id: str | None = None
    plan_revision: str | None = None


class ErrorEnvelope(StrictModel):
    """统一协议错误封装（P3-05）。

    Unified protocol-error envelope (P3-05).

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

    每个受管端点在协议 / HTTP 级失败时都返回该结构 —— 包括 Pydantic 校验（422）、
    认证（401/403）、未找到（404）、幂等冲突（409）、背压（429/503）与
    意外的内部错误（500）。合法的业务结果（READY / NEEDS_CONFIRMATION /
    INFEASIBLE / FAILED）保留各自的响应模型，绝不伪装成协议错误。

    ``retryable`` 由错误目录（domain/errors.py）决定，绝不从消息文本推断，
    因此客户端可编程判断是否重试。``details`` 只携带字段级、安全的信息 ——
    绝不含原始输入、堆栈跟踪或提供商负载。
    """

    status: int
    """HTTP status code (4xx/5xx) of the failing response.

    失败响应的 HTTP 状态码（4xx/5xx）。
    """

    error_code: str
    """Stable machine-readable code from the error catalog.

    来自错误目录的稳定机器可读代码。
    """

    message: str
    """Short, human-readable, non-sensitive description.

    简短、可读、不含敏感信息的描述。
    """

    correlation_id: str
    """Same value echoed in the X-Request-ID response header.

    与 X-Request-ID 响应头中回显的值相同。
    """

    details: dict[str, object] | list[dict[str, object]] | None = None
    """Field-level diagnostics only (validation loc/type, retry hint).

    仅字段级诊断信息（校验位置 / 类型、重试提示）。
    """

    retryable: bool = False
    """Whether clients may retry; decided by the error catalog.

    客户端是否可以重试；由错误目录决定。
    """


# ---------------------------------------------------------------------------
# Union type for polymorphic response
# 多态响应的联合类型
# ---------------------------------------------------------------------------

PlanResponse = ReadyPlanResponse | ConfirmationPlanResponse | InfeasiblePlanResponse | FailedPlanResponse
# ↑ 计划响应联合类型：四种业务结果之一

# ---------------------------------------------------------------------------
# Resolve forward references via model_rebuild()
# 通过 model_rebuild() 解析前向引用
# ---------------------------------------------------------------------------

# 以下类中使用了字符串形式的前向引用（如 "RecipeIR"、"LotAllocation" 等），
# 这些引用在类定义时可能尚未定义，因此需要在模块末尾显式调用 model_rebuild()
# 以重建模型、解析前向引用并完成校验准备。
EvidenceQuery.model_rebuild()
SafetyContext.model_rebuild()
IngredientFeasibility.model_rebuild()
GeneratePlanRequest.model_rebuild()
ReadyPlanResponse.model_rebuild()
ConfirmationPlanResponse.model_rebuild()
ReconciledEvidence.model_rebuild()
