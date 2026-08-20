# =============================================================================
# LangGraph 工作流状态模块（workflow/state）
# -----------------------------------------------------------------------------
# 按手册 8.2 定义工作流状态的 TypedDict。工作流状态使用 TypedDict（可序列化），
# 边界处的值使用 Pydantic 模型。Provider 客户端与密钥绝不存储于此。
# =============================================================================

"""LangGraph workflow state — TypedDict per handbook 8.2.

LangGraph 工作流状态 —— 按手册 8.2 使用 TypedDict。

Uses TypedDict for workflow state (serialisable) and Pydantic models
for values at boundaries. Provider clients and secrets are NOT stored here.

工作流状态使用 TypedDict（可序列化），边界处的值使用 Pydantic 模型。
Provider 客户端与密钥绝不存储于此。
"""

from typing import TypedDict

from cooking_plan_agent.domain.models import (
    Assumption,
    ConfirmationPlanResponse,
    CookingTask,
    ExtractedRecipeCandidate,
    FeasibilityReport,
    GeneratePlanRequest,
    PlanResponse,
    QuestionAnswer,
    RecipeGap,
    RecipeIR,
    ReconciledEvidence,
    RepairOption,
    SafetyPolicyRecord,
    SafetyReport,
    WorkflowError,
)
from cooking_plan_agent.preparation.task_graph import TaskGraph
from cooking_plan_agent.scheduling.models import (
    RepairAttemptRecord,
    ScheduleResult,
    VerificationReport,
)


class PlanState(TypedDict, total=False):
    """Workflow state carried between LangGraph nodes.

    在 LangGraph 节点之间传递的工作流状态。

    Only serialisable domain objects. No provider clients, request-scoped
    secrets, or OR-Tools model objects (handbook 8.2).

    仅包含可序列化的领域对象。不含 Provider 客户端、请求级密钥或
    OR-Tools 模型对象（手册 8.2）。

    total=False means every field is Optional — nodes only return the
    CHANGED subset, and downstream nodes use .get() to safely read.

    total=False 意味着每个字段都是 Optional —— 节点只返回被更改的子集，
    下游节点使用 .get() 安全读取。
    """

    # --- Input (populated once, never mutated) ---
    # --- 输入（只填充一次，绝不修改） ---
    request: GeneratePlanRequest

    # --- Parsing & inference ---
    # --- 解析与推断 ---
    # Candidates are raw extraction output; RecipeIR is the validated form
    # 候选是原始抽取输出；RecipeIR 是经过验证的形式
    extracted_candidates: tuple[ExtractedRecipeCandidate, ...]
    gaps: tuple[RecipeGap, ...]
    # Evidence stored as plain dicts to keep state serialisable (no dataclass coupling)
    # 证据以普通 dict 存储，以保持状态可序列化（无 dataclass 耦合）
    evidence: tuple[dict[str, object], ...]
    # Per-gap reconciled research results (keyed by gap_id for traceability)
    # 按缺口调和的研究结果（以 gap_id 为键，便于追溯）
    research_evidence: dict[str, ReconciledEvidence]
    # Evidence-backed assumptions produced by apply_research_evidence_node
    # (P1-01). Merged into RecipeIR.assumptions so provenance is traceable.
    # 由 apply_research_evidence_node 产生的有证据支撑的假设（P1-01）。
    # 合并进 RecipeIR.assumptions，使来源可追溯。
    research_assumptions: tuple[Assumption, ...]

    # --- Validation ---
    # --- 验证 ---
    parsed_recipes: tuple[RecipeIR, ...]
    safety_report: SafetyReport
    # P3-04: resolved regional policy provenance, recorded on terminal
    # responses and retained in state so historical plans stay auditable.
    # P3-04：已解析的区域政策来源，记录在终态响应上并保留在状态中，
    # 使历史计划保持可审计。
    safety_policy: SafetyPolicyRecord
    feasibility_report: FeasibilityReport
    # Repair options injected by check_feasibility when infeasible but fixable
    # 不可行但可修复时，由 check_feasibility 注入的修复选项
    repair_options: tuple[RepairOption, ...]

    # --- Preparation & scheduling ---
    # --- 预处理与排程 ---
    # Tasks are split into three categories so merge_preparation can
    # apply different dedup/priority rules per category
    # 任务拆分为三类，使 merge_preparation 可以按类别应用不同的去重 / 优先级规则
    recipe_tasks: tuple[CookingTask, ...]
    prep_tasks: tuple[CookingTask, ...]
    safety_tasks: tuple[CookingTask, ...]
    # P2-01: human-readable summaries of shared-prep merge/branch/isolate
    # decisions, for observability and regression tests.
    # P2-01：共享预处理 merge/branch/isolate 决策的人类可读摘要，
    # 用于可观测性与回归测试。
    prep_observations: tuple[str, ...]
    task_graph: TaskGraph
    schedule_result: ScheduleResult
    # Independent verification report (separate from solver's internal check)
    # 独立验证报告（与求解器内部检查分离）
    verification_report: VerificationReport
    # P4-01: additive schedule explanation set by explain_schedule_node
    # between verify and render READY. Display-only — never affects the
    # schedule. explanation_source ∈ {"llm", "deterministic", "disabled"}.
    # P4-01：由 explain_schedule_node 在 verify 与 render READY 之间设置的
    # 加法式排程解释。仅用于展示 —— 绝不影响排程。
    # explanation_source ∈ {"llm", "deterministic", "disabled"}。
    explanation: str | None
    explanation_source: str | None

    # --- Terminal output ---
    # --- 终态输出 ---
    # Exactly ONE terminal field is populated by the terminal node that fires
    # 恰好一个终态字段由触发的终态节点填充
    response: PlanResponse
    needs_confirmation: bool
    confirmation_context: ConfirmationPlanResponse

    # --- Error ---
    # --- 错误 ---
    # Set by any node that encounters a recoverable/terminal error;
    # the graph uses this to route to error terminal nodes
    # 由任何遇到可恢复 / 终态错误的节点设置；图用它路由到错误终态节点
    error: WorkflowError

    # --- P5: agent trace (shared by all agentic phases) ---
    # --- P5：agent 留痕（所有 agentic 阶段共享） ---
    # 每步 agent 决策（repair / tool_call / question）的留痕，保持可序列化。
    # 元素为 plain dict（{"step": int, "action": str, "detail": dict}）。
    agent_trace: tuple[dict[str, object], ...]

    # --- P5-3: schedule repair loop ---
    # --- P5-3：排程修复循环 ---
    # 已验证失败的修复尝试计数；超过 max_attempts 则 FAILED（保证终止）。
    repair_attempts: int
    repair_history: tuple[RepairAttemptRecord, ...]
    # 请求级求解覆盖（如 {"optimization_level": "phase12"}）——不污染全局 Settings。
    solver_overrides: dict[str, object]

    # --- P5-2: ReAct controller ---
    # --- P5-2：ReAct 控制器 ---
    # 对话/推理消息（role/content），供 LLM 控制器与未来对话阶段复用。
    messages: tuple[dict[str, str], ...]
    # 当前循环步数；超过 agent_max_steps 强制落回确定性 DAG。
    agent_step: int
    # "controller"（LLM 编排）| "deterministic"（回退 DAG）。
    agent_mode: str
    # 每轮工具调用与观察留痕（agent_trace 的具体化）。
    tool_calls: tuple[dict[str, object], ...]
    observations: tuple[dict[str, object], ...]
    # 控制器上一步决策（{"type": "tool_call"|"final"|"fallback", ...}），
    # 由 run_tool_node 消费后清空。
    pending_decision: dict[str, object]

    # --- P5-4: confirmation dialog (multi-turn) ---
    # --- P5-4：确认对话（多轮） ---
    # 用户提交的确认答复（question_id -> value），由 apply_confirmation
    # 校验并映射为 approved decisions。
    confirmation_answers: tuple[QuestionAnswer, ...]
    # 答复是否已成功应用（校验通过并产出新 request）。
    confirmation_applied: bool
    # 答复应用后的续接方向："parse_recipes" | "solve_schedule" |
    # "render_ready_response"。
    confirmation_route: str
    # 字段级校验错误（复用 ConfirmationAnswersError 的 issue 列表）。
    confirmation_error: tuple[str, ...]
