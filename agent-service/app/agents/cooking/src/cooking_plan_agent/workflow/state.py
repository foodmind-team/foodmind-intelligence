"""LangGraph workflow state — TypedDict per handbook 8.2.

Uses TypedDict for workflow state (serialisable) and Pydantic models
for values at boundaries. Provider clients and secrets are NOT stored here.
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

    Only serialisable domain objects. No provider clients, request-scoped
    secrets, or OR-Tools model objects (handbook 8.2).

    total=False means every field is Optional — nodes only return the
    CHANGED subset, and downstream nodes use .get() to safely read.
    """

    # --- Input (populated once, never mutated) ---
    request: GeneratePlanRequest

    # --- Parsing & inference ---
    # Candidates are raw extraction output; RecipeIR is the validated form
    extracted_candidates: tuple[ExtractedRecipeCandidate, ...]
    gaps: tuple[RecipeGap, ...]
    # Evidence stored as plain dicts to keep state serialisable (no dataclass coupling)
    evidence: tuple[dict[str, object], ...]
    # Per-gap reconciled research results (keyed by gap_id for traceability)
    research_evidence: dict[str, ReconciledEvidence]
    # Evidence-backed assumptions produced by apply_research_evidence_node
    # (P1-01). Merged into RecipeIR.assumptions so provenance is traceable.
    research_assumptions: tuple[Assumption, ...]

    # --- Validation ---
    parsed_recipes: tuple[RecipeIR, ...]
    safety_report: SafetyReport
    # P3-04: resolved regional policy provenance, recorded on terminal
    # responses and retained in state so historical plans stay auditable.
    safety_policy: SafetyPolicyRecord
    feasibility_report: FeasibilityReport
    # Repair options injected by check_feasibility when infeasible but fixable
    repair_options: tuple[RepairOption, ...]

    # --- Preparation & scheduling ---
    # Tasks are split into three categories so merge_preparation can
    # apply different dedup/priority rules per category
    recipe_tasks: tuple[CookingTask, ...]
    prep_tasks: tuple[CookingTask, ...]
    safety_tasks: tuple[CookingTask, ...]
    # P2-01: human-readable summaries of shared-prep merge/branch/isolate
    # decisions, for observability and regression tests.
    prep_observations: tuple[str, ...]
    task_graph: TaskGraph
    schedule_result: ScheduleResult
    # Independent verification report (separate from solver's internal check)
    verification_report: VerificationReport
    # P4-01: additive schedule explanation set by explain_schedule_node
    # between verify and render READY. Display-only — never affects the
    # schedule. explanation_source ∈ {"llm", "deterministic", "disabled"}.
    explanation: str | None
    explanation_source: str | None

    # --- Terminal output ---
    # Exactly ONE terminal field is populated by the terminal node that fires
    response: PlanResponse
    needs_confirmation: bool
    confirmation_context: ConfirmationPlanResponse

    # --- Error ---
    # Set by any node that encounters a recoverable/terminal error;
    # the graph uses this to route to error terminal nodes
    error: WorkflowError

    # --- P5: agent trace (shared by all agentic phases) ---
    # 每步 agent 决策（repair / tool_call / question）的留痕，保持可序列化。
    # 元素为 plain dict（{"step": int, "action": str, "detail": dict}）。
    agent_trace: tuple[dict[str, object], ...]

    # --- P5-3: schedule repair loop ---
    # 已验证失败的修复尝试计数；超过 max_attempts 则 FAILED（保证终止）。
    repair_attempts: int
    repair_history: tuple[RepairAttemptRecord, ...]
    # 请求级求解覆盖（如 {"optimization_level": "phase12"}）——不污染全局 Settings。
    solver_overrides: dict[str, object]

    # --- P5-2: ReAct controller ---
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
