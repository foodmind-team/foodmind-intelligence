# =============================================================================
# 响应渲染器模块（rendering/responses）
# -----------------------------------------------------------------------------
# 把工作流状态转换为“终态”PlanResponse 对象（手册 11.6–11.10）。
# 每个渲染器都是纯函数：接收 PlanState，返回恰好一个 PlanResponse 子类型。
# validate_terminal_response 保证响应在离开图之前格式正确。
# 核心：
#   - render_ready_response        ：READY 响应（含时间线、备料、完成清单）
#   - render_confirmation_response ：NEEDS_CONFIRMATION 响应（结构化确认问题）
#   - render_infeasible_response   ：INFEASIBLE 响应（不可行原因）
#   - render_failed_response       ：FAILED 响应（从公共消息目录解析）
#   - validate_terminal_response   ：终态响应校验
# =============================================================================

"""Response renderers — convert workflow state into terminal PlanResponse objects.

响应渲染器 —— 把工作流状态转换为终态 PlanResponse 对象。

Handbook sections 11.6–11.10: each renderer is a pure function that takes
PlanState and returns exactly one PlanResponse subtype. validate_terminal_response
ensures the response is well-formed before it exits the graph.

手册 11.6–11.10：每个渲染器都是纯函数，接收 PlanState 并返回恰好一个
PlanResponse 子类型。validate_terminal_response 保证响应在离开图之前格式正确。
"""

from __future__ import annotations

from uuid import uuid4

from cooking_plan_agent.domain.errors import (
    DomainErrorCode,
    is_known_error_code,
    public_message_for,
)
from cooking_plan_agent.domain.models import (
    ApprovedDecision,
    Assumption,
    CompletionItem,
    ConfirmationPlanResponse,
    ConfirmationQuestion,
    FailedPlanResponse,
    InfeasiblePlanResponse,
    PlanResponse,
    QuestionOption,
    QuestionResponseType,
    ReadyPlanResponse,
    RepairOption,
)
from cooking_plan_agent.rendering.builder import (
    build_dish_completion_summary,
    build_execution_flow,
    build_mise_en_place,
    build_timeline,
)
from cooking_plan_agent.workflow.state import PlanState

# =============================================================================
# 11.6  READY response
# 11.6  READY 响应
# =============================================================================


def render_ready_response(state: PlanState) -> ReadyPlanResponse:
    """从工作流状态构建完整的 READY 响应。

    Build a complete READY response from workflow state.

    Aggregates: timeline from schedule + tasks, mise en place from
    prep tasks, dish completions, and completion checklist from
    the reservation proposal.

    聚合：来自 schedule + tasks 的时间线、来自 prep 任务的 mise en place、
    菜品完成、以及来自预留方案的完成清单。

    Args:
        state: Workflow state at the verify_schedule → READY transition.
            state：verify_schedule → READY 转换处的工作流状态。

    Returns:
        ReadyPlanResponse with full schedule details.
        含完整调度详情的 ReadyPlanResponse。
    """
    request = state["request"]
    schedule = state.get("schedule_result")

    # Collect all tasks from the task graph
    # 从任务图收集所有任务
    task_graph = state.get("task_graph")
    all_tasks = task_graph.tasks if task_graph else ()

    # Build timeline
    # 构建时间线
    timeline: tuple[dict[str, object], ...] = ()
    makespan = 0
    solver_status = "UNKNOWN"
    if schedule:
        solver_status = schedule.status.value
        makespan = schedule.makespan_minutes or 0
        timeline = build_timeline(schedule, all_tasks)

    # P0-05: when the caller supplied an absolute serving instant, attach
    # it to the timeline as display context (plan start derives from it).
    # This is a display aid only — scheduling itself uses integer minutes.
    # P0-05：当调用方提供了绝对开餐时刻时，把它附加到时间线作为展示上下文
    # （计划开始时间由此推导）。这仅是展示辅助 —— 调度本身用整数分钟。
    if timeline and request.serving_at is not None:
        serving_iso = request.serving_at.isoformat()
        timeline = tuple(
            {
                **entry,
                "serving_at": serving_iso,
                "offset_from_serving_minutes": -int(str(entry["end_minute"])),
            }
            if isinstance(entry, dict)
            else entry
            for entry in timeline
        )

    # Build mise en place from prep tasks
    # 从 prep 任务构建 mise en place
    prep_tasks = state.get("prep_tasks", ())
    safety_tasks = state.get("safety_tasks", ())
    recipe_tasks = state.get("recipe_tasks", ())
    combined = recipe_tasks + prep_tasks + safety_tasks
    mise_en_place = build_mise_en_place(combined) if combined else ()

    # Build dish completion summary
    # 构建菜品完成汇总
    dish_completions: tuple[dict[str, object], ...] = ()
    if schedule:
        dish_completions = build_dish_completion_summary(schedule, all_tasks)

    # Build completion checklist from feasibility report allocations.
    # Uses the FULL ingredient_results (satisfied + short) so a READY plan
    # always carries the inventory consumption plan — previously it was
    # empty because ingredient_shortages only retains short items.
    # 从可行性报告的分配构建完成清单。使用完整的 ingredient_results（满足 + 短缺），
    # 使 READY 计划始终携带库存消耗方案 —— 之前为空是因为 ingredient_shortages 只保留短缺项。
    completion_checklist: tuple[CompletionItem, ...] = ()
    feasibility = state.get("feasibility_report")
    if feasibility and (feasibility.ingredient_results or feasibility.ingredient_shortages):
        from cooking_plan_agent.inventory.feasibility import build_reservation_proposal

        proposal = build_reservation_proposal(feasibility)
        completion_checklist = proposal.items

    return ReadyPlanResponse(
        plan_id=request.request_id,
        status="READY",
        solver_status=solver_status,
        makespan_minutes=makespan,
        timeline=timeline,
        execution_flow=build_execution_flow(all_tasks),
        completion_checklist=completion_checklist,
        mise_en_place=mise_en_place,
        dish_completions=dish_completions,
        # P3-04: regional safety-policy provenance (region/version/sources).
        # P3-04：区域安全策略溯源（地区 / 版本 / 来源）
        safety_policy=state.get("safety_policy"),
        # P4-01: additive schedule explanation (llm | deterministic | disabled).
        # P4-01：附加调度解释（llm | deterministic | disabled）
        explanation=state.get("explanation"),
        explanation_source=state.get("explanation_source"),
    )


# =============================================================================
# 11.7  CONFIRMATION response
# 11.7  CONFIRMATION 响应
# =============================================================================


def _stable_question_key(*parts: str) -> str:
    """从领域字段派生稳定、有界的 question 键（P4-02 D6）。

    Derive a stable, bounded question key from domain fields (P4-02 D6).

    SHA-256 of the joined stable keys so the same input always reproduces
    the same question_id, and the ID never depends on array position or on
    random identifiers (e.g. the random suffix inside a gap_id).

    对拼接的稳定键做 SHA-256，使相同输入始终重现相同 question_id，且 ID 绝不依赖
    数组位置或随机标识（如 gap_id 内的随机后缀）。
    """
    import hashlib

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _gap_question_field_path(recipe_id: str, field_path: str) -> str:
    """返回同时标识菜谱与字段的自包含路径。"""
    if field_path.startswith("recipe."):
        return field_path
    return f"recipe.{recipe_id}.{field_path}"


def _servings_option_label(payload: dict[str, object]) -> str:
    servings = payload.get("servings")
    if servings is None:
        return "Reduce servings"
    count = int(str(servings))
    noun = "serving" if count == 1 else "servings"
    return f"Reduce to {count} {noun}"


def _append_repair_strategy_question(
    questions: list[ConfirmationQuestion],
    decisions: tuple[ApprovedDecision, ...],
) -> set[str]:
    """把 purchase + reduce_servings 折叠为一个库存策略选择。

    Collapse purchase + reduce_servings into one inventory strategy choice.

    Creates a strategy question whenever a purchase or reduce_servings
    decision exists. Individual per-item repair
    questions are suppressed when a strategy question is emitted so the
    user never sees a long list of Apply/Do-not-apply items.

    当存在 purchase 或 reduce_servings 决策时创建一个策略问题。发出策略问题时
    抑制单个逐项修复问题，使用户绝不看到一长串 Apply / Do-not-apply 项。
    """
    purchase = next((decision for decision in decisions if decision.option_type == "purchase"), None)
    reduce = next((decision for decision in decisions if decision.option_type == "reduce_servings"), None)
    if purchase is None and reduce is None:
        return set()

    options: list[QuestionOption] = []
    suggested_value: str | None = None
    if reduce is not None:
        options.append(
            QuestionOption(value=reduce.option_id, label=_servings_option_label(reduce.payload), suggested=True)
        )
        suggested_value = reduce.option_id
    if purchase is not None:
        options.append(
            QuestionOption(value=purchase.option_id, label="Buy missing ingredients", suggested=reduce is None)
        )
        if suggested_value is None:
            suggested_value = purchase.option_id
    # Strategy question is emitted whenever a plan-level repair exists —
    # even a single one (e.g. only purchase) — so the user always decides
    # at plan level and never sees per-item Apply/Do-not-apply questions.
    # 只要存在计划级修复就发出策略问题 —— 即使只有一个（如仅 purchase）——
    # 使用户始终在计划级决策，绝不看到逐项 Apply / Do-not-apply 问题。
    prompt = "Some ingredients are missing. Choose how to continue."
    questions.append(
        ConfirmationQuestion(
            question_id="repair:strategy",
            field_path="repair_strategy",
            prompt=prompt,
            response_type=QuestionResponseType.CHOICE,
            options=tuple(options),
            required=True,
            suggested_value=suggested_value,
        )
    )
    grouped_ids: set[str] = set()
    if purchase is not None:
        grouped_ids.add(purchase.option_id)
    if reduce is not None:
        grouped_ids.add(reduce.option_id)
    return grouped_ids


def _build_confirmation_questions(
    state: PlanState,
    repair_options: tuple[RepairOption, ...],
    decisions: tuple[ApprovedDecision, ...],
) -> tuple[ConfirmationQuestion, ...]:
    """构建字段级结构化确认表单（P4-02）。

    Build the field-level structured confirmation form (P4-02).

    Question sources, in order:
      1. Blocking gaps — every unresolved critical / safety_critical gap
         becomes exactly one required TEXT question asking for the missing
         value. question_id is derived from stable domain keys
         (recipe_id + field_path), never from array position (D6).
      2. Repair options — each supported RepairOption becomes one CHOICE
         question (apply / skip); the apply option's value is the presented
         decision's option_id so the answer maps back to the decision
         verbatim (D9).

    问题来源（按顺序）：
      1. 阻塞缺口 —— 每个未解决的关键 / 安全关键缺口变成恰好一个必答 TEXT 问题，
         询问缺失值。question_id 从稳定领域键（recipe_id + field_path）派生，
         绝不来自数组位置（D6）。
      2. 修复选项 —— 每个支持的 RepairOption 变成一个 CHOICE 问题（apply / skip）；
         apply 选项的 value 是所呈现决策的 option_id，使答案原样映射回决策（D9）。

    Legacy ``questions`` strings are derived from these structured
    questions (dual-emit for old clients).

    旧 ``questions`` 字符串由这些结构化问题派生（为旧客户端双发）。
    """
    questions: list[ConfirmationQuestion] = []

    # Backend-preprocessed requests normally arrive complete. They may still
    # contain a gap when the LLM and deterministic fallback could not infer a
    # reasonable value, so actual blocking gaps must never be hidden merely
    # because a preprocess call happened.
    # 后端预处理的请求通常完整。当 LLM 与确定性兜底都无法推断合理值时，仍可能含缺口，
    # 因此绝不能仅因发生过预处理调用就隐藏真实的阻塞缺口。
    # 1. Blocking gaps → one required TEXT question per gap (one-to-one).
    # 1. 阻塞缺口 → 每个缺口一个必答 TEXT 问题（一对一）
    for gap in state.get("gaps", ()):
        if gap.gap_class not in ("critical", "safety_critical"):
            continue
        questions.append(
            ConfirmationQuestion(
                question_id=f"gap:{_stable_question_key(gap.recipe_id, gap.field_path)}",
                field_path=_gap_question_field_path(gap.recipe_id, gap.field_path),
                prompt=(
                    f"The {gap.field_path} for recipe '{gap.recipe_id}' is missing "
                    f"({gap.description}). Please provide the correct value."
                ),
                response_type=QuestionResponseType.TEXT,
                required=True,
                suggested_value=gap.current_value,
            )
        )

    # Assumptions remain in the response as audit provenance, but are
    # accepted automatically. Operational model/rule defaults are not user
    # decisions; only unresolved blocking facts and repair strategy are.
    # 假设作为审计溯源保留在响应中，但自动接受。操作模型 / 规则默认值不是用户决策；
    # 只有未解决的阻塞事实与修复策略才是。

    # 2. Repair options → strategy question when any plan-level repair
    # (purchase / reduce_servings / substitute) exists; otherwise one CHOICE
    # question per supported decision. When a strategy question is emitted,
    # per-item repair questions are never shown — the user decides at plan
    # level only.
    # 2. 修复选项 → 当存在计划级修复（purchase / reduce_servings / substitute）时
    # 发策略问题；否则每个支持的决策一个 CHOICE 问题。发出策略问题时，
    # 绝不显示逐项修复问题 —— 用户只在计划级决策。
    label_by_option_id = {option.option_id: option.description for option in repair_options}
    grouped_decision_ids = _append_repair_strategy_question(questions, decisions)
    if not grouped_decision_ids:
        for decision in decisions:
            label = label_by_option_id.get(decision.option_id, decision.option_type)
            questions.append(
                ConfirmationQuestion(
                    question_id=f"repair:{decision.option_id}",
                    field_path="repair_options",
                    prompt=f"Apply the repair option '{label}'?",
                    response_type=QuestionResponseType.CHOICE,
                    options=(
                        QuestionOption(value=decision.option_id, label="Apply", suggested=True),
                        QuestionOption(value="__skip__", label="Do not apply"),
                    ),
                    required=False,
                    suggested_value=decision.option_id,
                )
            )

    return tuple(questions)


def render_confirmation_response(state: PlanState) -> ConfirmationPlanResponse:
    """构建带假设与修复选项的 NEEDS_CONFIRMATION 响应。

    Build a NEEDS_CONFIRMATION response with assumptions and repair options.

    Sources assumptions from parsed recipes (inferred fields) and repair
    options from the feasibility check. Generates user-facing questions.

    从解析菜谱（推断字段）收集假设，从可行性检查收集修复选项。生成面向用户的问题。

    P0-06: also emits structured, client-submittable ApprovedDecisions so
    the client can resubmit them verbatim instead of opaque string IDs.

    P0-06：也发出结构化、可提交的 ApprovedDecisions，使客户端能原样重提，
    而非不透明的字符串 ID。

    P4-02: also emits ``confirmation_questions`` — a field-level structured
    form the client renders and answers directly. The legacy ``questions``
    strings are derived from it (dual-emit, deprecated since P4-02).

    P4-02：也发出 confirmation_questions —— 客户端直接渲染并作答的字段级结构化表单。
    旧 questions 字符串由它派生（双发，自 P4-02 起弃用）。

    Args:
        state: Workflow state at any CONFIRMATION transition.
            state：任一 CONFIRMATION 转换处的工作流状态。

    Returns:
        ConfirmationPlanResponse with assumptions, options, decisions,
        structured confirmation_questions, and legacy questions.
        含假设、选项、决策、结构化 confirmation_questions 与旧 questions 的 ConfirmationPlanResponse。
    """
    request = state["request"]

    # Collect assumptions from parsed recipes
    # 从解析菜谱收集假设
    parsed = state.get("parsed_recipes", ())
    assumptions: list[Assumption] = []
    for recipe in parsed:
        assumptions.extend(recipe.assumptions)

    # P1-01: research evidence that was applied but still warrants
    # confirmation (e.g. conflict over threshold) is surfaced here with its
    # EvidenceRef provenance — never silently dropped.
    # P1-01：已应用但仍需确认的研究证据（如超过阈值的冲突）在此处连同其
    # EvidenceRef 溯源一起呈现 —— 绝不静默丢弃。
    assumptions.extend(state.get("research_assumptions", ()))

    # Collect repair options
    # 收集修复选项
    repair_options = state.get("repair_options", ())

    # P0-06: plan revision — the confirmation the client will answer.
    # P0-06：计划修订 —— 客户端将应答的确认
    plan_revision = f"{request.request_id}:v1"

    from cooking_plan_agent.repair.options import build_approved_decisions

    decisions = build_approved_decisions(repair_options, plan_revision)

    # P4-02: field-level structured form + legacy plain-string dual-emit.
    # P4-02：字段级结构化表单 + 旧纯字符串双发
    confirmation_questions = _build_confirmation_questions(state, repair_options, decisions)
    if confirmation_questions:
        questions = tuple(f"{q.prompt} ({q.question_id})" for q in confirmation_questions)
    else:
        # No field-level question is meaningful — keep the legacy fallback.
        # 无有意义的字段级问题 —— 保留旧兜底
        questions = ("Would you like to proceed with these options?",)

    return ConfirmationPlanResponse(
        plan_id=request.request_id,
        status="NEEDS_CONFIRMATION",
        assumptions=tuple(assumptions),
        repair_options=repair_options,
        questions=questions,
        confirmation_questions=confirmation_questions,
        decisions=decisions,
        plan_revision=plan_revision,
        # P3-04: regional safety-policy provenance (region/version/sources).
        # P3-04：区域安全策略溯源（地区 / 版本 / 来源）
        safety_policy=state.get("safety_policy"),
    )


# =============================================================================
# 11.8  INFEASIBLE response
# 11.8  INFEASIBLE 响应
# =============================================================================


def render_infeasible_response(state: PlanState) -> InfeasiblePlanResponse:
    """构建带安全 / 可行性 / 错误原因的 INFEASIBLE 响应。

    Build an INFEASIBLE response with reasons from safety/feasibility/errors.

    Merges reasons from all sources, ordered by severity:
      safety → feasibility → scheduling → generic.

    合并所有来源的原因，按严重程度排序：安全 → 可行性 → 调度 → 通用。

    Args:
        state: Workflow state at any INFEASIBLE transition.
            state：任一 INFEASIBLE 转换处的工作流状态。

    Returns:
        InfeasiblePlanResponse with ordered reasons.
        含排序原因的 InfeasiblePlanResponse。
    """
    request = state["request"]
    error = state.get("error")
    safety = state.get("safety_report")
    feasibility = state.get("feasibility_report")
    schedule = state.get("schedule_result")

    reasons: list[str] = []

    # Safety reasons (strongest blockers)
    # 安全原因（最强阻塞）
    if safety is not None and not safety.is_safe:
        reasons.extend(f.description for f in safety.findings)

    # Feasibility reasons
    # 可行性原因
    if feasibility is not None and not feasibility.is_feasible:
        for shortage in feasibility.ingredient_shortages:
            reasons.append(
                f"Insufficient '{shortage.ingredient_name}': "
                f"need {shortage.required} {shortage.unit}, "
                f"have {shortage.available} {shortage.unit}"
            )
        for res in feasibility.missing_resources:
            reasons.append(f"Missing equipment: {res}")

    # Scheduling reasons
    # 调度原因
    if schedule is not None:
        status = schedule.status.value
        if status == "INFEASIBLE":
            reasons.append("The scheduler proved no feasible timeline exists with current constraints.")
        elif status == "MODEL_INVALID":
            reasons.append("The scheduling model is invalid — likely a data inconsistency.")

    # Fallback reason
    # 兜底原因
    if not reasons:
        if error:
            reasons.append(error.message)
        else:
            reasons.append("The plan cannot be fulfilled with current constraints.")

    return InfeasiblePlanResponse(
        plan_id=request.request_id if request else "unknown",
        status="INFEASIBLE",
        reasons=tuple(reasons),
        safe_alternatives=(),
    )


# =============================================================================
# 11.9  FAILED response
# 11.9  FAILED 响应
# =============================================================================


def render_failed_response(state: PlanState) -> FailedPlanResponse:
    """从公共消息目录构建 FAILED 响应（P2-03）。

    Build a FAILED response from the public message catalog (P2-03).

    Client-facing text always comes from the catalog — never from the node's
    internal ``message``. An error code without a catalog row fails closed to
    INTERNAL_ERROR instead of echoing raw exception text.

    面向客户端的文案始终来自目录 —— 绝不来自节点内部的 message。没有目录行的错误码
    失败即关闭到 INTERNAL_ERROR，而非回显原始异常文本。

    Graceful fallback: if no error is in state, returns INTERNAL_ERROR.

    优雅兜底：若状态中无错误，返回 INTERNAL_ERROR。

    Args:
        state: Workflow state at any FAILED transition.
            state：任一 FAILED 转换处的工作流状态。

    Returns:
        FailedPlanResponse with stable error code, public message and
        correlation ID for support traceability.
        含稳定错误码、公共消息与 correlation ID（用于支持追溯）的 FailedPlanResponse。
    """
    request = state["request"]
    error = state.get("error")

    if error is None:
        correlation_id = request.request_id if request else str(uuid4().hex[:8])
        return FailedPlanResponse(
            status="FAILED",
            error_code=DomainErrorCode.INTERNAL_ERROR.value,
            correlation_id=correlation_id,
            message=public_message_for(DomainErrorCode.INTERNAL_ERROR.value),
        )

    # Missing catalog row is an invariant break — fail closed to
    # INTERNAL_ERROR (P2-03), never echo the node's raw message.
    # 缺少目录行是不变量破坏 —— 失败即关闭到 INTERNAL_ERROR（P2-03），
    # 绝不回显节点的原始消息。
    error_code = error.error_code if is_known_error_code(error.error_code) else DomainErrorCode.INTERNAL_ERROR.value
    message = error.public_message or public_message_for(error_code)

    return FailedPlanResponse(
        status="FAILED",
        error_code=error_code,
        correlation_id=error.correlation_id,
        message=message,
    )


# =============================================================================
# 11.10  Terminal response validation
# 11.10  终态响应校验
# =============================================================================


def validate_terminal_response(response: PlanResponse) -> PlanResponse:
    """在离开图之前校验终态响应格式正确。

    Validate that a terminal response is well-formed before graph exit.

    Checks:
      - READY: makespan > 0, solver_status is non-empty
      - CONFIRMATION: has at least one assumption or repair_option or question
      - INFEASIBLE: has at least one reason
      - FAILED: error_code and correlation_id are non-empty

    检查：
      - READY：makespan > 0，solver_status 非空
      - CONFIRMATION：至少有一个假设或修复选项或问题
      - INFEASIBLE：至少有一个原因
      - FAILED：error_code 与 correlation_id 非空

    Returns the response unchanged on success. Raises ValueError on
    validation failure (which the caller should catch and map to FAILED).

    成功时原样返回响应。校验失败抛 ValueError（调用方应捕获并映射到 FAILED）。

    Args:
        response: Any PlanResponse subtype.
            response：任意 PlanResponse 子类型。

    Returns:
        The same response object if valid.
        若有效则返回同一响应对象。

    Raises:
        ValueError: If the response fails validation.
        ValueError：响应校验失败时抛出。
    """
    status = getattr(response, "status", None)

    if status == "READY":
        if isinstance(response, ReadyPlanResponse):
            if not response.solver_status:
                raise ValueError("READY response: solver_status is empty")
            if response.makespan_minutes <= 0:
                raise ValueError(f"READY response: makespan must be > 0, got {response.makespan_minutes}")

    elif status == "NEEDS_CONFIRMATION":
        if isinstance(response, ConfirmationPlanResponse):
            if not response.confirmation_questions:
                raise ValueError("CONFIRMATION response: must contain at least one actionable structured question")

    elif status == "INFEASIBLE":
        if isinstance(response, InfeasiblePlanResponse):
            if not response.reasons:
                raise ValueError("INFEASIBLE response: must have at least one reason")

    elif status == "FAILED":
        if isinstance(response, FailedPlanResponse):
            if not response.error_code.strip():
                raise ValueError("FAILED response: error_code is empty")
            if not response.correlation_id.strip():
                raise ValueError("FAILED response: correlation_id is empty")

    else:
        raise ValueError(f"Unknown response status: {status!r}")

    return response
