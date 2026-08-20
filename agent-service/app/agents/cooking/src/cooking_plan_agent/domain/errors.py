"""

Domain error types for the Cooking Plan Agent.

This module defines all stable, traceable error codes and the base exception
class for the domain/application layer—the first layer of the three-level
error boundary (handbook 8.10). Every domain service exposes errors through
the types defined here, ensuring consistent semantics across service and
module boundaries so downstream consumers (LangGraph workflow nodes, FastAPI
exception handlers) can route them deterministically.

Core responsibilities:
  1. DomainErrorCode — enumerates every known domain error scenario. Each code
     maps to a specific, auditable business failure reason.
  2. WorkflowException — base domain exception carrying an error code and a
     human-readable description. Raised by domain services, caught by workflow
     nodes, and routed to the corresponding terminal response.

Author: cooking-plan-agent team
Created: 2026-07
"""

from enum import StrEnum

# =============================================================================
# 领域错误类型模块（domain/errors）
# -----------------------------------------------------------------------------
# 本文件定义烹饪计划 Agent 的「领域错误契约」，是三层错误边界（handbook 8.10）
# 中的第一层（领域/应用层）。核心由四部分组成：
#   1. DomainErrorCode  —— 稳定、可追溯的领域错误码枚举（StrEnum）
#   2. WorkflowException —— 携带错误码 + 人类可读描述的领域基础异常
#   3. 可重试语义目录   —— 客户端能否重试的单一权威来源（与文案解耦，D9）
#   4. 公共消息目录     —— 面向客户端的安全、脱敏文案（P2-03）
# 下游（LangGraph workflow 节点、FastAPI 异常处理器）依赖这些类型做确定性路由。
# =============================================================================

# ===========================================================================
# DomainErrorCode — stable domain error code enumeration (handbook 3.8)
# 领域错误码枚举 —— 稳定、可追溯（handbook 3.8）
# ===========================================================================


class DomainErrorCode(StrEnum):
    """

    Stable identifiers for every known domain-level failure scenario.

    Design principles (handbook 3.8):
      - Error codes are stable; their semantics must not change across
        refactors.
      - Each code represents a specific business failure reason, never a
        technical implementation detail.
      - Workflow nodes and the API layer route decisions based on these codes.
      - No catch-all "uncategorized" or "other" codes are defined; every error
        must have a clear home.

    """

    # ------------------------------------------------------------------
    # Recipe input errors (handbook 4.2, 4.6)
    # 菜谱输入错误（handbook 4.2, 4.6）
    # ------------------------------------------------------------------

    # The input text cannot form a usable recipe: empty content, pure binary,
    # oversized file, or preprocessing yielded no recognizable
    # ingredient/step information.
    # 输入文本无法构成可用菜谱：空内容、纯二进制、文件过大，或预处理后
    # 未能识别出任何食材/步骤信息。
    INVALID_RECIPE_TEXT = "INVALID_RECIPE_TEXT"

    # ------------------------------------------------------------------
    # Request-level input validation errors (P0-03)
    # 请求级输入校验错误（P0-03）
    # ------------------------------------------------------------------

    # Two or more recipes in the request share the same recipe_id.  Recipe
    # identity must be unique so ingredient demands and task graphs can be
    # attributed unambiguously.
    # 请求中有两个及以上菜谱共享同一 recipe_id。菜谱身份必须唯一，
    # 以便食材需求与任务图能被无歧义地归属。
    DUPLICATE_RECIPE_ID = "DUPLICATE_RECIPE_ID"

    # The request exceeds the maximum number of recipes the service is
    # configured to accept (Settings.max_recipe_count).
    # 请求超出服务配置允许的最大菜谱数（Settings.max_recipe_count）。
    TOO_MANY_RECIPES = "TOO_MANY_RECIPES"

    # A recipe's raw text exceeds the configured byte limit
    # (Settings.max_recipe_text_bytes).  Rejected to bound memory and
    # extraction cost.
    # 某菜谱原始文本超出配置的字节上限（Settings.max_recipe_text_bytes）。
    # 拒绝以约束内存与提取成本。
    RECIPE_TEXT_TOO_LARGE = "RECIPE_TEXT_TOO_LARGE"

    # The request as a whole exceeds the configured size limit
    # (Settings.max_request_bytes).  Rejected at the workflow boundary.
    # 整个请求超出配置的大小上限（Settings.max_request_bytes）。
    # 在 workflow 边界被拒绝。
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"

    # The request's schema_version is not in the supported set
    # (Settings.supported_schema_versions).  The caller must upgrade.
    # 请求的 schema_version 不在支持集合内（Settings.supported_schema_versions）。
    # 调用方必须升级。
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"

    # A time-related request field is invalid (e.g. negative
    # time_limit_minutes).
    # 与时间相关的请求字段非法（例如 time_limit_minutes 为负数）。
    INVALID_TIME_LIMIT = "INVALID_TIME_LIMIT"

    # The serving time is malformed or ambiguous: not a valid HH:MM string,
    # or a serving_at instant without a timezone offset (P0-05). The client
    # must resubmit a well-formed time expression.
    # 上菜时间格式错误或存在歧义：不是合法的 HH:MM 字符串，或 serving_at
    # 时刻缺少时区偏移（P0-05）。客户端必须重新提交格式正确的时间表达式。
    INVALID_SERVING_TIME = "INVALID_SERVING_TIME"

    # An approved decision in the request is invalid: unsupported type,
    # conflicting combination, unknown/stale plan_revision, or malformed
    # payload (P0-06).
    # 请求中的审批决策非法：不支持的类型、冲突的组合、未知/过期的
    # plan_revision，或负载格式错误（P0-06）。
    INVALID_APPROVED_DECISION = "INVALID_APPROVED_DECISION"

    # The request body failed Pydantic schema validation at the HTTP
    # boundary (P3-05). Used by the RequestValidationError handler; never
    # raised from domain services.
    # 请求体在 HTTP 边界未能通过 Pydantic 模式校验（P3-05）。由
    # RequestValidationError 处理器使用；领域服务永不抛出此码。
    REQUEST_VALIDATION_ERROR = "REQUEST_VALIDATION_ERROR"

    # ------------------------------------------------------------------
    # Unit conversion errors (handbook 5.3, 5.4)
    # 单位转换错误（handbook 5.3, 5.4）
    # ------------------------------------------------------------------

    # A requested unit conversion lacks required product-specific data.
    # Example: converting "1 onion" to grams without density or average-weight
    # data for onions. The system should request user confirmation rather than
    # applying an unreliable default.
    # 请求的单位转换缺少所需的产品专属数据。例如：在没有洋葱密度或平均重量
    # 数据的情况下把「1 个洋葱」转换为克。系统应请求用户确认，而非套用
    # 不可靠的默认值。
    UNSUPPORTED_UNIT_CONVERSION = "UNSUPPORTED_UNIT_CONVERSION"

    # ------------------------------------------------------------------
    # Safety constraint errors (handbook 5.7, 5.8, 5.9)
    # 安全约束错误（handbook 5.7, 5.8, 5.9）
    # ------------------------------------------------------------------

    # A hard safety rule was triggered and cannot be repaired automatically.
    # Typical scenarios: cross-contamination risk (raw meat sharing a board
    # with ready-to-eat ingredients and no sanitisation task can be inserted),
    # allergen matches, or missing safe-cooking endpoint temperatures that
    # cannot be filled from a trusted source.
    # This error typically produces an INFEASIBLE terminal response.
    # 触发了硬性安全规则且无法自动修复。典型场景：交叉污染风险（生肉与即食
    # 食材共用砧板且无法插入消毒任务）、过敏原匹配，或无法从可信来源补齐的
    # 安全烹饪终点温度。此错误通常产生 INFEASIBLE 终态响应。
    SAFETY_CONSTRAINT_VIOLATION = "SAFETY_CONSTRAINT_VIOLATION"

    # The requested regional food-safety policy pack cannot be applied:
    # unknown region, unknown version, not yet effective, or missing official
    # sources (P3-04 D6). Never silently falls back to another region.
    # Routes to a FAILED response — a plan must not enter READY under an
    # unverifiable policy.
    # 请求的区域食品安全策略包无法应用：未知区域、未知版本、尚未生效，或
    # 缺少官方来源（P3-04 D6）。绝不静默回退到其它区域。路由到 FAILED 响应
    # —— 在策略不可验证的情况下，计划不得进入 READY。
    SAFETY_POLICY_UNAVAILABLE = "SAFETY_POLICY_UNAVAILABLE"

    # ------------------------------------------------------------------
    # Inventory and resource errors (handbook 5.5, 5.6)
    # 库存与资源错误（handbook 5.5, 5.6）
    # ------------------------------------------------------------------

    # Required consumable ingredient quantity is insufficient.
    # A shortage remains even after applying the FEFO (First-Expired-First-Out)
    # allocation strategy. The system should generate RepairOptions for the
    # user to select (reduce servings, substitute ingredients, etc.).
    # 所需消耗性食材数量不足。即便应用 FEFO（先过期先出）分配策略后仍短缺。
    # 系统应为用户生成 RepairOptions 供选择（减少份数、替换食材等）。
    INSUFFICIENT_INVENTORY = "INSUFFICIENT_INVENTORY"

    # No compatible kitchen equipment instance can perform a required task.
    # Example: the recipe requires oven baking but the user's kitchen has no
    # oven, or the only available oven's capacity is below the minimum.
    # Distinct from a scheduling resource conflict: this error occurs during
    # the feasibility check phase and means "does not exist at all" rather
    # than "temporarily occupied".
    # 没有兼容的厨房设备实例能执行所需任务。例如：菜谱需要烤箱烘焙但用户
    # 厨房没有烤箱，或唯一可用烤箱的容量低于最低要求。区别于调度资源冲突：
    # 此错误发生在可行性检查阶段，含义是「根本不存在」而非「暂时被占用」。
    NO_COMPATIBLE_RESOURCE = "NO_COMPATIBLE_RESOURCE"

    # ------------------------------------------------------------------
    # Task graph errors (handbook 6.8, 6.9)
    # 任务图错误（handbook 6.8, 6.9）
    # ------------------------------------------------------------------

    # A cycle was detected in the generated task dependency graph.
    # Typically discovered during topological sort (Kahn's algorithm) inside
    # build_task_graph(). A cyclic graph must never be passed to the CP-SAT
    # solver.
    # 在生成的任务依赖图中检测到环。通常在 build_task_graph() 内部的拓扑排序
    # （Kahn 算法）中发现。带环的图绝不能被传给 CP-SAT 求解器。
    TASK_GRAPH_CYCLE = "TASK_GRAPH_CYCLE"

    # ------------------------------------------------------------------
    # Scheduling solver errors (handbook 7.6, 7.7, 7.8)
    # 调度求解器错误（handbook 7.6, 7.7, 7.8）
    # ------------------------------------------------------------------

    # The CP-SAT solver proved that no feasible schedule exists under the
    # current constraints. Typical causes: excessively tight time windows,
    # insufficient resource capacity, or conflicting task lag constraints.
    # May lead to an INFEASIBLE response or trigger repair options such as
    # time relaxation or recipe replacement.
    # CP-SAT 求解器证明：在当前约束下不存在可行调度。典型原因：时间窗口过紧、
    # 资源容量不足，或任务间隔约束相互冲突。可能导致 INFEASIBLE 响应，或触发
    # 修复选项（如放宽时间、替换菜谱）。
    SCHEDULE_INFEASIBLE = "SCHEDULE_INFEASIBLE"

    # The solver could not determine feasibility or infeasibility before
    # timing out. Unlike INFEASIBLE: UNKNOWN means the solver "found no
    # answer" rather than "proved no solution exists". The schedule result is
    # unusable—return a FAILED response.
    # This is a direct mapping of OR-Tools CpSolverStatus.UNKNOWN.
    # 求解器在超时前既无法判定可行也无法判定不可行。与 INFEASIBLE 不同：
    # UNKNOWN 表示求解器「没找到答案」，而非「证明无解」。调度结果不可用
    # —— 返回 FAILED 响应。此码是 OR-Tools CpSolverStatus.UNKNOWN 的直接映射。
    SCHEDULE_UNKNOWN = "SCHEDULE_UNKNOWN"

    # The CP-SAT model itself is malformed: contradictory constraints, invalid
    # variables, or a scheduling problem shape the builder cannot express.
    # Distinct from SCHEDULE_INFEASIBLE — INFEASIBLE means the solver PROVED no
    # solution exists for a valid model; MODEL_INVALID means the model was
    # never valid (a construction bug). Both are independent of the solver's
    # own status enum but map to a FAILED response (P1-04).
    # CP-SAT 模型本身构建错误：约束矛盾、变量非法，或调度问题形态超出构建器
    # 表达能力。区别于 SCHEDULE_INFEASIBLE —— INFEASIBLE 表示求解器「证明」
    # 有效模型无解；MODEL_INVALID 表示模型「从未有效」（构建 bug）。两者都
    # 独立于求解器自身状态枚举，但都映射到 FAILED 响应（P1-04）。
    SCHEDULE_MODEL_INVALID = "SCHEDULE_MODEL_INVALID"

    # The independent verifier (ScheduleVerifier) rejected the solver's output.
    # This indicates the solver produced a result that appears valid but
    # violates constraints—a possible signal of a bug in CP-SAT model
    # construction or a numerical-precision issue. Rejected results must never
    # be returned to the user with READY status.
    # 独立校验器（ScheduleVerifier）拒绝了求解器的输出。这表明求解器产出了
    # 看似有效却违反约束的结果——可能是 CP-SAT 模型构建存在 bug 或数值精度
    # 问题。被拒绝的结果绝不允许以 READY 状态返回给用户。
    SCHEDULE_VERIFICATION_FAILED = "SCHEDULE_VERIFICATION_FAILED"

    # ------------------------------------------------------------------
    # External dependency errors (handbook 4.11, 10.4)
    # 外部依赖错误（handbook 4.11, 10.4）
    # ------------------------------------------------------------------

    # The LLM provider or web search service is unavailable after bounded
    # retries. Used when recipe parsing or gap research requires an external
    # call and all attempts have failed. Nodes should map this to a stable
    # FAILED response—never expose raw provider exceptions to the client.
    # LLM 供应商或 web 搜索服务在有界重试后仍不可用。用于菜谱解析或缺口调研
    # 需要外部调用且所有尝试均失败时。节点应将其映射为稳定的 FAILED 响应
    # —— 绝不向客户端暴露原始供应商异常。
    EXTERNAL_PROVIDER_UNAVAILABLE = "EXTERNAL_PROVIDER_UNAVAILABLE"

    # ------------------------------------------------------------------
    # System-level errors
    # 系统级错误
    # ------------------------------------------------------------------

    # P5-4: the confirmation dialog was not enabled / no checkpointer, so
    # there is no paused conversation to resume. A stable FAILED outcome,
    # never a silent re-run.
    # P5-4：确认对话未启用 / 没有 checkpointer，因此没有可恢复的暂停对话。
    # 稳定的 FAILED 结果，绝不静默重跑。
    CONFIRMATION_DIALOG_UNAVAILABLE = "CONFIRMATION_DIALOG_UNAVAILABLE"

    # An unexpected internal error that cannot be classified into any of the
    # above categories. Reserved for the FastAPI global exception handler as a
    # last resort; workflow nodes should prefer more specific error codes.
    # Must carry a correlation_id in the client response for investigation.
    # 无法归入上述任何类别的意外内部错误。保留给 FastAPI 全局异常处理器作
    # 最后兜底；workflow 节点应优先使用更具体的错误码。客户端响应必须携带
    # correlation_id 以便排查。
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ===========================================================================
# WorkflowException — base domain exception (handbook 8.10 layer 1)
# WorkflowException —— 基础领域异常（handbook 8.10 第一层）
# ===========================================================================


class WorkflowException(Exception):
    """

    Unified base class for all domain-level exceptions.

    Design intent (handbook 8.10 three-level error boundary):
      Layer 1 (this layer): domain/application services raise
        WorkflowException carrying a DomainErrorCode and a human-readable
        description.
      Layer 2: LangGraph workflow nodes catch WorkflowException and write a
        typed WorkflowError into the state, routing to the appropriate
        terminal response node.
      Layer 3: the FastAPI global exception handler catches unexpected
        exceptions, logs the correlation_id, and returns a generic
        INTERNAL_ERROR.

    Typical triggers:
      - Recipe preprocessing yields invalid content
      - Safety rule engine detects an unrepairable safety violation
      - CP-SAT solver returns infeasible or unknown status
      - Independent verifier rejects the solver output
      - External LLM / search provider is unavailable

    Usage conventions:
      - Always carry an explicit DomainErrorCode; never use a meaningless
        generic code.
      - The message should be a short, human-readable description—never
        include stack traces, provider prompts, or secrets.
      - Do not implement automatic retry or fallback logic in this base class;
        those responsibilities belong to the workflow node layer.

    """

    # Domain error code identifying the specific failure category
    # (a DomainErrorCode enum member).
    # 领域错误码，标识具体的失败类别（DomainErrorCode 枚举成员）。
    code: DomainErrorCode
    # Human-readable error description used in logs and terminal responses.
    # Must not contain provider prompts, API keys, or user private data.
    # 人类可读的错误描述，用于日志与终态响应。不得包含供应商 prompt、
    # API 密钥或用户隐私数据。
    message: str

    def __init__(self, code: DomainErrorCode, message: str) -> None:
        """

        Initialise a domain exception instance.

        Args:
            code: Domain error code; must be a member of DomainErrorCode,
                  indicating the business category of the failure.
            message: Human-readable error description string for logging and
                     terminal response rendering. Keep it short and free of
                     sensitive information.

        Initialisation behaviour:
          1. Store code and message as instance attributes for downstream
             consumers.
          2. Call the parent Exception constructor with a formatted error
             string in the pattern "[ERROR_CODE] message"
             (e.g. "[TASK_GRAPH_CYCLE] Cyclic dependency detected").
             This ensures that even when the exception is not explicitly
             caught, its string representation contains all critical
             diagnostic information.

        """
        self.code = code
        self.message = message
        # Build a standardised error string with the error code prefix for
        # easy log retrieval.
        # 构建带错误码前缀的标准化错误字符串，便于日志检索。
        super().__init__(f"[{code.value}] {message}")


# ===========================================================================
# Error catalog — retryable semantics (P3-05)
# 错误目录 —— 可重试语义（P3-05）
# ===========================================================================
# The catalog is the single source of truth for whether a client may retry
# after receiving a given error_code. It is deliberately decoupled from the
# human-readable message text (D9), so retry semantics stay stable and
# auditable. Unknown codes default to non-retryable (safe: fail loudly).
# 该目录是「客户端在收到某 error_code 后能否重试」的单一权威来源。它有意与
# 人类可读文案解耦（D9），使重试语义保持稳定、可审计。未知码默认不可重试
# （安全：响亮失败）。

# Codes that indicate a transient condition where a later retry may succeed.
# These are protocol-level (backpressure, shutdown) or provider-level
# (external LLM/search) failures, never business-logic rejections.
# 这些码表示「暂时性状态、稍后重试可能成功」。它们是协议级（背压、停机）或
# 供应商级（外部 LLM/搜索）故障，绝非业务逻辑拒绝。
_RETRYABLE_ERROR_CODES: frozenset[str] = frozenset(
    {
        DomainErrorCode.EXTERNAL_PROVIDER_UNAVAILABLE.value,
        DomainErrorCode.SCHEDULE_UNKNOWN.value,
        "OVERLOADED",  # P1-03 backpressure 503
        "SHUTTING_DOWN",  # graceful shutdown 503
        "SCHEDULE_MODEL_INVALID",  # transient model construction fault
    }
)


def is_retryable(error_code: str) -> bool:
    """Return whether a client may retry for the given error code.

    The decision comes exclusively from the error catalog — never from
    message-text heuristics (P3-05 D9). Unknown codes are non-retryable.
    """
    # 返回客户端对给定错误码是否可重试。
    # 判定完全来自错误目录——绝不依赖消息文本启发式（P3-05 D9）。
    # 未知码不可重试。
    return error_code in _RETRYABLE_ERROR_CODES


def retryable_error_codes() -> tuple[str, ...]:
    """Return the sorted set of catalogued retryable codes (audit/report)."""
    # 返回已登记可重试码的排序集合（用于审计/报告）。
    return tuple(sorted(_RETRYABLE_ERROR_CODES))


# ===========================================================================
# Public message catalog (P2-03) — stable, sanitised client-facing text
# 公共消息目录（P2-03）—— 稳定、脱敏的面向客户端文案
# ===========================================================================
# Single source of truth for FAILED response messages. Every registered
# error code has one stable public message that is free of secrets, provider
# payloads, recipe text and raw exception details. Nodes never build their
# own client-facing strings; the renderers resolve them here. Unknown codes
# fail closed to INTERNAL_ERROR instead of echoing raw message text.
#
# Retry semantics deliberately live in _RETRYABLE_ERROR_CODES (P3-05) — this
# catalog is content-only to keep a single source of truth for retryability.
# FAILED 响应文案的单一权威来源。每个已登记错误码都有一条稳定的公共文案，
# 不含密钥、供应商负载、菜谱原文及原始异常细节。节点绝不自行拼装面向客户端的
# 字符串，统一由渲染器在此解析。未知码 fail closed 到 INTERNAL_ERROR，
# 而不是回显原始消息文本。
#
# 可重试语义有意放在 _RETRYABLE_ERROR_CODES（P3-05）中——本目录仅负责内容，
# 以保持「可重试性」的单一权威来源。

_PUBLIC_MESSAGES: dict[str, str] = {
    DomainErrorCode.INVALID_RECIPE_TEXT.value: ("The recipe text could not be parsed into a usable recipe."),
    DomainErrorCode.DUPLICATE_RECIPE_ID.value: ("The request contains duplicate recipe identifiers."),
    DomainErrorCode.TOO_MANY_RECIPES.value: ("The request contains more recipes than the service allows."),
    DomainErrorCode.RECIPE_TEXT_TOO_LARGE.value: ("One or more recipe texts exceed the allowed size limit."),
    DomainErrorCode.REQUEST_TOO_LARGE.value: ("The request exceeds the allowed total size."),
    DomainErrorCode.UNSUPPORTED_SCHEMA_VERSION.value: ("The request schema version is not supported."),
    DomainErrorCode.INVALID_TIME_LIMIT.value: "The time limit is invalid.",
    DomainErrorCode.INVALID_SERVING_TIME.value: "The serving time is invalid.",
    DomainErrorCode.INVALID_APPROVED_DECISION.value: ("One or more approved decisions are invalid or conflicting."),
    DomainErrorCode.REQUEST_VALIDATION_ERROR.value: ("The request failed validation."),
    DomainErrorCode.UNSUPPORTED_UNIT_CONVERSION.value: ("A required unit conversion is not supported."),
    DomainErrorCode.SAFETY_CONSTRAINT_VIOLATION.value: ("A food-safety constraint cannot be satisfied."),
    DomainErrorCode.SAFETY_POLICY_UNAVAILABLE.value: (
        "The food-safety policy for the requested region is unavailable."
    ),
    DomainErrorCode.INSUFFICIENT_INVENTORY.value: ("There is not enough inventory to fulfil the plan."),
    DomainErrorCode.NO_COMPATIBLE_RESOURCE.value: ("No compatible kitchen resource is available."),
    DomainErrorCode.TASK_GRAPH_CYCLE.value: ("The task dependency graph is invalid."),
    DomainErrorCode.SCHEDULE_INFEASIBLE.value: ("No feasible schedule exists under the current constraints."),
    DomainErrorCode.SCHEDULE_UNKNOWN.value: (
        "The scheduler could not determine a feasible schedule within the time limit."
    ),
    DomainErrorCode.SCHEDULE_MODEL_INVALID.value: ("The scheduling model is invalid."),
    DomainErrorCode.SCHEDULE_VERIFICATION_FAILED.value: ("The generated schedule failed verification."),
    DomainErrorCode.EXTERNAL_PROVIDER_UNAVAILABLE.value: ("An external service is temporarily unavailable."),
    DomainErrorCode.CONFIRMATION_DIALOG_UNAVAILABLE.value: (
        "The confirmation dialog is not available for this request."
    ),
    DomainErrorCode.INTERNAL_ERROR.value: "An unexpected internal error occurred.",
}


def public_message_for(error_code: str) -> str:
    """Return the stable client-facing message for ``error_code``.

    Unknown codes fail closed to the INTERNAL_ERROR message (P2-03) — the
    caller must never fall back to echoing raw exception text.
    """
    # 返回 ``error_code`` 对应的稳定面向客户端文案。
    # 未知码 fail closed 到 INTERNAL_ERROR 文案（P2-03）——调用方绝不
    # 回退到回显原始异常文本。
    return _PUBLIC_MESSAGES.get(
        error_code,
        _PUBLIC_MESSAGES[DomainErrorCode.INTERNAL_ERROR.value],
    )


def is_known_error_code(error_code: str) -> bool:
    """True when ``error_code`` has a registered public message row."""
    # 当 ``error_code`` 存在已注册的公共文案条目时返回 True。
    return error_code in _PUBLIC_MESSAGES
