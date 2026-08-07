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

# ===========================================================================
# DomainErrorCode — stable domain error code enumeration (handbook 3.8)
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
    # ------------------------------------------------------------------

    # The input text cannot form a usable recipe: empty content, pure binary,
    # oversized file, or preprocessing yielded no recognizable
    # ingredient/step information.
    INVALID_RECIPE_TEXT = "INVALID_RECIPE_TEXT"

    # ------------------------------------------------------------------
    # Request-level input validation errors (P0-03)
    # ------------------------------------------------------------------

    # Two or more recipes in the request share the same recipe_id.  Recipe
    # identity must be unique so ingredient demands and task graphs can be
    # attributed unambiguously.
    DUPLICATE_RECIPE_ID = "DUPLICATE_RECIPE_ID"

    # The request exceeds the maximum number of recipes the service is
    # configured to accept (Settings.max_recipe_count).
    TOO_MANY_RECIPES = "TOO_MANY_RECIPES"

    # A recipe's raw text exceeds the configured byte limit
    # (Settings.max_recipe_text_bytes).  Rejected to bound memory and
    # extraction cost.
    RECIPE_TEXT_TOO_LARGE = "RECIPE_TEXT_TOO_LARGE"

    # The request as a whole exceeds the configured size limit
    # (Settings.max_request_bytes).  Rejected at the workflow boundary.
    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE"

    # The request's schema_version is not in the supported set
    # (Settings.supported_schema_versions).  The caller must upgrade.
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"

    # A time-related request field is invalid (e.g. negative
    # time_limit_minutes).
    INVALID_TIME_LIMIT = "INVALID_TIME_LIMIT"

    # The serving time is malformed or ambiguous: not a valid HH:MM string,
    # or a serving_at instant without a timezone offset (P0-05). The client
    # must resubmit a well-formed time expression.
    INVALID_SERVING_TIME = "INVALID_SERVING_TIME"

    # An approved decision in the request is invalid: unsupported type,
    # conflicting combination, unknown/stale plan_revision, or malformed
    # payload (P0-06).
    INVALID_APPROVED_DECISION = "INVALID_APPROVED_DECISION"

    # The request body failed Pydantic schema validation at the HTTP
    # boundary (P3-05). Used by the RequestValidationError handler; never
    # raised from domain services.
    REQUEST_VALIDATION_ERROR = "REQUEST_VALIDATION_ERROR"

    # ------------------------------------------------------------------
    # Unit conversion errors (handbook 5.3, 5.4)
    # ------------------------------------------------------------------

    # A requested unit conversion lacks required product-specific data.
    # Example: converting "1 onion" to grams without density or average-weight
    # data for onions. The system should request user confirmation rather than
    # applying an unreliable default.
    UNSUPPORTED_UNIT_CONVERSION = "UNSUPPORTED_UNIT_CONVERSION"

    # ------------------------------------------------------------------
    # Safety constraint errors (handbook 5.7, 5.8, 5.9)
    # ------------------------------------------------------------------

    # A hard safety rule was triggered and cannot be repaired automatically.
    # Typical scenarios: cross-contamination risk (raw meat sharing a board
    # with ready-to-eat ingredients and no sanitisation task can be inserted),
    # allergen matches, or missing safe-cooking endpoint temperatures that
    # cannot be filled from a trusted source.
    # This error typically produces an INFEASIBLE terminal response.
    SAFETY_CONSTRAINT_VIOLATION = "SAFETY_CONSTRAINT_VIOLATION"

    # The requested regional food-safety policy pack cannot be applied:
    # unknown region, unknown version, not yet effective, or missing official
    # sources (P3-04 D6). Never silently falls back to another region.
    # Routes to a FAILED response — a plan must not enter READY under an
    # unverifiable policy.
    SAFETY_POLICY_UNAVAILABLE = "SAFETY_POLICY_UNAVAILABLE"

    # ------------------------------------------------------------------
    # Inventory and resource errors (handbook 5.5, 5.6)
    # ------------------------------------------------------------------

    # Required consumable ingredient quantity is insufficient.
    # A shortage remains even after applying the FEFO (First-Expired-First-Out)
    # allocation strategy. The system should generate RepairOptions for the
    # user to select (reduce servings, substitute ingredients, etc.).
    INSUFFICIENT_INVENTORY = "INSUFFICIENT_INVENTORY"

    # No compatible kitchen equipment instance can perform a required task.
    # Example: the recipe requires oven baking but the user's kitchen has no
    # oven, or the only available oven's capacity is below the minimum.
    # Distinct from a scheduling resource conflict: this error occurs during
    # the feasibility check phase and means "does not exist at all" rather
    # than "temporarily occupied".
    NO_COMPATIBLE_RESOURCE = "NO_COMPATIBLE_RESOURCE"

    # ------------------------------------------------------------------
    # Task graph errors (handbook 6.8, 6.9)
    # ------------------------------------------------------------------

    # A cycle was detected in the generated task dependency graph.
    # Typically discovered during topological sort (Kahn's algorithm) inside
    # build_task_graph(). A cyclic graph must never be passed to the CP-SAT
    # solver.
    TASK_GRAPH_CYCLE = "TASK_GRAPH_CYCLE"

    # ------------------------------------------------------------------
    # Scheduling solver errors (handbook 7.6, 7.7, 7.8)
    # ------------------------------------------------------------------

    # The CP-SAT solver proved that no feasible schedule exists under the
    # current constraints. Typical causes: excessively tight time windows,
    # insufficient resource capacity, or conflicting task lag constraints.
    # May lead to an INFEASIBLE response or trigger repair options such as
    # time relaxation or recipe replacement.
    SCHEDULE_INFEASIBLE = "SCHEDULE_INFEASIBLE"

    # The solver could not determine feasibility or infeasibility before
    # timing out. Unlike INFEASIBLE: UNKNOWN means the solver "found no
    # answer" rather than "proved no solution exists". The schedule result is
    # unusable—return a FAILED response.
    # This is a direct mapping of OR-Tools CpSolverStatus.UNKNOWN.
    SCHEDULE_UNKNOWN = "SCHEDULE_UNKNOWN"

    # The CP-SAT model itself is malformed: contradictory constraints, invalid
    # variables, or a scheduling problem shape the builder cannot express.
    # Distinct from SCHEDULE_INFEASIBLE — INFEASIBLE means the solver PROVED no
    # solution exists for a valid model; MODEL_INVALID means the model was
    # never valid (a construction bug). Both are independent of the solver's
    # own status enum but map to a FAILED response (P1-04).
    SCHEDULE_MODEL_INVALID = "SCHEDULE_MODEL_INVALID"

    # The independent verifier (ScheduleVerifier) rejected the solver's output.
    # This indicates the solver produced a result that appears valid but
    # violates constraints—a possible signal of a bug in CP-SAT model
    # construction or a numerical-precision issue. Rejected results must never
    # be returned to the user with READY status.
    SCHEDULE_VERIFICATION_FAILED = "SCHEDULE_VERIFICATION_FAILED"

    # ------------------------------------------------------------------
    # External dependency errors (handbook 4.11, 10.4)
    # ------------------------------------------------------------------

    # The LLM provider or web search service is unavailable after bounded
    # retries. Used when recipe parsing or gap research requires an external
    # call and all attempts have failed. Nodes should map this to a stable
    # FAILED response—never expose raw provider exceptions to the client.
    EXTERNAL_PROVIDER_UNAVAILABLE = "EXTERNAL_PROVIDER_UNAVAILABLE"

    # ------------------------------------------------------------------
    # System-level errors
    # ------------------------------------------------------------------

    # P5-4: the confirmation dialog was not enabled / no checkpointer, so
    # there is no paused conversation to resume. A stable FAILED outcome,
    # never a silent re-run.
    CONFIRMATION_DIALOG_UNAVAILABLE = "CONFIRMATION_DIALOG_UNAVAILABLE"

    # An unexpected internal error that cannot be classified into any of the
    # above categories. Reserved for the FastAPI global exception handler as a
    # last resort; workflow nodes should prefer more specific error codes.
    # Must carry a correlation_id in the client response for investigation.
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ===========================================================================
# WorkflowException — base domain exception (handbook 8.10 layer 1)
# ===========================================================================


class WorkflowException(Exception):
    """

    Unified base class for all domain-level exceptions.

    Design intent (handbook 8.10 three-level error boundary):
      Layer 1 (this layer): domain/application services raise
        WorkflowException carrying a DomainErrorCode and a human-readable
        description.
      Layer 2: LangGraph workflow nodes catch WorkflowException, convert it
        to NodeExecutionError, and route to the appropriate terminal response
        node.
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
    code: DomainErrorCode
    # Human-readable error description used in logs and terminal responses.
    # Must not contain provider prompts, API keys, or user private data.
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
        super().__init__(f"[{code.value}] {message}")


# ===========================================================================
# Error catalog — retryable semantics (P3-05)
# ===========================================================================
# The catalog is the single source of truth for whether a client may retry
# after receiving a given error_code. It is deliberately decoupled from the
# human-readable message text (D9), so retry semantics stay stable and
# auditable. Unknown codes default to non-retryable (safe: fail loudly).

# Codes that indicate a transient condition where a later retry may succeed.
# These are protocol-level (backpressure, shutdown) or provider-level
# (external LLM/search) failures, never business-logic rejections.
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
    return error_code in _RETRYABLE_ERROR_CODES


def retryable_error_codes() -> tuple[str, ...]:
    """Return the sorted set of catalogued retryable codes (audit/report)."""
    return tuple(sorted(_RETRYABLE_ERROR_CODES))


# ===========================================================================
# Public message catalog (P2-03) — stable, sanitised client-facing text
# ===========================================================================
# Single source of truth for FAILED response messages. Every registered
# error code has one stable public message that is free of secrets, provider
# payloads, recipe text and raw exception details. Nodes never build their
# own client-facing strings; the renderers resolve them here. Unknown codes
# fail closed to INTERNAL_ERROR instead of echoing raw message text.
#
# Retry semantics deliberately live in _RETRYABLE_ERROR_CODES (P3-05) — this
# catalog is content-only to keep a single source of truth for retryability.

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
    return _PUBLIC_MESSAGES.get(
        error_code,
        _PUBLIC_MESSAGES[DomainErrorCode.INTERNAL_ERROR.value],
    )


def is_known_error_code(error_code: str) -> bool:
    """True when ``error_code`` has a registered public message row."""
    return error_code in _PUBLIC_MESSAGES
