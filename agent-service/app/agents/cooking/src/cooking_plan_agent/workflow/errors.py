"""Workflow-specific error types for node-to-graph communication.

Nodes raise these; the graph routes to error terminal nodes.
Three-level error boundary per handbook 8.10:
  1. Domain/application services raise typed errors (WorkflowException)
  2. Node wrappers convert expected errors into NodeExecutionError
  3. FastAPI global handler catches unexpected failures
"""

from cooking_plan_agent.domain.errors import DomainErrorCode


class NodeExecutionError(Exception):
    """Expected error from a workflow node — routes to a stable response.

    Nodes wrap domain/application errors into this type, which the
    graph uses for conditional routing to error terminal nodes
    (render_failed_response, render_infeasible_response).

    The three-level error boundary (handbook 8.10):
      1. Domain services raise typed errors (DomainErrorCode)
      2. Node wrappers catch and emit NodeExecutionError (this type)
      3. FastAPI global handler catches unexpected/uncaught failures
    """

    def __init__(
        self,
        code: DomainErrorCode,
        message: str,
        correlation_id: str = "",
        recoverable: bool = False,
    ) -> None:
        self.code = code
        self.message = message
        # correlation_id links the error back to the original request for debugging
        self.correlation_id = correlation_id
        # recoverable=True means the graph may retry rather than terminate
        self.recoverable = recoverable
        super().__init__(f"[{code.value}] {message}")
