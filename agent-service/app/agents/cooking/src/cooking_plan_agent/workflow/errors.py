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
        self.correlation_id = correlation_id
        self.recoverable = recoverable
        super().__init__(f"[{code.value}] {message}")
