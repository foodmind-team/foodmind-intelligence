"""FastAPI API layer — HTTP boundary and request/response handling.

This package owns the web-facing contract between Spring Boot and the
cooking-plan agent. It validates service authentication and transforms
HTTP concerns into domain-layer calls.

Handbook sections: 9.1–9.13 (FastAPI and Spring Boot Integration).

Re-exports the public API so callers only need one import:
  >>> from cooking_plan_agent.api import router, register_exception_handlers
"""

from cooking_plan_agent.api.dependencies import (
    extract_correlation_id,
    require_internal_service,
)
from cooking_plan_agent.api.errors import register_exception_handlers
from cooking_plan_agent.api.router import router

__all__ = [
    "extract_correlation_id",
    "register_exception_handlers",
    "require_internal_service",
    "router",
]
