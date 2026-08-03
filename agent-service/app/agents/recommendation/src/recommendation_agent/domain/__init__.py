"""Domain types and errors."""

from recommendation_agent.domain.errors import AgentError, ErrorCode
from recommendation_agent.domain.models import RecommendationType

__all__ = ["AgentError", "ErrorCode", "RecommendationType"]
