"""Ports owned by the application layer."""

from typing import Protocol

from recommendation_agent.domain.models import (
    InferenceCommand,
    InferenceResult,
    ReasonedCandidate,
    RenderedCandidate,
    SelectedCandidate,
)
from recommendation_agent.schemas.agent_v2 import AgentRequest, AgentResponse
from recommendation_agent.time.budget import DeadlineBudget


class InferencePort(Protocol):
    """The Agent's single bounded external capability."""

    async def score(self, command: InferenceCommand, *, budget: DeadlineBudget) -> InferenceResult: ...


class ResultSelector(Protocol):
    async def select(self, request: AgentRequest, result: InferenceResult) -> tuple[SelectedCandidate, ...]: ...


class ReasonDeriver(Protocol):
    async def derive(
        self,
        request: AgentRequest,
        result: InferenceResult,
        selections: tuple[SelectedCandidate, ...],
    ) -> tuple[ReasonedCandidate, ...]: ...


class ExplanationRenderer(Protocol):
    async def render(self, candidates: tuple[ReasonedCandidate, ...]) -> tuple[RenderedCandidate, ...]: ...


class AgentWorkflow(Protocol):
    """A bounded workflow installed by the Recommendation Agent composition root."""

    async def run(self, request: AgentRequest, *, agent_trace_id: str) -> AgentResponse:
        """Return a validated response or raise a stable AgentError."""
        ...

    async def aclose(self) -> None:
        """Release workflow resources."""
        ...
