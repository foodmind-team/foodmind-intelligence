"""Deterministic test-only workflow ports."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from conftest import AGENT_FIXTURES, REPOSITORY_ROOT
from pydantic import SecretStr

from recommendation_agent.application.ports import ExplanationRenderer, InferencePort, ReasonDeriver, ResultSelector
from recommendation_agent.config.settings import Settings
from recommendation_agent.domain.errors import AgentError
from recommendation_agent.domain.models import (
    CollaborativeSignal,
    InferenceEvidence,
    InferenceResult,
    ReasonCode,
    ReasonedCandidate,
    RecommendationType,
    RenderedCandidate,
    ScoredCandidate,
    SelectedCandidate,
)
from recommendation_agent.schemas.agent_v2 import AgentRequest
from recommendation_agent.schemas.inference_v1 import InferenceSuccess
from recommendation_agent.time.budget import DeadlineBudget
from recommendation_agent.workflow.context import WorkflowContext

INFERENCE_FIXTURE = (
    REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures/valid-hybrid.json"
)


@dataclass
class FakeClock:
    now: datetime = datetime(2030, 1, 1, tzinfo=UTC)
    tick: float = 100.0

    def utc_now(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.tick


def canonical_request() -> AgentRequest:
    return AgentRequest.model_validate_json((AGENT_FIXTURES / "valid-normal.json").read_text(encoding="utf-8"))


def canonical_result() -> InferenceResult:
    return result_from_fixture(INFERENCE_FIXTURE)


def result_from_fixture(path: Path) -> InferenceResult:
    response = InferenceSuccess.model_validate_json(path.read_text(encoding="utf-8"))
    return InferenceResult(
        model_version=response.model_version,
        model_package_version=response.model_package_version,
        feature_schema_version=response.feature_schema_version,
        inference_contract_version=response.contract_version,
        model_key_version=response.model_key_version,
        candidates=tuple(
            ScoredCandidate(
                candidate_id=item.candidate_id,
                probability=item.probability,
                model_score=item.model_score,
                user_cf=CollaborativeSignal(item.user_cf.available, item.user_cf.score, item.user_cf.neighbor_support),
                item_cf=CollaborativeSignal(
                    item.item_cf.available,
                    item.item_cf.score,
                    item.item_cf.supporting_item_count,
                ),
                evidence=InferenceEvidence(
                    preference_match=item.signals.preference_match,
                    want_to_try=item.signals.want_to_try,
                    group_preference_rate=item.signals.group_preference_rate,
                    group_eligible_member_count=item.signals.group_eligible_member_count,
                    context_match=item.signals.context_match,
                    cleanliness_observed=item.signals.cleanliness_observed,
                ),
            )
            for item in response.predictions
        ),
    )


@dataclass
class SpyInference(InferencePort):
    result: InferenceResult = field(default_factory=canonical_result)
    error: AgentError | None = None
    calls: int = 0

    async def score(self, _command: object, *, budget: DeadlineBudget) -> InferenceResult:
        del budget
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@dataclass
class FixtureSelector(ResultSelector):
    calls: int = 0
    malicious_candidate_id: str | None = None

    async def select(self, _request: AgentRequest, result: InferenceResult) -> tuple[SelectedCandidate, ...]:
        self.calls += 1
        types = (RecommendationType.PERSONAL, RecommendationType.EXPLORATORY, RecommendationType.GROUP_INSPIRED)
        return tuple(
            SelectedCandidate(
                candidate_id=self.malicious_candidate_id or candidate.candidate_id,
                recommendation_type=types[index],
                probability=candidate.probability,
                model_score=candidate.model_score,
            )
            for index, candidate in enumerate(result.candidates[:3])
        )


@dataclass
class FixtureReasons(ReasonDeriver):
    calls: int = 0

    async def derive(
        self,
        _request: AgentRequest,
        _result: InferenceResult,
        selections: tuple[SelectedCandidate, ...],
    ) -> tuple[ReasonedCandidate, ...]:
        self.calls += 1
        return tuple(ReasonedCandidate(selection, (ReasonCode.PREFERENCE_MATCH,)) for selection in selections)


@dataclass
class FixtureRenderer(ExplanationRenderer):
    calls: int = 0

    async def render(
        self,
        candidates: tuple[ReasonedCandidate, ...],
        *,
        budget: DeadlineBudget | None = None,
    ) -> tuple[RenderedCandidate, ...]:
        del budget
        self.calls += 1
        return tuple(RenderedCandidate(candidate, "This matches your saved preferences.") for candidate in candidates)


def workflow_context(
    *,
    inference: InferencePort | None = None,
    selector: ResultSelector | None = None,
    reasons: ReasonDeriver | None = None,
    renderer: ExplanationRenderer | None = None,
    clock: FakeClock | None = None,
) -> WorkflowContext:
    return WorkflowContext(
        inference=inference or SpyInference(),
        selector=selector or FixtureSelector(),
        reason_deriver=reasons or FixtureReasons(),
        renderer=renderer or FixtureRenderer(),
        settings=Settings(
            app_env="test",
            internal_service_token=SecretStr("workflow-test-token"),
            inference_service_token=SecretStr("workflow-inference-test-token"),
        ),
        clock=clock or FakeClock(),
    )
