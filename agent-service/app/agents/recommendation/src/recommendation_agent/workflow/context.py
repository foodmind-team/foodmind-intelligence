"""Immutable dependencies captured outside graph state."""

from dataclasses import dataclass, field

from recommendation_agent.application.ports import ExplanationRenderer, InferencePort, ReasonDeriver, ResultSelector
from recommendation_agent.config.settings import Settings
from recommendation_agent.observability.metrics import MetricsRegistry
from recommendation_agent.time.budget import Clock


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    inference: InferencePort
    selector: ResultSelector
    reason_deriver: ReasonDeriver
    renderer: ExplanationRenderer
    settings: Settings
    clock: Clock
    metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
