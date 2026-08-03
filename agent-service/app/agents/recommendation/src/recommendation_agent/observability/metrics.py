"""Constant-memory low-cardinality metrics with an OpenTelemetry-friendly shape."""

from dataclasses import dataclass, field
from threading import Lock

from recommendation_agent.domain.errors import ErrorCode

STAGES = frozenset(
    {
        "validate_envelope",
        "score_once",
        "validate_compatibility",
        "select_results",
        "derive_reasons",
        "render_explanations",
        "build_success",
        "build_failure",
    }
)
RESULTS = frozenset({"success", "failure"})
INFERENCE_RESULTS = frozenset({"success", *(code.value for code in ErrorCode)})
_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


@dataclass(slots=True)
class Histogram:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    buckets: list[int] = field(default_factory=lambda: [0] * len(_BUCKETS))

    def observe(self, value: float) -> None:
        bounded = max(0.0, value)
        self.count += 1
        self.total += bounded
        self.maximum = max(self.maximum, bounded)
        for index, boundary in enumerate(_BUCKETS):
            if bounded <= boundary:
                self.buckets[index] += 1


class MetricsRegistry:
    """A process-local registry whose only labels come from frozen allow-lists."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests = {result: 0 for result in RESULTS}
        self._failures = {code.value: 0 for code in ErrorCode}
        self._inference = {result: 0 for result in INFERENCE_RESULTS}
        self._request_duration = Histogram()
        self._inference_duration = Histogram()
        self._stage_duration = {stage: Histogram() for stage in STAGES}
        self._input_candidates = Histogram()
        self._output_results = Histogram()
        self._readiness = 0

    def record_request(self, *, result: str, duration_seconds: float, candidates: int, outputs: int) -> None:
        if result not in RESULTS:
            raise ValueError("request result label is not allow-listed")
        with self._lock:
            self._requests[result] += 1
            self._request_duration.observe(duration_seconds)
            self._input_candidates.observe(float(candidates))
            self._output_results.observe(float(outputs))

    def record_stage(self, *, stage: str, duration_seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError("stage label is not allow-listed")
        with self._lock:
            self._stage_duration[stage].observe(duration_seconds)

    def record_inference(self, *, result: str, duration_seconds: float) -> None:
        if result not in INFERENCE_RESULTS:
            raise ValueError("inference result label is not allow-listed")
        with self._lock:
            self._inference[result] += 1
            self._inference_duration.observe(duration_seconds)

    def record_failure(self, code: ErrorCode) -> None:
        with self._lock:
            self._failures[code.value] += 1

    def set_readiness(self, ready: bool) -> None:
        with self._lock:
            self._readiness = int(ready)

    def snapshot(self) -> dict[str, object]:
        """Return safe aggregates only; no arbitrary labels or correlations."""

        with self._lock:
            return {
                "requests": dict(self._requests),
                "failures": dict(self._failures),
                "inference": dict(self._inference),
                "requestCount": self._request_duration.count,
                "inferenceCount": self._inference_duration.count,
                "stageCounts": {stage: histogram.count for stage, histogram in self._stage_duration.items()},
                "readiness": self._readiness,
            }
