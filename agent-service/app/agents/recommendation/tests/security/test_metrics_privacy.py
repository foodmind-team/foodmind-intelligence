import pytest

from recommendation_agent.domain.errors import ErrorCode
from recommendation_agent.observability.metrics import MetricsRegistry


def test_metrics_accept_only_frozen_low_cardinality_labels() -> None:
    registry = MetricsRegistry()
    registry.record_request(result="success", duration_seconds=0.01, candidates=5, outputs=3)
    registry.record_stage(stage="score_once", duration_seconds=0.005)
    registry.record_inference(result="success", duration_seconds=0.004)
    registry.record_failure(ErrorCode.INFERENCE_TIMEOUT)
    snapshot = registry.snapshot()
    assert snapshot["requestCount"] == 1
    assert snapshot["inferenceCount"] == 1
    assert "requestId" not in repr(snapshot)
    with pytest.raises(ValueError):
        registry.record_stage(stage="candidate-controlled-canary", duration_seconds=0.1)
    with pytest.raises(ValueError):
        registry.record_request(result="request-123", duration_seconds=0.1, candidates=1, outputs=1)
