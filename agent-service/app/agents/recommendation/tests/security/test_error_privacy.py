from recommendation_agent.domain.errors import AgentError, ErrorCode


def test_raw_exception_context_is_never_retained_by_typed_error() -> None:
    canaries = ("raw-upstream-body", "signed-url?credential=secret", "model-key-123", "feature-vector-456")
    try:
        raise RuntimeError(" ".join(canaries))
    except RuntimeError as raw:
        safe = AgentError(ErrorCode.INFERENCE_MALFORMED_RESPONSE, http_status=502)
        safe.__cause__ = raw
    assert all(canary not in str(safe) for canary in canaries)
    assert str(safe) == "INFERENCE_MALFORMED_RESPONSE"
