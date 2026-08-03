from recommendation_agent.domain.errors import AgentError, ErrorCode


def test_typed_inference_error_contains_no_raw_sensitive_context() -> None:
    canaries = ("signed-url-canary", "model-key-canary", "feature-vector-canary", "token-canary")
    error = AgentError(ErrorCode.INFERENCE_CONNECTION_FAILED, http_status=502, retryable=True)
    rendered = repr(error)
    assert all(canary not in rendered for canary in canaries)
