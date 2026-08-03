import logging

from recommendation_agent.observability.logging import JsonFormatter


def test_structured_log_fields_redact_sensitive_canaries() -> None:
    canary = "CANARY-MODEL-KEY-123"
    record = logging.LogRecord("recommendation", logging.INFO, __file__, 1, "safe event", (), None)
    record.fields = {"modelUserKey": canary, "featureVector": [canary], "safe": "ok"}
    rendered = JsonFormatter().format(record)
    assert canary not in rendered
    assert "[REDACTED]" in rendered
