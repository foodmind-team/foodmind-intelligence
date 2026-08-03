"""Minimal structured JSON logging configuration."""

import json
import logging
from datetime import UTC, datetime

from recommendation_agent.observability.redaction import redact


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": redact(record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload["fields"] = redact(fields)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.handlers.clear()
        root.addHandler(handler)
