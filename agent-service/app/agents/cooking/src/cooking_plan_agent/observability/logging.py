"""Safe structured logging configuration for the application."""

import json
import logging
import os
import sys
from datetime import UTC, datetime

from cooking_plan_agent.observability.redaction import (
    REDACTION_APPLIED_FIELD,
    REDACTION_FAIL_FIELD,
    redact_with_stats,
)


class RedactingJsonFormatter(logging.Formatter):
    """JSON log formatter that redacts ALL content per Handbook 12.7.

    P4-03 (补 P2-05): upgraded from "redact only by extra field name" to a
    uniform recursive redactor (observability/redaction.py) covering the
    message text, exception strings, URLs and nested objects. Diagnostic
    fields (request_id / node / error_code / correlation_id / duration_ms
    / status / level / logger / timestamp) survive via the whitelist, and
    every line carries redaction counters for the P2-05 monitoring matrix.
    """

    # Standard LogRecord attributes that are never echoed into the JSON line.
    _LOG_ATTR_SKIP = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "process",
            "processName",
            "taskName",
            "message",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        message, msg_stats = redact_with_stats(record.getMessage())
        log_entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # P2-05: exception keeps only the safe type name + cleaned summary —
        # the raw traceback is never written, so it cannot leak secrets.
        applied = msg_stats.applied
        failed = msg_stats.failed
        if record.exc_info and record.exc_info[1]:
            exc = record.exc_info[1]
            exc_summary, exc_stats = redact_with_stats(str(exc))
            log_entry["exception"] = f"{type(exc).__name__}: {exc_summary}"
            applied += exc_stats.applied
            failed += exc_stats.failed

        # Every extra attribute is recursively redacted; the whitelist keeps
        # diagnostic fields and the redactor hides everything else.
        for key, value in record.__dict__.items():
            if key in self._LOG_ATTR_SKIP:
                continue
            redacted, stats = redact_with_stats(value)
            log_entry[key] = redacted
            applied += stats.applied
            failed += stats.failed

        # P2-05 monitoring matrix: fail-closed redaction is observable.
        log_entry[REDACTION_APPLIED_FIELD] = applied
        log_entry[REDACTION_FAIL_FIELD] = failed

        return json.dumps(log_entry, default=str)


def configure_structured_logging() -> None:
    """Configure the root logger with JSON formatting for production.

    Handbook 12.7: structured logs with request_id, node, duration_ms, etc.
    In production, set COOKING_PLAN_LOG_FORMAT=json to enable.
    """
    log_format = os.environ.get("COOKING_PLAN_LOG_FORMAT", "text")
    if log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(RedactingJsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(os.environ.get("COOKING_PLAN_LOG_LEVEL", "INFO").upper())
