"""Unit tests for the public message catalog (P2-03).

The catalog is the single source of truth for client-facing FAILED messages:
- Every DomainErrorCode must have exactly one registered public message.
- Public messages are stable and free of sensitive placeholders.
- Unknown codes fail closed to INTERNAL_ERROR.

Retry semantics are covered by P3-05's ``is_retryable`` catalog and are NOT
duplicated here (single source of truth).
"""

import re

from cooking_plan_agent.domain.errors import (
    DomainErrorCode,
    is_known_error_code,
    public_message_for,
)

# INVALID_SERVING_TIME is a registered string code (not yet an enum member
# in some baselines); it must still resolve to a stable message.
_EXTRA_CODES = {"INVALID_SERVING_TIME"}

_SENSITIVE_PATTERNS = (
    re.compile(r"\b(sk-|pk-|ak-)[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r"bearer\s+\S+", re.IGNORECASE),
    re.compile(r"(api[_-]?key|password|secret|token)\b", re.IGNORECASE),
    re.compile(r"https?://\S+"),
)

# Codes that only appear on non-FAILED paths or are covered elsewhere.
_EXCLUDED_ENUM = {"INVALID_SERVING_TIME"}  # superset guard; enum may lack it


class TestCatalogCompleteness:
    def test_every_domain_error_code_has_a_public_message(self):
        declared = {code.value for code in DomainErrorCode} - _EXCLUDED_ENUM
        for code in declared:
            assert is_known_error_code(code), f"Missing public message for {code}"

    def test_extra_registered_codes_have_messages(self):
        for code in _EXTRA_CODES:
            assert is_known_error_code(code), f"Missing public message for {code}"

    def test_public_messages_are_sensitive_free(self):
        codes = [code.value for code in DomainErrorCode] + list(_EXTRA_CODES)
        for code in codes:
            if not is_known_error_code(code):
                continue
            message = public_message_for(code)
            assert message.strip()
            for pattern in _SENSITIVE_PATTERNS:
                assert not pattern.search(message), f"catalog row {code} contains sensitive content: {message!r}"


class TestCatalogMessages:
    def test_internal_error_snapshot(self):
        assert public_message_for(DomainErrorCode.INTERNAL_ERROR.value) == ("An unexpected internal error occurred.")

    def test_schedule_unknown_snapshot(self):
        assert public_message_for("SCHEDULE_UNKNOWN") == (
            "The scheduler could not determine a feasible schedule within the time limit."
        )

    def test_external_provider_snapshot(self):
        assert public_message_for("EXTERNAL_PROVIDER_UNAVAILABLE") == (
            "An external service is temporarily unavailable."
        )

    def test_serving_time_message(self):
        assert public_message_for("INVALID_SERVING_TIME") == "The serving time is invalid."


class TestFailClosed:
    def test_unknown_code_returns_internal_error_message(self):
        assert public_message_for("NO_SUCH_CODE") == "An unexpected internal error occurred."

    def test_known_code_detection(self):
        assert is_known_error_code("TASK_GRAPH_CYCLE") is True
        assert is_known_error_code("NO_SUCH_CODE") is False
