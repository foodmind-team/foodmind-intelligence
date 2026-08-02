"""Security tests for P4-03 — deep log redaction (补 P2-05).

Asserts that the recursive redactor AND the production JSON formatter can
never emit secret canaries (credentials, tokens, keyed URLs), that unknown
structures fail closed to ``[REDACTED]``, and that diagnostic fields
(request_id / node / error_code / correlation_id) stay traceable.

Test surface (P4-03 验证):
  - 单元六类：mapping / sequence / 嵌套 JSON / 大小写 / 多 token 前缀 /
    多值 query / URL（userinfo、敏感 query）/ exception。
  - 安全：注入已知 secret canary，capture handler 全量断言不可检索原文。
  - fail-closed：非预期对象结构输出 [REDACTED]。
  - 诊断保留：request_id / node / error_code 仍在日志中可检索。
"""

import json
import logging
import os

# main 的模块级 import 会触发 get_settings()（internal_service_token 必填）。
# 与 tests/smoke/test_docker_smoke.py 一致：在导入 main 前先注入测试 token。
os.environ.setdefault("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", "test-token-redaction")

from cooking_plan_agent.main import _RedactingJsonFormatter  # noqa: E402
from cooking_plan_agent.observability.redaction import (  # noqa: E402
    REDACTED,
    RedactionStats,
    redact,
    redact_with_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CaptureHandler(logging.Handler):
    """Collects records formatted by the production JSON formatter."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(_RedactingJsonFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def _format_record(
    message: str,
    *,
    extra: dict[str, object] | None = None,
    exc_info: tuple[type[BaseException], BaseException, object] | None = None,
) -> dict[str, object]:
    """Format one log record through the production formatter and parse it."""
    record = logging.LogRecord(
        name="test.log.redaction",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
        func="test",
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    handler = _CaptureHandler()
    handler.emit(record)
    return json.loads(handler.lines[0])


class _UnknownObject:
    """Deliberately unprocessable by the redactor (fail-closed case)."""

    def __str__(self) -> str:
        return "sk-should-never-leak-via-str"


# ---------------------------------------------------------------------------
# 1. Redactor 单元测试
# ---------------------------------------------------------------------------


def test_mapping_credential_keys_redacted() -> None:
    """Credential-style mapping keys have their whole value hidden."""
    value = redact(
        {
            "api_key": "sk-live-1234567890",
            "authorization": "Bearer abcdef",
            "password": "hunter2",
            "client_secret": "sec-1",
            "internal_service_token": "tok-1",
            "recipe_name": "kept",
            "duration_minutes": 10,
        }
    )
    assert isinstance(value, dict)
    assert value["api_key"] == REDACTED
    assert value["authorization"] == REDACTED
    assert value["password"] == REDACTED
    assert value["client_secret"] == REDACTED
    assert value["internal_service_token"] == REDACTED
    assert value["recipe_name"] == "kept"
    assert value["duration_minutes"] == 10


def test_key_matching_is_case_insensitive_and_substring() -> None:
    """x-api-key / API_KEY / X-Authorization variants are all caught."""
    value = redact(
        {
            "X-Api-Key": "k1",
            "x-api-key": "k2",
            "API_KEY": "k3",
            "X-Authorization": "Bearer tok",
            "DB_PASSWORD": "p",
            "ACCESS_TOKEN": "t",
            "headers.authorization": "Bearer z",
        }
    )
    assert isinstance(value, dict)
    assert set(value.values()) == {REDACTED}


def test_content_keys_fully_redacted() -> None:
    """Recipe text / prompt / response extras stay hidden (legacy fields)."""
    value = redact(
        {
            "recipe_text": "Cook salmon at 180C with sk-abcdef1234",
            "prompt": "extract from: <full recipe>",
            "inventory": {"lot": "L-1"},
            "dietary_rules": "no nuts",
            "log_level": "INFO",
        }
    )
    assert isinstance(value, dict)
    assert value["recipe_text"] == REDACTED
    assert value["prompt"] == REDACTED
    assert value["inventory"] == REDACTED
    assert value["dietary_rules"] == REDACTED
    assert value["log_level"] == "INFO"


def test_nested_structures_recursively_redacted() -> None:
    """Sensitive keys hidden at any nesting depth (dict/list/mixed)."""
    value = redact(
        {
            "outer": [
                {"inner": {"token": "secret-token", "value": "ok"}},
                {"api_key": "sk-abc", "tags": ["a", "b"]},
            ],
            "payload": {"headers": {"Authorization": "Bearer abc", "Accept": "json"}},
        }
    )
    assert value["outer"][0]["inner"]["token"] == REDACTED
    assert value["outer"][0]["inner"]["value"] == "ok"
    assert value["outer"][1]["api_key"] == REDACTED
    assert value["outer"][1]["tags"] == ["a", "b"]
    assert value["payload"]["headers"]["Authorization"] == REDACTED
    assert value["payload"]["headers"]["Accept"] == "json"


def test_sequence_items_redacted() -> None:
    """Sequence/set items are redacted item-by-item."""
    value = redact(["safe", "Bearer 12345.67890", {"password": "pw"}])
    assert value[0] == "safe"
    assert REDACTED in value[1]
    assert "12345.67890" not in value[1]
    assert value[2]["password"] == REDACTED


def test_multi_token_prefixes_masked() -> None:
    """Bearer / sk- / JWT / GitHub / AWS prefixes are all masked."""
    value = redact(
        "call with Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig and sk-abcdefghijklmn and "
        "ghp_1234567890abcdefghijk and AKIA1234567890ABCDEF"
    )
    assert isinstance(value, str)
    assert "eyJhbGciOiJIUzI1NiJ9" not in value
    assert "sk-abcdefghijklmn" not in value
    assert "ghp_1234567890abcdefghijk" not in value
    assert "AKIA1234567890ABCDEF" not in value
    # Bearer keeps its scheme word for readability.
    assert "Bearer " in value


def test_bearer_token_masked() -> None:
    """A full `Bearer <token>` never survives; the scheme word stays."""
    value = redact("Authorization: Bearer abc.def.ghi")
    assert "abc.def.ghi" not in value
    assert value.count(REDACTED) >= 1


def test_url_userinfo_removed() -> None:
    """URL userinfo is stripped, scheme://host[:port]/path preserved."""
    value = redact("see https://user:pass@example.com:8443/path?a=1 for details")
    assert "user:pass" not in value
    assert "https://example.com:8443/path?a=1" in value


def test_url_sensitive_query_params_dropped() -> None:
    """Sensitive query params (token/key/signature/auth) are dropped."""
    value = redact("https://api.example.com/search?q=salmon&token=s3cr3t&key=k&sig=abc&page=2")
    assert "s3cr3t" not in value
    assert "abc" not in value  # the signature value must not survive either
    assert "q=salmon" in value
    assert "page=2" in value
    assert "token" not in value and "key=" not in value and "sig=" not in value


def test_multi_value_query_params_kept_for_benign() -> None:
    """Duplicate benign params survive while sensitive ones are dropped."""
    value = redact("https://host/endpoint?tag=x&tag=y&api_key=zzz&auth=1")
    assert "api_key=zzz" not in value
    assert "auth=1" not in value
    assert value.count("tag=x") == 1
    assert value.count("tag=y") == 1


def test_embedded_url_in_exception_text_cleaned() -> None:
    """Exception text containing a keyed URL is cleaned."""
    exc = ValueError("provider failed: https://user:token@api.com/v1?api_key=live")
    cleaned = redact(str(exc))
    assert "user:token" not in cleaned
    assert "api_key=live" not in cleaned
    assert "provider failed" in cleaned


def test_keep_keys_preserved() -> None:
    """Diagnostic whitelist fields survive redaction."""
    value = redact(
        {
            "request_id": "req-abc",
            "node": "solve_schedule",
            "error_code": "SCHEDULE_UNKNOWN",
            "correlation_id": "corr-1",
            "duration_ms": 123,
            "status": 200,
            "level": "WARNING",
            "logger": "cooking_plan_agent.workflow",
            "timestamp": "2026-08-02T00:00:00Z",
            "api_key": "sk-x",
        }
    )
    assert isinstance(value, dict)
    assert value["request_id"] == "req-abc"
    assert value["node"] == "solve_schedule"
    assert value["error_code"] == "SCHEDULE_UNKNOWN"
    assert value["correlation_id"] == "corr-1"
    assert value["duration_ms"] == 123
    assert value["status"] == 200
    assert value["level"] == "WARNING"
    assert value["logger"] == "cooking_plan_agent.workflow"
    assert value["timestamp"] == "2026-08-02T00:00:00Z"
    assert value["api_key"] == REDACTED


def test_fail_closed_unknown_object() -> None:
    """Unknown object structures are hidden wholesale and counted."""
    value, stats = redact_with_stats(_UnknownObject())
    assert value == REDACTED
    assert stats.failed >= 1
    assert isinstance(stats, RedactionStats)


def test_fail_closed_unknown_object_nested() -> None:
    """An unknown object nested inside a mapping is hidden, not leaked."""
    value = redact({"payload": _UnknownObject(), "keep": "yes"})
    assert isinstance(value, dict)
    assert value["payload"] == REDACTED
    assert value["keep"] == "yes"


def test_stats_count_applied_and_failed() -> None:
    """redact_with_stats reports applied/failed counters for metrics."""
    _, stats = redact_with_stats({"api_key": "sk-abcdefgh", "safe": "https://user:pw@host/x?token=1"})
    assert stats.applied >= 1
    assert stats.failed == 0


def test_primitives_pass_through() -> None:
    """Scalars never change shape or value."""
    assert redact(None) is None
    assert redact(True) is True
    assert redact(42) == 42
    assert redact(1.5) == 1.5


# ---------------------------------------------------------------------------
# 2. 生产 JSON formatter：全链路 canary 不可检索
# ---------------------------------------------------------------------------


def test_formatter_redacts_message_and_extra() -> None:
    """Message, nested extra and extra values are all redacted; diagnostics kept."""
    entry = _format_record(
        "request failed: Bearer sk-live-aaaaaaaa and https://u:p@host/x?token=zz",
        extra={
            "request_id": "req-42",
            "node": "parse_recipes",
            "error_code": "PARSE_FAILED",
            "nested": {"authorization": "Bearer hidden-1", "attempt": 1},
        },
    )
    rendered = json.dumps(entry, ensure_ascii=False)
    assert "sk-live-aaaaaaaa" not in rendered
    assert "hidden-1" not in rendered
    assert "u:p@" not in rendered
    assert "token=zz" not in rendered
    # Diagnostic fields remain searchable.
    assert entry["request_id"] == "req-42"
    assert entry["node"] == "parse_recipes"
    assert entry["error_code"] == "PARSE_FAILED"


def test_formatter_exception_keeps_type_and_cleaned_summary() -> None:
    """Exception lines carry type + redacted summary, never the raw text."""
    secret = "sk-live-0987654321"
    entry = _format_record(
        "workflow failed",
        exc_info=(ValueError, ValueError(f"provider body with {secret}"), None),
    )
    assert entry["exception"].startswith("ValueError: ")
    assert secret not in entry["exception"]
    # No raw traceback is emitted either.
    assert "Traceback" not in entry["exception"]


def test_formatter_canary_never_leaks_full_log_path() -> None:
    """A batch of canary-laden records never leaks through the formatter."""
    canaries = [
        "Bearer abcdef.ghijkl",
        "sk-proj-abcdefghijklmnopqrstuvwx",
        "https://user:pass@example.org/private?api_key=key-12345&mode=full",
        "eyJhbGciOiJIUzI1NiJ9.payload.signature",
    ]
    handler = _CaptureHandler()
    for idx, canary in enumerate(canaries):
        record = logging.LogRecord(
            name="test.canary",
            level=logging.WARNING,
            pathname=__file__,
            lineno=idx + 1,
            msg=f"event {idx}: {canary}",
            args=(),
            exc_info=None,
            func="test",
        )
        handler.emit(record)

    output = "\n".join(handler.lines)
    for canary in canaries:
        assert canary not in output
    # The sensitive fragments are individually unrecoverable too.
    for fragment in ("abcdef.ghijkl", "sk-proj", "user:pass", "api_key=key-12345", "signature"):
        assert fragment not in output


def test_formatter_counts_emitted_per_record() -> None:
    """Every structured line carries redaction counters (P2-05 metrics)."""
    entry = _format_record("token sk-abcdefghij", extra={"request_id": "r1"})
    applied = entry["redaction_applied_total"]
    failed = entry["redaction_fail_total"]
    assert isinstance(applied, int) and applied >= 1
    assert isinstance(failed, int) and failed == 0

    failed_entry = _format_record("boom", extra={"blob": _UnknownObject()})
    assert failed_entry["redaction_fail_total"] >= 1


def test_formatter_keeps_benign_message_unchanged() -> None:
    """Non-sensitive messages are preserved verbatim (regression safety)."""
    entry = _format_record("Cooking Plan Agent ready", extra={"request_id": "r9"})
    assert entry["message"] == "Cooking Plan Agent ready"
    assert entry["request_id"] == "r9"
