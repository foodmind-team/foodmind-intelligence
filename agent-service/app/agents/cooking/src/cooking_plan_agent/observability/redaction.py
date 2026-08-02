"""Recursive log redaction (P2-05 / P4-03).

Replaces the legacy "redact only by extra field name" formatter with a
uniform, recursive redactor that covers message text, exception strings,
URLs and arbitrarily nested objects. Design rules:

  - Key-based redaction: a mapping key that matches a sensitive key name
    (case-insensitive, substring match — e.g. ``x-api-key`` matches
    ``api_key``) has its whole value replaced by ``[REDACTED]``.
  - Content redaction: free-form user/provider payload keys the pipeline
    must never log (recipe text, prompts, responses, inventory) are fully
    hidden even when they carry no credential-like key name.
  - String cleaning: embedded URLs are stripped of userinfo and sensitive
    query parameters (token/key/signature/auth); common credential
    prefixes (``Bearer``, ``sk-``, JWT ``eyJ``, GitHub ``ghp_``, AWS
    ``AKIA``) are masked.
  - Fail-closed (D8): structures the redactor cannot process are replaced
    wholesale by ``[REDACTED]`` and counted in ``RedactionStats.failed``.
  - Diagnostic whitelist: request_id / node / error_code / correlation_id
    / duration_ms / status / level / logger / timestamp are preserved so
    incidents stay traceable (P2-05: correlation-ID based tracing).

The module is side-effect free: ``redact()`` is a pure function and
``redact_with_stats()`` additionally reports how many redactions were
applied / failed, which the JSON formatter attaches to every log line
as ``redaction_applied_total`` / ``redaction_fail_total``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"

# Log-output field names carrying the redaction counters (P2-05 monitoring
# matrix: `cooking_plan_redaction_fail_total` / `redaction_applied_total`).
REDACTION_APPLIED_FIELD = "redaction_applied_total"
REDACTION_FAIL_FIELD = "redaction_fail_total"

# ---------------------------------------------------------------------------
# Key sets
# ---------------------------------------------------------------------------
# Credential-style keys — matched case-insensitively and by substring so
# variants like `x-api-key`, `X-Authorization` or `DB_PASSWORD` are caught.
_REDACT_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "passwd",
        "credential",
        "client_secret",
        "private_key",
        "cookie",
        "session_id",
        "internal_service_token",
    }
)

# Free-form content keys carried over from the legacy formatter — full-value
# hiding, exact (normalised) name match only.
_CONTENT_KEYS = frozenset(
    {
        "recipe_text",
        "recipes",
        "inventory",
        "dietary_rules",
        "dietary_restrictions",
        "prompt",
        "response",
        "provider_payload",
    }
)

# Diagnostic fields that must survive redaction (values still pass through
# string-level cleaning, so even these cannot leak an embedded secret).
_KEEP_KEYS = frozenset(
    {
        "request_id",
        "node",
        "error_code",
        "correlation_id",
        "duration_ms",
        "status",
        "level",
        "logger",
        "timestamp",
    }
)


def _normalise_key(key: str) -> str:
    """Lowercase and strip separators so key matching is case/format agnostic."""
    return re.sub(r"[\s\-_.]", "", key).lower()


_NORM_REDACT_KEYS = frozenset(_normalise_key(k) for k in _REDACT_KEYS)
_NORM_CONTENT_KEYS = frozenset(_normalise_key(k) for k in _CONTENT_KEYS)
_NORM_KEEP_KEYS = frozenset(_normalise_key(k) for k in _KEEP_KEYS)


def _is_sensitive_key(norm: str) -> bool:
    """Substring match against the credential key set (D8: prefer hiding)."""
    return any(redact_key in norm for redact_key in _NORM_REDACT_KEYS)


# ---------------------------------------------------------------------------
# String-level cleaning
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

# Query parameters whose presence in a URL makes the URL sensitive
# (substring match; "sig" covers the standard "signature" abbreviation).
_SENSITIVE_QUERY_KEYS = frozenset({"token", "key", "signature", "sig", "auth"})

# Common credential prefixes. Bearer keeps its scheme word for readability;
# everything else is masked wholesale.
_BEARER_RE = re.compile(r"\b(Bearer)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),  # OpenAI-style keys
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),  # JWT
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub PAT
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key IDs
)


def _clean_url(url: str, stats: RedactionStats | None = None) -> str:
    """Strip userinfo and sensitive query params from a URL.

    Keeps ``scheme://host[:port]/path`` only. Unparseable URLs fail closed
    (counted as a redaction failure when stats are supplied).
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        if stats is not None:
            stats.failed += 1
        return REDACTED

    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        netloc = f"[{host}]"
    else:
        netloc = host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"

    raw_query = parts.query
    pairs = parse_qsl(raw_query, keep_blank_values=True)
    if pairs and any(_is_sensitive_query_key(k) for k, _ in pairs):
        kept = [(k, v) for k, v in pairs if not _is_sensitive_query_key(k)]
        raw_query = urlencode(kept, doseq=True)

    return urlunsplit((parts.scheme, netloc, parts.path, raw_query, parts.fragment))


def _is_sensitive_query_key(key: str) -> bool:
    norm = _normalise_key(key)
    return any(sensitive in norm for sensitive in _SENSITIVE_QUERY_KEYS)


def _redact_string(text: str, stats: RedactionStats) -> str:
    original = text
    cleaned = _URL_RE.sub(lambda match: _clean_url(match.group(0), stats), text)
    cleaned = _BEARER_RE.sub(lambda match: f"{match.group(1)} {REDACTED}", cleaned)
    for pattern in _TOKEN_PATTERNS:
        cleaned = pattern.sub(REDACTED, cleaned)
    if cleaned != original:
        stats.applied += 1
    return cleaned


# ---------------------------------------------------------------------------
# Recursive redaction
# ---------------------------------------------------------------------------


@dataclass
class RedactionStats:
    """Counters attached to one redaction pass (per log record)."""

    applied: int = 0
    failed: int = 0


def _redact_mapping_value(key: str, value: object, stats: RedactionStats) -> object:
    norm = _normalise_key(key)
    if norm in _NORM_KEEP_KEYS:
        # Diagnostic whitelist wins — but the value still passes through
        # string-level cleaning so even it cannot leak an embedded secret.
        return _redact(value, stats)
    if _is_sensitive_key(norm) or norm in _NORM_CONTENT_KEYS:
        stats.applied += 1
        return REDACTED
    return _redact(value, stats)


def _redact(value: object, stats: RedactionStats) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value, stats)
    if isinstance(value, bytes):
        try:
            return _redact_string(value.decode("utf-8"), stats)
        except UnicodeDecodeError:
            stats.failed += 1
            return REDACTED
    if isinstance(value, Mapping):
        return {str(key): _redact_mapping_value(str(key), item, stats) for key, item in value.items()}
    if isinstance(value, Sequence) or isinstance(value, set):
        # JSON-friendly shape; tuples/sets become lists.
        return [_redact(item, stats) for item in value]
    # Fail-closed (D8): unknown object structure must never be emitted raw.
    stats.failed += 1
    return REDACTED


def redact(value: object) -> object:
    """Pure recursive redaction; unprocessable structures become ``[REDACTED]``.

    Never raises — the formatter and call sites may rely on this.
    """
    return _redact(value, RedactionStats())


def redact_with_stats(value: object) -> tuple[object, RedactionStats]:
    """Redact and report how many redactions were applied / failed.

    The JSON formatter uses this to attach ``redaction_applied_total`` and
    ``redaction_fail_total`` to every structured log line (P2-05 metrics).
    """
    stats = RedactionStats()
    result = _redact(value, stats)
    return result, stats


__all__ = [
    "REDACTED",
    "REDACTION_APPLIED_FIELD",
    "REDACTION_FAIL_FIELD",
    "RedactionStats",
    "redact",
    "redact_with_stats",
]
