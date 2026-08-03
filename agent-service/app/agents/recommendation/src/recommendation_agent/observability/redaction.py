"""Deep redaction for structured log values."""

import re
from collections.abc import Mapping, Sequence
from typing import Final

_REDACTED: Final = "[REDACTED]"
_SENSITIVE_FRAGMENTS: Final = (
    "authorization",
    "body",
    "credential",
    "exception",
    "feature",
    "key",
    "secret",
    "token",
    "url",
    "uri",
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*")
_URI_QUERY = re.compile(r"(https?://[^\s?]+)\?[^\s]+", re.IGNORECASE)


def redact(value: object, *, key: str | None = None) -> object:
    """Return a recursively redacted value suitable for structured logging."""

    if key is not None and any(fragment in key.casefold() for fragment in _SENSITIVE_FRAGMENTS):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(item_key): redact(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _URI_QUERY.sub(r"\1?[REDACTED]", _BEARER.sub("Bearer [REDACTED]", value))
    if isinstance(value, bytes):
        return _REDACTED
    return value
