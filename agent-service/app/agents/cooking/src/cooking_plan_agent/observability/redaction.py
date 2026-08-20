# =============================================================================
# 递归日志脱敏模块（observability/redaction）
# -----------------------------------------------------------------------------
# 用统一、递归的脱敏器取代旧版“仅按 extra 字段名脱敏”，覆盖消息文本、
# 异常字符串、URL 与任意嵌套对象。设计规则：
#   - 基于键脱敏：映射键命中敏感键名（大小写不敏感、子串匹配，如 x-api-key
#     匹配 api_key）时，整个值替换为 [REDACTED]。
#   - 内容脱敏：流水线绝不可记录的“自由形式用户/provider 负载键”（菜谱文本、
#     prompt、response、库存）即使不含凭据类键名也整体隐藏。
#   - 字符串清洗：内嵌 URL 剥离 userinfo 与敏感查询参数（token/key/signature/auth）；
#     常见凭据前缀（Bearer、sk-、JWT eyJ、GitHub ghp_、AWS AKIA）被掩码。
#   - 失败即关闭（D8）：脱敏器无法处理的结构整体替换为 [REDACTED] 并计入 failed。
#   - 诊断白名单：request_id / node / error_code / correlation_id / duration_ms /
#     status / level / logger / timestamp 保留，使事故可追溯。
# 本模块无副作用：redact() 是纯函数；redact_with_stats() 额外报告
# 应用 / 失败次数，JSON 格式化器将其作为 redaction_applied_total /
# redaction_fail_total 附加到每行日志。
# =============================================================================

"""Recursive log redaction (P2-05 / P4-03).

递归日志脱敏（P2-05 / P4-03）。

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
# ↑ 统一的“已脱敏”占位符

# Log-output field names carrying the redaction counters (P2-05 monitoring
# matrix: `cooking_plan_redaction_fail_total` / `redaction_applied_total`).
# 承载脱敏计数器的日志输出字段名（P2-05 监控矩阵）。
REDACTION_APPLIED_FIELD = "redaction_applied_total"
REDACTION_FAIL_FIELD = "redaction_fail_total"

# ---------------------------------------------------------------------------
# Key sets
# 键集合
# ---------------------------------------------------------------------------
# Credential-style keys — matched case-insensitively and by substring so
# variants like `x-api-key`, `X-Authorization` or `DB_PASSWORD` are caught.
# 凭据类键 —— 大小写不敏感 + 子串匹配，使 x-api-key、X-Authorization、DB_PASSWORD
# 等变体都能被捕获。
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
# 从旧格式化器继承的“自由形式内容键” —— 整值隐藏，仅精确（规范化后）名称匹配。
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
# 必须保留的诊断字段（其值仍经过字符串级清洗，因此连这些也无法泄漏内嵌密钥）。
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
    """小写并剥离分隔符，使键匹配对大小写 / 格式无关。"""
    return re.sub(r"[\s\-_.]", "", key).lower()


_NORM_REDACT_KEYS = frozenset(_normalise_key(k) for k in _REDACT_KEYS)
_NORM_CONTENT_KEYS = frozenset(_normalise_key(k) for k in _CONTENT_KEYS)
_NORM_KEEP_KEYS = frozenset(_normalise_key(k) for k in _KEEP_KEYS)


def _is_sensitive_key(norm: str) -> bool:
    """对凭据键集合做子串匹配（D8：倾向隐藏）。"""
    return any(redact_key in norm for redact_key in _NORM_REDACT_KEYS)


# ---------------------------------------------------------------------------
# String-level cleaning
# 字符串级清洗
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

# Query parameters whose presence in a URL makes the URL sensitive
# (substring match; "sig" covers the standard "signature" abbreviation).
# 出现在 URL 中即让该 URL 敏感的查询参数（子串匹配；"sig" 覆盖 "signature" 缩写）。
_SENSITIVE_QUERY_KEYS = frozenset({"token", "key", "signature", "sig", "auth"})

# Common credential prefixes. Bearer keeps its scheme word for readability;
# everything else is masked wholesale.
# 常见凭据前缀。Bearer 保留其 scheme 词以保持可读；其余整体掩码。
_BEARER_RE = re.compile(r"\b(Bearer)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),  # OpenAI-style keys
    # ↑ OpenAI 风格密钥
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),  # JWT
    # ↑ JWT
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub PAT
    # ↑ GitHub 个人访问令牌
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key IDs
    # ↑ AWS 访问密钥 ID
)


def _clean_url(url: str, stats: RedactionStats | None = None) -> str:
    """从 URL 剥离 userinfo 与敏感查询参数。

    Strip userinfo and sensitive query params from a URL.

    Keeps ``scheme://host[:port]/path`` only. Unparseable URLs fail closed
    (counted as a redaction failure when stats are supplied).

    仅保留 scheme://host[:port]/path。无法解析的 URL 失败即关闭（提供 stats 时计为脱敏失败）。
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        if stats is not None:
            stats.failed += 1
        return REDACTED

    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        # ↑ IPv6 字面量
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
# 递归脱敏
# ---------------------------------------------------------------------------


@dataclass
class RedactionStats:
    """单次脱敏（每条日志记录）附加的计数器。"""

    applied: int = 0
    failed: int = 0


def _redact_mapping_value(key: str, value: object, stats: RedactionStats) -> object:
    norm = _normalise_key(key)
    if norm in _NORM_KEEP_KEYS:
        # Diagnostic whitelist wins — but the value still passes through
        # string-level cleaning so even it cannot leak an embedded secret.
        # 诊断白名单优先 —— 但值仍经过字符串级清洗，因此连它也无法泄漏内嵌密钥。
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
        # JSON 友好形状；tuple/set 变为 list。
        return [_redact(item, stats) for item in value]
    # Fail-closed (D8): unknown object structure must never be emitted raw.
    # 失败即关闭（D8）：未知对象结构绝不能原样输出。
    stats.failed += 1
    return REDACTED


def redact(value: object) -> object:
    """纯递归脱敏；无法处理的结构变为 ``[REDACTED]``。

    Pure recursive redaction; unprocessable structures become ``[REDACTED]``.

    Never raises — the formatter and call sites may rely on this.

    绝不抛异常 —— 格式化器与调用点可依赖这一点。
    """
    return _redact(value, RedactionStats())


def redact_with_stats(value: object) -> tuple[object, RedactionStats]:
    """脱敏并报告应用 / 失败次数。

    Redact and report how many redactions were applied / failed.

    The JSON formatter uses this to attach ``redaction_applied_total`` and
    ``redaction_fail_total`` to every structured log line (P2-05 metrics).

    JSON 格式化器用它把 redaction_applied_total 与 redaction_fail_total
    附加到每行结构化日志（P2-05 指标）。
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
