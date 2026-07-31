"""Hostile text defence (handbook 10.5).

Retrieved web content may contain prompt injection, scripts, or markup.
This module strips dangerous content at the provider/adapter boundary.
"""

import re

# ---------------------------------------------------------------------------
# Patterns to strip from retrieved text
# ---------------------------------------------------------------------------

# HTML/XML tags and their content
_HTML_TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
# JavaScript blocks
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
# CSS blocks
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
# HTML entities
_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
# Excessive whitespace
_WHITESPACE_RE = re.compile(r"\s+")
# Prompt injection markers — instruct/inject/system/override patterns
_INJECTION_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?)\b",
        r"\b(system|assistant|user)\s*:\s*",
        r"\b(you\s+are\s+now|you\s+must|your\s+new\s+role)\b",
        r"\b(override|bypass|ignore)\s+(system|safety|policy|rules?)\b",
    )
)


def strip_markup(text: str) -> str:
    """Remove HTML/XML markup, scripts, and style blocks from text.

    Must be called at the provider or adapter boundary — before any text
    reaches extraction or LLM prompts.
    """
    text = _SCRIPT_RE.sub(" ", text)
    text = _STYLE_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _ENTITY_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def detect_prompt_injection(text: str) -> bool:
    """Return True if the text contains known prompt injection patterns.

    This is a defence-in-depth measure — text flagged here should be
    treated as DATA ONLY, never as instructions.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_document_content(raw_content: str) -> tuple[str, bool]:
    """Sanitize raw web content: strip markup, detect injection.

    Returns (cleaned_text, has_injection_flag).
    If has_injection_flag is True, the content must be treated as data only,
    never as instructions — and extraction should be extra strict.
    """
    cleaned = strip_markup(raw_content)

    # Truncate to a reasonable size to bound extraction cost
    # (handbook 10.5: "Limit document count and content length")
    max_chars = 4000
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    has_injection = detect_prompt_injection(cleaned)
    return cleaned, has_injection
