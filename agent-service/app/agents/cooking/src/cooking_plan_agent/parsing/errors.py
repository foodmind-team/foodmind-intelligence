
# Top-level blank line — separates module docstring (none here) from first class.


# Top-level blank line — separates module header area from class definitions.


class PreprocessError(ValueError):  # Base exception for all deterministic preprocessing failures.
    """Base error for all preprocessing failures. Raised before LLM extraction."""


class DecodeError(PreprocessError):  # Raised when byte content cannot be decoded as valid UTF-8.
    """Bytes could not be decoded into a supported text encoding."""


class OversizedInputError(PreprocessError):  # Raised when input exceeds byte or character budget.
    """Input exceeds the configured byte or character limit."""


class EmptyContentError(PreprocessError):  # Raised when decoded text contains only whitespace.
    """Decoded text contains no usable content after noise removal."""

# Blank line — separates EmptyContentError from NULBytesError for readability.


class NULBytesError(PreprocessError):  # Raised when raw bytes contain NUL (0x00) — binary signal.
    """Binary-appearing input — NUL bytes detected in raw content."""
