import unicodedata  # Python stdlib — provides Unicode character category classification.

from cooking_plan_agent.parsing.errors import (  # Local error hierarchy for preprocessing stage.
    DecodeError,  # Bytes could not be decoded (invalid UTF-8).
    EmptyContentError,  # Text is empty or whitespace-only after cleaning.
    NULBytesError,  # Raw content contains NUL (0x00) — binary file signal.
    OversizedInputError,  # Content exceeds configured byte or character limit.
)

# ---------------------------------------------------------------------------
# 4.2.1  Byte → text decoding
# ---------------------------------------------------------------------------


def decode_txt(content: bytes, max_bytes: int = 512_000) -> str:  # Public API — decode raw bytes with safety gates.
    """Decode raw recipe bytes into a clean Unicode string.

    Accepts UTF-8 first (dominant encoding for recipe sources).  If a second
    encoding is added later, document it explicitly in this docstring.

    Rules (handbook 4.2):
      - Reject oversized input before decoding (defence against binary blobs).
      - Reject NUL bytes — strong signal of binary / non-text content.
      - Prefer explicit ``DecodeError`` variants so callers can branch without
        string-matching on ``UnicodeDecodeError``.

    Args:
        content: Raw bytes from a file upload, clipboard, or HTTP body.
        max_bytes: Hard byte-size cap.  Default 512 KB (~85 000 words in CJK).

    Returns:
        Decoded Unicode string, with surrogate escapes repaired to U+FFFD.

    Raises:
        OversizedInputError: ``len(content) > max_bytes``.
        NULBytesError: ``b'\\x00'`` present in ``content``.
        DecodeError: UTF-8 (and any future fallback) failed.
    """
    if len(content) > max_bytes:  # Gate 1: reject input that exceeds the byte budget.
        raise OversizedInputError(  # Fail fast — no decoding attempted on oversized payloads.
            f"input is {len(content)} bytes, exceeds limit of {max_bytes} bytes"  # Report actual vs limit.
        )

    if b"\x00" in content:  # Gate 2: detect NUL bytes — strong binary-file indicator.
        raise NULBytesError("input contains NUL bytes — likely binary, not recipe text")  # Reject before decode.

    try:  # Attempt strict UTF-8 decoding; any invalid byte sequence will raise UnicodeDecodeError.
        return content.decode("utf-8", errors="strict")  # Strict mode: no silent replacement of invalid bytes.
    except UnicodeDecodeError as exc:  # Catch invalid UTF-8 and wrap it in our domain exception.
        raise DecodeError(  # Wrap so callers can catch PreprocessError uniformly.
            f"UTF-8 decode failed at byte position {exc.start}"  # Include byte offset for diagnostics.
        ) from exc  # Chain the original exception for full traceback.


# ---------------------------------------------------------------------------
# 4.2.2  Line-ending normalisation
# ---------------------------------------------------------------------------


def normalise_line_endings(text: str) -> str:  # Public API — normalise all line endings to Unix LF.
    """Replace all common line-break conventions with a single ``\\n``.

    Covers:
      - Windows ``\\r\\n``
      - Legacy Mac ``\\r``
      - Unicode line / paragraph separators (U+2028, U+2029)

    Why: mixed line endings from copy-paste / different OS sources would
    cause blank-line collapse and step-splitting heuristics to misbehave.

    Args:
        text: Any recipe source string.

    Returns:
        Text with all line breaks normalised to ``\\n``.
    """
    # Order matters: \r\n must be handled before lone \r
    text = text.replace("\r\n", "\n")  # Step 1: collapse Windows CRLF → single LF (must precede CR step).
    text = text.replace("\r", "\n")  # Step 2: collapse legacy Mac CR → LF (safe now that CRLF is gone).
    # Unicode line / paragraph separators → \n
    text = text.replace("\u2028", "\n")  # Step 3: Unicode LINE SEPARATOR (U+2028) → LF (inline break).
    text = text.replace("\u2029", "\n\n")  # Step 4: Unicode PARAGRAPH SEPARATOR (U+2029) → two LFs (block break).
    return text  # Return the fully normalised string — all line endings are now \n.


# ---------------------------------------------------------------------------
# 4.2.3  Control-character removal
# ---------------------------------------------------------------------------


# Control characters to KEEP — they carry semantic meaning in recipes:
#   \n    line breaks (step boundaries, ingredient lines)
#   \t    horizontal tab (some sources use tab-separated ingredient columns)
_KEEP_CONTROL_CHARS = frozenset({"\n", "\t"})  # Immutable, hashable set — fast membership test for O(1) lookups.


def remove_control_characters(text: str) -> str:  # Public API — strip noise control characters, keep structural ones.
    """Strip control characters except those with structural value.

    *Removes*: ASCII C0 controls (except ``\\n``, ``\\t``), C1 controls,
    zero-width spaces, and Unicode ``Cc`` category chars that are not in the
    keep-set.

    *Preserves*:
      - ``\\n``, ``\\t`` — structural whitespace
      - fractions (½, ¼, ¾), degree symbol °, bullet points
      - all printable CJK, Latin, punctuation, and mathematical symbols
      - user warnings ("do not boil", "keep separate") — text-only filtering

    This is purely character-level; it does **not** interpret or rewrite
    recipe instructions.  Prompt-injection-style text is treated as data.

    Args:
        text: Normalised recipe string.

    Returns:
        Text with noise control characters replaced by `U+FFFD` (�).
    """
    result: list[str] = []  # Accumulate cleaned characters into a list (more efficient than repeated str concat).
    for ch in text:  # Iterate over every character in the input string.
        cat = unicodedata.category(ch)  # Query Unicode general category for this character (e.g. "Cc", "Lu").
        if cat == "Cc" and ch not in _KEEP_CONTROL_CHARS:  # "Cc" = Control; only strip if NOT in the keep-set.
            result.append("\ufffd")  # Replace stripped control char with U+FFFD (REPLACEMENT CHARACTER �).
        else:  # Non-control character OR a control char we explicitly keep (\n, \t).
            result.append(ch)  # Preserve the original character verbatim.
    return "".join(result)  # Join the cleaned character list back into a single string.


# ---------------------------------------------------------------------------
# 4.2.4  Blank-line collapse
# ---------------------------------------------------------------------------


def collapse_blank_lines(text: str) -> str:  # Public API — collapse 3+ consecutive blank lines down to 2.
    """Collapse runs of 3+ consecutive blank lines into exactly 2.

    Two blank lines are the maximum meaningful separator in recipe text
    (e.g. between ingredient block and step block).  More than two is
    almost certainly copy-paste noise.

    This is a deterministic, line-by-line pass — no regex engine involved.

    Args:
        text: Normalised, control-cleaned recipe string.

    Returns:
        Text with at most 2 consecutive blank lines between any two
        non-empty lines.
    """
    lines = text.split("\n")  # Split the full text into individual lines (preserves empty list entries).
    collapsed: list[str] = []  # Build the output line-by-line, skipping excess blanks.
    blank_run = 0  # Running counter of consecutive blank lines seen so far.

    for line in lines:  # Process each line in order to preserve original sequence.
        stripped = line.strip()  # Strip whitespace to test if this line is semantically blank.
        if stripped == "":  # This line is blank (contains no visible content).
            blank_run += 1  # Increment the consecutive-blank counter.
            if blank_run <= 2:  # Keep the blank line only if we haven't exceeded the cap of 2.
                collapsed.append(line)  # Preserve this blank line in the output.
        else:  # This line has visible content — it's a non-blank line.
            blank_run = 0  # Reset the blank counter (gap between content blocks).
            collapsed.append(line)  # Always preserve non-blank lines.

    return "\n".join(collapsed)  # Rejoin the filtered lines with LF separators.


# ---------------------------------------------------------------------------
# 4.2.5  Language detection
# ---------------------------------------------------------------------------


def detect_supported_language(text: str) -> str:  # Public API — heuristic CJK vs Latin language detection.
    """Heuristically detect whether the recipe is predominantly Chinese or English.

    Strategy: count characters in the CJK Unified Ideographs block
    (U+4E00–U+9FFF) and compare against ASCII-letter count.  Return the
    language tag that covers > 50 % of the meaningful character budget.

    If neither dominates (> 10 meaningful chars but no clear majority),
    return ``"mixed"``.

    If the text is effectively empty of CJK + Latin, return ``"und"``
    (undetermined).

    This is intentionally simple — we do NOT pull in ``langdetect`` or a
    heavy ML classifier for a preprocessing step.

    Args:
        text: Cleaned recipe string (after control removal and blank collapse).

    Returns:
        ISO 639-3 language code: ``"zho"``, ``"eng"``, ``"mixed"``, or ``"und"``.
    """
    cjk = 0  # Counter for characters in the CJK Unified Ideographs block (U+4E00–U+9FFF).
    latin = 0  # Counter for ASCII alphabetic characters (A–Z, a–z).
    for ch in text:  # Scan every character in the cleaned recipe text.
        cp = ord(ch)  # Get the Unicode code point (integer) for efficient range comparison.
        if 0x4E00 <= cp <= 0x9FFF:  # Character falls within the CJK Unified Ideographs range.
            cjk += 1  # Count as a CJK ideograph.
        elif ch.isascii() and ch.isalpha():  # Character is an ASCII letter (not digit, not punctuation).
            latin += 1  # Count as a Latin/English alphabetic character.

    meaningful = cjk + latin  # Total "meaningful" characters — only CJK and Latin contribute to language signal.
    if meaningful == 0:  # No script characters at all (pure numbers, symbols, or empty text).
        return "und"  # Undetermined — insufficient data to classify.

    cjk_ratio = cjk / meaningful  # Proportion of CJK characters in the meaningful character budget.
    if cjk_ratio > 0.5:  # CJK characters make up more than half of meaningful content.
        return "zho"  # Predominantly Chinese — return ISO 639-3 code for Chinese.
    if latin / meaningful > 0.5:  # Latin characters make up more than half of meaningful content.
        return "eng"  # Predominantly English — return ISO 639-3 code for English.
    return "mixed"  # Neither script exceeds 50% — return "mixed" for bilingual content.


# ---------------------------------------------------------------------------
# 4.2.6  Size validation
# ---------------------------------------------------------------------------


def validate_recipe_text_size(text: str, max_chars: int = 200_000) -> None:  # Public API — size gate check.
    """Reject recipe text that is empty or exceeds the character budget.

    Called *after* noise removal so that blank-only input is caught as empty.

    Raises:
        EmptyContentError: ``text`` is empty or contains only whitespace.
        OversizedInputError: ``len(text) > max_chars``.
    """
    if not text.strip():  # Check 1: strip all whitespace; if nothing remains, the text is effectively empty.
        raise EmptyContentError("recipe text is empty or whitespace-only")  # Reject empty/blank input.
    if len(text) > max_chars:  # Check 2: measure total character count against the configured budget.
        raise OversizedInputError(  # Fail fast — do not pass oversized text downstream to LLM extraction.
            f"recipe text is {len(text)} chars, exceeds limit of {max_chars}"  # Report actual vs limit.
        )


# ---------------------------------------------------------------------------
# 4.2    Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_recipe_text(  # Convenience pipeline — chains all 6 stages into one call.
    content: bytes,  # Raw recipe bytes (file upload, clipboard paste, HTTP body).
    max_bytes: int = 512_000,  # Byte-level size gate — defaults to 512 KB.
    max_chars: int = 200_000,  # Character-level size gate — defaults to ~100k CJK characters.
) -> tuple[str, str]:  # Returns (cleaned_text, language_code) for downstream LLM extraction.
    """Run the full deterministic preprocessing pipeline on raw recipe bytes.

    Stages (handbook 4.2 flow):
      1. ``decode_txt``     — bytes → Unicode
      2. ``normalise_line_endings``
      3. ``remove_control_characters``
      4. ``collapse_blank_lines``
      5. ``detect_supported_language`` — run before size check to surface language
      6. ``validate_recipe_text_size``

    Returns:
        (cleaned_text, language_code) tuple ready for LLM extraction.

    Raises:
        PreprocessError (or subclass) on any stage failure.
    """
    text = decode_txt(content, max_bytes=max_bytes)  # Stage 1: bytes → Unicode (with safety gates).
    text = normalise_line_endings(text)  # Stage 2: CRLF/CR/U+2028/U+2029 → \n.
    text = remove_control_characters(text)  # Stage 3: strip Cc control chars (keep \n, \t).
    text = collapse_blank_lines(text)  # Stage 4: fold 3+ consecutive blank lines → 2.
    language = detect_supported_language(text)  # Stage 5: classify as "zho", "eng", "mixed", or "und".
    validate_recipe_text_size(text, max_chars=max_chars)  # Stage 6: reject empty or oversized text.
    return text, language  # Return the fully cleaned text and its detected language code.
