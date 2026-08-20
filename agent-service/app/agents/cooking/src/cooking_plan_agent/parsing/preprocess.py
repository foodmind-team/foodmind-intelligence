# =============================================================================
# 预处理流水线模块（parsing/preprocess）
# -----------------------------------------------------------------------------
# 实现手册 4.2 的“确定性预处理”流水线，把原始菜谱字节清洗成干净的 Unicode 文本，
# 供下游 LLM / 规则提取使用。六个阶段：
#   1. decode_txt                 ：字节 → Unicode（含超限 / NUL 字节安全闸门）
#   2. normalise_line_endings     ：统一行尾为 \n（CRLF / CR / U+2028 / U+2029）
#   3. remove_control_characters  ：剥离噪声控制字符（保留 \n、\t）
#   4. collapse_blank_lines       ：把 3+ 连续空行折叠为 2
#   5. detect_supported_language  ：启发式检测语言（zho / eng / mixed / und）
#   6. validate_recipe_text_size  ：拒绝空文本或超限文本
# 设计：确定性、无 I/O、无 LLM 依赖 —— 每一步都可独立测试。
# =============================================================================

import unicodedata  # Python stdlib — provides Unicode character category classification.
# ↑ Python 标准库 —— 提供 Unicode 字符类别分类

from cooking_plan_agent.parsing.errors import (  # Local error hierarchy for preprocessing stage.
    DecodeError,  # Bytes could not be decoded (invalid UTF-8).
    EmptyContentError,  # Text is empty or whitespace-only after cleaning.
    NULBytesError,  # Raw content contains NUL (0x00) — binary file signal.
    OversizedInputError,  # Content exceeds configured byte or character limit.
)
# ↑ 预处理阶段的本地异常层级（DecodeError / EmptyContentError / NULBytesError / OversizedInputError）

# ---------------------------------------------------------------------------
# 4.2.1  Byte → text decoding
# 4.2.1  字节 → 文本解码
# ---------------------------------------------------------------------------


def decode_txt(content: bytes, max_bytes: int = 512_000) -> str:  # Public API — decode raw bytes with safety gates.
    """把原始菜谱字节解码为干净的 Unicode 字符串。

    Decode raw recipe bytes into a clean Unicode string.

    Accepts UTF-8 first (dominant encoding for recipe sources).  If a second
    encoding is added later, document it explicitly in this docstring.

    优先接受 UTF-8（菜谱来源的主流编码）。若后续加入第二种编码，须在本 docstring 中显式说明。

    Rules (handbook 4.2):
      - Reject oversized input before decoding (defence against binary blobs).
      - Reject NUL bytes — strong signal of binary / non-text content.
      - Prefer explicit ``DecodeError`` variants so callers can branch without
        string-matching on ``UnicodeDecodeError``.

    规则（手册 4.2）：
      - 解码前拒绝超限输入（防御二进制大块）。
      - 拒绝 NUL 字节 —— 二进制 / 非文本内容的强信号。
      - 优先使用显式 DecodeError 变体，使调用方无需对 UnicodeDecodeError 做字符串匹配即可分支。

    Args:
        content: Raw bytes from a file upload, clipboard, or HTTP body.
            content：来自文件上传、剪贴板或 HTTP 请求体的原始字节。
        max_bytes: Hard byte-size cap.  Default 512 KB (~85 000 words in CJK).
            max_bytes：硬性字节上限。默认 512 KB（约 85000 个中文字）。

    Returns:
        Decoded Unicode string, with surrogate escapes repaired to U+FFFD.
        解码后的 Unicode 字符串，代理项转义修复为 U+FFFD。

    Raises:
        OversizedInputError: ``len(content) > max_bytes``.
        NULBytesError: ``b'\\x00'`` present in ``content``.
        DecodeError: UTF-8 (and any future fallback) failed.
    """
    if len(content) > max_bytes:  # Gate 1: reject input that exceeds the byte budget.
        # ↑ 闸门 1：拒绝超过字节预算的输入
        raise OversizedInputError(  # Fail fast — no decoding attempted on oversized payloads.
            # ↑ 快速失败 —— 对超限负载不做任何解码尝试
            f"input is {len(content)} bytes, exceeds limit of {max_bytes} bytes"  # Report actual vs limit.
        )

    if b"\x00" in content:  # Gate 2: detect NUL bytes — strong binary-file indicator.
        # ↑ 闸门 2：检测 NUL 字节 —— 强二进制文件指示
        raise NULBytesError("input contains NUL bytes — likely binary, not recipe text")  # Reject before decode.

    try:  # Attempt strict UTF-8 decoding; any invalid byte sequence will raise UnicodeDecodeError.
        # ↑ 尝试严格 UTF-8 解码；任何非法字节序列都会抛 UnicodeDecodeError
        return content.decode("utf-8", errors="strict")  # Strict mode: no silent replacement of invalid bytes.
        # ↑ 严格模式：不静默替换非法字节
    except UnicodeDecodeError as exc:  # Catch invalid UTF-8 and wrap it in our domain exception.
        # ↑ 捕获非法 UTF-8 并包装成我们的领域异常
        raise DecodeError(  # Wrap so callers can catch PreprocessError uniformly.
            # ↑ 包装使调用方能统一捕获 PreprocessError
            f"UTF-8 decode failed at byte position {exc.start}"  # Include byte offset for diagnostics.
            # ↑ 附带字节偏移以便诊断
        ) from exc  # Chain the original exception for full traceback.


# ---------------------------------------------------------------------------
# 4.2.2  Line-ending normalisation
# 4.2.2  行尾规范化
# ---------------------------------------------------------------------------


def normalise_line_endings(text: str) -> str:  # Public API — normalise all line endings to Unix LF.
    """把所有常见换行约定统一为单个 ``\\n``。

    Replace all common line-break conventions with a single ``\\n``.

    Covers:
      - Windows ``\\r\\n``
      - Legacy Mac ``\\r``
      - Unicode line / paragraph separators (U+2028, U+2029)

    覆盖：
      - Windows ``\\r\\n``
      - 旧版 Mac ``\\r``
      - Unicode 行 / 段落分隔符（U+2028、U+2029）

    Why: mixed line endings from copy-paste / different OS sources would
    cause blank-line collapse and step-splitting heuristics to misbehave.

    原因：来自复制粘贴 / 不同操作系统来源的混合行尾会导致空行折叠与步骤切分启发式异常。

    Args:
        text: Any recipe source string.
            text：任意菜谱来源字符串。

    Returns:
        Text with all line breaks normalised to ``\\n``.
        所有换行都规范化为 ``\\n`` 的文本。
    """
    # Order matters: \r\n must be handled before lone \r
    # 顺序很重要：\r\n 必须在单独的 \r 之前处理
    text = text.replace("\r\n", "\n")  # Step 1: collapse Windows CRLF → single LF (must precede CR step).
    # ↑ 第 1 步：折叠 Windows CRLF → 单个 LF（必须先于 CR 步骤）
    text = text.replace("\r", "\n")  # Step 2: collapse legacy Mac CR → LF (safe now that CRLF is gone).
    # ↑ 第 2 步：折叠旧版 Mac CR → LF（CRLF 已处理后安全）
    # Unicode line / paragraph separators → \n
    # Unicode 行 / 段落分隔符 → \n
    text = text.replace("\u2028", "\n")  # Step 3: Unicode LINE SEPARATOR (U+2028) → LF (inline break).
    # ↑ 第 3 步：Unicode 行分隔符（U+2028）→ LF（行内断行）
    text = text.replace("\u2029", "\n\n")  # Step 4: Unicode PARAGRAPH SEPARATOR (U+2029) → two LFs (block break).
    # ↑ 第 4 步：Unicode 段落分隔符（U+2029）→ 两个 LF（块断行）
    return text  # Return the fully normalised string — all line endings are now \n.


# ---------------------------------------------------------------------------
# 4.2.3  Control-character removal
# 4.2.3  控制字符移除
# ---------------------------------------------------------------------------


# Control characters to KEEP — they carry semantic meaning in recipes:
#   \n    line breaks (step boundaries, ingredient lines)
#   \t    horizontal tab (some sources use tab-separated ingredient columns)
# 需保留的控制字符 —— 它们在菜谱中承载语义：
#   \n    换行（步骤边界、食材行）
#   \t    水平制表符（部分来源用制表符分隔食材列）
_KEEP_CONTROL_CHARS = frozenset({"\n", "\t"})  # Immutable, hashable set — fast membership test for O(1) lookups.
# ↑ 不可变、可哈希集合 —— O(1) 成员测试


def remove_control_characters(text: str) -> str:  # Public API — strip noise control characters, keep structural ones.
    """剥离控制字符，但保留有结构价值者。

    Strip control characters except those with structural value.

    *Removes*: ASCII C0 controls (except ``\\n``, ``\\t``), C1 controls,
    zero-width spaces, and Unicode ``Cc`` category chars that are not in the
    keep-set.

    *移除*：ASCII C0 控制（除 ``\\n``、``\\t``）、C1 控制、零宽空格、以及不在保留集中的 Unicode ``Cc`` 类字符。

    *Preserves*:
      - ``\\n``, ``\\t`` — structural whitespace
      - fractions (½, ¼, ¾), degree symbol °, bullet points
      - all printable CJK, Latin, punctuation, and mathematical symbols
      - user warnings ("do not boil", "keep separate") — text-only filtering

    *保留*：
      - ``\\n``、``\\t`` —— 结构空白
      - 分数（½、¼、¾）、度符号 °、项目符号
      - 所有可打印 CJK、拉丁、标点与数学符号
      - 用户警告（"do not boil"、"keep separate"）—— 仅做文本过滤

    This is purely character-level; it does **not** interpret or rewrite
    recipe instructions.  Prompt-injection-style text is treated as data.

    这纯属字符级；它不解释或改写菜谱指令。提示注入式文本被当作数据处理。

    Args:
        text: Normalised recipe string.
            text：规范化后的菜谱字符串。

    Returns:
        Text with noise control characters replaced by `U+FFFD` (�).
        噪声控制字符被替换为 U+FFFD（�）的文本。
    """
    result: list[str] = []  # Accumulate cleaned characters into a list (more efficient than repeated str concat).
    # ↑ 把清洗后的字符累加到列表（比反复字符串拼接高效）
    for ch in text:  # Iterate over every character in the input string.
        # ↑ 遍历输入字符串的每个字符
        cat = unicodedata.category(ch)  # Query Unicode general category for this character (e.g. "Cc", "Lu").
        # ↑ 查询该字符的 Unicode 通用类别（如 "Cc"、"Lu"）
        if cat == "Cc" and ch not in _KEEP_CONTROL_CHARS:  # "Cc" = Control; only strip if NOT in the keep-set.
            # ↑ "Cc" = 控制；仅当不在保留集中才剥离
            result.append("\ufffd")  # Replace stripped control char with U+FFFD (REPLACEMENT CHARACTER �).
            # ↑ 用 U+FFFD（替换字符 �）替换被剥离的控制字符
        else:  # Non-control character OR a control char we explicitly keep (\n, \t).
            # ↑ 非控制字符，或明确保留的控制字符（\n、\t）
            result.append(ch)  # Preserve the original character verbatim.
            # ↑ 原样保留该字符
    return "".join(result)  # Join the cleaned character list back into a single string.


# ---------------------------------------------------------------------------
# 4.2.4  Blank-line collapse
# 4.2.4  空行折叠
# ---------------------------------------------------------------------------


def collapse_blank_lines(text: str) -> str:  # Public API — collapse 3+ consecutive blank lines down to 2.
    """把 3+ 连续空行折叠为恰好 2 行。

    Collapse runs of 3+ consecutive blank lines into exactly 2.

    Two blank lines are the maximum meaningful separator in recipe text
    (e.g. between ingredient block and step block).  More than two is
    almost certainly copy-paste noise.

    两个空行是菜谱文本中最大有意义的间隔（如食材块与步骤块之间）。超过两个几乎肯定是复制粘贴噪声。

    This is a deterministic, line-by-line pass — no regex engine involved.

    这是确定性、逐行的处理 —— 不涉及正则引擎。

    Args:
        text: Normalised, control-cleaned recipe string.
            text：已规范化、已清理控制字符的菜谱字符串。

    Returns:
        Text with at most 2 consecutive blank lines between any two
        non-empty lines.
        任意两个非空行之间最多 2 个连续空行的文本。
    """
    lines = text.split("\n")  # Split the full text into individual lines (preserves empty list entries).
    # ↑ 把全文拆成单行（保留空条目）
    collapsed: list[str] = []  # Build the output line-by-line, skipping excess blanks.
    # ↑ 逐行构建输出，跳过多余空行
    blank_run = 0  # Running counter of consecutive blank lines seen so far.
    # ↑ 当前连续空行计数器

    for line in lines:  # Process each line in order to preserve original sequence.
        # ↑ 按顺序处理每行以保持原始顺序
        stripped = line.strip()  # Strip whitespace to test if this line is semantically blank.
        # ↑ 去空白以判断该行是否语义上为空
        if stripped == "":  # This line is blank (contains no visible content).
            # ↑ 该行为空（无可见内容）
            blank_run += 1  # Increment the consecutive-blank counter.
            if blank_run <= 2:  # Keep the blank line only if we haven't exceeded the cap of 2.
                # ↑ 仅当未超过 2 行上限时保留该空行
                collapsed.append(line)  # Preserve this blank line in the output.
        else:  # This line has visible content — it's a non-blank line.
            # ↑ 该行有可见内容 —— 非空行
            blank_run = 0  # Reset the blank counter (gap between content blocks).
            # ↑ 重置空行计数器（内容块之间的间隔）
            collapsed.append(line)  # Always preserve non-blank lines.

    return "\n".join(collapsed)  # Rejoin the filtered lines with LF separators.


# ---------------------------------------------------------------------------
# 4.2.5  Language detection
# 4.2.5  语言检测
# ---------------------------------------------------------------------------


def detect_supported_language(text: str) -> str:  # Public API — heuristic CJK vs Latin language detection.
    """启发式检测菜谱主要是中文还是英文。

    Heuristically detect whether the recipe is predominantly Chinese or English.

    Strategy: count characters in the CJK Unified Ideographs block
    (U+4E00–U+9FFF) and compare against ASCII-letter count.  Return the
    language tag that covers > 50 % of the meaningful character budget.

    策略：统计 CJK 统一表意文字块（U+4E00–U+9FFF）字符数，并与 ASCII 字母数比较。
    返回占“有意义字符预算”超过 50% 的语言标签。

    If neither dominates (> 10 meaningful chars but no clear majority),
    return ``"mixed"``.

    若两者都不占主导（有意义字符 > 10 但无明确多数），返回 ``"mixed"``。

    If the text is effectively empty of CJK + Latin, return ``"und"``
    (undetermined).

    若文本几乎没有 CJK + 拉丁字符，返回 ``"und"``（未确定）。

    This is intentionally simple — we do NOT pull in ``langdetect`` or a
    heavy ML classifier for a preprocessing step.

    这是刻意保持简单 —— 预处理步骤不引入 langdetect 或重型 ML 分类器。

    Args:
        text: Cleaned recipe string (after control removal and blank collapse).
            text：已清理的菜谱字符串（控制字符移除与空行折叠之后）。

    Returns:
        ISO 639-3 language code: ``"zho"``, ``"eng"``, ``"mixed"``, or ``"und"``.
        ISO 639-3 语言码：``"zho"``、``"eng"``、``"mixed"`` 或 ``"und"``。
    """
    cjk = 0  # Counter for characters in the CJK Unified Ideographs block (U+4E00–U+9FFF).
    # ↑ CJK 统一表意文字块（U+4E00–U+9FFF）字符计数器
    latin = 0  # Counter for ASCII alphabetic characters (A–Z, a–z).
    # ↑ ASCII 字母字符（A–Z、a–z）计数器
    for ch in text:  # Scan every character in the cleaned recipe text.
        cp = ord(ch)  # Get the Unicode code point (integer) for efficient range comparison.
        # ↑ 获取 Unicode 码点（整数）以便高效范围比较
        if 0x4E00 <= cp <= 0x9FFF:  # Character falls within the CJK Unified Ideographs range.
            # ↑ 字符落在 CJK 统一表意文字范围内
            cjk += 1  # Count as a CJK ideograph.
        elif ch.isascii() and ch.isalpha():  # Character is an ASCII letter (not digit, not punctuation).
            # ↑ 字符是 ASCII 字母（非数字、非标点）
            latin += 1  # Count as a Latin/English alphabetic character.

    meaningful = cjk + latin  # Total "meaningful" characters — only CJK and Latin contribute to language signal.
    # ↑ “有意义”字符总数 —— 只有 CJK 与拉丁贡献语言信号
    if meaningful == 0:  # No script characters at all (pure numbers, symbols, or empty text).
        # ↑ 完全无文字字符（纯数字、符号或空文本）
        return "und"  # Undetermined — insufficient data to classify.

    cjk_ratio = cjk / meaningful  # Proportion of CJK characters in the meaningful character budget.
    # ↑ CJK 字符在有意义字符预算中的占比
    if cjk_ratio > 0.5:  # CJK characters make up more than half of meaningful content.
        # ↑ CJK 字符超过有意义内容的一半
        return "zho"  # Predominantly Chinese — return ISO 639-3 code for Chinese.
    if latin / meaningful > 0.5:  # Latin characters make up more than half of meaningful content.
        # ↑ 拉丁字符超过有意义内容的一半
        return "eng"  # Predominantly English — return ISO 639-3 code for English.
    return "mixed"  # Neither script exceeds 50% — return "mixed" for bilingual content.
    # ↑ 两种文字都不超过 50% —— 对双语内容返回 "mixed"


# ---------------------------------------------------------------------------
# 4.2.6  Size validation
# 4.2.6  大小校验
# ---------------------------------------------------------------------------


def validate_recipe_text_size(text: str, max_chars: int = 200_000) -> None:  # Public API — size gate check.
    """拒绝为空或超过字符预算的菜谱文本。

    Reject recipe text that is empty or exceeds the character budget.

    Called *after* noise removal so that blank-only input is caught as empty.

    在噪声移除之后调用，使纯空白输入被识别为空。

    Raises:
        EmptyContentError: ``text`` is empty or contains only whitespace.
        OversizedInputError: ``len(text) > max_chars``.
    """
    if not text.strip():  # Check 1: strip all whitespace; if nothing remains, the text is effectively empty.
        # ↑ 检查 1：去所有空白；若什么都不剩，文本实际上为空
        raise EmptyContentError("recipe text is empty or whitespace-only")  # Reject empty/blank input.
    if len(text) > max_chars:  # Check 2: measure total character count against the configured budget.
        # ↑ 检查 2：把总字符数与配置预算比较
        raise OversizedInputError(  # Fail fast — do not pass oversized text downstream to LLM extraction.
            # ↑ 快速失败 —— 不把超限文本传给下游 LLM 提取
            f"recipe text is {len(text)} chars, exceeds limit of {max_chars}"  # Report actual vs limit.
        )


# ---------------------------------------------------------------------------
# 4.2    Full preprocessing pipeline
# 4.2    完整预处理流水线
# ---------------------------------------------------------------------------


def preprocess_recipe_text(  # Convenience pipeline — chains all 6 stages into one call.
    content: bytes,  # Raw recipe bytes (file upload, clipboard paste, HTTP body).
    max_bytes: int = 512_000,  # Byte-level size gate — defaults to 512 KB.
    max_chars: int = 200_000,  # Character-level size gate — defaults to ~100k CJK characters.
) -> tuple[str, str]:  # Returns (cleaned_text, language_code) for downstream LLM extraction.
    """对原始菜谱字节运行完整的确定性预处理流水线。

    Run the full deterministic preprocessing pipeline on raw recipe bytes.

    Stages (handbook 4.2 flow):
      1. ``decode_txt``     — bytes → Unicode
      2. ``normalise_line_endings``
      3. ``remove_control_characters``
      4. ``collapse_blank_lines``
      5. ``detect_supported_language`` — run before size check to surface language
      6. ``validate_recipe_text_size``

    阶段（手册 4.2 流程）：
      1. decode_txt —— 字节 → Unicode
      2. normalise_line_endings
      3. remove_control_characters
      4. collapse_blank_lines
      5. detect_supported_language —— 在大小检查前运行，先给出语言
      6. validate_recipe_text_size

    Returns:
        (cleaned_text, language_code) tuple ready for LLM extraction.
        供 LLM 提取使用的 (cleaned_text, language_code) 元组。

    Raises:
        PreprocessError (or subclass) on any stage failure.
        任一阶段失败时抛 PreprocessError（或其子类）。
    """
    text = decode_txt(content, max_bytes=max_bytes)  # Stage 1: bytes → Unicode (with safety gates).
    # ↑ 阶段 1：字节 → Unicode（含安全闸门）
    text = normalise_line_endings(text)  # Stage 2: CRLF/CR/U+2028/U+2029 → \n.
    # ↑ 阶段 2：CRLF/CR/U+2028/U+2029 → \n
    text = remove_control_characters(text)  # Stage 3: strip Cc control chars (keep \n, \t).
    # ↑ 阶段 3：剥离 Cc 控制字符（保留 \n、\t）
    text = collapse_blank_lines(text)  # Stage 4: fold 3+ consecutive blank lines → 2.
    # ↑ 阶段 4：把 3+ 连续空行折叠为 2
    language = detect_supported_language(text)  # Stage 5: classify as "zho", "eng", "mixed", or "und".
    # ↑ 阶段 5：分类为 "zho"、"eng"、"mixed" 或 "und"
    validate_recipe_text_size(text, max_chars=max_chars)  # Stage 6: reject empty or oversized text.
    # ↑ 阶段 6：拒绝空或超限文本
    return text, language  # Return the fully cleaned text and its detected language code.
    # ↑ 返回完全清洗后的文本及其检测到的语言码
