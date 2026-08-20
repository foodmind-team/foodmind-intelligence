# =============================================================================
# 预处理异常模块（parsing/errors）
# -----------------------------------------------------------------------------
# 定义菜谱“确定性预处理”阶段的异常层级，用于在 LLM 提取之前，
# 把各类输入失败（编码 / 超限 / 空内容 / 二进制信号）结构化地区分开。
#   - PreprocessError      ：所有预处理失败的基类
#   - DecodeError          ：字节内容无法解码为受支持的文本编码
#   - OversizedInputError  ：输入超过配置的字节 / 字符预算
#   - EmptyContentError    ：去噪后无可用的有效内容（仅空白）
#   - NULBytesError        ：原始字节含 NUL（0x00）—— 二进制信号
# =============================================================================

# Top-level blank line — separates module docstring (none here) from first class.


# Top-level blank line — separates module header area from class definitions.


class PreprocessError(ValueError):  # Base exception for all deterministic preprocessing failures.
    """所有预处理失败的基类。在 LLM 提取之前抛出。

    Base error for all preprocessing failures. Raised before LLM extraction.
    """


class DecodeError(PreprocessError):  # Raised when byte content cannot be decoded as valid UTF-8.
    """字节无法解码为受支持的文本编码。"""


class OversizedInputError(PreprocessError):  # Raised when input exceeds byte or character budget.
    """输入超过配置的字节或字符上限。"""


class EmptyContentError(PreprocessError):  # Raised when decoded text contains only whitespace.
    """解码文本在去噪后不含任何可用内容。"""


# Blank line — separates EmptyContentError from NULBytesError for readability.


class NULBytesError(PreprocessError):  # Raised when raw bytes contain NUL (0x00) — binary signal.
    """看起来像二进制输入 —— 原始内容中检测到 NUL 字节。"""
