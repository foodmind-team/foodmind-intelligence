# =============================================================================
# 安全结构化日志配置模块（observability/logging）
# -----------------------------------------------------------------------------
# 提供“安全的结构化日志”配置，核心职责：
#   - RedactingJsonFormatter      ：JSON 日志格式化器，对全部内容做脱敏（手册 12.7）
#   - configure_structured_logging：配置根 logger，生产环境用 JSON 格式
# 安全要点（P4-03 / 补 P2-05）：
#   - 对所有日志内容做“递归脱敏”，覆盖消息文本、异常字符串、URL、嵌套对象；
#   - 诊断字段（request_id / node / error_code / correlation_id / duration_ms /
#     status / level / logger / timestamp）经白名单保留；
#   - 每行日志携带脱敏计数器（redaction_applied / redaction_failed），
#     使“脱敏失败即关闭（fail-closed）”可观测（P2-05 监控矩阵）。
# =============================================================================

"""Safe structured logging configuration for the application.

应用的安全结构化日志配置。
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime

from cooking_plan_agent.observability.redaction import (
    REDACTION_APPLIED_FIELD,
    REDACTION_FAIL_FIELD,
    redact_with_stats,
)


class RedactingJsonFormatter(logging.Formatter):
    """JSON 日志格式化器：按手册 12.7 对全部内容脱敏。

    JSON log formatter that redacts ALL content per Handbook 12.7.

    P4-03 (补 P2-05): upgraded from "redact only by extra field name" to a
    uniform recursive redactor (observability/redaction.py) covering the
    message text, exception strings, URLs and nested objects. Diagnostic
    fields (request_id / node / error_code / correlation_id / duration_ms
    / status / level / logger / timestamp) survive via the whitelist, and
    every line carries redaction counters for the P2-05 monitoring matrix.

    P4-03（补 P2-05）：从“仅按 extra 字段名脱敏”升级为统一的递归脱敏器
    （observability/redaction.py），覆盖消息文本、异常字符串、URL 与嵌套对象。
    诊断字段（request_id / node / error_code / correlation_id / duration_ms /
    status / level / logger / timestamp）经白名单保留，每行日志携带脱敏计数器，
    用于 P2-05 监控矩阵。
    """

    # Standard LogRecord attributes that are never echoed into the JSON line.
    # 标准 LogRecord 属性：绝不回显到 JSON 行中。
    _LOG_ATTR_SKIP = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "process",
            "processName",
            "taskName",
            "message",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        message, msg_stats = redact_with_stats(record.getMessage())
        log_entry: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # P2-05: exception keeps only the safe type name + cleaned summary —
        # the raw traceback is never written, so it cannot leak secrets.
        # P2-05：异常只保留安全的类型名 + 清洗后的摘要 —— 原始 traceback 绝不写入，
        # 因此不会泄漏密钥。
        applied = msg_stats.applied
        failed = msg_stats.failed
        if record.exc_info and record.exc_info[1]:
            exc = record.exc_info[1]
            exc_summary, exc_stats = redact_with_stats(str(exc))
            log_entry["exception"] = f"{type(exc).__name__}: {exc_summary}"
            applied += exc_stats.applied
            failed += exc_stats.failed

        # Every extra attribute is recursively redacted; the whitelist keeps
        # diagnostic fields and the redactor hides everything else.
        # 每个 extra 属性都被递归脱敏；白名单保留诊断字段，脱敏器隐藏其余一切。
        for key, value in record.__dict__.items():
            if key in self._LOG_ATTR_SKIP:
                continue
            redacted, stats = redact_with_stats(value)
            log_entry[key] = redacted
            applied += stats.applied
            failed += stats.failed

        # P2-05 monitoring matrix: fail-closed redaction is observable.
        # P2-05 监控矩阵：使“脱敏失败即关闭”可观测。
        log_entry[REDACTION_APPLIED_FIELD] = applied
        log_entry[REDACTION_FAIL_FIELD] = failed

        return json.dumps(log_entry, default=str)


def configure_structured_logging() -> None:
    """为生产环境把根 logger 配置为 JSON 格式化。

    Configure the root logger with JSON formatting for production.

    Handbook 12.7: structured logs with request_id, node, duration_ms, etc.
    In production, set COOKING_PLAN_LOG_FORMAT=json to enable.

    手册 12.7：带 request_id、node、duration_ms 等的结构化日志。
    生产环境设置 COOKING_PLAN_LOG_FORMAT=json 启用。
    """
    log_format = os.environ.get("COOKING_PLAN_LOG_FORMAT", "text")
    if log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(RedactingJsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(os.environ.get("COOKING_PLAN_LOG_LEVEL", "INFO").upper())
