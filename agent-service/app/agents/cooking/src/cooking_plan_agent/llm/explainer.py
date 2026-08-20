# =============================================================================
# 调度解释模块（llm/explainer）
# -----------------------------------------------------------------------------
# 用 LLM 把“已求解的调度表”转成自然语言解释（“为什么是这个时间 / 顺序”）。
# 可选增强能力：READY 计划产出后，由本地 LLM 生成一段简短的调度解释
# （如并行烹饪、菜品节奏控制）。这是纯增量能力 —— 不改变 PlanResponse 的 schema。
# 当 LLM 被禁用或调用失败时，调用方回退到确定性摘要（_fallback）。
# =============================================================================

"""LLM-powered schedule explanation — turns a solved schedule into prose.

基于 LLM 的调度解释 —— 把已求解的调度表转成自然语言。

Optional enhancement: after a READY plan is produced, a local LLM generates
a short "why this schedule" explanation (e.g. parallel cooking, dish pacing).
This is an additive capability — it does not alter the PlanResponse schema.
When LLM is disabled or fails, callers fall back to a deterministic summary.

可选增强：READY 计划产出后，由本地 LLM 生成一段简短的“为何这样调度”解释
（如并行烹饪、菜品节奏）。这是增量能力 —— 不改变 PlanResponse 的 schema。
当 LLM 被禁用或失败时，调用方回退到确定性摘要。
"""

from __future__ import annotations

import logging
from typing import Any

from cooking_plan_agent.llm.client import LLMClient, LLMError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful cooking assistant. Explain a meal-plan schedule in 2-3 "
    "short, friendly sentences. Focus on WHY the timing/order makes sense "
    "(parallel cooking, resting time, last dish finishing on time). Respond with "
    'a JSON object only: {"explanation": string}. Use the caller\'s language.'
)
# ↑ 系统提示词：用 2-3 句友好的话解释调度，只输出 {"explanation": string}，并使用调用方语言


class LLMPlanExplainer:
    """为已求解的调度生成自然语言解释。"""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def explain(self, schedule_summary: dict[str, Any]) -> str:
        """返回调度的自然语言解释。

        Return a prose explanation of the schedule.

        Args:
            schedule_summary: Compact schedule facts:
                {"makespan_minutes": int, "dish_completions": [{"dish": str,
                 "completion_minute": int}], "parallel_groups": int}.
                schedule_summary：紧凑的调度事实：
                {"makespan_minutes": int, "dish_completions": [{"dish": str,
                 "completion_minute": int}], "parallel_groups": int}.

        Returns:
            Explanation string. On LLM failure returns a deterministic
            fallback sentence so the response pipeline is never blocked.
            解释字符串。LLM 失败时返回确定性兜底句子，响应管线绝不因此阻塞。
        """
        try:
            data = await self._client.chat_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": self._format(schedule_summary)},
                ]
            )
            explanation = str(data.get("explanation") or "").strip()
            if explanation:
                return explanation
        except LLMError:
            logger.warning("Schedule explanation LLM call failed — using fallback")
        except (KeyError, TypeError):
            logger.warning("Schedule explanation parsing failed — using fallback")
        return self._fallback(schedule_summary)

    # ------------------------------------------------------------------
    # Helpers
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format(summary: dict[str, Any]) -> str:
        """把调度摘要格式化为给 LLM 的可读文本。"""
        makespan = summary.get("makespan_minutes", "?")
        completions = summary.get("dish_completions", [])
        lines = [f"Total time: {makespan} minutes"]
        if completions:
            lines.append(
                "Dish completions: "
                + ", ".join(f"{c.get('dish', '?')} at {c.get('completion_minute', '?')} min" for c in completions)
            )
        if summary.get("parallel_groups"):
            lines.append(f"Parallel workstreams: {summary['parallel_groups']}")
        return "\n".join(lines)

    @staticmethod
    def _fallback(summary: dict[str, Any]) -> str:
        """确定性兜底解释：仅基于 makespan 生成一句话。"""
        makespan = summary.get("makespan_minutes", "?")
        return f"Plan completes in approximately {makespan} minutes."
