"""LLM-powered schedule explanation — turns a solved schedule into prose.

Optional enhancement: after a READY plan is produced, a local LLM generates
a short "why this schedule" explanation (e.g. parallel cooking, dish pacing).
This is an additive capability — it does not alter the PlanResponse schema.
When LLM is disabled or fails, callers fall back to a deterministic summary.
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


class LLMPlanExplainer:
    """Generate a natural-language explanation for a solved schedule."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def explain(self, schedule_summary: dict[str, Any]) -> str:
        """Return a prose explanation of the schedule.

        Args:
            schedule_summary: Compact schedule facts:
                {"makespan_minutes": int, "dish_completions": [{"dish": str,
                 "completion_minute": int}], "parallel_groups": int}.

        Returns:
            Explanation string. On LLM failure returns a deterministic
            fallback sentence so the response pipeline is never blocked.
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
    # ------------------------------------------------------------------

    @staticmethod
    def _format(summary: dict[str, Any]) -> str:
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
        makespan = summary.get("makespan_minutes", "?")
        return f"Plan completes in approximately {makespan} minutes."
