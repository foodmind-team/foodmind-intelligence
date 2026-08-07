"""P5-2: LLMReActController —— 基于 tool-calling 的真实 LLM 控制器。

实现 ``AgentController`` Protocol：把 LLM 的 ``tool_calls`` / 文本响应
翻译为节点可消费的结构化决策：
  {"type": "tool_call", "tool": str, "arguments": dict}
  {"type": "final", "response": dict}
  {"type": "fallback"}

安全红线（P5-2）：
  - LLM 只做"软决策"（选择工具/判断完成），工具内部仍由确定性服务
    执行；最终动作经 controller_nodes._apply_decision 白名单校验；
  - 任何失败路径（LLM 异常 / 非法 JSON / 空输出 / 未知工具名）都收敛
    为 fallback，由节点回退确定性 DAG，绝不静默放行；
  - 传给 LLM 的状态摘要是紧凑、非敏感的（D4）：只含 request_id /
    步数 / 工具调用计数，不携带菜谱原文、库存或用户身份。
"""

from __future__ import annotations

import json
import logging

from cooking_plan_agent.llm.client import LLMClient, LLMError
from cooking_plan_agent.tooling.schemas import RegisteredTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the orchestrating controller of a cooking-plan generation agent. "
    "You may call the provided tools to gather facts or perform steps. "
    "Your decisions must follow EXACTLY one of these JSON shapes and nothing else:\n"
    '1. {"type": "tool_call", "tool": "<tool name>", "arguments": {<valid JSON args>}}\n'
    '2. {"type": "final", "response": {"status": "READY"}}  (only when the task is complete)\n'
    '3. {"type": "fallback"}  (when you cannot safely continue)\n'
    "Never invent tool names. Never call a tool with arguments outside its schema. "
    "Never emit anything other than the three JSON shapes above."
)


def _to_openai_tool(tool: RegisteredTool) -> dict[str, object]:
    """把 RegisteredTool 转换为 OpenAI tools 参数格式（P5-1）。"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_decision(text: str | None, calls: object) -> dict[str, object]:
    """把 LLM 响应翻译为结构化决策（统一收敛为三种合法 type）。"""
    # 优先 tool_calls：工具调用是强结构化信号。
    if isinstance(calls, tuple) and calls:
        call = calls[0]
        name = getattr(call, "name", "")
        arguments = getattr(call, "arguments", {})
        if isinstance(name, str) and isinstance(arguments, dict):
            return {"type": "tool_call", "tool": name, "arguments": dict(arguments)}
        return {"type": "fallback"}

    # 无工具调用 → 尝试解析文本中的 final JSON。
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM controller returned non-JSON text | snippet=%s", text[:80])
            return {"type": "fallback"}
        if isinstance(payload, dict):
            # 显式 final 决策，或 LLM 直接给出完成 JSON（无工具调用即视为完成）。
            if payload.get("type") == "final":
                response = payload.get("response")
                if isinstance(response, dict):
                    return {"type": "final", "response": response}
            if "type" not in payload and payload:
                return {"type": "final", "response": payload}
        # JSON 但形状不符 → 保守降级。
        return {"type": "fallback"}

    # 空输出。
    return {"type": "fallback"}


class LLMReActController:
    """AgentController 的 LLM 实现 —— 通过 chat_with_tools 决定下一步动作。"""

    def __init__(
        self,
        client: LLMClient,
        tools: tuple[RegisteredTool, ...] = (),
    ) -> None:
        self._client = client
        self._tools = tools

    async def decide(self, state_summary: dict[str, object]) -> dict[str, object]:
        """基于状态摘要调用 LLM 并返回结构化决策（P5-2）。

        任何异常（网络 / provider / 解析）都被捕获并收敛为 fallback ——
        控制器失败由 controller_nodes 回退确定性 DAG，不中断主流程。
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(state_summary, ensure_ascii=False)},
        ]
        tool_specs = [_to_openai_tool(tool) for tool in self._tools]

        try:
            text, calls = await self._client.chat_with_tools(messages, tool_specs)
        except (LLMError, KeyError, TypeError) as exc:
            logger.warning(
                "LLM controller call failed | exception_type=%s",
                type(exc).__name__,
            )
            return {"type": "fallback"}

        return _parse_decision(text, calls)
