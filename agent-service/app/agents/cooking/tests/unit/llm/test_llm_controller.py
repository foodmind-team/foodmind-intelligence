"""P5-2: LLMReActController —— 真实 LLM 控制器（AgentController 实现）。

验证：把 LLM 的 tool_calls / 文本响应翻译为节点可消费的结构化决策；
异常 / 无有效输出统一降级为 fallback（绝不静默放行）。
"""

import pytest

from cooking_plan_agent.llm.client import LLMClient, ToolCall
from cooking_plan_agent.llm.controller import LLMReActController
from cooking_plan_agent.tooling.schemas import RegisteredTool


def _client() -> LLMClient:
    return LLMClient(base_url="http://llm.test", model="m")


def _tool() -> RegisteredTool:
    async def _execute(arguments: dict) -> dict:
        return {}

    return RegisteredTool(
        name="parse_recipe",
        description="Parse a recipe.",
        parameters={"type": "object", "properties": {"source_text": {"type": "string"}}, "required": ["source_text"]},
        executor=_execute,
    )


def _controller(client: LLMClient, tools: tuple[RegisteredTool, ...] = ()) -> LLMReActController:
    return LLMReActController(client, tools)


@pytest.mark.asyncio
async def test_tool_call_decision_translated(monkeypatch) -> None:
    """LLM 返回 tool_call → 决策 dict（type=tool_call / tool / arguments）。"""

    async def fake_chat_with_tools(messages, tools):
        return None, (ToolCall(id="c1", name="parse_recipe", arguments={"source_text": "300 g tofu"}),)

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    decision = await controller.decide({"request_id": "r1", "agent_step": 0})
    assert decision["type"] == "tool_call"
    assert decision["tool"] == "parse_recipe"
    assert decision["arguments"] == {"source_text": "300 g tofu"}


@pytest.mark.asyncio
async def test_final_decision_from_text(monkeypatch) -> None:
    """LLM 无工具调用、返回 JSON 文本 → final 决策。"""

    async def fake_chat_with_tools(messages, tools):
        return '{"status": "READY"}', ()

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    decision = await controller.decide({"request_id": "r1"})
    assert decision["type"] == "final"
    assert decision["response"] == {"status": "READY"}


@pytest.mark.asyncio
async def test_invalid_text_falls_back(monkeypatch) -> None:
    """LLM 返回非 JSON 文本（无工具调用）→ fallback（交给确定性 DAG）。"""

    async def fake_chat_with_tools(messages, tools):
        return "I don't know", ()

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    decision = await controller.decide({"request_id": "r1"})
    assert decision == {"type": "fallback"}


@pytest.mark.asyncio
async def test_llm_error_falls_back(monkeypatch) -> None:
    """LLM 调用抛异常 → fallback（控制器不崩溃，由节点回退 DAG）。"""
    from cooking_plan_agent.llm.client import LLMError

    async def fake_chat_with_tools(messages, tools):
        raise LLMError("provider down")

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    decision = await controller.decide({"request_id": "r1"})
    assert decision == {"type": "fallback"}


@pytest.mark.asyncio
async def test_empty_output_falls_back(monkeypatch) -> None:
    """LLM 无文本也无工具调用 → fallback。"""

    async def fake_chat_with_tools(messages, tools):
        return None, ()

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    decision = await controller.decide({"request_id": "r1"})
    assert decision == {"type": "fallback"}


@pytest.mark.asyncio
async def test_tools_serialised_to_openai_format(monkeypatch) -> None:
    """RegisteredTool 被转换为 OpenAI tool 格式传给 LLM。"""
    captured: dict = {}

    async def fake_chat_with_tools(messages, tools):
        captured["messages"] = messages
        captured["tools"] = tools
        return None, ()

    controller = _controller(_client(), (_tool(),))
    monkeypatch.setattr(controller._client, "chat_with_tools", fake_chat_with_tools)

    await controller.decide({"request_id": "r1", "agent_step": 0})
    assert captured["tools"][0]["type"] == "function"
    fn = captured["tools"][0]["function"]
    assert fn["name"] == "parse_recipe"
    assert "parameters" in fn
    # system prompt 存在，且 state_summary 以 JSON 传给 LLM。
    assert any(m["role"] == "system" for m in captured["messages"])
    assert captured["messages"][-1]["role"] == "user"
    assert "request_id" in captured["messages"][-1]["content"]
