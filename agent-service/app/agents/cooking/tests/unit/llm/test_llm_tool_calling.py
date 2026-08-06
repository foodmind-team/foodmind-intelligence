"""P5-1: LLMClient chat_with_tools —— tool-calling 请求与解析。"""
import pytest

from cooking_plan_agent.llm.client import LLMClient, ToolCall


class _FakeResponse:
    def __init__(self, message: dict) -> None:
        self._message = message

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": self._message}]}


@pytest.mark.asyncio
async def test_tools_go_into_request_body(monkeypatch) -> None:
    captured: dict = {}

    async def fake_post(url, json=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse({"content": "ok", "tool_calls": []})

    client = LLMClient(base_url="http://llm.test", model="m")
    monkeypatch.setattr(client._client, "post", fake_post)
    tools = [{"type": "function", "function": {"name": "parse_recipe", "parameters": {}}}]

    text, calls = await client.chat_with_tools([{"role": "user", "content": "hi"}], tools)

    assert "tools" in captured["payload"]
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == "auto"
    assert text == "ok"
    assert calls == ()


@pytest.mark.asyncio
async def test_parses_tool_calls(monkeypatch) -> None:
    message = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "parse_recipe",
                    "arguments": '{"source_text": "300 g tofu"}',
                },
            },
            {
                "id": "call_2",
                "function": {
                    "name": "solve_schedule",
                    "arguments": "not-json{",
                },
            },
        ],
    }

    async def fake_post(url, json=None, headers=None):
        return _FakeResponse(message)

    client = LLMClient(base_url="http://llm.test", model="m")
    monkeypatch.setattr(client._client, "post", fake_post)

    text, calls = await client.chat_with_tools([{"role": "user", "content": "go"}], [])

    assert text is None
    assert isinstance(calls[0], ToolCall)
    assert calls[0].id == "call_1"
    assert calls[0].name == "parse_recipe"
    assert calls[0].arguments == {"source_text": "300 g tofu"}
    # 非法 JSON 参数回退为空 dict，不抛错。
    assert calls[1].arguments == {}
