# =============================================================================
# LLM 客户端模块（llm/client）
# -----------------------------------------------------------------------------
# 提供“与具体厂商无关”的 OpenAI 兼容 Chat Completions 客户端，核心职责：
#   - chat             ：发送聊天补全请求，返回助手文本（可选 JSON 模式 + 重试）
#   - chat_json        ：请求 JSON 对象响应并解析为 dict
#   - chat_with_tools  ：OpenAI 兼容的 function/tool 调用请求（P5-1）
# 设计：直接用 httpx 打 OpenAI 兼容的 /chat/completions 端点（如本地 Ollama），
#       无需厂商 SDK，任意兼容提供方（Ollama / LM Studio / vLLM / 云）都可通过配置切换。
# 安全：日志中绝不记录原始异常文本（P4-03 / 补 P2-05），只输出经 redact 脱敏后的摘要，
#       防止 provider 错误体里可能夹带的密钥或带签名 URL 泄漏到日志。
# =============================================================================

"""LLM client — provider-neutral OpenAI-compatible chat completions.

LLM 客户端 —— 与厂商无关的 OpenAI 兼容聊天补全。

Uses httpx directly against an OpenAI-compatible /chat/completions endpoint
(e.g. local Ollama at http://localhost:11434/v1). No vendor SDK is required,
so any OpenAI-compatible provider (Ollama, LM Studio, vLLM, cloud) can be
swapped via settings — consistent with the project's provider-neutral stance.

直接用 httpx 打 OpenAI 兼容的 /chat/completions 端点（如本地 Ollama 的
http://localhost:11434/v1）。无需厂商 SDK，因此任意 OpenAI 兼容提供方
（Ollama / LM Studio / vLLM / 云）都可通过配置切换 —— 契合项目“厂商中立”立场。
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

import httpx

from cooking_plan_agent.observability.redaction import redact

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """当 LLM 调用失败（网络 / HTTP / 畸形输出）时抛出。"""


class ToolCall(NamedTuple):
    """LLM 请求中的单次工具调用（P5-1）。

    A single tool call within an LLM request (P5-1).
    """

    id: str
    name: str
    arguments: dict[str, object]


class LLMClient:
    """极简的 OpenAI 兼容聊天补全异步客户端。

    Minimal async client for OpenAI-compatible chat completions.

    Supports optional JSON-mode (response_format={"type": "json_object"})
    for deterministic structured output, and a fixed retry policy.

    支持可选的 JSON 模式（response_format={"type": "json_object"}）以获取确定性
    结构化输出，并内置固定重试策略。

    P1-02: the client owns ONE lifecycle-level ``httpx.AsyncClient`` (created
    once, closed once via ``aclose()``) so connection pools are reused across
    calls instead of being rebuilt per request. The app lifespan registers
    this client for clean shutdown.

    P1-02：客户端持有**一个**生命周期级 ``httpx.AsyncClient``（创建一次、通过
    ``aclose()`` 关闭一次），使连接池在多次调用间复用，而非每次请求重建。
    应用 lifespan 注册此客户端以便优雅关闭。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        connection_pool_size: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        # Lifecycle-level transport: bounded pool, reused across every call.
        # 生命周期级传输层：有界连接池，跨所有调用复用。
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size,
            ),
        )

    async def aclose(self) -> None:
        """关闭共享的 httpx 客户端（幂等安全）。"""
        try:
            await self._client.aclose()
        except httpx.TransportError:  # already closed / transport gone
            # ↑ 已关闭 / 传输层已失效
            return

    # ------------------------------------------------------------------
    # Public API
    # 公共 API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发送聊天补全请求并返回助手文本。

        Send a chat completion request and return the assistant text.

        Args:
            messages: OpenAI-style message list ([{"role": ..., "content": ...}]).
                messages：OpenAI 风格消息列表（[{"role": ..., "content": ...}]）。
            json_mode: If True, request a JSON object response and return it raw.
                json_mode：为 True 时请求 JSON 对象响应并原样返回。
            temperature: Optional per-call override (defaults to instance value).
                temperature：可选的按调用覆盖（默认用实例值）。
            max_tokens: Optional per-call output budget override. When None the
                instance default (llm_max_output_tokens) is used. Recipe import
                may raise it so multi-dish JSON output is never truncated.
                max_tokens：可选的按调用输出预算覆盖。None 时用实例默认值。
                菜谱导入可能调高它，避免多菜 JSON 输出被截断。

        Returns:
            The assistant's text content (JSON string when json_mode=True).
            助手文本内容（json_mode=True 时为 JSON 字符串）。

        Raises:
            LLMError: On non-2xx HTTP, network error, or empty response.
            LLMError：非 2xx HTTP、网络错误或空响应时抛出。
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_output_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise LLMError("LLM returned empty content")
                return str(content).strip()
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, IndexError, LLMError) as exc:
                # P4-03 (补 P2-05): never log the raw exception text — the
                # provider error body may embed credentials or keyed URLs.
                # Only the safe exception type + redacted summary is emitted.
                # P4-03（补 P2-05）：绝不记录原始异常文本 —— provider 错误体可能夹带
                # 密钥或带签名 URL。只输出安全的异常类型 + 脱敏摘要。
                logger.warning(
                    "LLM call failed | attempt=%d/%d | exception_type=%s | error=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    type(exc).__name__,
                    redact(str(exc)),
                )
                if attempt < self._max_retries:
                    continue
                raise LLMError(f"LLM request failed after retries: {exc}") from exc
        raise LLMError("LLM request failed")  # unreachable — satisfies mypy
        # ↑ 不可达 —— 仅为满足 mypy

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """请求 JSON 对象响应并解析为 dict。

        Request a JSON object response and parse it.

        Args:
            messages: OpenAI-style message list.
                messages：OpenAI 风格消息列表。
            temperature: Optional per-call override.
                temperature：可选的按调用覆盖。
            max_tokens: Optional per-call output budget override (see ``chat``).
                max_tokens：可选的按调用输出预算覆盖（见 ``chat``）。

        Returns:
            Parsed JSON object.
            解析后的 JSON 对象。

        Raises:
            LLMError: If the response is not valid JSON or not an object.
            LLMError：响应不是合法 JSON 或不是对象时抛出。
        """
        raw = await self.chat(messages, json_mode=True, temperature=temperature, max_tokens=max_tokens)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"LLM JSON response is not an object: {type(parsed).__name__}")
        return parsed

    async def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]],
    ) -> tuple[str | None, tuple[ToolCall, ...]]:
        """OpenAI 兼容的工具调用请求（P5-1）。

        OpenAI-compatible tool-calling request (P5-1).

        Args:
            messages: OpenAI-style message list.
                messages：OpenAI 风格消息列表。
            tools: OpenAI tool-format tool descriptions.
                tools：OpenAI 工具格式的工具描述。

        Returns:
            (assistant_text, tool_calls). ``tool_calls`` is an empty tuple when
            there are no tool calls. The text may be None on a pure tool-call turn.
            (assistant_text, tool_calls)。无工具调用时 ``tool_calls`` 为空元组。
            纯工具调用回合时文本可能为 None。
        """
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": False,
            "tools": tools,
            "tool_choice": "auto",
        }
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    arguments=dict(arguments),
                )
            )
        return (str(content).strip() if content else None, tuple(calls))
