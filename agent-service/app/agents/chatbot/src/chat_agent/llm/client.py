"""Small provider-neutral OpenAI-compatible chat-completions client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


class LLMError(Exception):
    """Safe boundary error for provider failures."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMChatResult:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        temperature: float,
        thinking_enabled: bool,
        max_output_tokens: int,
        connection_pool_size: int,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
        self._thinking_enabled = thinking_enabled
        self._max_output_tokens = max_output_tokens
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size,
            ),
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
    ) -> str | LLMChatResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "thinking": {"type": "enabled" if self._thinking_enabled else "disabled"},
            "max_tokens": self._max_output_tokens,
            "stream": False,
        }
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with asyncio.timeout(min(self._timeout_seconds, timeout_seconds or self._timeout_seconds)):
                for attempt in range(self._max_retries + 1):
                    try:
                        response = await self._client.post(self._url, json=payload, headers=headers)
                        response.raise_for_status()
                        body = response.json()
                        message = body["choices"][0]["message"]
                        content = message.get("content")
                        if content is not None and not isinstance(content, str):
                            raise LLMError("provider returned invalid content")
                        raw_calls = message.get("tool_calls") or []
                        if not isinstance(raw_calls, list):
                            raise LLMError("provider returned invalid tool calls")
                        calls: list[ToolCall] = []
                        for raw_call in raw_calls:
                            function = raw_call.get("function") if isinstance(raw_call, dict) else None
                            if not isinstance(function, dict):
                                raise LLMError("provider returned invalid tool call")
                            call_id, name, arguments = (
                                raw_call.get("id"),
                                function.get("name"),
                                function.get("arguments"),
                            )
                            if not all(isinstance(item, str) and item for item in (call_id, name, arguments)):
                                raise LLMError("provider returned invalid tool call")
                            calls.append(ToolCall(call_id, name, arguments))
                        if tools is not None:
                            if not calls and (not isinstance(content, str) or not content.strip()):
                                raise LLMError("provider returned neither content nor tool calls")
                            return LLMChatResult(
                                content=content.strip() if isinstance(content, str) and content.strip() else None,
                                tool_calls=tuple(calls),
                                reasoning_content=message.get("reasoning_content")
                                if isinstance(message.get("reasoning_content"), str)
                                else None,
                            )
                        if not isinstance(content, str) or not content.strip():
                            raise LLMError("provider returned empty content")
                        return content.strip()
                    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, LLMError) as exc:
                        if attempt == self._max_retries:
                            raise LLMError("provider request failed") from exc
        except TimeoutError as exc:
            raise LLMError("provider request timed out") from exc
        raise LLMError("provider request failed")

    async def aclose(self) -> None:
        await self._client.aclose()
