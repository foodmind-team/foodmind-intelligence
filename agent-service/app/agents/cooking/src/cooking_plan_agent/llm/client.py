"""LLM client — provider-neutral OpenAI-compatible chat completions.

Uses httpx directly against an OpenAI-compatible /chat/completions endpoint
(e.g. local Ollama at http://localhost:11434/v1). No vendor SDK is required,
so any OpenAI-compatible provider (Ollama, LM Studio, vLLM, cloud) can be
swapped via settings — consistent with the project's provider-neutral stance.
"""

from __future__ import annotations

import json
import logging
from typing import Any, NamedTuple

import httpx

from cooking_plan_agent.observability.redaction import redact

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails (network, HTTP, or malformed output)."""


class ToolCall(NamedTuple):
    """A single tool call within an LLM request (P5-1)."""

    id: str
    name: str
    arguments: dict[str, object]


class LLMClient:
    """Minimal async client for OpenAI-compatible chat completions.

    Supports optional JSON-mode (response_format={"type": "json_object"})
    for deterministic structured output, and a fixed retry policy.

    P1-02: the client owns ONE lifecycle-level ``httpx.AsyncClient`` (created
    once, closed once via ``aclose()``) so connection pools are reused across
    calls instead of being rebuilt per request. The app lifespan registers
    this client for clean shutdown.
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
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size,
            ),
        )

    async def aclose(self) -> None:
        """Close the shared httpx client (idempotent-safe)."""
        try:
            await self._client.aclose()
        except httpx.TransportError:  # already closed / transport gone
            return

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant text.

        Args:
            messages: OpenAI-style message list ([{"role": ..., "content": ...}]).
            json_mode: If True, request a JSON object response and return it raw.
            temperature: Optional per-call override (defaults to instance value).
            max_tokens: Optional per-call output budget override. When None the
                instance default (llm_max_output_tokens) is used. Recipe import
                may raise it so multi-dish JSON output is never truncated.

        Returns:
            The assistant's text content (JSON string when json_mode=True).

        Raises:
            LLMError: On non-2xx HTTP, network error, or empty response.
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

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Request a JSON object response and parse it.

        Args:
            messages: OpenAI-style message list.
            temperature: Optional per-call override.
            max_tokens: Optional per-call output budget override (see ``chat``).

        Returns:
            Parsed JSON object.

        Raises:
            LLMError: If the response is not valid JSON or not an object.
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
        """OpenAI-compatible tool-calling request (P5-1).

        Args:
            messages: OpenAI-style message list.
            tools: OpenAI tool-format tool descriptions.

        Returns:
            (assistant_text, tool_calls). ``tool_calls`` is an empty tuple when
            there are no tool calls. The text may be None on a pure tool-call turn.
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
