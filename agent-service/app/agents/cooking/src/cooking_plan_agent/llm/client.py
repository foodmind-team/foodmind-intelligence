"""LLM client — provider-neutral OpenAI-compatible chat completions.

Uses httpx directly against an OpenAI-compatible /chat/completions endpoint
(e.g. local Ollama at http://localhost:11434/v1). No vendor SDK is required,
so any OpenAI-compatible provider (Ollama, LM Studio, vLLM, cloud) can be
swapped via settings — consistent with the project's provider-neutral stance.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails (network, HTTP, or malformed output)."""


class LLMClient:
    """Minimal async client for OpenAI-compatible chat completions.

    Supports optional JSON-mode (response_format={"type": "json_object"})
    for deterministic structured output, and a fixed retry policy.
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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant text.

        Args:
            messages: OpenAI-style message list ([{"role": ..., "content": ...}]).
            json_mode: If True, request a JSON object response and return it raw.
            temperature: Optional per-call override (defaults to instance value).

        Returns:
            The assistant's text content (JSON string when json_mode=True).

        Raises:
            LLMError: On non-2xx HTTP, network error, or empty response.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
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
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise LLMError("LLM returned empty content")
                return str(content).strip()
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, IndexError, LLMError) as exc:
                logger.warning(
                    "LLM call failed | attempt=%d/%d | error=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
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
    ) -> dict[str, Any]:
        """Request a JSON object response and parse it.

        Args:
            messages: OpenAI-style message list.
            temperature: Optional per-call override.

        Returns:
            Parsed JSON object.

        Raises:
            LLMError: If the response is not valid JSON or not an object.
        """
        raw = await self.chat(messages, json_mode=True, temperature=temperature)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"LLM JSON response is not an object: {type(parsed).__name__}")
        return parsed
