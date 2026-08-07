"""Small provider-neutral OpenAI-compatible chat-completions client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class LLMError(Exception):
    """Safe boundary error for provider failures."""


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
        max_output_tokens: int,
        connection_pool_size: int,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
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

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with asyncio.timeout(self._timeout_seconds):
                for attempt in range(self._max_retries + 1):
                    try:
                        response = await self._client.post(self._url, json=payload, headers=headers)
                        response.raise_for_status()
                        body = response.json()
                        content = body["choices"][0]["message"]["content"]
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
