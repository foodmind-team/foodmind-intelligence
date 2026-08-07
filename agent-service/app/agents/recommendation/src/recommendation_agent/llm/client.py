"""Provider-neutral OpenAI-compatible JSON chat client."""

from __future__ import annotations

import json
from typing import Any

import httpx


class LLMError(Exception):
    """Safe provider-boundary error."""


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
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
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size,
            ),
        )

    async def chat_json(self, messages: list[dict[str, str]], *, timeout_seconds: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        timeout = min(self._timeout_seconds, timeout_seconds)
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise LLMError("provider JSON content is not an object")
                return parsed
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, LLMError) as exc:
                if attempt == self._max_retries:
                    raise LLMError("provider request failed") from exc
        raise LLMError("provider request failed")

    async def aclose(self) -> None:
        await self._client.aclose()

