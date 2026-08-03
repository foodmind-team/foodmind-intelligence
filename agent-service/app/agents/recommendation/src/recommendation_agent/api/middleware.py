"""ASGI request/response byte bounds and correlation propagation."""

import json
import uuid
from typing import Any, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from recommendation_agent.config.settings import Settings

_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            try:
                return cast(bytes, value).decode("ascii")
            except UnicodeDecodeError:
                return None
    return None


def _safe_id(value: str | None) -> str | None:
    if value is None or not value or len(value) > 64:
        return None
    return value if value[0].isalnum() and all(char in _SAFE for char in value) else None


async def _send_json(send: Send, status: int, code: str, request_id: str, *, retryable: bool = False) -> None:
    body = json.dumps(
        {
            "contractVersion": "recommendation-agent-v2",
            "requestId": request_id,
            "sessionId": "unavailable",
            "traceId": request_id,
            "agentTraceId": f"agent-{uuid.uuid4().hex}",
            "status": "failure",
            "error": {"code": code, "retryable": retryable},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BoundaryMiddleware:
    """Buffer bounded private JSON messages so limits apply before parsing/sending."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _safe_id(_header(scope, b"x-request-id")) or uuid.uuid4().hex
        content_length = _header(scope, b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._settings.max_request_bytes:
                    await _send_json(send, 413, "REQUEST_TOO_LARGE", request_id)
                    return
            except ValueError:
                await _send_json(send, 400, "INVALID_REQUEST", request_id)
                return

        request_messages: list[Message] = []
        request_size = 0
        while True:
            message = await receive()
            request_messages.append(message)
            if message["type"] != "http.request":
                break
            request_size += len(message.get("body", b""))
            if request_size > self._settings.max_request_bytes:
                await _send_json(send, 413, "REQUEST_TOO_LARGE", request_id)
                return
            if not message.get("more_body", False):
                break
        message_index = 0

        async def replay() -> Message:
            nonlocal message_index
            if message_index < len(request_messages):
                message = request_messages[message_index]
                message_index += 1
                return message
            return {"type": "http.disconnect"}

        response_messages: list[Message] = []

        async def buffer_send(message: Message) -> None:
            response_messages.append(message)

        await self._app(scope, replay, buffer_send)
        response_size = sum(len(message.get("body", b"")) for message in response_messages)
        if response_size > self._settings.max_response_bytes:
            await _send_json(send, 500, "RESPONSE_TOO_LARGE", request_id)
            return

        state = scope.get("state", {})
        correlation: Any = state.get("correlation") if isinstance(state, dict) else None
        response_request_id = _safe_id(str(getattr(correlation, "request_id", request_id))) or request_id
        response_trace_id = _safe_id(str(getattr(correlation, "trace_id", response_request_id))) or response_request_id
        for message in response_messages:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-request-id", response_request_id.encode("ascii")),
                        (b"x-trace-id", response_trace_id.encode("ascii")),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)
