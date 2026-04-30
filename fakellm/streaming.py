"""Streaming response generators for OpenAI and Anthropic SSE formats."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from .responder import _deterministic_echo


async def build_stream(
    rule: dict[str, Any] | None, body: dict[str, Any], api: str
) -> AsyncIterator[str]:
    respond = (rule or {}).get("respond", {})
    content = respond.get("content")
    if content is None:
        content = _deterministic_echo(body)

    if api == "openai":
        async for chunk in _openai_stream(content, body):
            yield chunk
    else:
        async for chunk in _anthropic_stream(content, body):
            yield chunk


async def _openai_stream(content: str, body: dict[str, Any]) -> AsyncIterator[str]:
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = body.get("model", "gpt-4o-mini")
    created = int(time.time())

    # First chunk: role
    first = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(first)}\n\n"

    # Content chunks: token-by-token (approximated as words)
    words = content.split(" ")
    for i, word in enumerate(words):
        piece = word if i == 0 else " " + word
        chunk = {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.01)

    # Final chunk
    final = {
        "id": msg_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _anthropic_stream(content: str, body: dict[str, Any]) -> AsyncIterator[str]:
    """Anthropic uses typed SSE events. This emits the full event sequence."""
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    model = body.get("model", "claude-sonnet-4-5")

    def event(event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    yield event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    )

    yield event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )

    words = content.split(" ")
    for i, word in enumerate(words):
        piece = word if i == 0 else " " + word
        yield event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            },
        )
        await asyncio.sleep(0.01)

    yield event("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": max(1, len(content) // 4)},
        },
    )

    yield event("message_stop", {"type": "message_stop"})
