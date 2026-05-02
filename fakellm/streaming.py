"""Streaming response generators for OpenAI and Anthropic SSE formats."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from ._util import deterministic_echo


async def build_stream(
    rule: dict[str, Any] | None, body: dict[str, Any], api: str
) -> AsyncIterator[str]:
    respond = (rule or {}).get("respond", {})
    content = respond.get("content")
    tool_calls = respond.get("tool_calls")

    # If neither content nor tool_calls is configured, fall back to echo text.
    if content is None and not tool_calls:
        content = deterministic_echo(body)

    if api == "openai":
        async for chunk in _openai_stream(content, tool_calls, body):
            yield chunk
    else:
        async for chunk in _anthropic_stream(content, tool_calls, body):
            yield chunk


async def _openai_stream(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    body: dict[str, Any],
) -> AsyncIterator[str]:
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = body.get("model", "gpt-4o-mini")
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return "data: " + json.dumps(
            {
                "id": msg_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        ) + "\n\n"

    # Role chunk
    yield chunk({"role": "assistant"})

    if tool_calls:
        # Stream each tool call: first chunk announces id+name, subsequent chunks
        # stream the JSON arguments piece by piece.
        for index, tc in enumerate(tool_calls):
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": ""},
                        }
                    ]
                }
            )

            args_json = json.dumps(tc.get("arguments", {}))
            # Split arguments into a few chunks to exercise streaming-parse code paths.
            for piece in _split_for_streaming(args_json):
                yield chunk(
                    {
                        "tool_calls": [
                            {"index": index, "function": {"arguments": piece}}
                        ]
                    }
                )
                await asyncio.sleep(0.01)

        yield chunk({}, finish_reason="tool_calls")
    else:
        # Content chunks: word-by-word.
        words = (content or "").split(" ")
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            yield chunk({"content": piece})
            await asyncio.sleep(0.01)
        yield chunk({}, finish_reason="stop")

    yield "data: [DONE]\n\n"


async def _anthropic_stream(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    body: dict[str, Any],
) -> AsyncIterator[str]:
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

    block_index = 0
    output_tokens = 1

    if content:
        yield event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": block_index,
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
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": piece},
                },
            )
            await asyncio.sleep(0.01)
        yield event(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )
        output_tokens = max(1, len(content) // 4)
        block_index += 1

    if tool_calls:
        for tc in tool_calls:
            yield event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": tc["name"],
                        "input": {},
                    },
                },
            )

            args_json = json.dumps(tc.get("arguments", {}))
            for piece in _split_for_streaming(args_json):
                yield event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": piece},
                    },
                )
                await asyncio.sleep(0.01)

            yield event(
                "content_block_stop",
                {"type": "content_block_stop", "index": block_index},
            )
            block_index += 1

    stop_reason = "tool_use" if tool_calls else "end_turn"

    yield event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )

    yield event("message_stop", {"type": "message_stop"})


def _split_for_streaming(s: str, max_pieces: int = 4) -> list[str]:
    """Split a string into roughly equal-sized pieces for streaming.

    We split tool-call argument JSON across multiple chunks so that
    tests exercise the SDK's streaming-JSON parsing code paths rather
    than receiving the whole blob in one chunk.
    """
    if not s:
        return [""]
    n = min(max_pieces, len(s))
    if n <= 1:
        return [s]
    size = len(s) // n
    pieces = [s[i * size : (i + 1) * size] for i in range(n - 1)]
    pieces.append(s[(n - 1) * size :])  # remainder
    return pieces
