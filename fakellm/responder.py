"""Build non-streaming responses in OpenAI or Anthropic format."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from .matcher import extract_messages


def build_response(
    rule: dict[str, Any] | None, body: dict[str, Any], api: str
) -> tuple[int, dict[str, Any]]:
    """Return (status_code, response_body)."""
    respond = (rule or {}).get("respond", {})

    # Error response shortcut
    status = respond.get("status", 200)
    if status >= 400:
        return status, _error_body(respond, api)

    content = respond.get("content")
    if content is None:
        content = _deterministic_echo(body)

    tool_calls = respond.get("tool_calls")

    if api == "openai":
        return 200, _openai_response(body, content, tool_calls)
    return 200, _anthropic_response(body, content, tool_calls)


def _deterministic_echo(body: dict[str, Any]) -> str:
    """A stable, fake-but-plausible response based on a hash of the request."""
    seed = hashlib.sha256(repr(sorted(body.items())).encode()).hexdigest()[:8]
    model = body.get("model", "unknown")
    return f"[mock response for {model}, fingerprint {seed}]"


def _error_body(respond: dict[str, Any], api: str) -> dict[str, Any]:
    message = respond.get("error", "Mock error")
    if api == "openai":
        return {"error": {"message": message, "type": "mock_error", "code": None}}
    return {"type": "error", "error": {"type": "mock_error", "message": message}}


def _openai_response(
    body: dict[str, Any], content: str, tool_calls: list[dict[str, Any]] | None
) -> dict[str, Any]:
    prompt_tokens = _count_tokens_from_messages(extract_messages(body, "openai"))
    completion_tokens = _approx_tokens(content) if content else 0

    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"

    if tool_calls:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": _json_dump(tc.get("arguments", {})),
                },
            }
            for tc in tool_calls
        ]
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _anthropic_response(
    body: dict[str, Any], content: str, tool_calls: list[dict[str, Any]] | None
) -> dict[str, Any]:
    prompt_tokens = _count_tokens_from_messages(extract_messages(body, "anthropic"))
    completion_tokens = _approx_tokens(content) if content else 0

    blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    if content:
        blocks.append({"type": "text", "text": content})

    if tool_calls:
        for tc in tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:12]}",
                    "name": tc["name"],
                    "input": tc.get("arguments", {}),
                }
            )
        stop_reason = "tool_use"

    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": body.get("model", "claude-sonnet-4-5"),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


def _approx_tokens(text: str) -> int:
    """Rough token count. Real version should use tiktoken / Anthropic tokenizer."""
    return max(1, len(text) // 4)


def _count_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += _approx_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += _approx_tokens(block["text"])
    return total


def _json_dump(obj: Any) -> str:
    import json

    return json.dumps(obj)
