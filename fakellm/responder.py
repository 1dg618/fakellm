"""Build non-streaming responses in OpenAI or Anthropic format."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ._util import approx_tokens, count_tokens_from_messages, deterministic_echo
from .config import Rule, normalize_rule
from .matcher import extract_messages


def build_response(
    rule: Rule | dict[str, Any] | None, body: dict[str, Any], api: str
) -> tuple[int, dict[str, Any]]:
    """Return (status_code, response_body).

    `rule` may be a Rule, a dict in the YAML rule shape, or None. Dicts are
    accepted for back-compat with tests and older callers that construct
    rules by hand.
    """
    rule = normalize_rule(rule)
    respond = rule.respond if rule is not None else None

    status = respond.status if respond is not None else 200
    if status >= 400:
        return status, _error_body(respond, api)

    content = respond.content if respond is not None else None
    if content is None:
        content = deterministic_echo(body)

    tool_calls = respond.tool_calls if respond is not None else None

    if api == "openai":
        return 200, _openai_response(body, content, tool_calls)
    return 200, _anthropic_response(body, content, tool_calls)


def _error_body(respond: Any, api: str) -> dict[str, Any]:
    message = respond.error if respond is not None and respond.error else "Mock error"
    if api == "openai":
        return {"error": {"message": message, "type": "mock_error", "code": None}}
    return {"type": "error", "error": {"type": "mock_error", "message": message}}


def _openai_response(
    body: dict[str, Any], content: str, tool_calls: list[dict[str, Any]] | None
) -> dict[str, Any]:
    prompt_tokens = count_tokens_from_messages(extract_messages(body, "openai"))
    completion_tokens = approx_tokens(content) if content else 0

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
                    "arguments": json.dumps(tc.get("arguments", {})),
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
    prompt_tokens = count_tokens_from_messages(extract_messages(body, "anthropic"))
    completion_tokens = approx_tokens(content) if content else 0

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
