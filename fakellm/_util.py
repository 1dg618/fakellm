"""Shared utilities used by both responder and streaming modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def deterministic_echo(body: dict[str, Any]) -> str:
    """A stable, fake-but-plausible response based on a hash of the request.

    Uses json.dumps with sort_keys=True so nested dicts produce the same
    fingerprint regardless of key insertion order — same logical request,
    same fingerprint, every time.
    """
    serialized = json.dumps(body, sort_keys=True, default=str)
    seed = hashlib.sha256(serialized.encode()).hexdigest()[:8]
    model = body.get("model", "unknown")
    return f"[mock response for {model}, fingerprint {seed}]"


def approx_tokens(text: str) -> int:
    """Rough token count.

    TODO: replace with tiktoken / Anthropic tokenizer for accurate counts.
    See https://github.com/1dg618/fakellm/issues for the tracking issue.
    """
    return max(1, len(text) // 4)


def count_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += approx_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += approx_tokens(block["text"])
    return total
