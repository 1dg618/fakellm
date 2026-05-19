"""Shared utilities used by both responder and streaming modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Optional dependency. If tiktoken is installed we use it for accurate
# OpenAI counts and as a reasonable approximation for Anthropic models
# (Anthropic's tokenizer isn't fully public; cl100k_base is close enough
# for testing purposes). If not installed, we fall back to the original
# len // 4 heuristic — no error, no warning, fakellm just works without it.
try:
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover — exercised by the fallback path
    # We catch broadly here on purpose. Possible failure modes include:
    #   - tiktoken not installed (ImportError)
    #   - tiktoken installed but BPE files can't be downloaded (sandboxed CI,
    #     no internet, expired cache) — raises requests.HTTPError
    #   - tiktoken's loader changing shape in a future version
    # None of these should crash a mock server at import time. Fall back
    # silently to the heuristic in approx_tokens().
    _ENCODING = None


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
    """Token count for `text`.

    Uses tiktoken's cl100k_base if available (exact for modern OpenAI models,
    approximate for Anthropic). Falls back to len // 4 if tiktoken isn't
    installed. Always returns at least 1 for non-empty strings.
    """
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return max(1, len(_ENCODING.encode(text)))
        except Exception:
            # tiktoken can raise on some unusual inputs; fall back rather
            # than break the whole request over a token count.
            pass
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
