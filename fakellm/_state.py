"""Per-process conversation state.

Tracks turn counts keyed by conversation ID. State lives at module level
because fakellm runs as a single-worker uvicorn process — same caveat as
server.py. Reset via /_fakellm/reset.

Thread-safety: a single lock guards all mutation. Reads of immutable
snapshots (the dict of dicts returned by snapshot()) are safe without
holding the lock because we copy.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """State for a single conversation across multiple turns."""

    turn: int = 0  # incremented on each request; first request makes it 1
    seen_tool_results: list[str] = field(default_factory=list)
    last_assistant_response: dict[str, Any] | None = None


_lock = threading.Lock()
_conversations: dict[str, ConversationState] = {}


def derive_conversation_id(body: dict[str, Any], headers: dict[str, str]) -> str:
    """Get a stable conversation ID for this request.

    Priority:
      1. X-Fakellm-Conversation-Id header (explicit, for tests)
      2. Hash of the first user message content (implicit, conversation-stable)

    Why the first user message and not all messages: as a conversation grows,
    later turns include more messages than earlier ones, so hashing all of
    them would give every turn a different ID. The first user message is the
    one stable thing that's present on every turn of the same conversation.
    """
    explicit = headers.get("x-fakellm-conversation-id")
    if explicit:
        return explicit

    messages = body.get("messages", [])
    first_user = next(
        (m for m in messages if m.get("role") == "user"),
        None,
    )
    if first_user is None:
        # No user message yet — use a hash of the whole body as a fallback.
        # Unlikely to be re-hit, but keeps behavior defined.
        serialized = json.dumps(body, sort_keys=True, default=str)
        return "anon-" + hashlib.sha256(serialized.encode()).hexdigest()[:12]

    content = first_user.get("content", "")
    if isinstance(content, list):
        # Multimodal: concatenate text blocks for hashing.
        content = " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif not isinstance(content, str):
        content = str(content)

    return "conv-" + hashlib.sha256(content.encode()).hexdigest()[:12]


def advance(conversation_id: str) -> ConversationState:
    """Increment turn count and return the updated state.

    Called once per request, before matching. Returns a *copy* of the state
    so callers can read it without holding the lock.
    """
    with _lock:
        state = _conversations.setdefault(conversation_id, ConversationState())
        state.turn += 1
        # Return a shallow copy so the matcher can't accidentally mutate
        # shared state. seen_tool_results is intentionally shared by reference
        # because it's read-only from the matcher's perspective.
        return ConversationState(
            turn=state.turn,
            seen_tool_results=list(state.seen_tool_results),
            last_assistant_response=state.last_assistant_response,
        )


def record_tool_results(conversation_id: str, body: dict[str, Any]) -> None:
    """Extract tool results from the request body and remember them.

    Called before advance() so the *current* turn can match on tool results
    that arrived in this request's message history.
    """
    tool_texts = _extract_tool_result_texts(body)
    if not tool_texts:
        return
    with _lock:
        state = _conversations.setdefault(conversation_id, ConversationState())
        for t in tool_texts:
            if t not in state.seen_tool_results:
                state.seen_tool_results.append(t)


def reset() -> int:
    """Clear all conversation state. Returns the number of conversations cleared."""
    with _lock:
        n = len(_conversations)
        _conversations.clear()
        return n


def snapshot() -> dict[str, dict[str, Any]]:
    """Return a serializable snapshot of all conversations (for the dashboard)."""
    with _lock:
        return {
            cid: {
                "turn": s.turn,
                "tool_results_seen": len(s.seen_tool_results),
            }
            for cid, s in _conversations.items()
        }


def _extract_tool_result_texts(body: dict[str, Any]) -> list[str]:
    """Pull text out of tool-result messages in either OpenAI or Anthropic shape."""
    out: list[str] = []
    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")

        # OpenAI: role="tool", content is a string
        if role == "tool" and isinstance(content, str):
            out.append(content)
            continue

        # Anthropic: role="user", content is a list with tool_result blocks
        if role == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                inner = block.get("content", "")
                if isinstance(inner, str):
                    out.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            out.append(sub.get("text", ""))
    return out
