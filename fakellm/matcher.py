"""Match incoming requests against configured rules."""

from __future__ import annotations

import re
from typing import Any

from ._state import ConversationState
from .config import Config


def match_request(
    body: dict[str, Any],
    headers: dict[str, str],
    config: Config,
    api: str,
    state: ConversationState | None = None,
) -> dict[str, Any] | None:
    """Walk rules top-to-bottom. Return the first matching rule, or None.

    `state` carries multi-turn context. If None (e.g. callers from older code
    paths or tests), conversation-aware matchers (turn, previous_*, etc.)
    behave as if this were turn 1 with no prior context.
    """
    if state is None:
        state = ConversationState(turn=1)

    messages = extract_messages(body, api)

    for rule in config.rules:
        if _rule_matches(rule, body, messages, headers, state):
            return rule

    return None


def _rule_matches(
    rule: dict[str, Any],
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    headers: dict[str, str],
    state: ConversationState,
) -> bool:
    when = rule.get("when", {})
    if not when:
        return True  # rule with no conditions matches everything

    if "messages_contain" in when:
        text = _flatten_messages(messages)
        if when["messages_contain"].lower() not in text.lower():
            return False

    if "model_matches" in when:
        pattern = when["model_matches"].replace("*", ".*")
        if not re.match(f"^{pattern}$", body.get("model", "")):
            return False

    if "tools_include" in when:
        tool_names = _extract_tool_names(body)
        if when["tools_include"] not in tool_names:
            return False

    # ---- Conversation-aware matchers ----

    if "turn" in when:
        if state.turn != int(when["turn"]):
            return False

    if "turn_in" in when:
        # Inclusive range: [low, high]
        low, high = when["turn_in"]
        if not (int(low) <= state.turn <= int(high)):
            return False

    if "previous_message_role" in when:
        prev = _previous_message(messages)
        if prev is None or prev.get("role") != when["previous_message_role"]:
            return False

    if "previous_message_contains" in when:
        prev = _previous_message(messages)
        if prev is None:
            return False
        prev_text = _message_text(prev)
        if when["previous_message_contains"].lower() not in prev_text.lower():
            return False

    if "tool_result_contains" in when:
        needle = when["tool_result_contains"].lower()
        # Check both: tool results in this request's messages, AND any we've
        # seen previously in this conversation. Either source can satisfy
        # the matcher — useful for rules that fire several turns after a
        # tool was called.
        in_request = any(
            needle in t.lower() for t in _tool_result_texts_from_messages(messages)
        )
        in_history = any(needle in t.lower() for t in state.seen_tool_results)
        if not (in_request or in_history):
            return False

    # Header matchers: any key starting with "header." matches that header
    for key, value in when.items():
        if key.startswith("header."):
            header_name = key[len("header.") :].lower()
            if headers.get(header_name) != value:
                return False

    return True


def extract_messages(body: dict[str, Any], api: str) -> list[dict[str, Any]]:
    """Normalize message format across OpenAI and Anthropic."""
    if api == "openai":
        return body.get("messages", [])

    # Anthropic: system is a separate field, prepend it as a message
    msgs = list(body.get("messages", []))
    system = body.get("system")
    if system:
        if isinstance(system, list):
            # Anthropic system blocks: [{"type": "text", "text": "..."}]
            sys_text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        else:
            sys_text = system
        msgs.insert(0, {"role": "system", "content": sys_text})
    return msgs


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(_message_text(m))
    return " ".join(parts)


def _message_text(message: dict[str, Any]) -> str:
    """Extract text from a single message, handling string and block content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    parts.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(sub.get("text", ""))
        return " ".join(parts)
    return ""


def _previous_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the message immediately before the current request would respond to.

    For an incoming request, the "current" message is the last one in the
    list (typically the latest user/tool message). The "previous" one is
    the second-to-last.
    """
    if len(messages) < 2:
        return None
    return messages[-2]


def _tool_result_texts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool-result text from messages in either OpenAI or Anthropic shape."""
    out: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")

        # OpenAI: role="tool", content is a string
        if role == "tool" and isinstance(content, str):
            out.append(content)
            continue

        # Anthropic: role="user", content list contains tool_result blocks
        if isinstance(content, list):
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


def _extract_tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for t in body.get("tools", []):
        if not isinstance(t, dict):
            continue
        # OpenAI shape: {"type": "function", "function": {"name": "..."}}
        if "function" in t and isinstance(t["function"], dict):
            name = t["function"].get("name")
            if name:
                names.append(name)
        # Anthropic shape: {"name": "...", "input_schema": {...}}
        elif "name" in t:
            names.append(t["name"])
    return names
