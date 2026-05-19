"""Match incoming requests against configured rules.

Rules are walked top-to-bottom; first match wins. Per-rule precomputation
(lowercased needles, compiled regexes) happens at config load time — see
config._build_rule — so this hot path is straight comparisons.
"""

from __future__ import annotations

from typing import Any

from ._state import ConversationState
from .config import Config, Rule


def match_request(
    body: dict[str, Any],
    headers: dict[str, str],
    config: Config,
    api: str,
    state: ConversationState | None = None,
) -> Rule | None:
    """Walk rules top-to-bottom. Return the first matching Rule, or None.

    `state` carries multi-turn context. If None (e.g. older callers, tests),
    conversation-aware matchers behave as if this were turn 1 with no prior
    context.
    """
    if state is None:
        state = ConversationState(turn=1)

    messages = extract_messages(body, api)

    # Flatten + lowercase once per request, not once per rule. Many rules
    # share the same messages_contain / previous_message_contains check
    # against the same flattened text — recomputing it inside the loop was
    # the single biggest source of redundant work on the hot path.
    flat_text_lower = _flatten_messages(messages).lower()
    prev_message = messages[-2] if len(messages) >= 2 else None
    prev_text_lower = _message_text(prev_message).lower() if prev_message else ""
    model_str = str(body.get("model", ""))
    tool_names = _extract_tool_names(body)
    request_tool_result_texts_lower = [
        t.lower() for t in _tool_result_texts_from_messages(messages)
    ]

    for rule in config.rules:
        if _rule_matches(
            rule,
            flat_text_lower=flat_text_lower,
            prev_message=prev_message,
            prev_text_lower=prev_text_lower,
            model_str=model_str,
            tool_names=tool_names,
            request_tool_result_texts_lower=request_tool_result_texts_lower,
            headers=headers,
            state=state,
        ):
            return rule

    return None


def _rule_matches(
    rule: Rule,
    *,
    flat_text_lower: str,
    prev_message: dict[str, Any] | None,
    prev_text_lower: str,
    model_str: str,
    tool_names: list[str],
    request_tool_result_texts_lower: list[str],
    headers: dict[str, str],
    state: ConversationState,
) -> bool:
    when = rule.when

    # Order checks cheapest-first: O(1) scalar / dict compares before any
    # substring or regex scans. The previous version did messages_contain
    # first, which is the most expensive check.

    if when.turn is not None and state.turn != when.turn:
        return False

    if when.turn_in is not None:
        low, high = when.turn_in
        if not (low <= state.turn <= high):
            return False

    for hk, hv in when.headers.items():
        if headers.get(hk) != hv:
            return False

    if when.previous_message_role is not None:
        if prev_message is None or prev_message.get("role") != when.previous_message_role:
            return False

    if when.tools_include is not None and when.tools_include not in tool_names:
        return False

    # model_matches: compiled at load time, anchored regex preserving glob
    # semantics. re.match is anchored at start; the regex has $ at end.
    if rule._model_regex is not None:
        if not rule._model_regex.match(model_str):
            return False

    if rule._previous_message_contains_lower is not None:
        if rule._previous_message_contains_lower not in prev_text_lower:
            return False

    if rule._messages_contain_lower is not None:
        if rule._messages_contain_lower not in flat_text_lower:
            return False

    if rule._tool_result_contains_lower is not None:
        needle = rule._tool_result_contains_lower
        in_request = any(needle in t for t in request_tool_result_texts_lower)
        in_history = any(needle in t.lower() for t in state.seen_tool_results)
        if not (in_request or in_history):
            return False

    return True


def extract_messages(body: dict[str, Any], api: str) -> list[dict[str, Any]]:
    """Normalize message format across OpenAI and Anthropic."""
    if api == "openai":
        return body.get("messages", [])

    # Anthropic: system is a separate field, prepend it as a message.
    msgs = list(body.get("messages", []))
    system = body.get("system")
    if system:
        if isinstance(system, list):
            sys_text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        else:
            sys_text = str(system)
        msgs.insert(0, {"role": "system", "content": sys_text})
    return msgs


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return " ".join(_message_text(m) for m in messages)


def _message_text(message: dict[str, Any] | None) -> str:
    """Extract text from a single message, handling string and block content."""
    if message is None:
        return ""
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


def _tool_result_texts_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool-result text from messages in either OpenAI or Anthropic shape."""
    out: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")

        if role == "tool" and isinstance(content, str):
            out.append(content)
            continue

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
        if "function" in t and isinstance(t["function"], dict):
            name = t["function"].get("name")
            if name:
                names.append(name)
        elif "name" in t:
            names.append(t["name"])
    return names
