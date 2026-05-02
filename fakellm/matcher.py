"""Match incoming requests against configured rules."""

from __future__ import annotations

import re
from typing import Any

from .config import Config


def match_request(
    body: dict[str, Any],
    headers: dict[str, str],
    config: Config,
    api: str,
) -> dict[str, Any] | None:
    """Walk rules top-to-bottom. Return the first matching rule, or None."""
    messages = extract_messages(body, api)

    for rule in config.rules:
        if _rule_matches(rule, body, messages, headers):
            return rule

    return None


def _rule_matches(
    rule: dict[str, Any],
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    headers: dict[str, str],
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
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # multimodal/blocks: pull out text parts
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


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
