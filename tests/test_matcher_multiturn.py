"""Tests for fakellm.matcher — focus on the new conversation-aware matchers."""

from __future__ import annotations

from fakellm._state import ConversationState
from fakellm.config import Config
from fakellm.matcher import match_request


def _cfg(*rules: dict) -> Config:
    return Config(rules=list(rules), defaults={})


# ---------------- backward compat: existing matchers still work ----------------


def test_messages_contain_still_works():
    cfg = _cfg(
        {
            "name": "greet",
            "when": {"messages_contain": "hello"},
            "respond": {"content": "hi"},
        }
    )
    body = {"messages": [{"role": "user", "content": "hello there"}]}
    rule = match_request(body, {}, cfg, api="openai")
    assert rule is not None
    assert rule["name"] == "greet"


def test_no_state_provided_defaults_to_turn_one():
    """Calling match_request without state should let turn=1 rules match."""
    cfg = _cfg(
        {
            "name": "first",
            "when": {"turn": 1},
            "respond": {"content": "hi"},
        }
    )
    body = {"messages": [{"role": "user", "content": "anything"}]}
    rule = match_request(body, {}, cfg, api="openai")
    assert rule is not None
    assert rule["name"] == "first"


# ---------------- turn matcher ----------------


def test_turn_matcher_matches_exact_turn():
    cfg = _cfg(
        {
            "name": "first_turn",
            "when": {"turn": 1},
            "respond": {"content": "hello"},
        },
        {
            "name": "second_turn",
            "when": {"turn": 2},
            "respond": {"content": "again"},
        },
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}

    r1 = match_request(body, {}, cfg, api="openai", state=ConversationState(turn=1))
    r2 = match_request(body, {}, cfg, api="openai", state=ConversationState(turn=2))

    assert r1["name"] == "first_turn"
    assert r2["name"] == "second_turn"


def test_turn_matcher_no_match_falls_through():
    cfg = _cfg(
        {
            "name": "only_first",
            "when": {"turn": 1},
            "respond": {"content": "hi"},
        }
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}
    rule = match_request(body, {}, cfg, api="openai", state=ConversationState(turn=5))
    assert rule is None


# ---------------- turn_in matcher ----------------


def test_turn_in_range_inclusive():
    cfg = _cfg(
        {
            "name": "early",
            "when": {"turn_in": [1, 3]},
            "respond": {"content": "early"},
        }
    )
    body = {"messages": [{"role": "user", "content": "hi"}]}

    for t in (1, 2, 3):
        rule = match_request(
            body, {}, cfg, api="openai", state=ConversationState(turn=t)
        )
        assert rule is not None, f"turn {t} should match"

    rule = match_request(body, {}, cfg, api="openai", state=ConversationState(turn=4))
    assert rule is None


# ---------------- previous_message_role / contains ----------------


def test_previous_message_role():
    cfg = _cfg(
        {
            "name": "after_tool",
            "when": {"previous_message_role": "tool"},
            "respond": {"content": "summarized"},
        }
    )
    body = {
        "messages": [
            {"role": "user", "content": "do a search"},
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "search results"},
            {"role": "user", "content": "what did you find?"},
        ]
    }
    # Previous-to-last is the tool message
    rule = match_request(body, {}, cfg, api="openai")
    assert rule is not None
    assert rule["name"] == "after_tool"


def test_previous_message_role_no_match():
    cfg = _cfg(
        {
            "name": "after_tool",
            "when": {"previous_message_role": "tool"},
            "respond": {"content": "x"},
        }
    )
    body = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    }
    assert match_request(body, {}, cfg, api="openai") is None


def test_previous_message_contains():
    cfg = _cfg(
        {
            "name": "react_to_sunny",
            "when": {"previous_message_contains": "sunny"},
            "respond": {"content": "great weather!"},
        }
    )
    body = {
        "messages": [
            {"role": "tool", "content": "Weather: sunny and warm"},
            {"role": "user", "content": "what should I wear?"},
        ]
    }
    rule = match_request(body, {}, cfg, api="openai")
    assert rule is not None
    assert rule["name"] == "react_to_sunny"


def test_previous_message_with_too_few_messages():
    cfg = _cfg(
        {
            "name": "needs_prev",
            "when": {"previous_message_role": "tool"},
            "respond": {"content": "x"},
        }
    )
    body = {"messages": [{"role": "user", "content": "first ever message"}]}
    assert match_request(body, {}, cfg, api="openai") is None


# ---------------- tool_result_contains ----------------


def test_tool_result_contains_in_request_messages():
    cfg = _cfg(
        {
            "name": "weather_followup",
            "when": {"tool_result_contains": "sunny"},
            "respond": {"content": "great!"},
        }
    )
    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking"},
            {"role": "tool", "content": "It is sunny in Paris"},
        ]
    }
    rule = match_request(body, {}, cfg, api="openai")
    assert rule is not None
    assert rule["name"] == "weather_followup"


def test_tool_result_contains_in_history():
    """Match should hit even if the tool result is no longer in this turn's
    messages, as long as it's in the conversation's seen history."""
    cfg = _cfg(
        {
            "name": "remembers_sunny",
            "when": {"tool_result_contains": "sunny"},
            "respond": {"content": "still sunny"},
        }
    )
    state = ConversationState(turn=3, seen_tool_results=["earlier: sunny day"])
    body = {"messages": [{"role": "user", "content": "back to me"}]}
    rule = match_request(body, {}, cfg, api="openai", state=state)
    assert rule is not None


def test_tool_result_contains_anthropic_shape():
    cfg = _cfg(
        {
            "name": "match",
            "when": {"tool_result_contains": "rainy"},
            "respond": {"content": "got it"},
        }
    )
    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [{"type": "tool_use"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "very rainy today",
                    }
                ],
            },
        ]
    }
    rule = match_request(body, {}, cfg, api="anthropic")
    assert rule is not None


# ---------------- combined matchers ----------------


def test_combining_turn_and_tool_result():
    """A rule with multiple conditions only fires when ALL match."""
    cfg = _cfg(
        {
            "name": "specific",
            "when": {"turn": 3, "tool_result_contains": "found it"},
            "respond": {"content": "ok"},
        },
        {
            "name": "fallback",
            "respond": {"content": "default"},
        },
    )
    # Right turn, wrong tool result
    state_a = ConversationState(turn=3, seen_tool_results=["nothing"])
    body = {"messages": [{"role": "user", "content": "x"}]}
    assert match_request(body, {}, cfg, api="openai", state=state_a)["name"] == "fallback"

    # Right turn, right tool result
    state_b = ConversationState(turn=3, seen_tool_results=["found it!"])
    assert match_request(body, {}, cfg, api="openai", state=state_b)["name"] == "specific"

    # Wrong turn, right tool result
    state_c = ConversationState(turn=2, seen_tool_results=["found it!"])
    assert match_request(body, {}, cfg, api="openai", state=state_c)["name"] == "fallback"


def test_rule_order_first_match_wins():
    cfg = _cfg(
        {
            "name": "a",
            "when": {"turn": 1},
            "respond": {"content": "a"},
        },
        {
            "name": "b",
            "when": {"turn": 1},
            "respond": {"content": "b"},
        },
    )
    body = {"messages": [{"role": "user", "content": "x"}]}
    rule = match_request(body, {}, cfg, api="openai", state=ConversationState(turn=1))
    assert rule["name"] == "a"
