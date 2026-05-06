"""Tests for fakellm._state — conversation ID, turn counting, tool results, reset."""

from __future__ import annotations

import pytest

from fakellm import _state


@pytest.fixture(autouse=True)
def _clear_state():
    """Reset module-level state before every test."""
    _state.reset()
    yield
    _state.reset()


# ---------------- derive_conversation_id ----------------


def test_explicit_header_wins():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    headers = {"x-fakellm-conversation-id": "my-session-42"}
    assert _state.derive_conversation_id(body, headers) == "my-session-42"


def test_same_first_user_message_same_id():
    body1 = {"messages": [{"role": "user", "content": "Plan a trip to Paris"}]}
    body2 = {
        "messages": [
            {"role": "user", "content": "Plan a trip to Paris"},
            {"role": "assistant", "content": "Sure, when?"},
            {"role": "user", "content": "Next month"},
        ]
    }
    # Same conversation, more turns: ID should be stable.
    assert _state.derive_conversation_id(body1, {}) == _state.derive_conversation_id(
        body2, {}
    )


def test_different_first_user_message_different_id():
    body1 = {"messages": [{"role": "user", "content": "Plan a trip"}]}
    body2 = {"messages": [{"role": "user", "content": "Write a poem"}]}
    assert _state.derive_conversation_id(body1, {}) != _state.derive_conversation_id(
        body2, {}
    )


def test_anthropic_multimodal_first_message():
    """First user message with block content should still produce a stable ID."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image", "source": {"data": "..."}},
                ],
            }
        ]
    }
    cid = _state.derive_conversation_id(body, {})
    assert cid.startswith("conv-")
    # Same content -> same id
    assert cid == _state.derive_conversation_id(body, {})


def test_no_user_message_falls_back_to_anon():
    body = {"messages": [{"role": "system", "content": "You are a helper"}]}
    cid = _state.derive_conversation_id(body, {})
    assert cid.startswith("anon-")


# ---------------- advance ----------------


def test_advance_increments_turn():
    cid = "test-conv"
    s1 = _state.advance(cid)
    s2 = _state.advance(cid)
    s3 = _state.advance(cid)
    assert (s1.turn, s2.turn, s3.turn) == (1, 2, 3)


def test_advance_isolates_conversations():
    a = _state.advance("conv-a")
    b = _state.advance("conv-b")
    a2 = _state.advance("conv-a")
    assert a.turn == 1
    assert b.turn == 1
    assert a2.turn == 2


def test_advance_returns_copy_not_shared_reference():
    """Mutating the returned state shouldn't affect future calls."""
    cid = "test-conv"
    s1 = _state.advance(cid)
    s1.seen_tool_results.append("garbage")
    s2 = _state.advance(cid)
    assert s2.seen_tool_results == []  # not contaminated


# ---------------- record_tool_results ----------------


def test_record_openai_tool_result():
    cid = "openai-conv"
    body = {
        "messages": [
            {"role": "user", "content": "search for X"},
            {"role": "assistant", "tool_calls": [...]},
            {"role": "tool", "content": "Results: sunny in Paris"},
        ]
    }
    _state.record_tool_results(cid, body)
    state = _state.advance(cid)
    assert state.seen_tool_results == ["Results: sunny in Paris"]


def test_record_anthropic_tool_result_string():
    cid = "anthropic-conv"
    body = {
        "messages": [
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": [{"type": "tool_use"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "rainy"}
                ],
            },
        ]
    }
    _state.record_tool_results(cid, body)
    assert _state.advance(cid).seen_tool_results == ["rainy"]


def test_record_anthropic_tool_result_block_content():
    """Anthropic also allows tool_result content to be a list of text blocks."""
    cid = "anthropic-conv-2"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "stormy"}],
                    }
                ],
            }
        ]
    }
    _state.record_tool_results(cid, body)
    assert _state.advance(cid).seen_tool_results == ["stormy"]


def test_record_dedupes_tool_results():
    """Recording the same tool result twice doesn't duplicate it."""
    cid = "dedupe-conv"
    body = {
        "messages": [{"role": "tool", "content": "Result A"}],
    }
    _state.record_tool_results(cid, body)
    _state.record_tool_results(cid, body)
    assert _state.advance(cid).seen_tool_results == ["Result A"]


# ---------------- reset / snapshot ----------------


def test_reset_clears_everything():
    _state.advance("a")
    _state.advance("b")
    n = _state.reset()
    assert n == 2
    assert _state.snapshot() == {}


def test_snapshot_shape():
    _state.record_tool_results("c1", {"messages": [{"role": "tool", "content": "x"}]})
    _state.advance("c1")
    snap = _state.snapshot()
    assert "c1" in snap
    assert snap["c1"]["turn"] == 1
    assert snap["c1"]["tool_results_seen"] == 1
