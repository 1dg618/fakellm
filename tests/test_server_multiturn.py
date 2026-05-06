"""End-to-end tests: drive the FastAPI app in-process and verify the full
multi-turn flow over HTTP, including conversation IDs, turn counting,
tool-result tracking, the /_fakellm/reset endpoint, and conversation isolation.

Uses httpx.ASGITransport so no real socket is needed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport


# A YAML config exercising every new matcher the multi-turn feature ships with.
CONFIG_YAML = """
version: 1

defaults:
  fallback: deterministic_echo

rules:
  - name: agent_turn_1_search
    when:
      turn: 1
      messages_contain: "research"
    respond:
      tool_calls:
        - name: web_search
          arguments: {query: "fakellm"}

  - name: agent_turn_2_summarize
    when:
      turn: 2
      tool_result_contains: "found"
    respond:
      content: "Based on the search, I found what you were looking for."

  - name: turn_three_or_later
    when:
      turn_in: [3, 99]
    respond:
      content: "Continuing the conversation..."

  - name: greeting
    when:
      messages_contain: "hello"
    respond:
      content: "Hi there!"
"""


@pytest.fixture
def app():
    """Load the app with our test config and a clean state, return the app."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "fakellm.yaml"
        config_path.write_text(CONFIG_YAML)
        os.environ["FAKELLM_CONFIG"] = str(config_path)

        # Import after FAKELLM_CONFIG is set so module-level load_config picks it up.
        # Reload the server module if it was already imported.
        import importlib

        from fakellm import _state, server

        importlib.reload(server)
        _state.reset()

        yield server.app

        _state.reset()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


# ---------------- conversation ID is returned and stable ----------------


@pytest.mark.asyncio
async def test_response_includes_conversation_id_header(client):
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    cid = r.headers.get("x-fakellm-conversation-id")
    assert cid and cid.startswith("conv-")


@pytest.mark.asyncio
async def test_same_conversation_keeps_same_id(client):
    body1 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Plan a trip to Paris"}],
    }
    body2 = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Plan a trip to Paris"},
            {"role": "assistant", "content": "Sure, what dates?"},
            {"role": "user", "content": "Next month"},
        ],
    }
    r1 = await client.post("/v1/chat/completions", json=body1)
    r2 = await client.post("/v1/chat/completions", json=body2)
    assert (
        r1.headers["x-fakellm-conversation-id"]
        == r2.headers["x-fakellm-conversation-id"]
    )


@pytest.mark.asyncio
async def test_explicit_conversation_id_header(client):
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "anything"}],
    }
    r = await client.post(
        "/v1/chat/completions",
        json=body,
        headers={"X-Fakellm-Conversation-Id": "my-test-session"},
    )
    assert r.headers["x-fakellm-conversation-id"] == "my-test-session"


# ---------------- multi-turn agent flow ----------------


@pytest.mark.asyncio
async def test_agent_two_turn_flow(client):
    """Turn 1: user asks for research → mock returns tool_call.
    Turn 2: code returns the tool result → mock returns summary text.

    This is the core use case the multi-turn feature exists for.
    """
    # Turn 1: initial request
    turn1_body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Please research fakellm"}],
    }
    r1 = await client.post("/v1/chat/completions", json=turn1_body)
    assert r1.status_code == 200
    data1 = r1.json()

    # Should have come back with a tool call, not text
    msg = data1["choices"][0]["message"]
    assert msg.get("content") is None
    assert msg.get("tool_calls"), "expected a tool_call on turn 1"
    tool_call = msg["tool_calls"][0]
    assert tool_call["function"]["name"] == "web_search"

    # Turn 2: simulate the agent loop calling the tool and returning the result
    turn2_body = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Please research fakellm"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "Search results: found 696 downloads on pepy.tech",
            },
        ],
    }
    r2 = await client.post("/v1/chat/completions", json=turn2_body)
    assert r2.status_code == 200
    data2 = r2.json()

    # Should match agent_turn_2_summarize
    content2 = data2["choices"][0]["message"]["content"]
    assert "found what you were looking for" in content2

    # And same conversation ID throughout
    assert (
        r1.headers["x-fakellm-conversation-id"]
        == r2.headers["x-fakellm-conversation-id"]
    )


@pytest.mark.asyncio
async def test_turn_in_range_matcher_fires_for_later_turns(client):
    """Once we're past turn 2, the turn_in:[3,99] rule should win."""
    msgs = [{"role": "user", "content": "Please research fakellm"}]

    # Turn 1: tool call
    r = await client.post(
        "/v1/chat/completions", json={"model": "gpt-4", "messages": msgs}
    )
    assert r.status_code == 200

    # Turn 2: tool result -> summary
    msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [r.json()["choices"][0]["message"]["tool_calls"][0]],
        }
    )
    msgs.append(
        {
            "role": "tool",
            "tool_call_id": msgs[-1]["tool_calls"][0]["id"],
            "content": "found stuff",
        }
    )
    r = await client.post(
        "/v1/chat/completions", json={"model": "gpt-4", "messages": msgs}
    )
    assert r.status_code == 200

    # Turn 3: should now match turn_three_or_later
    msgs.append({"role": "assistant", "content": r.json()["choices"][0]["message"]["content"]})
    msgs.append({"role": "user", "content": "anything else?"})
    r = await client.post(
        "/v1/chat/completions", json={"model": "gpt-4", "messages": msgs}
    )
    assert r.status_code == 200
    assert "Continuing" in r.json()["choices"][0]["message"]["content"]


# ---------------- conversation isolation ----------------


@pytest.mark.asyncio
async def test_two_conversations_have_independent_turn_counters(client):
    """Different first messages = different conversations = independent turn counts."""
    body_a = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Conversation A topic"}],
    }
    body_b = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Conversation B topic"}],
    }

    # Make 3 requests for A
    for _ in range(3):
        await client.post("/v1/chat/completions", json=body_a)

    # Check stats — A should be at turn 3, B not present yet
    stats = (await client.get("/_fakellm/stats")).json()
    convs = stats["conversations"]
    assert len(convs) == 1
    a_id = next(iter(convs))
    assert convs[a_id]["turn"] == 3

    # Now make 1 request for B
    r = await client.post("/v1/chat/completions", json=body_b)
    b_id = r.headers["x-fakellm-conversation-id"]

    convs = (await client.get("/_fakellm/conversations")).json()
    assert convs[a_id]["turn"] == 3
    assert convs[b_id]["turn"] == 1
    assert a_id != b_id


# ---------------- /_fakellm/reset ----------------


@pytest.mark.asyncio
async def test_reset_clears_conversations(client):
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "kickoff"}],
    }
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)

    convs_before = (await client.get("/_fakellm/conversations")).json()
    assert len(convs_before) == 1

    r = await client.post("/_fakellm/reset")
    assert r.status_code == 200
    assert r.json()["cleared_conversations"] == 1

    convs_after = (await client.get("/_fakellm/conversations")).json()
    assert convs_after == {}


@pytest.mark.asyncio
async def test_reset_resets_turn_count(client):
    """After reset, the same conversation starts back at turn 1."""
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Please research fakellm"}],
    }
    # Bump turn to 2
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)

    await client.post("/_fakellm/reset")

    # Same first-user-message → same conversation id, but turn should be 1 again,
    # and the turn:1 rule should fire again (returning a tool call).
    r = await client.post("/v1/chat/completions", json=body)
    msg = r.json()["choices"][0]["message"]
    assert msg.get("tool_calls"), "after reset, turn-1 rule should fire again"


# ---------------- Anthropic API ----------------


@pytest.mark.asyncio
async def test_anthropic_endpoint_also_tracks_turns(client):
    body1 = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Please research fakellm"}],
        "max_tokens": 100,
    }
    r1 = await client.post("/v1/messages", json=body1)
    assert r1.status_code == 200
    data1 = r1.json()
    # Turn 1 rule emits a tool call → Anthropic format has tool_use blocks
    blocks = data1["content"]
    assert any(b.get("type") == "tool_use" for b in blocks)

    # Build turn 2 with tool_result block
    tool_use = next(b for b in blocks if b["type"] == "tool_use")
    body2 = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Please research fakellm"},
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": "found 696 downloads",
                    }
                ],
            },
        ],
    }
    r2 = await client.post("/v1/messages", json=body2)
    assert r2.status_code == 200
    text_blocks = [b for b in r2.json()["content"] if b.get("type") == "text"]
    assert text_blocks
    assert "found what you were looking for" in text_blocks[0]["text"]


# ---------------- regression: existing matchers still work ----------------


@pytest.mark.asyncio
async def test_messages_contain_still_works(client):
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    # Note: the turn:1 + research rule won't match (no "research"), and
    # the greeting rule should win.
    assert "Hi there" in r.json()["choices"][0]["message"]["content"]
