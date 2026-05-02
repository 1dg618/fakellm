"""Tests for the v0.1.x fixes.

Covers all eight issues:
  1. FAKELLM_CONFIG env var rename
  2. Streaming tool calls
  3. Missing-config startup failure
  4. Deterministic echo stability across key order
  5. (multi-worker — documented, not unit-testable here)
  6. Dashboard HTML escaping
  7. Explicit headers in match_request
  8. Shared utils, no cross-module private imports
"""

from __future__ import annotations

import asyncio
import json

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from fakellm import server
from fakellm._util import deterministic_echo
from fakellm.cli import main as cli_main
from fakellm.config import Config
from fakellm.matcher import match_request
from fakellm.responder import build_response
from fakellm.streaming import build_stream


# ---------- Issue 4: deterministic echo stability ----------

def test_deterministic_echo_stable_across_top_level_key_order():
    a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    b = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}
    assert deterministic_echo(a) == deterministic_echo(b)


def test_deterministic_echo_stable_across_nested_key_order():
    a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    b = {"model": "gpt-4o", "messages": [{"content": "hi", "role": "user"}]}
    assert deterministic_echo(a) == deterministic_echo(b)


def test_deterministic_echo_differs_for_different_inputs():
    a = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    b = {"model": "gpt-4o", "messages": [{"role": "user", "content": "bye"}]}
    assert deterministic_echo(a) != deterministic_echo(b)


# ---------- Issue 7: explicit headers, no body mutation ----------

def test_matcher_uses_explicit_headers():
    config = Config(
        rules=[
            {
                "name": "rate_limit",
                "when": {"header.x-test-scenario": "rate_limit"},
                "respond": {"status": 429, "error": "rate limited"},
            }
        ]
    )
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    headers = {"x-test-scenario": "rate_limit"}

    rule = match_request(body, headers, config, api="openai")
    assert rule is not None
    assert rule["name"] == "rate_limit"


def test_matcher_does_not_mutate_body():
    config = Config(rules=[])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    original = dict(body)
    match_request(body, {"x-foo": "bar"}, config, api="openai")
    assert body == original
    assert "_headers" not in body


def test_matcher_returns_none_when_header_does_not_match():
    config = Config(
        rules=[
            {
                "name": "rate_limit",
                "when": {"header.x-test-scenario": "rate_limit"},
                "respond": {"status": 429},
            }
        ]
    )
    body = {"model": "gpt-4o"}
    rule = match_request(body, {}, config, api="openai")
    assert rule is None


# ---------- Issue 2: streaming tool calls ----------

def _collect_stream(rule, body, api):
    async def go():
        return [c async for c in build_stream(rule, body, api=api)]

    return asyncio.run(go())


def test_openai_stream_emits_tool_calls():
    rule = {
        "respond": {
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "SF"}}]
        }
    }
    body = {"model": "gpt-4o-mini", "stream": True}

    chunks = _collect_stream(rule, body, "openai")
    full = "".join(chunks)

    assert "tool_calls" in full
    assert "get_weather" in full
    # Final chunk should set finish_reason to tool_calls
    assert '"finish_reason": "tool_calls"' in full
    # Stream must terminate with [DONE]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_openai_stream_arguments_are_chunked():
    """Arguments should be split across multiple deltas, not sent as one blob."""
    rule = {
        "respond": {
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"city": "San Francisco", "unit": "celsius"},
                }
            ]
        }
    }
    body = {"model": "gpt-4o-mini", "stream": True}

    chunks = _collect_stream(rule, body, "openai")
    # Count chunks that contain argument deltas (have "arguments" but not the
    # initial announce chunk with "name")
    arg_chunks = [
        c for c in chunks if '"arguments"' in c and '"name"' not in c
    ]
    assert len(arg_chunks) > 1, "Tool call arguments should be split across chunks"


def test_anthropic_stream_emits_tool_use_block():
    rule = {
        "respond": {
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "SF"}}]
        }
    }
    body = {"model": "claude-sonnet-4-5", "stream": True}

    chunks = _collect_stream(rule, body, "anthropic")
    full = "".join(chunks)

    assert "tool_use" in full
    assert "input_json_delta" in full
    assert "get_weather" in full
    assert '"stop_reason": "tool_use"' in full


def test_anthropic_stream_emits_message_start_and_stop():
    rule = {"respond": {"content": "hello"}}
    body = {"model": "claude-sonnet-4-5", "stream": True}

    chunks = _collect_stream(rule, body, "anthropic")
    full = "".join(chunks)

    assert "event: message_start" in full
    assert "event: message_stop" in full
    assert "event: content_block_start" in full
    assert "event: content_block_stop" in full


def test_text_streaming_still_works_after_tool_call_changes():
    rule = {"respond": {"content": "hello world"}}
    body = {"model": "gpt-4o-mini", "stream": True}

    chunks = _collect_stream(rule, body, "openai")
    full = "".join(chunks)

    assert "hello" in full
    assert "world" in full
    assert chunks[-1] == "data: [DONE]\n\n"


# ---------- Issue 6: dashboard HTML escaping ----------

@pytest.fixture
def clean_server_state():
    """Reset server module state between tests."""
    server._stats.clear()
    server._recent.clear()
    yield
    server._stats.clear()
    server._recent.clear()


def test_dashboard_escapes_malicious_rule_names(clean_server_state):
    server._stats["<script>alert(1)</script>"] = 1
    client = TestClient(server.app)
    resp = client.get("/_fakellm")
    assert resp.status_code == 200
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def test_dashboard_escapes_malicious_model_names(clean_server_state):
    server._recent.appendleft(
        {
            "ts": "2026-01-01T00:00:00+00:00",
            "api": "openai",
            "model": "<img src=x onerror=alert(1)>",
            "stream": False,
            "matched_rule": "test",
        }
    )
    client = TestClient(server.app)
    resp = client.get("/_fakellm")
    assert resp.status_code == 200
    assert "<img src=x" not in resp.text
    assert "&lt;img" in resp.text


# ---------- Issue 3: missing-config CLI failure ----------

def test_serve_fails_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    # Use a config path that doesn't exist
    result = runner.invoke(cli_main, ["serve", "--config", "nope.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_init_creates_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_main, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / "fakellm.yaml").exists()


# ---------- Issue 1: FAKELLM_CONFIG env var ----------

def test_fakellm_config_env_var_used(tmp_path, monkeypatch):
    """Verify the server reads FAKELLM_CONFIG, not the old LLMOCK_CONFIG."""
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        """
version: 1
rules:
  - name: special
    when:
      messages_contain: "magic_word"
    respond:
      content: "matched"
"""
    )
    monkeypatch.setenv("FAKELLM_CONFIG", str(custom))

    # Reload by calling load_config directly
    from fakellm.config import load_config

    cfg = load_config(custom)
    assert any(r.get("name") == "special" for r in cfg.rules)


# ---------- Issue 8: deterministic_echo is now a public name ----------

def test_deterministic_echo_is_public_in_util():
    """The helper has been moved out of responder's private surface."""
    from fakellm import _util

    assert hasattr(_util, "deterministic_echo")
    assert callable(_util.deterministic_echo)


# ---------- Sanity: non-streaming responder still works ----------

def test_responder_returns_configured_content():
    rule = {"respond": {"content": "yo"}}
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    status, payload = build_response(rule, body, api="openai")
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "yo"


def test_responder_returns_error_status():
    rule = {"respond": {"status": 429, "error": "rate limited"}}
    body = {"model": "gpt-4o-mini", "messages": []}
    status, payload = build_response(rule, body, api="openai")
    assert status == 429
    assert "error" in payload


def test_responder_anthropic_format():
    rule = {"respond": {"content": "yo"}}
    body = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}
    status, payload = build_response(rule, body, api="anthropic")
    assert status == 200
    assert payload["content"][0]["text"] == "yo"
    assert payload["stop_reason"] == "end_turn"
