"""Smoke tests. Run with: pytest tests/"""

from __future__ import annotations

from fastapi.testclient import TestClient

from fakellm.config import Config
from fakellm.matcher import match_request
from fakellm.responder import build_response


def test_messages_contain_matcher():
    config = Config(
        rules=[
            {
                "name": "hello_rule",
                "when": {"messages_contain": "hello"},
                "respond": {"content": "hi back"},
            }
        ]
    )
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello world"}]}
    rule = match_request(body, {}, config, api="openai")
    assert rule is not None
    assert rule["name"] == "hello_rule"


def test_no_match_returns_none():
    config = Config(
        rules=[
            {
                "name": "specific",
                "when": {"messages_contain": "foobar"},
                "respond": {"content": "ok"},
            }
        ]
    )
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    assert match_request(body, {}, config, api="openai") is None


def test_model_matches_glob():
    config = Config(
        rules=[
            {
                "name": "haiku_only",
                "when": {"model_matches": "*haiku*"},
                "respond": {"content": "fast"},
            }
        ]
    )
    body1 = {"model": "claude-haiku-4-5", "messages": []}
    body2 = {"model": "gpt-4o", "messages": []}
    assert match_request(body1, {}, config, api="openai") is not None
    assert match_request(body2, {}, config, api="openai") is None


def test_openai_response_shape():
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    status, payload = build_response(None, body, api="openai")
    assert status == 200
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in payload


def test_anthropic_response_shape():
    body = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}]}
    status, payload = build_response(None, body, api="anthropic")
    assert status == 200
    assert payload["type"] == "message"
    assert payload["content"][0]["type"] == "text"
    assert "usage" in payload


def test_error_status_passthrough():
    rule = {
        "name": "fail",
        "when": {},
        "respond": {"status": 429, "error": "slow down"},
    }
    body = {"model": "gpt-4o-mini", "messages": []}
    status, payload = build_response(rule, body, api="openai")
    assert status == 429
    assert "error" in payload


def test_server_openai_endpoint():
    from fakellm.server import app

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]


def test_server_anthropic_endpoint():
    from fakellm.server import app

    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["content"][0]["type"] == "text"


def test_deterministic_echo_is_stable():
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "abc"}]}
    _, r1 = build_response(None, body, api="openai")
    _, r2 = build_response(None, body, api="openai")
    assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]
