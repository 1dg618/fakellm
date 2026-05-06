"""FastAPI server. Mounts OpenAI and Anthropic compatible endpoints.

Note: this server stores per-process state (config, stats, recent requests,
conversations) at module level. Running with multiple uvicorn workers will
partition that state across workers and is not currently supported. Run with
a single worker.
"""

from __future__ import annotations

import html
import os
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import _state
from .config import Config, load_config
from .matcher import match_request
from .responder import build_response
from .streaming import build_stream

app = FastAPI(title="fakellm", description="A mock LLM server for testing.")

# Module-level state. Reload via /_fakellm/reload, reset conversations via
# /_fakellm/reset.
# NOTE: not safe across multiple uvicorn workers — see module docstring.
_MAX_RECENT = 50
_config: Config = load_config(os.environ.get("FAKELLM_CONFIG", "fakellm.yaml"))
_stats: Counter[str] = Counter()
_recent: Deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT)


def _record(
    api: str,
    body: dict[str, Any],
    rule: dict[str, Any] | None,
    conversation_id: str,
    turn: int,
) -> None:
    rule_name = rule.get("name", "<unnamed>") if rule else "<fallthrough>"
    _stats[rule_name] += 1
    _recent.appendleft(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "api": api,
            "model": body.get("model"),
            "stream": bool(body.get("stream")),
            "matched_rule": rule_name,
            "conversation_id": conversation_id,
            "turn": turn,
        }
    )


async def _handle(request: Request, api: str):
    body = await request.json()
    headers = {k.lower(): v for k, v in request.headers.items()}

    conversation_id = _state.derive_conversation_id(body, headers)
    # Record any tool results in the incoming history *before* advancing,
    # so the current turn's matchers can see them.
    _state.record_tool_results(conversation_id, body)
    state = _state.advance(conversation_id)

    rule = match_request(body, headers, _config, api=api, state=state)
    _record(api, body, rule, conversation_id, state.turn)

    if body.get("stream"):
        return StreamingResponse(
            build_stream(rule, body, api=api),
            media_type="text/event-stream",
            headers={"X-Fakellm-Conversation-Id": conversation_id},
        )

    status, payload = build_response(rule, body, api=api)
    return JSONResponse(
        payload,
        status_code=status,
        headers={"X-Fakellm-Conversation-Id": conversation_id},
    )


@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    return await _handle(request, api="openai")


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    return await _handle(request, api="anthropic")


@app.get("/_fakellm/stats")
async def stats():
    return {
        "total_requests": sum(_stats.values()),
        "by_rule": dict(_stats),
        "recent": list(_recent),
        "conversations": _state.snapshot(),
    }


@app.get("/_fakellm/conversations")
async def conversations():
    return _state.snapshot()


@app.post("/_fakellm/reload")
async def reload_config():
    global _config
    _config = load_config(os.environ.get("FAKELLM_CONFIG", "fakellm.yaml"))
    return {"reloaded": True, "rules": len(_config.rules)}


@app.post("/_fakellm/reset")
async def reset():
    """Clear all conversation state. Stats and recent requests are preserved."""
    cleared = _state.reset()
    return {"cleared_conversations": cleared}


@app.get("/_fakellm", response_class=HTMLResponse)
async def dashboard():
    def esc(value: Any) -> str:
        return html.escape(str(value)) if value is not None else ""

    rows = "".join(
        f"<tr><td>{esc(r['ts'])}</td><td>{esc(r['api'])}</td><td>{esc(r['model'])}</td>"
        f"<td>{'yes' if r['stream'] else 'no'}</td>"
        f"<td>{esc(r.get('conversation_id', ''))[:18]}</td>"
        f"<td>{esc(r.get('turn', ''))}</td>"
        f"<td>{esc(r['matched_rule'])}</td></tr>"
        for r in _recent
    )
    rule_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{count}</td></tr>"
        for name, count in _stats.most_common()
    )
    convs = _state.snapshot()
    conv_rows = "".join(
        f"<tr><td>{esc(cid)[:24]}</td><td>{info['turn']}</td>"
        f"<td>{info['tool_results_seen']}</td></tr>"
        for cid, info in sorted(convs.items(), key=lambda kv: -kv[1]["turn"])
    )
    return f"""<!doctype html>
<html><head><title>fakellm</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }}
th {{ background: #f7f7f7; }}
.tot {{ font-size: 1.2rem; }}
</style></head><body>
<h1>fakellm dashboard</h1>
<p class="tot">Total requests: <strong>{sum(_stats.values())}</strong> &nbsp;|&nbsp;
Active conversations: <strong>{len(convs)}</strong></p>
<h2>Matches by rule</h2>
<table><tr><th>Rule</th><th>Count</th></tr>{rule_rows or '<tr><td colspan=2>none yet</td></tr>'}</table>
<h2>Conversations</h2>
<table><tr><th>ID</th><th>Turns</th><th>Tool results seen</th></tr>
{conv_rows or '<tr><td colspan=3>none yet</td></tr>'}</table>
<h2>Recent requests</h2>
<table><tr><th>Time</th><th>API</th><th>Model</th><th>Stream</th><th>Conversation</th><th>Turn</th><th>Matched</th></tr>
{rows or '<tr><td colspan=7>none yet</td></tr>'}</table>
</body></html>"""
