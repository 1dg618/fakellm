"""FastAPI server. Mounts OpenAI and Anthropic compatible endpoints."""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import Config, load_config
from .matcher import match_request
from .responder import build_response
from .streaming import build_stream

app = FastAPI(title="fakellm", description="A mock LLM server for testing.")

# Module-level state. Reload via /_fakellm/reload.
_config: Config = load_config(os.environ.get("LLMOCK_CONFIG", "fakellm.yaml"))
_stats: Counter[str] = Counter()
_recent: list[dict[str, Any]] = []
_MAX_RECENT = 50


def _record(api: str, body: dict[str, Any], rule: dict[str, Any] | None) -> None:
    rule_name = rule.get("name", "<unnamed>") if rule else "<fallthrough>"
    _stats[rule_name] += 1
    _recent.insert(
        0,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "api": api,
            "model": body.get("model"),
            "stream": bool(body.get("stream")),
            "matched_rule": rule_name,
        },
    )
    del _recent[_MAX_RECENT:]


async def _handle(request: Request, api: str):
    body = await request.json()
    body["_headers"] = {k.lower(): v for k, v in request.headers.items()}

    rule = match_request(body, _config, api=api)
    _record(api, body, rule)

    body.pop("_headers", None)

    if body.get("stream"):
        return StreamingResponse(
            build_stream(rule, body, api=api),
            media_type="text/event-stream",
        )

    status, payload = build_response(rule, body, api=api)
    return JSONResponse(payload, status_code=status)


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
        "recent": _recent,
    }


@app.post("/_fakellm/reload")
async def reload_config():
    global _config
    _config = load_config(os.environ.get("LLMOCK_CONFIG", "fakellm.yaml"))
    return {"reloaded": True, "rules": len(_config.rules)}


@app.get("/_fakellm", response_class=HTMLResponse)
async def dashboard():
    rows = "".join(
        f"<tr><td>{r['ts']}</td><td>{r['api']}</td><td>{r['model']}</td>"
        f"<td>{'yes' if r['stream'] else 'no'}</td><td>{r['matched_rule']}</td></tr>"
        for r in _recent
    )
    rule_rows = "".join(
        f"<tr><td>{name}</td><td>{count}</td></tr>"
        for name, count in _stats.most_common()
    )
    return f"""<!doctype html>
<html><head><title>fakellm</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }}
th {{ background: #f7f7f7; }}
.tot {{ font-size: 1.2rem; }}
</style></head><body>
<h1>fakellm dashboard</h1>
<p class="tot">Total requests: <strong>{sum(_stats.values())}</strong></p>
<h2>Matches by rule</h2>
<table><tr><th>Rule</th><th>Count</th></tr>{rule_rows or '<tr><td colspan=2>none yet</td></tr>'}</table>
<h2>Recent requests</h2>
<table><tr><th>Time</th><th>API</th><th>Model</th><th>Stream</th><th>Matched</th></tr>
{rows or '<tr><td colspan=5>none yet</td></tr>'}</table>
</body></html>"""
