# fakellm

**A mock OpenAI/Anthropic server for testing LLM apps without burning API credits.**

fakellm speaks the OpenAI and Anthropic HTTP APIs and returns whatever responses
you tell it to. Point your code at it instead of the real APIs in tests, CI,
and local development. Define behavior in a YAML file — including multi-turn
agent flows where turn 1 returns a tool call, turn 2 returns a summary, and
turn N returns whatever you want.

```bash
pip install fakellm
fakellm init      # creates fakellm.yaml
fakellm serve     # starts on http://127.0.0.1:9999
```

Then point your client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9999/v1", api_key="not-used")
```

---

## Why fakellm

Testing code that calls an LLM is annoying. Real APIs cost money and rate-limit
you. Recording-and-replay tools (VCR-style) go stale and can't cover error
paths. `unittest.mock.patch` works for unit tests but falls apart the moment
you have an agent that loops through tool calls.

fakellm fits between those:

| | Real API in tests | `unittest.mock` | VCR-style replay | **fakellm** |
|---|---|---|---|---|
| Free / fast | ❌ | ✅ | ✅ | ✅ |
| Multi-turn agent flows | ✅ | painful | ❌ | ✅ |
| Test error paths (429, 500, malformed) | hard to trigger | ✅ | ❌ | ✅ |
| Test streaming | ✅ | painful | partial | ✅ |
| No code changes vs. production | ✅ | ❌ | ✅ | ✅ |
| Shareable across services / languages | n/a | ❌ | ❌ | ✅ |

**What fakellm does and doesn't test.** fakellm verifies that *your code* handles
LLM responses correctly — parsing, tool-call dispatch, streaming, error handling,
and multi-turn agent loops. It does **not** tell you whether a real model returns
the right answer to your prompt, since the responses are the ones you define. Use
fakellm for fast, deterministic coverage of your plumbing, and pair it with a
small suite of real-API evals to catch prompt and model-behavior regressions.

---

## Multi-turn agents in 20 lines (new in 0.2)

Most mock servers can answer "what does turn N look like in isolation."
fakellm can describe a whole agent flow as data:

**fakellm.yaml**

```yaml
rules:
  # Turn 1: user asks for research → return a tool call
  - name: kickoff_research
    when:
      turn: 1
      messages_contain: "research"
    respond:
      tool_calls:
        - name: web_search
          arguments: {query: "fakellm"}

  # Turn 2: tool result came back → return a summary
  - name: summarize_results
    when:
      turn: 2
      tool_result_contains: "found"
    respond:
      content: "Based on the search, I found what you were looking for."
```

**test_my_agent.py**

```python
import httpx
import pytest
from openai import OpenAI

@pytest.fixture(autouse=True)
def reset_fakellm():
    httpx.post("http://127.0.0.1:9999/_fakellm/reset")

def test_agent_handles_search():
    client = OpenAI(base_url="http://127.0.0.1:9999/v1", api_key="not-used")
    result = run_my_agent(client, prompt="Please research fakellm")
    assert "found what you were looking for" in result
```

That's it. No mocks, no recordings, no real API calls. The agent loop runs
end-to-end against fakellm and you assert on the output.

---

## Features

- **Speaks both APIs.** Drop-in replacement for the OpenAI and Anthropic HTTP
  APIs — same request shapes, same response shapes, same SSE streaming formats.
  Point the OpenAI SDK at `http://127.0.0.1:9999/v1` and the Anthropic SDK at
  `http://127.0.0.1:9999` (its messages endpoint is served at `/v1/messages`).
- **Rules engine.** Match requests on prompt content, model name, tools, headers,
  conversation turn, previous message role/content, or tool-result content.
  First match wins.
- **Multi-turn aware.** Conversations are tracked across requests so rules can
  fire on "turn 2 after a tool result mentioned X."
- **Tool/function calls.** Mock tool calls in either OpenAI or Anthropic shape,
  including streaming chunked arguments.
- **Streaming.** Both `data: ...` SSE for OpenAI and the typed event sequence
  (`message_start`, `content_block_delta`, etc.) for Anthropic.
- **Error injection.** Per-rule status codes for 4xx/5xx testing.
- **Live dashboard.** Visit `http://127.0.0.1:9999/_fakellm` to see request
  history, matched rules, and active conversations.
- **Hot reload.** `POST /_fakellm/reload` re-reads the YAML without restarting.

---

## Installation

```bash
pip install fakellm
```

Requires Python 3.10+.

For exact, tiktoken-based token counts (instead of the default approximation),
install the `accurate` extra:

```bash
pip install fakellm[accurate]
```

---

## Quickstart

```bash
fakellm init       # creates fakellm.yaml in the current directory
fakellm serve      # starts the server on 127.0.0.1:9999
```

Edit `fakellm.yaml` to add rules. Either restart the server or
`curl -X POST http://127.0.0.1:9999/_fakellm/reload` to pick up changes.

---

## Endpoints

### LLM-compatible

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/messages` | Anthropic messages |

Both support `stream: true`.

### Admin

| Method | Path | Purpose |
|---|---|---|
| GET | `/_fakellm` | HTML dashboard |
| GET | `/_fakellm/stats` | JSON: request counts, recent requests, conversations |
| GET | `/_fakellm/conversations` | JSON: turn count + tool results per conversation |
| POST | `/_fakellm/reload` | Re-read the YAML config |
| POST | `/_fakellm/reset` | Clear all conversation state |

Every response also includes an `X-Fakellm-Conversation-Id` header so clients
can see which conversation they were bucketed into.

---

## Config reference

### Top-level structure

```yaml
version: 1

defaults:
  fallback: deterministic_echo  # what to return when no rule matches

rules:
  - name: my_rule
    when: { ... }      # conditions (all must match)
    respond: { ... }   # what to return
```

### Conditions (`when:`)

All conditions in a `when:` block must match for the rule to fire. Rules are
evaluated top-to-bottom; first match wins. A rule with no `when:` block
matches everything.

| Condition | Type | Description |
|---|---|---|
| `messages_contain` | string | Case-insensitive substring across all message content. |
| `model_matches` | glob | e.g. `gpt-4*`, `claude-*-sonnet-*`. |
| `tools_include` | string | Match if a tool with this name is defined in the request. |
| `turn` | int | Match the Nth turn of this conversation (1-indexed). |
| `turn_in` | `[low, high]` | Match a turn in this inclusive range. |
| `previous_message_role` | string | Role of the message immediately before the latest one (`user`, `assistant`, `tool`). |
| `previous_message_contains` | string | Substring match on the previous message's text. |
| `tool_result_contains` | string | Match if any tool result — in this request or earlier in this conversation — contains the substring. |
| `header.<name>` | string | Match a request header (e.g. `header.x-test-scenario: rate_limit`). |

### Responses (`respond:`)

| Key | Type | Description |
|---|---|---|
| `content` | string | Assistant text content. |
| `tool_calls` | list | List of `{name, arguments}` to return as tool calls. |
| `status` | int | HTTP status. Default 200. Set to 4xx/5xx for error responses. |
| `error` | string | Error message body (used when `status >= 400`). |

If neither `content` nor `tool_calls` is set, fakellm returns a deterministic
echo response derived from a hash of the request — useful for "I just need
*some* response" tests.

---

## Conversations

A conversation is identified by a stable hash of the first user message in
the request. Adding more turns doesn't change the ID, so the same conversation
keeps the same ID across all its turns.

To override the ID (useful in tests where you want explicit control), send
the `X-Fakellm-Conversation-Id` header with any value you want:

```python
client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    extra_headers={"X-Fakellm-Conversation-Id": "test-session-42"},
)
```

Between tests, call `POST /_fakellm/reset` to clear all conversation state.
Stats and request history are preserved.

---

## CLI

```
fakellm init                  # create fakellm.yaml
fakellm serve                 # start the server
fakellm serve --port 8080     # custom port
fakellm serve --host 0.0.0.0  # custom host
fakellm serve --config x.yaml # custom config path
fakellm serve --reload        # auto-reload on code changes (dev only)
fakellm serve --workers N     # uvicorn workers (see caveat below; default 1)
```

`--workers` defaults to 1 and should stay there for normal use — fakellm stores
state in process memory, so more than one worker partitions conversations and
stats across workers. The flag exists for advanced cases only and prints a
warning when set above 1.

---

## Caveats

- Single-worker only. fakellm stores config and conversation state in process
  memory; running with multiple uvicorn workers will partition that state
  across workers. Stick with the default single worker.
- Token counts are approximate (`len(text) // 4`) by default. Install the
  `accurate` extra for tiktoken-based counts: `pip install fakellm[accurate]`.
- Not for production traffic. fakellm is built for tests; it's not a
  production-ready proxy.

---

## Contributing

Issues and PRs welcome. See `CONTRIBUTING.md`.

---

## License

See `LICENSE`.
