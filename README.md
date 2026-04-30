# fakellm

**Run your LLM tests offline. Free, fast, deterministic.**

A mock server that speaks the OpenAI and Anthropic APIs. Point your test code at it and your tests stop being slow, expensive, and flaky.

```
pip install fakellm
fakellm init
fakellm serve
```

Then in your tests:

```python
import os
os.environ["OPENAI_BASE_URL"] = "http://localhost:9999/v1"
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:9999"

# your existing code runs unchanged
```

## Why

Three bad options exist for testing LLM code today:

1. **Hit the real API** — slow, expensive, flaky.
2. **Mock by hand** — brittle, drifts, doesn't exercise streaming or tool-call code paths.
3. **Record-and-replay cassettes** — go stale, blow up when prompts change.

`fakellm` is a fourth option. A local server that returns plausible responses in the right shape, controlled by a small YAML file. Same prompt → same response, every time.

## Configure

`fakellm.yaml`:

```yaml
version: 1

defaults:
  fallback: deterministic_echo

rules:
  - name: greeting
    when:
      messages_contain: "hello"
    respond:
      content: "Hi there!"

  - name: weather_tool
    when:
      tools_include: get_weather
    respond:
      tool_calls:
        - name: get_weather
          arguments: { location: "San Francisco" }

  - name: only_haiku
    when:
      model_matches: "*haiku*"
    respond:
      content: "Short response."

  - name: rate_limit_test
    when:
      header.x-test-scenario: rate_limit
    respond:
      status: 429
      error: "Rate limit exceeded"
```

Rules are walked top-to-bottom. First match wins. If nothing matches, you get a stable fingerprint response — same input gives the same output, forever.

## Use with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9999/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)  # → "Hi there!"
```

## Use with the Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:9999", api_key="not-needed")
resp = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.content[0].text)
```

## Streaming works

Both APIs stream chunks the way the real ones do. Your streaming code paths get exercised.

## Simulate failures

Set a header in your test to trigger a specific scenario:

```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    extra_headers={"x-test-scenario": "rate_limit"},
)
# raises a 429 just like the real API
```

## Dashboard

Visit `http://localhost:9999/_fakellm` to see which rules are matching and which requests are falling through. Useful for tightening up your config.

## What's in v0.1

- OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`
- Streaming for both
- Matchers: `messages_contain`, `model_matches`, `tools_include`, `header.*`
- Tool call responses
- Error/status code responses
- Deterministic fallback
- Live dashboard

## Roadmap

- Multi-turn response sequences for agentic tests
- Recorded fixture mode (point at real API, capture, replay)
- pytest plugin with inline rule definitions
- More matchers (semantic similarity, JSON schema)

## License

MIT
