# Changelog

All notable changes to `fakellm` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-05-02

First maintenance release. Closes eight issues from an internal code review covering correctness bugs, a feature gap in streaming, and several refactors.

### Added
- Streaming tool calls for both OpenAI and Anthropic formats. Rules with `respond.tool_calls` now work correctly when the request has `stream: true`. Previously these were silently dropped and the deterministic echo fallback streamed instead.
- OpenAI streams emit `tool_calls` deltas with indexed function calls and chunked argument JSON, exercising SDK streaming-parse code paths.
- Anthropic streams emit `tool_use` content blocks with `input_json_delta` events and a `stop_reason` of `tool_use`.
- `fakellm serve` now validates that the config file exists at startup and fails loudly with a clear error message if it does not. Previously the server would start silently and every request would fall through to the deterministic echo, leaving users confused about why their rules weren't matching.

### Changed
- **Breaking (internal):** `match_request()` now takes `headers` as an explicit parameter rather than reading from a magic `_headers` key on the request body. Callers should update from `match_request(body, config, api=...)` to `match_request(body, headers, config, api=...)`. This eliminates request body mutation and makes the matcher's dependencies explicit.
- Renamed the configuration environment variable from `LLMOCK_CONFIG` to `FAKELLM_CONFIG` to match the project name. The old name is no longer recognized.
- Extracted `deterministic_echo`, `approx_tokens`, and `count_tokens_from_messages` from `responder.py` into a new `_util.py` module. Both `responder.py` and `streaming.py` now import from this shared location instead of cross-importing private names.
- The recent-requests log in `server.py` now uses `collections.deque(maxlen=50)` for O(1) appends instead of a list with manual trimming.

### Fixed
- `deterministic_echo` produced different fingerprints for semantically identical requests when nested dictionary keys had different insertion order. The hash now uses `json.dumps(body, sort_keys=True)` so the same logical request always produces the same fingerprint, regardless of how the JSON was constructed.
- The dashboard at `/_fakellm` now HTML-escapes all user-controlled values (rule names, model strings, request metadata) before rendering. Previously a rule named `<script>alert(1)</script>` would be rendered as live HTML in the dashboard. Risk was low in practice since the dashboard binds to localhost by default, but the fix is a one-line addition of `html.escape()`.
- The streaming code path no longer leaks the internal `_headers` key into the request body passed to downstream handlers. Resolved as a side effect of the `match_request` signature change.

### Known limitations
- Module-level state (config, stats, recent requests) is per-process, so running with multiple uvicorn workers will partition that state across workers. Run with a single worker. This is now documented in the `server.py` module docstring; a future release may add shared state.
- Token counts use a `len(text) // 4` approximation rather than a real tokenizer. Accuracy is low for short inputs and non-English text. Tracked for a future release.

## [0.1.0] — 2026-04-30

Initial public release.

### Added
- Mock server speaking OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` API formats.
- YAML-driven rule configuration with `messages_contain`, `model_matches`, `tools_include`, and `header.*` matchers.
- Streaming responses for text content in both API formats.
- Tool call responses (non-streaming).
- Configurable error and status code responses for failure-injection testing.
- Deterministic fallback responses based on request fingerprint when no rule matches.
- Live dashboard at `/_fakellm` showing match counts and recent requests.
- `fakellm init` command to create a starter `fakellm.yaml`.
- `fakellm serve` command to start the server.
- PyPI package published as `fakellm`.

[0.1.1]: https://github.com/1dg618/fakellm/releases/tag/v0.1.1
[0.1.0]: https://github.com/1dg618/fakellm/releases/tag/v0.1.0
