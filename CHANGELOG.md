# Changelog

All notable changes to fakellm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-06

The headline feature: rules can now match on conversation state, not just on
individual request shape. This unlocks realistic agent testing — your mock
can return a tool call on turn 1, a summary on turn 2, and a follow-up on
turn 3, all driven from a single YAML file.

### Added

- **Conversation tracking.** Requests are bucketed into conversations by a
  stable hash of the first user message. Turn count and tool-result history
  are tracked per conversation.
- **New `when:` matchers:**
  - `turn: N` — match the Nth turn of a conversation.
  - `turn_in: [low, high]` — match a turn within an inclusive range.
  - `previous_message_role: tool` — match the role of the message immediately
    before the latest one.
  - `previous_message_contains: "..."` — substring match on the previous
    message's text.
  - `tool_result_contains: "..."` — match if any tool result, in this request
    or earlier in this conversation, contains the substring.
- **`X-Fakellm-Conversation-Id` header.** Sent on every response so clients
  can see which conversation they were bucketed into. Clients can also send
  this header on requests to override the auto-derived ID — useful for tests.
- **`POST /_fakellm/reset`** — clear all conversation state. Stats and
  recent-request history are preserved. Use between tests.
- **`GET /_fakellm/conversations`** — JSON snapshot of active conversations
  with turn count and tool-result count for each.
- **Dashboard updates.** `/_fakellm` now shows active conversations and
  per-request turn numbers in the recent-requests table.

### Changed

- `match_request()` now accepts an optional `state: ConversationState`
  argument carrying multi-turn context. When omitted (e.g. from older callers),
  it defaults to a turn-1 state with no history, so all existing matchers
  behave exactly as before.

### Backward compatibility

All existing matchers (`messages_contain`, `model_matches`, `tools_include`,
`header.*`) continue to work unchanged. Existing config files run without
modification — the new matchers are purely additive.

## [0.1.1] — 2026-04-30

Initial public release.

### Added

- OpenAI and Anthropic API-compatible endpoints (`/v1/chat/completions`,
  `/v1/messages`).
- YAML-based rules engine with `messages_contain`, `model_matches`,
  `tools_include`, and `header.*` matchers.
- Streaming support for both OpenAI (`data: ...` SSE) and Anthropic
  (typed event sequence) formats.
- Tool call mocking in both API shapes, with chunked argument streaming.
- Per-rule status codes for error response testing.
- HTML dashboard at `/_fakellm` showing request stats and history.
- Hot config reload via `POST /_fakellm/reload`.
- CLI: `fakellm init`, `fakellm serve`.

[0.2.0]: https://github.com/yourname/fakellm/releases/tag/v0.2.0
[0.1.1]: https://github.com/yourname/fakellm/releases/tag/v0.1.1
