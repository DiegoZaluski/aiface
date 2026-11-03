# `connector::ConnectorError`

The single error type for the whole connector surface.

```rust,ignore
pub enum ConnectorError {
    UnknownProvider(String),        // bad name passed to Connector::new
    MissingApiKey(String),          // no key in keyring or env for the provider
    HttpError { status: u16, body: String },
    ContextWindowExceeded { status: u16, body: String, window_tokens: Option<usize> },
    Deserialization(String),        // response body wasn't valid JSON
    Network(String),                // transport-level failure
    NoChoices,                      // empty `choices` array in chat response
    NoContent,                      // choice with no `content` field
    NoEmbeddings,                   // empty `data` array in embedding response
    NotImplemented(&'static str),   // e.g. "embedding" for Claude
    StreamTerminated,               // SSE stream ended without [DONE]
}
```

Every variant `Display`s to a descriptive message (`Unknown provider: ...`,
`API key not set for provider: ...`, `HTTP {status} - {body}`, ...) and the
type implements `std::error::Error`. `reqwest::Error` and
`serde_json::Error` convert into `Network` / `Deserialization` via `From`.

## The context-window classifier

The interesting piece is `classify_http`:

```rust,ignore
pub fn classify_http(status: u16, body: String) -> Self
```

Every non-2xx path in the SDK funnels through it. It inspects the **body**
(case-insensitive) for context-window overflow markers — `"context length"`,
`"maximum context"`, `"prompt is too long"`, `"too many tokens"`,
`"token limit"`, `"exceeds the maximum"`, `"reduce the length"`, … — and
returns `ContextWindowExceeded` when one is found, otherwise a plain
`HttpError`.

**Why the body and not the status?** The reliable signal for a context-window
overflow is the message text, not the HTTP code:

- OpenAI/Anthropic-style overflows arrive as **400** — but so do bad JSON,
  invalid tool arguments, and auth errors;
- payload-too-large arrives as **413**;
- **SSE error frames and OpenAI-compatible JSON error responses arrive with
  status 200.**

So a 200 with a "maximum context length is 64000 tokens" body is still
classified as `ContextWindowExceeded`, while a 400 with "invalid api key" is
correctly left as `HttpError`.

`window_tokens` is a heuristic hint: the smallest token-sized number (≥ 1000)
near a size keyword (`"maximum"`, `"limit"`, `"window"`, `"context"`,
`"max"`) in the body — e.g. `128000` from OpenAI's "maximum context length is
128000 tokens ... resulted in 150000 tokens". Falls back to the largest
token-sized number in the body when no keyword is present. It's for local
budget checks only, never a hard contract.

Use `err.is_context_window()` to test, or match on the variant directly.

---

Next: [provider — registry and keys](provider.md).

---

## Summary

- `ConnectorError` is the single error type for the whole connector surface, with variants for unknown provider, missing API key, HTTP errors, context window overflow, deserialization failures, and more.
- The interesting piece is `classify_http`, which inspects response bodies for context-window overflow markers (case-insensitive phrases like "context length", "maximum context", "too many tokens").
- Body inspection (not status code) is the reliable signal because: OpenAI/Anthropic overflows arrive as 400 (same as other errors), payload-too-large is 413, and SSE error frames can arrive with status 200.
- `ContextWindowExceeded` includes a heuristic `window_tokens` hint — the smallest token-sized number (≥ 1000) near a size keyword in the body.
- Use `err.is_context_window()` to test for overflow, or match on the variant directly; all variants implement `Display` and `std::error::Error`.
