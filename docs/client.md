# `connector::Connector` — the client

The [`Connector`] is the whole interface. Construct it with a provider name,
configure it with builder methods, then call one of the request methods.

```rust,ignore
pub struct Connector { /* provider config + accumulated params */ }
```

It is `Clone`, `Debug`, and cheap to share (the params are the only state).
The provider config is `'static` — a connector never holds a borrowed
lifetime.

## Construction and builder methods

```rust,ignore
pub fn new(provider: &str) -> Result<Self, ConnectorError>   // UnknownProvider for bad names
```

Every `with_*` method returns `Self` (builder style) and is `#[must_use]`:

| Method | Parameter | Notes |
|---|---|---|
| `with_model` | `impl Into<String>` | Overrides the provider's default model |
| `with_max_tokens` | `u32` | `const fn` |
| `with_temperature` | `f32` | 0.0–2.0 |
| `with_top_p` | `f32` | nucleus sampling |
| `with_stop` | `serde_json::Value` | stop sequences |
| `with_frequency_penalty` / `with_presence_penalty` | `f32` | −2.0–2.0 |
| `with_seed` | `i64` | deterministic sampling |
| `with_response_format` | `ResponseFormat` | e.g. `json_object()` |
| `with_logprobs` / `with_top_logprobs` | `bool` / `u32` | |
| `with_tools` / `set_tools` | `Vec<ToolDefinition>` | builder / in-place |
| `with_tool_choice` | `serde_json::Value` | `"auto"`, `"none"`, specific tool |
| `with_reasoning_effort` | `impl Into<String>` | `"low"`/`"medium"`/`"high"`; mapped per family |
| `with_user` | `impl Into<String>` | end-user id for monitoring |
| `with_base_url` | `impl Into<String>` | override the endpoint |
| `with_api_key` | `impl Into<String>` | explicit key (highest priority) |
| `with_service_keyring` | `impl Into<String>` | keyring service override (default `"cosh"`) |

Only `new` is fallible. Everything else just accumulates into `Parameters` —
the request is validated against the provider at call time, not build time.

## Request methods (async)

| Method | Returns | Purpose |
|---|---|---|
| `chat(prompt)` | `Result<ChatOutput>` | single user prompt |
| `chat_with_system(prompt, system)` | `Result<ChatOutput>` | user + system prompt |
| `embed(input)` | `Result<Vec<f32>>` | embedding vector (not Claude) |
| `stream_chat(prompt)` | `Result<ChatStream>` | SSE stream of tokens |
| `stream_chat_with_system(prompt, system)` | `Result<ChatStream>` | |
| `stream_chat_with_system_no_tools(prompt, system)` | `Result<ChatStream>` | tools stripped from the request |
| `stream_chat_with_messages(system, &[ChatMessage])` | `Result<ChatStream>` | full message history with tool-call roles |
| `list_models()` | `Result<LsOutput>` | available models from the provider |

**`stream_chat_with_system_no_tools`** exists for a specific reason: a
sub-agent (e.g. the summarizer) must never see the main loop's tool schemas,
or it will answer with tool calls instead of prose. It clones the params and
clears `tools`/`tool_choice`, leaving the shared connector's tools intact.

**`stream_chat_with_messages`** is the tool-use workhorse: it takes a full
[`ChatMessage`](params.md) array so the model sees the conversation with
proper `role: "assistant"` tool-call messages and `role: "tool"` results.

`embed` returns `NotImplemented("embedding")` for Claude — the Anthropic API
has no embeddings endpoint.

## Streaming

```rust,ignore
use tokio_stream::StreamExt;

let mut stream = connector.stream_chat("hello").await?;
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    print!("{}", chunk.token());          // text delta
    if let Some(r) = chunk.reasoning() {  // reasoning/thinking delta
        print!("[{}]", r);
    }
}
let meta = stream.raw().await?;           // last SSE frame: usage, finish_reason
```

A [`ChatStream`](output.md) yields `Result<StreamChunk>` items; after the
stream ends, `raw()` returns the last SSE frame (usage, finish_reason, etc.).

## Token accounting and introspection

```rust,ignore
pub fn tokens(&self, raw: &str) -> Option<u32>   // completion tokens from a raw response
pub fn provider_name(&self) -> Option<&'static str>
pub fn is_local(&self) -> bool                   // base_url is localhost/127.0.0.1
pub fn model(&self) -> Option<&str>              // the explicit override, if set
pub fn effective_model(&self) -> Option<&str>    // override, else provider default
pub fn has_tools(&self) -> bool                  // non-empty tools registered
```

`tokens` reads the provider-specific usage field from a raw response
(`usage.completion_tokens` for OpenAI-compatible, `usage.output_tokens` for
Claude, `usageMetadata.candidatesTokenCount` for Gemini).

`is_local()` matters for tool-calling behavior: local model servers (ollama,
lmstudio, vllm, llamacpp) may lack reliable native function calling, so the
harness keeps the legacy inline-JSON tool prompt for them while cloud
providers get native function-calling instructions.

---

Next: [params — messages, tools, formats](params.md).

---

## Summary

- `Connector` is the main client interface — construct with a provider name, configure with builder methods, then call request methods.
- Builder methods (`with_model`, `with_temperature`, etc.) are `#[must_use]` and accumulate into `Parameters`; only `new` is fallible.
- Request methods: `chat` (single user prompt), `chat_with_system` (user + system), `embed` (embedding vector, not Claude), and streaming variants.
- Streaming yields `StreamChunk` items with `token()` (text delta) and `reasoning()` (thinking delta); after the stream ends, `raw()` returns the final SSE frame.
- Special streaming methods: `stream_chat_with_system_no_tools` (sub-agents must not see tool schemas) and `stream_chat_with_messages` (full conversation history with tool-call roles).
- Token accounting: `tokens()` reads provider-specific usage fields from raw responses; `is_local()` detects localhost servers for tool-calling behavior differences.
- Introspection methods: `provider_name()`, `is_local()`, `model()`, `effective_model()`, and `has_tools()` provide metadata about the connector's configuration.
