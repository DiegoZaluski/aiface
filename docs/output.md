# `connector::output` — responses and streams

The types returned by the [`Connector`](client.md) request methods.

## `ChatOutput` — one-shot reply

```rust,ignore
pub struct ChatOutput { /* raw, message, images */ }

impl ChatOutput {
    pub fn message(&self) -> &str    // extracted text of the model's reply
    pub fn raw(&self) -> &str        // raw JSON response, unmodified
    pub fn images(&self) -> &[GeneratedImage]  // images the model embedded in the answer
}
```

`raw()` is the escape hatch: usage, finish_reason, tool_calls, and anything
else the API returned is there to parse manually.

`images()` carries multi-modal OUTPUT — pixels the model generated INTO the
reply. Where they come from is a per-family wire shape:

- **Gemini** image-capable models (`gemini-2.5-flash-image`, …) answer
  `generateContent` with interleaved text/`inlineData` parts; every image
  arrives decoded. A withheld generation (`finishReason != "STOP"`)
  contributes nothing and the text survives.
- **OpenRouter-style gateways** carry data-URI images in
  `message.images[]` (Vercel AI Gateway and LiteLLM follow the same
  normalization); streaming frames use `delta.images[]`. Live http(s)
  URLs are NEVER fetched — the pixels must arrive inline.
- **OpenAI** chat messages never carry images; the Responses API's
  `image_generation` tool output (`image_generation_call` items) does.

Claude and Copilot have no image-output surface and always yield an
empty slice — `images().is_empty()` is the universal "no image this
turn" check, the same contract as `generate_image` safety blocks.

## `StreamChunk` — one item in a stream

```rust,ignore
pub struct StreamChunk { /* raw, token, reasoning, finish_reason, thinking_blocks, tool_call */ }

impl StreamChunk {
    pub fn token(&self) -> &str                      // text delta
    pub fn reasoning(&self) -> &str                  // reasoning/thinking delta
    pub fn thinking_blocks(&self) -> Option<&[ClaudeThinkingBlock]>  // Claude only
    pub fn finish_reason(&self) -> Option<&str>      // "stop", "length", ...
    pub fn tool_call(&self) -> Option<&NativeToolCall>  // provider-delivered structured call
}
```

- `reasoning()` is the streamed thinking text (many reasoning models emit it
  separately from the visible token). The TUI shows it in a collapsible
  "Thought" block and never echoes it back to the model.
- `thinking_blocks()` carries Claude extended-thinking blocks (text +
  signature) for verbatim replay — see [`ClaudeThinkingBlock`](params.md).
- `finish_reason()` is `Some` only on the final chunk: `"stop"`, `"length"`,
  `"content_filter"`, `"tool_calls"`, or provider-specific values.
- `tool_call()` carries a tool call the provider delivered NATIVELY
  (structured `tool_calls`/`tool_use`/`functionCall` parts — id, name, the
  raw accumulated `arguments` JSON text, and the Gemini `thought_signature`
  when present). The harness routes it straight to the extractor's native
  validation funnel (`ExtractAction::register_native_call`) — no
  inline-JSON round trip. It is `None` for text/reasoning chunks and for the
  legacy inline-JSON path (local providers, which arrive as `token` text).

## `ChatStream` — the streaming response

```rust,ignore
pub struct ChatStream { /* boxed async stream + last_raw */ }

impl Stream for ChatStream { type Item = Result<StreamChunk, ConnectorError>; }
impl ChatStream {
    pub async fn raw(&mut self) -> Result<&str, ConnectorError>  // last SSE frame
    pub async fn images(&mut self) -> &[GeneratedImage]          // generated images, wire order
}
```

Yields `Result<StreamChunk>` items (use `tokio_stream::StreamExt` to drive
it). After the stream ends, `raw()` returns the last SSE `data:` frame —
typically the one carrying `usage`, `finish_reason`, and the concatenated
content. If the stream was not fully consumed, `raw()` drains it first.
Returns `StreamTerminated` if the stream ended without ever receiving a
frame (e.g. connection reset before any data).

## `LsOutput` + `ModelInfo` — model listing

```rust,ignore
pub struct LsOutput { /* raw, models */ }
impl LsOutput {
    pub fn models(&self) -> &[ModelInfo]
    pub fn raw(&self) -> &str
}

pub struct ModelInfo { id: String }
impl ModelInfo { pub fn id(&self) -> &str }
```

`Connector::list_models()` returns the provider's model list with the raw
JSON alongside.

## `GeneratedImage` + `ImageOutput` — image payloads

```rust,ignore
pub struct GeneratedImage { /* bytes, mime_type, revised_prompt, raw */ }
impl GeneratedImage {
    pub fn bytes(&self) -> &[u8]            // DECODED pixels, ready to write
    pub fn mime_type(&self) -> &str         // "image/png", "image/jpeg", …
    pub fn extension(&self) -> &str         // "png", "jpg", "webp", …
    pub fn revised_prompt(&self) -> Option<&str>  // DALL·E prompt rewrites
    pub fn raw(&self) -> &str               // full response JSON
}

pub struct ImageOutput { /* images */ }
impl ImageOutput {
    pub fn images(&self) -> &[GeneratedImage]  // all images, provider order
    pub fn first(&self) -> Option<&GeneratedImage>
}
```

Two sources share the type: [`Connector::generate_image`](client.md)
(the dedicated Images route — `n` images per call) and the chat answers
above (multi-modal output embedded in the reply). Bytes are always
decoded before they reach the caller — never an expiring URL, never a
data-URI prefix.

---

Next: [error — ConnectorError and classification](error.md).

---

## Summary

- `ChatOutput` wraps a one-shot reply with `message()` (extracted text) and `raw()` (unmodified JSON response).
- `StreamChunk` represents one streaming item with `token()` (text delta), `reasoning()` (thinking delta), `thinking_blocks()` (Claude extended thinking), `finish_reason()` (final chunk only), and `tool_call()` (a provider-delivered native tool call, validated directly by the harness without a text round trip).
- `ChatStream` is the streaming response implementing `Stream<Item = Result<StreamChunk>>`; after the stream ends, `raw()` returns the last SSE frame (usage, finish_reason, etc.).
- `LsOutput` and `ModelInfo` handle model listing: `list_models()` returns available models with raw JSON alongside.
- `images()` on `ChatOutput`/`ChatStream` (and `ImageOutput` from `generate_image`) surface decoded image bytes with a MIME type; `is_empty()` is the "no image this turn" check.
- `thinking_blocks()` carries Claude extended-thinking blocks for verbatim replay (cryptographic signature validation); `skip_serializing_if` keeps this off other providers' wire formats.
- Streaming uses `tokio_stream::StreamExt` to drive; incomplete streams return `StreamTerminated` when no frame was ever received.
