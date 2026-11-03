use super::super::common::{
    SSE_CHUNK_TIMEOUT, SseBuffer, apply_provider_headers, apply_session_headers, retry_mid_stream,
    send_get_request, send_request, send_request_stream, send_with_retry, shared_client,
};
use super::super::error::ConnectorError;
use super::super::output::{
    ChatOutput, ChatStream, LsOutput, ModelInfo, StreamChunk, normalize_finish_reason,
};
use super::super::params::{Parameters, ResponseFormat, ToolCallMode, ToolDefinition};
use super::super::provider::{ProviderConfig, get_api_key, is_local_provider};
use crate::connector::NativeToolCall;

use async_stream::stream;
use std::pin::Pin;
use tokio_stream::Stream;

// Public ChatMessage from params module with native tool-call support.
use super::super::params::{
    ChatMessage as ApiChatMessage, system_message, user_message as pub_user_message,
};

/// DeepSeek/GLM-style thinking-mode toggle — `thinking: {type: "enabled"}`.
/// Sent ONLY for providers whose APIs use this exact wire shape (deepseek,
/// zai) alongside a reasoning effort: without the toggle, such deployments
/// ignore `reasoning_effort` entirely and the model never enters thinking
/// mode. Other providers must never receive it (OpenAI rejects unknown
/// top-level fields with 400).
#[derive(serde::Serialize)]
struct ThinkingToggle {
    #[serde(rename = "type")]
    kind: String,
}

/// NVIDIA NIM's thinking-mode opt-in body (`chat_template_kwargs`).
#[derive(serde::Serialize)]
struct ChatTemplateKwargs {
    thinking: bool,
}

/// Ask the server to include a final `usage` object in the stream. Without
/// this, OpenAI-compatible backends send only text deltas and no token
/// accounting ever reaches the consumer of a streamed response.
#[derive(serde::Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Kimi's prompt-cache write policy (`prompt_cache_options`, docs:
/// platform.kimi.ai/docs/guide/context-caching). `mode` is currently only
/// `"implicit"` (the prefix is written automatically when it meets the hit
/// conditions) and `ttl` selects the cache-write billing tier: `5m` (the
/// default when the field is omitted) or `1h`. Sent ONLY for the `kimi`
/// provider — OpenAI rejects unknown top-level fields with 400.
#[derive(serde::Serialize)]
struct PromptCacheOptions {
    mode: &'static str,
    ttl: &'static str,
}

#[derive(serde::Serialize)]
struct ChatRequest {
    model: String,
    /// Pre-serialized message array. Typed in Rust (`Vec<ChatMessage>`),
    /// lowered to JSON by [`lower_images_to_wire`] right before the request:
    /// a message carrying inline images becomes a multimodal content-parts
    /// array (`[{type: "text"...}, {type: "image_url"...}]`) — the plain
    /// `skip_serializing` on `ChatMessage.images` means the typed struct
    /// alone would silently drop the bytes.
    messages: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// Only meaningful together with `stream: true` (some servers reject
    /// the field on non-streaming requests, hence `skip_serializing_if`).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingToggle>,
    /// Kimi's prompt-cache write policy (`prompt_cache_options`). Sent ONLY
    /// for the `kimi` provider (see [`PromptCacheOptions`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<PromptCacheOptions>,
    /// NVIDIA NIM hybrid-thinking opt-in — `chat_template_kwargs:
    /// {"thinking": true}`. NIM ignores `reasoning_effort` for its
    /// hybrid-thinking models (DeepSeek, Qwen, …) unless this is present;
    /// sent ONLY for the nvidia provider for the same unknown-field reason
    /// as `thinking` above.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
#[allow(clippy::struct_field_names)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ToolCallFunction,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ToolCallFunction {
    name: String,
    arguments: String,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ApiErrorDetail {
    message: String,
    #[serde(default)]
    r#type: Option<String>,
}

impl ApiErrorDetail {
    fn error_type(&self) -> Option<&str> {
        self.r#type.as_deref()
    }
}

/// Whether the provider's error type code names a transient server-side
/// condition worth retrying. Mirrors fantasy's `TransientStreamErrorTypes`.
/// Mid-stream SSE error events ride inside an already-successful 200
/// response, so the HTTP status code cannot signal retryability — the
/// providers classify the payload against this set.
fn is_transient_error_type(err: &ApiErrorDetail) -> bool {
    matches!(
        err.error_type(),
        Some(
            "server_error"
                | "internal_error"
                | "overloaded_error"
                | "api_error"
                | "rate_limit_error"
        )
    )
}

/// Resolve the request's base URL. An explicit `params.base_url` (connector
/// `with_base_url`) always wins; the fallback goes through
/// [`resolved_provider_base_url`](super::super::provider::resolved_provider_base_url)
/// instead of reading the registry URL directly so the `bedrock` provider can
/// re-point its endpoint to the ambient AWS region (`AWS_REGION` /
/// `AWS_DEFAULT_REGION`) — every other provider's registry URL is returned
/// verbatim.
fn resolve_base_url(
    params: &Parameters,
    config: &ProviderConfig,
) -> std::borrow::Cow<'static, str> {
    if let Some(explicit) = params.base_url.as_deref() {
        return std::borrow::Cow::Owned(explicit.to_owned());
    }
    super::super::provider::resolved_provider_base_url(config.name)
}

/// Classify a JSON error body from either the OpenAI shape (`{"error":
/// {"message": ..., "type": ...}}`) or the AWS shape the Bedrock endpoints
/// return (`{"__type":"ThrottlingException","message":"..."}`, sometimes with
/// an `"Output"` wrapper). Returns `None` when the body is not an error
/// shape at all — the caller then reports the raw deserialization failure.
fn classify_error_body(body: &str) -> Option<ConnectorError> {
    // OpenAI shape first (the family's native contract).
    if let Ok(api_err) = serde_json::from_str::<ApiErrorResponse>(body) {
        let transient = is_transient_error_type(&api_err.error);
        let err = ConnectorError::classify_http(200, api_err.error.message);
        return Some(if transient { err.mark_transient() } else { err });
    }
    // AWS shape: `{"__type": "ThrottlingException", "message": "..."}` —
    // flat or wrapped in `"Output"`. `__type` may also carry the namespace
    // prefix (`com.amazonaws.bedrock#ThrottlingException`).
    //
    // FAMILY-WIDE, not bedrock-scoped: the caller routes every
    // empty-choices SSE frame and every deserialization failure through
    // here regardless of provider, so any backend whose error body carries
    // a top-level `code` string also classifies through this branch. No
    // known OpenAI-compatible error frame does, but the shape is accepted
    // deliberately — classification beats a raw parse failure either way.
    let aws: serde_json::Value = serde_json::from_str(body).ok()?;
    let aws_err = aws.get("Output").unwrap_or(&aws);
    let kind = aws_err
        .get("__type")
        .or_else(|| aws_err.get("code"))
        .and_then(serde_json::Value::as_str)?;
    let kind = kind.rsplit(['#', ':']).next().unwrap_or(kind);
    let message = aws_err
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(kind);
    // Throttling and service-side failures are the AWS analogues of the
    // transient types above — safe to retry; everything else (auth,
    // validation, access) is deterministic. The non-suffix spellings
    // (`ServiceUnavailable`) occur on AWS endpoints that name the condition
    // without the `Exception` suffix.
    let transient = matches!(
        kind,
        "ThrottlingException"
            | "TooManyRequestsException"
            | "TooManyRequests"
            | "ServiceUnavailableException"
            | "ServiceUnavailable"
            | "InternalServerException"
            | "InternalServerError"
            | "InternalFailure"
    );
    let err = ConnectorError::classify_http(200, message.to_string());
    Some(if transient { err.mark_transient() } else { err })
}

//  SSE streaming types

#[derive(serde::Deserialize)]
struct ChatChunkResponse {
    /// The final usage-only frame (with `stream_options.include_usage`)
    /// carries an empty or absent `choices` array.
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    /// Present on the usage-only frame; absent on text-delta frames and
    /// keep-alives.
    #[serde(default)]
    usage: Option<Usage>,
    /// The OpenCode Zen/Go gateways end the stream with a non-standard
    /// trailer `{"choices":[],"cost":"0"}` — the REAL billed cost outside
    /// the usage object, typed as a JSON string (anomalyco/opencode#42918).
    /// Present only there (and never on keep-alives).
    #[serde(default)]
    cost: Option<serde_json::Value>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ChunkChoice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning/thinking text delta (DeepSeek/GLM-style reasoning models).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// Alternate reasoning field name used by some OpenAI-compatible models.
    #[serde(default)]
    reasoning: Option<String>,
    /// Streaming tool call deltas (OpenAI-compatible API).
    /// Each chunk may carry a partial tool call for one or more indices.
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallChunk>>,
}

/// One element inside `delta.tool_calls` in a streaming SSE frame.
/// Fields are absent on all chunks except the first one for a given index.
#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct ToolCallChunk {
    #[serde(default)]
    index: Option<i32>,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    function: Option<ToolCallFunctionChunk>,
}

#[derive(serde::Deserialize)]
struct ToolCallFunctionChunk {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulated state for one tool call received over multiple SSE chunks.
#[derive(Default, Debug)]
struct PendingToolCall {
    /// The `tool_call_id` from the API's native mechanism.
    id: String,
    name: String,
    arguments: String,
}

/// Drain accumulated tool calls and return structured [`StreamChunk`] items
/// carrying the provider's native call verbatim (no inline-JSON round trip:
/// the harness extractor validates the structured fields directly).
fn flush_tool_calls(
    pending: &mut Vec<PendingToolCall>,
    last_raw: &mut Option<String>,
) -> Vec<StreamChunk> {
    let raw = last_raw.take().unwrap_or_default();
    let mut out = Vec::new();
    for tc in pending.drain(..) {
        if tc.name.is_empty() {
            continue;
        }
        out.push(StreamChunk {
            raw: raw.clone(),
            token: String::new(),
            reasoning: String::new(),
            finish_reason: Some("tool_calls".to_string()),
            thinking_blocks: None,
            tool_call: Some(NativeToolCall {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
                thought_signature: String::new(),
            }),
            reset: false,
        });
    }
    out
}

// Embedding types

#[derive(serde::Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

#[derive(serde::Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(serde::Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

fn build_messages(prompt: &str, system_prompt: Option<&str>) -> Vec<ApiChatMessage> {
    let mut messages = Vec::new();
    if let Some(system) = system_prompt {
        messages.push(ApiChatMessage {
            role: "system".to_string(),
            content: Some(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
            thinking_blocks: None,
            images: None,
        });
    }
    messages.push(pub_user_message(prompt));
    messages
}

/// Build the full messages array including system message, conversation
/// history, and tool results with native tool-call formatting.
pub fn build_full_messages(system: &str, messages: &[ApiChatMessage]) -> Vec<ApiChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(system_message(system));
    out.extend_from_slice(messages);
    out
}

/// Drop the internal transport-only fields before the messages reach the
/// OpenAI-compatible wire format (which has no such fields — the API rejects
/// unknown keys with 400): the Gemini `thought_signature` on tool calls and
/// the Claude `thinking_blocks` on assistant messages. Both are internal
/// details only their native callers read.
fn strip_internal_fields(messages: &[ApiChatMessage]) -> Vec<ApiChatMessage> {
    let has_internal = messages.iter().any(|m| {
        m.thinking_blocks.is_some()
            || m.tool_calls
                .as_ref()
                .is_some_and(|tcs| tcs.iter().any(|tc| tc.thought_signature.is_some()))
    });
    // The common path (no cross-provider history) must not pay a full clone
    // of the message array just to strip nothing.
    if !has_internal {
        return messages.to_vec();
    }
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut msg = msg.clone();
        msg.thinking_blocks = None;
        if let Some(tcs) = &mut msg.tool_calls {
            for tc in tcs {
                tc.thought_signature = None;
            }
        }
        out.push(msg);
    }
    out
}

// ── Multimodal image lowering (chat-completions wire) ─────────────────

/// Lower the internal `images` channel onto the chat-completions wire.
///
/// The chat-completions schema allows content-parts arrays (text +
/// `image_url` data URLs) on user messages — but NOT on `role: "tool"`
/// messages, whose content may only contain text parts. So an image-bearing
/// tool result keeps its plain text content and the images are emitted as a
/// SEPARATE user message immediately after it, which every VLM backend
/// (OpenAI, vLLM, llama.cpp, …) accepts:
///
/// ```json
/// [{"role":"tool","tool_call_id":"c1","content":"screenshot 1568x882 …"},
///  {"role":"user","content":[{"type":"text","text":"[computer_screenshot call_1]"},
///                            {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]
/// ```
///
/// The internal `ChatMessage.images` field never serializes itself
/// (`skip_serializing`), so this pass runs in Rust before the request body
/// is built — without it the bytes would be silently dropped and a VLM
/// would only ever see the text envelope. Messages without images pass
/// through unchanged (plain string content), so the common text-only
/// conversation pays nothing.
fn lower_images_to_wire(messages: &[ApiChatMessage]) -> serde_json::Value {
    let has_images = messages
        .iter()
        .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));
    if !has_images {
        return serde_json::to_value(messages).unwrap_or_else(|_| serde_json::json!([]));
    }
    let mut out = Vec::<serde_json::Value>::with_capacity(messages.len() + 1);
    for msg in messages {
        let Some(images) = msg.images.as_ref().filter(|imgs| !imgs.is_empty()) else {
            out.push(serde_json::to_value(msg).unwrap_or_else(|_| serde_json::json!({})));
            continue;
        };
        // The image-bearing message itself stays exactly as it serializes
        // (text-only) — never an inline image part.
        out.push(serde_json::to_value(msg).unwrap_or_else(|_| serde_json::json!({})));
        // Follow-up user message carrying the pixels: one text part naming
        // the source tool call, then one image part per image.
        let mut parts = Vec::with_capacity(images.len() + 1);
        let label = match msg.tool_call_id.as_deref() {
            Some(id) => format!("[{id}]"),
            None => String::new(),
        };
        parts.push(serde_json::json!({
            "type": "text",
            "text": format!("Image attached from tool result {label}")
        }));
        for image in images {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.media_type, image.data)
                }
            }));
        }
        out.push(serde_json::json!({
            "role": "user",
            "content": parts
        }));
    }
    serde_json::Value::Array(out)
}

/// Shared SSE processing loop for streaming chat completions.
///
/// Handles parsing SSE frames, accumulating tool call deltas,
/// yielding text tokens, and flushing tool calls on completion.
/// Used by both [`chat_stream`] and [`chat_stream_with_messages`].
fn process_sse_response(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    let buf = SseBuffer::new();

    Box::pin(stream! {
        let mut response = response;
        let mut buf = buf;
        // Accumulate streaming tool calls across chunks.
        let mut pending_tool_calls: Vec<PendingToolCall> = Vec::new();
        // Keep the raw text of the last non-DONE data frame so we can
        // use it when yielding synthetic tool-call tokens.
        let mut last_raw: Option<String> = None;
        // Set once the finish_reason chunk was emitted; after that the
        // stream may end (or be closed by the server) WITHOUT [DONE] —
        // that is a normal end, not a termination failure. The server may
        // also send one more usage-only frame (include_usage) which we keep
        // consuming so `last_raw` captures it.
        let mut saw_finish = false;
        loop {
            let (frames, ended) =
                match tokio::time::timeout(SSE_CHUNK_TIMEOUT, response.chunk()).await {
                    Ok(Ok(Some(c))) => (buf.push_and_drain(&c), false),
                    Ok(Ok(None)) => {
                        // End-of-stream: emit any final frame that arrived
                        // without a trailing blank line (the SSE spec
                        // dispatches pending data on EOF) instead of
                        // silently dropping it.
                        (buf.flush(), true)
                    }
                    Ok(Err(e)) => {
                        // A connection reset AFTER finish_reason is just the
                        // server closing early (mocks and some proxies do
                        // this) — the turn already completed successfully.
                        if saw_finish {
                            return;
                        }
                        yield Err(ConnectorError::Network(e.to_string()));
                        return;
                    }
                    Err(_) => {
                        if saw_finish {
                            return;
                        }
                        yield Err(ConnectorError::Network(format!(
                            "stream timed out after {}s",
                            SSE_CHUNK_TIMEOUT.as_secs()
                        )));
                        return;
                    }
                };
            for data in frames {
                if data == "[DONE]" {
                    // Flush any accumulated tool calls before ending.
                    for chunk in flush_tool_calls(&mut pending_tool_calls, &mut last_raw) {
                        yield Ok(chunk);
                    }
                    // Normal end: `last_raw` already holds the usage-only
                    // frame (or the finish_reason frame when the server
                    // doesn't send one).
                    return;
                }
                match serde_json::from_str::<ChatChunkResponse>(&data) {
                    Ok(ccr) => {
                        last_raw = Some(data.clone());

                        // An empty-`choices` frame is either an API error
                        // event (`{"error": {...}}`) or a usage-only /
                        // keep-alive frame. Errors surface immediately;
                        // anything else is kept only via `last_raw`.
                        if ccr.choices.is_empty() {
                            // OpenAI-shaped AND AWS-shaped (Bedrock) error
                            // bodies both classify here — including the AWS
                            // transient kinds (ThrottlingException, …).
                            if let Some(err) = classify_error_body(&data) {
                                yield Err(err);
                                return;
                            }
                            // Usage-only frame (include_usage): surface it
                            // as an empty-content chunk so consumers tracking
                            // the last raw frame see it (it carries usage).
                            // Same for the OpenCode Zen/Go cost trailer
                            // (`{"choices":[],"cost":"0"}` — the gateway's
                            // real billed amount rides there, NOT inside
                            // usage). Pure keep-alives have neither usage
                            // nor cost and are skipped.
                            if ccr.usage.is_some() || ccr.cost.is_some() {
                                yield Ok(StreamChunk {
                                    raw: data,
                                    token: String::new(),
                                    reasoning: String::new(),
                                    finish_reason: None,
                                    thinking_blocks: None,
                                    tool_call: None,
                                    reset: false,
                                });
                            }
                            continue;
                        }

                        // Accumulate tool call deltas from this chunk.
                        if let Some(tcs) = ccr.choices.first()
                            .and_then(|c| c.delta.tool_calls.as_ref())
                        {
                            for tc in tcs {
                                let idx = tc.index.unwrap_or(0).unsigned_abs() as usize;
                                if idx >= pending_tool_calls.len() {
                                    pending_tool_calls.resize_with(idx + 1, Default::default);
                                }
                                let ptc = &mut pending_tool_calls[idx];
                                // Capture the tool_call_id from the first
                                // chunk for this index (it only appears on
                                // the first chunk of each tool call).
                                if let Some(ref id) = tc.id {
                                    id.clone_into(&mut ptc.id);
                                }
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        name.clone_into(&mut ptc.name);
                                    }
                                    if let Some(ref args) = func.arguments {
                                        ptc.arguments.push_str(args);
                                    }
                                }
                            }
                        }

                        let token = ccr.choices.first()
                            .and_then(|c| c.delta.content.as_deref())
                            .unwrap_or("")
                            .to_owned();
                        let reasoning = ccr.choices.first()
                            .and_then(|c| c.delta.reasoning.as_deref().or(c.delta.reasoning_content.as_deref()))
                            .unwrap_or("")
                            .to_owned();
                        // Idempotent for this provider's canonical values,
                        // applied for uniformity across callers.
                        let finish_reason = normalize_finish_reason(
                            ccr.choices
                                .first()
                                .and_then(|c| c.finish_reason.as_deref())
                                .map(String::from),
                        );
                        let should_stop = finish_reason.is_some();

                        // Emit text token if there is any.
                        if !token.is_empty() {
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token,
                                reasoning: String::new(),
                                finish_reason: None,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                        }

                        // Emit a reasoning delta whenever the model streams one
                        // (even without text), so the TUI can show it live.
                        if !reasoning.is_empty() {
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: String::new(),
                                reasoning,
                                finish_reason: None,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                        }

                        // If the model signalled a tool call via the API-level
                        // mechanism, convert the accumulated tool calls to inline
                        // JSON text so the harness extractor can parse it.
                        if should_stop && !pending_tool_calls.is_empty() {
                            for chunk in flush_tool_calls(&mut pending_tool_calls, &mut last_raw) {
                                yield Ok(chunk);
                            }
                            return;
                        }

                        if should_stop {
                            saw_finish = true;
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: String::new(),
                                reasoning: String::new(),
                                finish_reason,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                            // With `stream_options.include_usage` the server
                            // sends ONE MORE frame after this one — a
                            // usage-only frame with empty `choices` — before
                            // [DONE]. Keep consuming so `last_raw` (what
                            // `ChatStream::raw()` and `token_usage()` see)
                            // captures it; only tool-call turns end here.
                            continue;
                        }
                    }
                    Err(e) => {
                        // A parse failure is also the shape of an AWS error
                        // body (`{"__type":...}` does not match the OpenAI
                        // response schema) — classify before reporting the
                        // raw deserialization error.
                        if let Some(err) = classify_error_body(&data) {
                            yield Err(err);
                            return;
                        }
                        yield Err(ConnectorError::Deserialization(format!(
                            "{e}. Raw chunk: {data}"
                        )));
                        return;
                    }
                }
            }
            if ended {
                break;
            }
        }
        // Stream ended without [DONE]
        for chunk in flush_tool_calls(&mut pending_tool_calls, &mut last_raw) {
            yield Ok(chunk);
        }
        if saw_finish {
            return;
        }
        yield Err(ConnectorError::StreamTerminated);
    })
}

fn build_chat_request(
    model: String,
    messages: Vec<ApiChatMessage>,
    params: &Parameters,
    stream: bool,
    provider: &str,
) -> ChatRequest {
    // Providers whose hybrid-thinking deployments ignore `reasoning_effort`
    // without an explicit toggle — scoped so no other OpenAI-compatible
    // backend ever receives the non-standard field.
    // Kimi's prompt-cache write policy: `5m` is the server default and
    // costs nothing extra, so it needs no explicit field. The `1h` TTL only
    // pays off for sessions whose turns are spaced more than 5 minutes
    // apart (the extra write fee nets positive from the second hit — docs:
    // platform.kimi.ai/docs/guide/context-caching). The connector has no
    // wall-clock knowledge of the caller's turn cadence, so it stays on the
    // cheap default and lets cache hits renew it for free; pass
    // `with_base_url`-style overrides through a proxy if a workload needs
    // `1h`. Sent only for `kimi` — the field is a Kimi extension.
    let prompt_cache_options = (provider == "kimi").then_some(PromptCacheOptions {
        mode: "implicit",
        ttl: "5m",
    });
    // Kimi K3 accepts the OpenAI-style top-level `reasoning_effort`
    // (`low`/`high`/`max`) directly — no `thinking` toggle needed (docs:
    // platform.kimi.ai/docs/guide/use-reasoning-effort). K2.x models take a
    // `thinking` object instead; sending K3 a `thinking` field is
    // explicitly warned against, so kimi stays out of the toggle set.
    let thinking = match provider {
        "deepseek" | "zai" if params.reasoning_effort.is_some() => Some(ThinkingToggle {
            kind: "enabled".to_string(),
        }),
        _ => None,
    };
    // NVIDIA NIM needs a different opt-in shape for its hybrid-thinking
    // models (DeepSeek/Qwen on integrate.api.nvidia.com).
    // NVIDIA NIM needs a different opt-in shape for its hybrid-thinking
    // models (DeepSeek/Qwen on integrate.api.nvidia.com). NOTE: like the
    // DeepSeek toggle above, this keys off the provider name — a generic
    // openai connector pointed at a NIM base URL will not receive it.
    let chat_template_kwargs = (provider == "nvidia" && params.reasoning_effort.is_some())
        .then_some(ChatTemplateKwargs { thinking: true });
    // Lower the internal image channel onto the chat-completions wire
    // BEFORE building the request: a tool result carrying a screenshot
    // becomes a multimodal content-parts array here (see
    // [`lower_images_to_wire`]).
    let messages = lower_images_to_wire(&messages);
    // In inline mode the request must NOT carry the native `tools` array:
    // the model is instructed to write tool calls as JSON into its text
    // response, and the harness parses them. Without `tools` the API can
    // never produce structured tool calls, so the two delivery paths are
    // mutually exclusive at the wire level.
    let native_tools = params.tool_call_mode == ToolCallMode::Native;
    ChatRequest {
        model,
        messages,
        stream: if stream { Some(true) } else { None },
        stream_options: stream.then_some(StreamOptions {
            include_usage: true,
        }),
        max_tokens: params.max_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        stop: params.stop.clone(),
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        seed: params.seed,
        response_format: params.response_format.clone(),
        logprobs: params.logprobs,
        top_logprobs: params.top_logprobs,
        tools: if native_tools {
            params.tools.clone()
        } else {
            None
        },
        tool_choice: if native_tools {
            params.tool_choice.clone()
        } else {
            None
        },
        reasoning_effort: params.reasoning_effort.clone(),
        thinking,
        prompt_cache_options,
        chat_template_kwargs,
        user: params.user.clone(),
    }
}

/// Resolve the API key for a request, if any.
///
/// Local providers are configured by URL and never require a key (most
/// servers ignore the `Authorization` header entirely); cloud providers fall
/// back to the OS keyring / environment and must have a key — including the
/// OpenCode `zen` and `opencode-go` gateways. Returns an error when a cloud
/// provider has no key configured.
fn resolve_api_key(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Result<Option<String>, ConnectorError> {
    let key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service));
    if !is_local_provider(config.name) && key.is_none() {
        return Err(ConnectorError::MissingApiKey(config.name.to_string()));
    }
    Ok(key)
}

/// Send a non-streaming chat completion request.
///
/// Resolves the API key and model, builds the request body, and parses the
/// OpenAI-compatible JSON response.
pub async fn chat(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatOutput, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());

    let messages = build_messages(prompt, system_prompt);
    let request = build_chat_request(model, messages, params, false, config.name);
    let url = format!("{}/chat/completions", resolve_base_url(params, config));

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(a) = &auth {
        headers.push(("Authorization", a.to_string()));
    }
    let response_text = send_request(config, &url, &request, &headers).await?;
    let chat_response: ChatResponse = match serde_json::from_str(&response_text) {
        Ok(r) => r,
        Err(e) => {
            // The 200-with-error-body path: OpenAI-shaped and AWS-shaped
            // (Bedrock) error bodies are classified here, including the AWS
            // transient kinds (ThrottlingException, …).
            if let Some(err) = classify_error_body(&response_text) {
                return Err(err);
            }
            return Err(ConnectorError::Deserialization(format!(
                "{e}. Raw response: {response_text}"
            )));
        }
    };
    let first_choice = chat_response
        .choices
        .first()
        .ok_or(ConnectorError::NoChoices)?;
    let message = first_choice
        .message
        .content
        .clone()
        .ok_or(ConnectorError::NoContent)?;
    Ok(ChatOutput {
        raw: response_text,
        message,
    })
}

/// Send a streaming chat completion request.
///
/// Returns an async [`Stream`] of content chunks parsed from SSE frames.
/// The stream terminates with `StreamTerminated` if the connection closes
/// without a `[DONE]` signal.
pub async fn chat_stream(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());

    let messages = build_messages(prompt, system_prompt);
    let request = build_chat_request(model, messages, params, true, config.name);
    let url = format!("{}/chat/completions", resolve_base_url(params, config));

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    log::debug!("chat_stream: sending request to {url}");
    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&request)?;
        let mut request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        );
        if let Some(a) = &auth {
            request_builder = request_builder.header("Authorization", a.as_str());
        }
        request_builder = apply_session_headers(request_builder, params.session_id.as_deref());
        let response = send_with_retry(
            &request_builder,
            json_body.clone(),
            params.retry_delay_override,
            params.max_retries,
        )
        .await?;
        retry_mid_stream(
            response,
            request_builder,
            json_body,
            params.retry_delay_override,
            params.max_retries,
            process_sse_response,
        )
    } else {
        let mut headers: Vec<(&str, String)> = Vec::new();
        if let Some(sid) = params.session_id.as_deref() {
            headers.push(("x-session-id", sid.to_string()));
            headers.push(("x-session-affinity", sid.to_string()));
        }
        if let Some(a) = &auth {
            headers.push(("Authorization", a.to_string()));
        }
        let response = send_request_stream(config, &url, &request, &headers).await?;
        process_sse_response(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Send a streaming chat completion with a full messages array including
/// system, user, assistant (with `tool_calls`), and tool (with `tool_call_id`)
/// messages.
///
/// This is the native tool-calling path: tool definitions are sent via the
/// API's `tools` parameter, tool calls arrive as structured `delta.tool_calls`,
/// and tool results are returned as `role: "tool"` messages.
pub async fn chat_stream_with_messages(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: &[ApiChatMessage],
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());

    // OpenAI's wire format has no `thought_signature` — a Gemini-originated
    // call (carried internally on ToolCallMsg) must be stripped before
    // serialization, otherwise the request would be rejected with 400.
    let all_messages = build_full_messages(system, messages);
    let all_messages = strip_internal_fields(&all_messages);
    let request = build_chat_request(model, all_messages, params, true, config.name);
    let url = format!("{}/chat/completions", resolve_base_url(params, config));

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    log::debug!("chat_stream_with_messages: sending request to {url}");
    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&request)?;
        let mut request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        );
        if let Some(a) = &auth {
            request_builder = request_builder.header("Authorization", a.as_str());
        }
        request_builder = apply_session_headers(request_builder, params.session_id.as_deref());
        let response = send_with_retry(
            &request_builder,
            json_body.clone(),
            params.retry_delay_override,
            params.max_retries,
        )
        .await?;
        retry_mid_stream(
            response,
            request_builder,
            json_body,
            params.retry_delay_override,
            params.max_retries,
            process_sse_response,
        )
    } else {
        let mut headers: Vec<(&str, String)> = Vec::new();
        if let Some(sid) = params.session_id.as_deref() {
            headers.push(("x-session-id", sid.to_string()));
            headers.push(("x-session-affinity", sid.to_string()));
        }
        if let Some(a) = &auth {
            headers.push(("Authorization", a.to_string()));
        }
        let response = send_request_stream(config, &url, &request, &headers).await?;
        process_sse_response(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Send an embedding request and return the embedding vector.
///
/// Resolves the API key, builds the request body, and extracts the first
/// embedding from the OpenAI-compatible response.
pub async fn embed(
    config: &ProviderConfig,
    params: &Parameters,
    input: &str,
    service: Option<&str>,
) -> Result<Vec<f32>, ConnectorError> {
    if super::super::provider::is_bedrock(config.name) && params.base_url.is_none() {
        return Err(ConnectorError::NotImplemented(
            "embeddings for bedrock (the OpenAI-compatible endpoints carry no embedding API; use Titan via the Invoke API or a compatible proxy base URL)",
        ));
    }
    // Kimi documents no embedding endpoint and no embedding models
    // (platform.kimi.ai/docs/api/overview lists the operations; none is
    // `/embeddings`, and /docs/models catalogs chat models only) — a
    // request would fail at runtime with the confusing OpenAI default
    // model, so gate it up front like the bedrock check above.
    if config.name == "kimi" && params.base_url.is_none() {
        return Err(ConnectorError::NotImplemented(
            "embeddings for kimi (api.moonshot.ai documents no embedding endpoint; point with_base_url at a compatible proxy that serves one)",
        ));
    }
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| "text-embedding-3-small".to_string());

    let request = EmbeddingRequest {
        model,
        input: input.to_string(),
    };
    let url = format!("{}/embeddings", resolve_base_url(params, config));

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(a) = &auth {
        headers.push(("Authorization", a.to_string()));
    }
    let response_text = send_request(config, &url, &request, &headers).await?;
    let embed_response: EmbeddingResponse = serde_json::from_str(&response_text)?;
    let first_data = embed_response
        .data
        .first()
        .ok_or(ConnectorError::NoEmbeddings)?;
    Ok(first_data.embedding.clone())
}

#[derive(serde::Deserialize)]
struct ListModelsData {
    id: String,
}

#[derive(serde::Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelsData>,
}

/// Fetch the list of available models from the provider.
///
/// Resolves the API key, sends a GET to `{base_url}/models`, and parses
/// the OpenAI-compatible JSON response into a [`LsOutput`].
///
/// The `bedrock-runtime` endpoint does NOT implement the OpenAI-compatible
/// `GET /models` operation (AWS documents `ListFoundationModels` /
/// `ListInferenceProfiles` instead); only the `bedrock-mantle` compatibility
/// endpoint serves the model catalog. A Bedrock `list_models` therefore
/// fails up front unless the user pointed an explicit base URL at Mantle
/// (`https://bedrock-mantle.{region}.api.aws/v1`) — the gate mirrors the
/// embedding one.
pub async fn list_models(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Result<LsOutput, ConnectorError> {
    if super::super::provider::is_bedrock(config.name) && params.base_url.is_none() {
        return Err(ConnectorError::NotImplemented(
            "model listing for bedrock-runtime (the endpoint has no GET /models; point with_base_url at https://bedrock-mantle.{region}.api.aws/v1 for the catalog)",
        ));
    }
    let api_key = resolve_api_key(config, params, service)?;

    let url = format!("{}/models", resolve_base_url(params, config));

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(a) = &auth {
        headers.push(("Authorization", a.to_string()));
    }
    let response_text = send_get_request(config, &url, &headers).await?;

    let list: ListModelsResponse = serde_json::from_str(&response_text)?;
    let models: Vec<ModelInfo> = list
        .data
        .into_iter()
        .map(|item| ModelInfo { id: item.id })
        .collect();

    Ok(LsOutput::new(response_text, models))
}

#[cfg(test)]
mod image_lowering_tests {
    use super::lower_images_to_wire;
    use crate::connector::params::{ChatMessage, ImageBlock, tool_result_message_with_images};

    fn sample_image() -> ImageBlock {
        ImageBlock::png(b"\x89PNG fake bytes")
    }

    /// Regression: the openai_compatible caller serializes `ChatMessage`
    /// directly, and `images` is `skip_serializing` — without the lowering
    /// pass the screenshot bytes were silently dropped and a VLM only ever
    /// saw the text envelope (width/height/bytes metadata).
    ///
    /// The chat-completions schema allows content-parts (with `image_url`)
    /// ONLY on user messages — a `role: "tool"` message may carry text
    /// parts exclusively — so the lowering must keep the tool result
    /// text-only and emit the pixels as a follow-up user message.
    #[test]
    fn tool_result_with_images_becomes_tool_plus_user_image_message() {
        let msg =
            tool_result_message_with_images("call_1", "screenshot 1568x882", vec![sample_image()]);
        let lowered = lower_images_to_wire(std::slice::from_ref(&msg));
        let arr = lowered.as_array().expect("messages array");
        assert_eq!(arr.len(), 2, "tool result + follow-up user message");

        // Message 1: the tool result, TEXT-ONLY (schema: no image parts).
        assert_eq!(arr[0]["role"], "tool");
        assert_eq!(arr[0]["tool_call_id"], "call_1");
        assert_eq!(arr[0]["content"], "screenshot 1568x882");
        assert!(arr[0].get("images").is_none(), "images must never leak");
        assert!(arr[0]["content"].is_string(), "tool content stays a string");

        // Message 2: the synthesized user message carrying the pixels.
        assert_eq!(arr[1]["role"], "user");
        let content = arr[1]["content"].as_array().expect("content parts");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        let label = content[0]["text"].as_str().unwrap();
        assert!(label.contains("call_1"), "label names the source: {label}");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "url: {url}");
    }

    /// Text-only messages keep their plain string content (no parts array,
    /// no data-URL inflation of the common path).
    #[test]
    fn text_only_messages_serialize_unchanged() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            thinking_blocks: None,
            images: None,
        };
        let lowered = lower_images_to_wire(std::slice::from_ref(&msg));
        assert_eq!(lowered[0]["content"], "hello");
        assert!(lowered[0].get("images").is_none());
    }

    /// A conversation where ONLY one message carries images: the plain
    /// message stays string-content and is NOT followed by anything; the
    /// image-bearing tool result gains exactly one follow-up user message
    /// (no extra empty parts, no double emission).
    #[test]
    fn mixed_conversation_appends_user_message_only_after_tool_result() {
        let plain = ChatMessage {
            role: "user".to_string(),
            content: Some("take a screenshot".to_string()),
            tool_calls: None,
            tool_call_id: None,
            thinking_blocks: None,
            images: None,
        };
        let tool = tool_result_message_with_images("call_2", "shot", vec![sample_image()]);
        let both = [plain, tool];
        let lowered = lower_images_to_wire(&both);
        assert_eq!(lowered.as_array().map(Vec::len), Some(3));
        assert_eq!(lowered[0]["content"], "take a screenshot");
        assert_eq!(lowered[1]["role"], "tool");
        assert_eq!(lowered[1]["content"], "shot");
        assert_eq!(lowered[2]["role"], "user");
        assert!(lowered[2]["content"].is_array());
    }

    /// A tool result whose image list is EMPTY (payload drained or never
    /// staged): no follow-up message is synthesized for it.
    #[test]
    fn empty_image_list_keeps_string_content() {
        let msg = ChatMessage {
            role: "tool".to_string(),
            content: Some("plain".to_string()),
            tool_calls: None,
            tool_call_id: Some("c".to_string()),
            thinking_blocks: None,
            images: Some(Vec::new()),
        };
        let lowered = lower_images_to_wire(std::slice::from_ref(&msg));
        assert_eq!(lowered.as_array().map(Vec::len), Some(1));
        assert_eq!(lowered[0]["content"], "plain");
    }
}
