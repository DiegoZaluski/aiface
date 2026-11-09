use super::super::common::{
    SSE_CHUNK_TIMEOUT, SseBuffer, apply_provider_headers, apply_session_headers, retry_mid_stream,
    send_request, send_request_stream, send_with_retry, shared_client,
};
use super::super::error::ConnectorError;
use super::super::output::{ChatOutput, ChatStream, GeneratedImage, StreamChunk};
use super::super::params::{ChatMessage, Parameters, ToolCallMode};
use super::super::provider::{ProviderConfig, get_api_key, is_local_provider};
use base64::Engine as _;
use crate::connector::NativeToolCall;

use async_stream::stream;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use tokio_stream::Stream;

/// Responses-API reasoning configuration. `summary: "auto"` opts into
/// reasoning-summary text (streamed as `response.reasoning_summary_text.delta`
/// events), which is what the TUI renders as the "Thought" block.
#[derive(serde::Serialize)]
struct ReasoningConfig {
    effort: String,
    summary: String,
}

/// A Responses-API function tool definition. The `openai_compatible` module's
/// `ToolDefinition` carries the same data under `type`/`function`, but the
/// Responses API flattens the tool object — name, description and parameters
/// live directly on the tool.
#[derive(serde::Serialize)]
struct ResponseTool {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
    strict: bool,
}

/// Streaming state for one in-flight tool call, keyed by the Responses item
/// id (`item_id`), finalized when `response.output_item.done` delivers the
/// complete `function_call` item.
#[derive(Default)]
struct PendingCall {
    call_id: String,
    name: String,
    arguments: String,
}

// ── Non-streaming response types ─────────────────────────────────────────

#[derive(serde::Deserialize)]
struct ResponsesResponse {
    output: Vec<Value>,
}

// ── Streaming event types ────────────────────────────────────────────────

#[derive(serde::Deserialize, Default)]
struct ResponseMeta {
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(serde::Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(serde::Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: String,
        #[serde(default)]
        arguments: String,
    },
    #[serde(other)]
    #[default]
    Other,
}

/// Typed Responses-API streaming events. `#[serde(other)]` absorbs the many
/// lifecycle/keep-alive events (`response.created`, `response.in_progress`,
/// `response.content_part.*`, `response.output_text.done`, …) that carry no
/// data we act on, so an unknown or new event type never aborts the stream.
#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[serde(default)]
        item_id: String,
        #[serde(default)]
        delta: String,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        #[serde(default)]
        item: OutputItem,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        #[serde(default)]
        item: OutputItem,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete {
        #[serde(default)]
        response: ResponseMeta,
    },
    #[serde(rename = "response.failed")]
    ResponseFailed,
    #[serde(rename = "error")]
    Error,
    #[serde(other)]
    Other,
}

// ── Error classification ─────────────────────────────────────────────────

/// Extract `(message, is_transient)` from an error-carrying SSE frame.
///
/// The error object lives in three places depending on the event:
/// - the `error` event nests it as `{"error": {...}}`;
/// - the `response.failed` event nests it as `{"response": {"error": {...}}}`;
/// - a flat `error` event carries `message`/`code` at the top level.
fn extract_error(data: &str) -> (String, bool) {
    let v: Value = serde_json::from_str(data).unwrap_or(Value::Null);
    let top = v.get("error").filter(|e| !e.is_null());
    let nested = v
        .get("response")
        .and_then(|r| r.get("error"))
        .filter(|e| !e.is_null());
    let source = top.or(nested).unwrap_or(&v);
    let message = source
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| source.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown error")
        .to_string();
    let transient = matches!(
        source
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| source.get("code").and_then(Value::as_str)),
        Some(
            "server_error"
                | "internal_error"
                | "overloaded_error"
                | "api_error"
                | "rate_limit_error"
                | "service_unavailable_error"
                | "upstream_error"
        )
    );
    (message, transient)
}

// ── Request building ─────────────────────────────────────────────────────

/// Convert a conversation `ChatMessage` into Responses-API input items.
fn message_to_input_items(msg: &ChatMessage, out: &mut Vec<Value>) {
    // Tool result → `function_call_output` (must link back via call_id).
    if msg.role == "tool" {
        if let Some(call_id) = &msg.tool_call_id {
            // Multimodal tool result: `output` accepts a content-part ARRAY
            // (`input_text` / `input_image`) instead of a plain string —
            // the documented shape for functions that return images. An
            // EMPTY image vector is treated as no images, preserving the
            // text content instead of collapsing it to `""`.
            let has_images = msg.images.as_ref().is_some_and(|i| !i.is_empty());
            let output = if has_images {
                let mut parts = Vec::with_capacity(msg.images.as_ref().map_or(0, Vec::len) + 1);
                if let Some(text) = msg.content.as_deref().filter(|t| !t.is_empty()) {
                    parts.push(serde_json::json!({"type": "input_text", "text": text}));
                }
                for image in msg.images.as_ref().into_iter().flatten() {
                    parts.push(serde_json::json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", image.media_type, image.data),
                    }));
                }
                Value::Array(parts)
            } else {
                Value::String(msg.content.clone().unwrap_or_default())
            };
            out.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }));
        }
        return;
    }
    // Assistant messages may carry both prose and tool calls: emit any prose
    // first (as an `output_text` message item), then the `function_call`
    // items. In practice the harness sends one or the other, but a combined
    // message must not silently drop its text.
    if msg.role == "assistant" {
        if let Some(content) = &msg.content
            && !content.is_empty()
        {
            out.push(serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }));
        }
        for tc in msg.tool_calls.as_ref().into_iter().flatten() {
            out.push(serde_json::json!({
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }));
        }
        return;
    }
    // Plain message. System/user content is `input_text`; attached images
    // become sibling `input_image` parts (data URLs).
    let has_images = msg.images.as_ref().is_some_and(|i| !i.is_empty());
    let Some(content) = &msg.content else {
        if has_images {
            let role = if msg.role == "user" {
                "user"
            } else {
                "assistant"
            };
            let parts: Vec<Value> = msg
                .images
                .as_ref()
                .into_iter()
                .flatten()
                .map(|image| {
                    serde_json::json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", image.media_type, image.data),
                    })
                })
                .collect();
            out.push(serde_json::json!({
                "type": "message",
                "role": role,
                "content": parts,
            }));
        }
        return;
    };
    let (role, content_type) = match msg.role.as_str() {
        "system" => ("system", "input_text"),
        "user" => ("user", "input_text"),
        _ => ("assistant", "output_text"),
    };
    let mut blocks = Vec::new();
    if !content.is_empty() {
        blocks.push(if content_type == "output_text" {
            serde_json::json!({"type": "output_text", "text": content, "annotations": []})
        } else {
            serde_json::json!({"type": "input_text", "text": content})
        });
    }
    // Attached images become sibling `input_image` parts — a message may
    // carry BOTH text and images ("analyse this screenshot: ..."); the
    // images must not be dropped just because text is present.
    for image in msg.images.as_ref().into_iter().flatten() {
        blocks.push(serde_json::json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", image.media_type, image.data),
        }));
    }
    out.push(serde_json::json!({
        "type": "message",
        "role": role,
        "content": blocks,
    }));
}

/// Build the `input` array: the system prompt first, then the conversation.
fn build_input_items(system: &str, messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    if !system.is_empty() {
        out.push(serde_json::json!({
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": system}],
        }));
    }
    for msg in messages {
        message_to_input_items(msg, &mut out);
    }
    out
}

/// Build a Responses-API request body. `store: false` keeps the request
/// stateless (no server-side response retention). `reasoning` is only sent
/// when an effort was chosen; `temperature`/`top_p` are dropped alongside it
/// because OpenAI's reasoning models reject both.
///
/// Fields the Chat Completions API accepted but the Responses API does not
/// carry (`stop`, `frequency_penalty`, `presence_penalty`, `seed`,
/// `response_format`, `logprobs`, `top_logprobs`) are intentionally omitted.
fn build_request(
    model: &str,
    system: &str,
    messages: &[ChatMessage],
    params: &Parameters,
    stream: bool,
) -> Value {
    let mut req = serde_json::json!({
        "model": model,
        "input": build_input_items(system, messages),
        "store": false,
    });
    let map = req.as_object_mut().expect("request is an object");
    if stream {
        map.insert("stream".to_string(), Value::Bool(true));
    }
    if let Some(max_tokens) = params.max_tokens {
        map.insert("max_output_tokens".to_string(), Value::from(max_tokens));
    }
    if let Some(effort) = &params.reasoning_effort {
        map.insert(
            "reasoning".to_string(),
            serde_json::to_value(ReasoningConfig {
                effort: effort.clone(),
                summary: "auto".to_string(),
            })
            .expect("reasoning serializes"),
        );
    } else {
        if let Some(temperature) = params.temperature {
            map.insert("temperature".to_string(), Value::from(temperature));
        }
        if let Some(top_p) = params.top_p {
            map.insert("top_p".to_string(), Value::from(top_p));
        }
    }
    // In inline mode the request must NOT carry native tools: the model is
    // instructed to write tool calls as JSON into its text, and the harness
    // parses them (the same contract as the openai_compatible caller).
    if params.tool_call_mode == ToolCallMode::Native {
        if let Some(tools) = &params.tools {
            let tools_json: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::to_value(ResponseTool {
                        kind: "function",
                        name: t.function.name.clone(),
                        description: t.function.description.clone(),
                        parameters: t.function.parameters.clone(),
                        strict: false,
                    })
                    .expect("tool serializes")
                })
                .collect();
            map.insert("tools".to_string(), Value::Array(tools_json));
        }
        if let Some(tool_choice) = &params.tool_choice {
            map.insert("tool_choice".to_string(), tool_choice.clone());
        }
    }
    if let Some(user) = &params.user {
        map.insert("user".to_string(), Value::String(user.clone()));
    }
    // Prompt-cache controls: the affinity key routes the session's requests
    // to the machines holding its cached prefixes (required for reliable
    // matching on GPT-5.6+); the retention extends how long those prefixes
    // stay cached. Both are Responses-API top-level fields; unset = the
    // provider's default behavior.
    if let Some(key) = &params.prompt_cache_key {
        map.insert("prompt_cache_key".to_string(), Value::String(key.clone()));
    }
    if let Some(retention) = &params.openai_cache_retention {
        map.insert(
            "prompt_cache_retention".to_string(),
            Value::String(retention.clone()),
        );
    }
    req
}

/// Resolve the API key for a request (explicit key, then keyring/environment).
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

// ── Streaming parser ─────────────────────────────────────────────────────

/// Shared SSE processing loop for Responses-API streaming.
fn process_sse_response(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    let buf = SseBuffer::new();

    Box::pin(stream! {
        let mut response = response;
        let mut buf = buf;
        // Accumulate in-flight tool calls by item_id.
        let mut pending: HashMap<String, PendingCall> = HashMap::new();
        // Set once a terminal event (`response.completed` /
        // `response.incomplete`) was seen; after that the stream may end
        // WITHOUT [DONE] — that is a normal end, not a termination failure.
        let mut saw_finish = false;
        // A tool call already ended the turn (`finish_reason: "tool_calls"`);
        // the trailing `response.completed` then carries no finish reason so
        // the turn keeps the tool-call signal (mirrors openai_compatible,
        // which stops after flushing tool calls).
        let mut emitted_tool_call = false;
        loop {
            let (frames, ended) =
                match tokio::time::timeout(SSE_CHUNK_TIMEOUT, response.chunk()).await {
                    Ok(Ok(Some(c))) => (buf.push_and_drain(&c), false),
                    Ok(Ok(None)) => (buf.flush(), true),
                    Ok(Err(e)) => {
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
                    return;
                }
                let Ok(event) = serde_json::from_str::<StreamEvent>(&data) else {
                    // Not a typed Responses event: surface it as an error if
                    // it looks like an API error, otherwise skip it.
                    if let Ok(v) = serde_json::from_str::<Value>(&data)
                        && v.get("error").is_some()
                    {
                        yield Err(ConnectorError::Deserialization(format!(
                            "Unrecognized Responses event: {data}"
                        )));
                        return;
                    }
                    continue;
                };
                match event {
                    StreamEvent::OutputTextDelta { delta } => {
                        if !delta.is_empty() {
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: delta,
                                reasoning: String::new(),
                                finish_reason: None,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                        }
                    }
                    StreamEvent::ReasoningSummaryTextDelta { delta } => {
                        if !delta.is_empty() {
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: String::new(),
                                reasoning: delta,
                                finish_reason: None,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                        }
                    }
                    StreamEvent::RefusalDelta { delta } => {
                        if !delta.is_empty() {
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: delta,
                                reasoning: String::new(),
                                finish_reason: None,
                                thinking_blocks: None,
                                tool_call: None,
                                reset: false,
                            });
                        }
                    }
                    StreamEvent::FunctionCallArgumentsDelta { item_id, delta } => {
                        if !item_id.is_empty() && !delta.is_empty() {
                            pending.entry(item_id).or_default().arguments.push_str(&delta);
                        }
                    }
                    StreamEvent::OutputItemAdded { item } => {
                        if let OutputItem::FunctionCall {
                            id: Some(item_id),
                            call_id,
                            name,
                            ..
                        } = item
                        {
                            let entry = pending.entry(item_id).or_default();
                            if entry.call_id.is_empty() {
                                entry.call_id = call_id.unwrap_or_default();
                            }
                            if entry.name.is_empty() {
                                entry.name = name;
                            }
                        }
                    }
                    StreamEvent::OutputItemDone { item } => {
                        if let OutputItem::FunctionCall {
                            id,
                            call_id,
                            name,
                            arguments,
                        } = item
                        {
                            let key = id.clone().unwrap_or_default();
                            let (resolved_call_id, final_name, final_arguments) = {
                                let entry = pending.entry(key.clone()).or_default();
                                let resolved_call_id = if call_id.is_some() {
                                    call_id.unwrap_or_default()
                                } else if entry.call_id.is_empty() {
                                    id.unwrap_or_default()
                                } else {
                                    entry.call_id.clone()
                                };
                                if !name.is_empty() {
                                    entry.name = name;
                                }
                                if !arguments.is_empty() {
                                    entry.arguments = arguments;
                                }
                                (
                                    resolved_call_id,
                                    entry.name.clone(),
                                    entry.arguments.clone(),
                                )
                            };
                            pending.remove(&key);
                            // A malformed item with no tool name has nothing
                            // to dispatch (mirrors the openai_compatible
                            // flush guard).
                            if final_name.is_empty() {
                                continue;
                            }
                            emitted_tool_call = true;
                            yield Ok(StreamChunk {
                                raw: data.clone(),
                                token: String::new(),
                                reasoning: String::new(),
                                finish_reason: Some("tool_calls".to_string()),
                                thinking_blocks: None,
                                tool_call: Some(NativeToolCall {
                                    id: resolved_call_id,
                                    name: final_name,
                                    arguments: final_arguments,
                                    thought_signature: String::new(),
                                }),
                                reset: false,
                            });
                        }
                    }
                    StreamEvent::ResponseCompleted => {
                        saw_finish = true;
                        yield Ok(StreamChunk {
                            raw: data.clone(),
                            token: String::new(),
                            reasoning: String::new(),
                            finish_reason: if emitted_tool_call {
                                None
                            } else {
                                Some("stop".to_string())
                            },
                            thinking_blocks: None,
                            tool_call: None,
                            reset: false,
                        });
                    }
                    StreamEvent::ResponseIncomplete { response: meta } => {
                        saw_finish = true;
                        // The harness treats `length` as "cut at max tokens"
                        // and continues the loop instead of finishing the turn.
                        let reason = match meta
                            .incomplete_details
                            .as_ref()
                            .and_then(|d| d.reason.as_deref())
                        {
                            Some("max_output_tokens") => "length",
                            Some(other) => other,
                            None => "incomplete",
                        };
                        yield Ok(StreamChunk {
                            raw: data.clone(),
                            token: String::new(),
                            reasoning: String::new(),
                            finish_reason: Some(reason.to_string()),
                            thinking_blocks: None,
                            tool_call: None,
                            reset: false,
                        });
                    }
                    StreamEvent::ResponseFailed | StreamEvent::Error => {
                        let (message, transient) = extract_error(&data);
                        let err = ConnectorError::classify_http(200, message);
                        yield Err(if transient { err.mark_transient() } else { err });
                        return;
                    }
                    StreamEvent::Other => {}
                }
            }
            if ended {
                break;
            }
        }
        if saw_finish {
            return;
        }
        yield Err(ConnectorError::StreamTerminated);
    })
}

// ── Callers ──────────────────────────────────────────────────────────────

/// Extract images a chat ANSWER carries (Responses API): the `result`
/// base64 of every `image_generation_call` output item — the bytes the
/// `image_generation` tool produced inside the response.
///
/// OpenAI first-party NEVER returns images in chat messages (assistant
/// content is text/refusal only, per the schema dumps of every SDK) —
/// the only inline-in-response surface is this tool output item. A
/// `result` is `null` until the call completes; incomplete/failed items
/// simply contribute nothing.
pub(crate) fn extract_output_images(raw: &str) -> Vec<GeneratedImage> {
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(output) = parsed.get("output").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("image_generation_call") {
            continue;
        }
        let Some(result) = item.get("result").and_then(Value::as_str) else {
            continue;
        };
        let mime = item
            .get("output_format")
            .and_then(Value::as_str)
            .map_or_else(
                || "image/png".to_string(),
                |fmt| format!("image/{fmt}"),
            );
        // `result` is bare standard base64 (no data-URI prefix, per the
        // Responses API reference's image_generation_call item schema).
        let bytes = match base64::engine::general_purpose::STANDARD.decode(result) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::debug!("skipping undecodable image_generation_call result: {e}");
                continue;
            }
        };
        out.push(GeneratedImage::new(bytes, mime, None, raw.to_string()));
    }
    out
}

/// Extract the visible text of a non-streaming Responses response: the
/// `output_text` blocks of every `message` item, in order.
fn response_text(response: &ResponsesResponse) -> Option<String> {
    let mut text = String::new();
    for item in &response.output {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        for block in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(t) = block.get("text").and_then(Value::as_str) {
                text.push_str(t);
            }
        }
    }
    (!text.is_empty()).then_some(text)
}

/// Send a non-streaming Responses request.
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
    let system = system_prompt.unwrap_or("");
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(prompt.to_string()),
        tool_calls: None,
        tool_call_id: None,
        thinking_blocks: None,
        images: None,
    }];
    let request = build_request(&model, system, &messages, params, false);
    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/responses");

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(a) = &auth {
        headers.push(("Authorization", a.to_string()));
    }
    let raw = send_request(config, &url, &request, &headers).await?;
    let parsed: ResponsesResponse = match serde_json::from_str(&raw) {
        Ok(r) => r,
        Err(e) => {
            let error_val = serde_json::from_str::<Value>(&raw).ok();
            if error_val.as_ref().and_then(|v| v.get("error")).is_some() {
                let (message, transient) = extract_error(&raw);
                let err = ConnectorError::classify_http(200, message);
                return Err(if transient { err.mark_transient() } else { err });
            }
            return Err(ConnectorError::Deserialization(format!(
                "{e}. Raw response: {raw}"
            )));
        }
    };
    let message = response_text(&parsed).ok_or(ConnectorError::NoContent)?;
    // The `image_generation` tool, when enabled, delivers its output as
    // `image_generation_call` items in the same response — extract them
    // alongside the text.
    let images = extract_output_images(&raw);
    Ok(ChatOutput { raw, message, images })
}

/// Build and send a streaming Responses request, returning the parsed stream.
async fn stream_responses(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: Vec<ChatMessage>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());
    let request = build_request(&model, system, &messages, params, true);
    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/responses");

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    log::debug!("openai responses stream: sending request to {url}");
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

/// Send a streaming Responses request with a user prompt.
pub async fn chat_stream(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let system = system_prompt.unwrap_or("");
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(prompt.to_string()),
        tool_calls: None,
        tool_call_id: None,
        thinking_blocks: None,
        images: None,
    }];
    stream_responses(config, params, system, messages, service).await
}

/// Send a streaming Responses request with a full messages array (native
/// tool-call format: user/assistant `function_call` and `function_call_output`
/// history).
pub async fn chat_stream_with_messages(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: &[ChatMessage],
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    // `stream_responses` prepends the system prompt itself and re-resolves the
    // API key; the Responses wire format has no `thought_signature` /
    // `thinking_blocks` fields, and `message_to_input_items` ignores them.
    stream_responses(config, params, system, messages.to_vec(), service).await
}
#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn chat_answer_images_parse_from_image_generation_calls() {
        let json = r#"{"output":[
            {"type":"reasoning","summary":[]},
            {"type":"message","content":[{"type":"output_text","text":"done"}]},
            {"type":"image_generation_call","result":"aGVsbG8=","output_format":"png"},
            {"type":"image_generation_call","result":"aGVsbG8=","output_format":"webp"}
        ]}"#;
        let out = extract_output_images(json);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].mime_type(), "image/png");
        assert_eq!(out[0].bytes(), b"hello");
        assert_eq!(out[1].mime_type(), "image/webp");
    }

    #[test]
    fn chat_answer_incomplete_calls_contribute_nothing() {
        // `result` is null until the call completes; failed items too.
        let json = r#"{"output":[
            {"type":"image_generation_call","result":null,"status":"in_progress"},
            {"type":"image_generation_call"}
        ]}"#;
        assert!(extract_output_images(json).is_empty());
    }

    #[test]
    fn chat_answer_plain_responses_are_empty() {
        let json = r#"{"output":[{"type":"message","content":[
            {"type":"output_text","text":"words"}
        ]}]}"#;
        assert!(extract_output_images(json).is_empty());
        assert!(extract_output_images("not json").is_empty());
    }
}
