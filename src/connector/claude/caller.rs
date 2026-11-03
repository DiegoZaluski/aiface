use super::super::common::{
    SSE_CHUNK_TIMEOUT, SseBuffer, apply_provider_headers, apply_session_headers, retry_mid_stream,
    send_get_request, send_request, send_request_stream, send_with_retry, shared_client,
};
use super::super::error::ConnectorError;
use super::super::output::{
    ChatOutput, ChatStream, LsOutput, ModelInfo, StreamChunk, normalize_finish_reason,
};
use super::super::params::{ClaudeThinkingBlock, Parameters, ToolCallMode, ToolDefinition};
use super::super::provider::{ProviderConfig, get_api_key};
use crate::connector::NativeToolCall;

// Public ChatMessage type for structured conversation history.
use super::super::params::ChatMessage as ApiChatMessage;

use async_stream::stream;
use std::pin::Pin;
use tokio_stream::Stream;

// Request types

/// A content block within a Claude message.
#[derive(Clone, serde::Serialize)]
#[serde(untagged)]
enum ClaudeContentBlock {
    Text {
        #[serde(rename = "type")]
        kind: String,
        text: String,
    },
    /// Inline base64 image (multimodal delivery). Valid both as a top-level
    /// block in a user message and as an item inside a `tool_result` content
    /// array (screenshot tool results).
    Image {
        #[serde(rename = "type")]
        kind: String,
        source: ClaudeImageSource,
    },
    ToolUse {
        #[serde(rename = "type")]
        kind: String,
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        #[serde(rename = "type")]
        kind: String,
        tool_use_id: String,
        content: String,
    },
    /// `tool_result` whose content is a BLOCK ARRAY (text + inline images)
    /// instead of a plain string — how the Anthropic API carries images in
    /// tool results.
    ToolResultWithImages {
        #[serde(rename = "type")]
        kind: String,
        tool_use_id: String,
        content: Vec<ClaudeContentBlock>,
    },
    /// A thinking block from a PREVIOUS response, replayed verbatim (text +
    /// signature) at the start of the assistant message so the API can
    /// validate the reasoning continuity.
    Thinking {
        #[serde(rename = "type")]
        kind: String,
        thinking: String,
        signature: String,
    },
}

/// `source` object of a Claude base64 image block.
#[derive(Clone, serde::Serialize)]
struct ClaudeImageSource {
    #[serde(rename = "type")]
    kind: String,
    media_type: String,
    data: String,
}

/// Build a Claude image block from a connector [`ImageBlock`].
fn claude_image_block(image: &crate::connector::params::ImageBlock) -> ClaudeContentBlock {
    ClaudeContentBlock::Image {
        kind: "image".to_string(),
        source: ClaudeImageSource {
            kind: "base64".to_string(),
            media_type: image.media_type.clone(),
            data: image.data.clone(),
        },
    }
}

#[derive(serde::Serialize)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContentBlock>,
}

/// Simple message format used by the legacy `chat` and `chat_stream` functions.
/// Content is a plain text string (not content blocks).
#[derive(serde::Serialize)]
struct MessageContent {
    role: String,
    content: String,
}

#[derive(serde::Serialize)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_schema: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

/// Claude thinking configuration — either manual extended thinking
/// (`thinking: { type: "enabled", budget_tokens: N }`, the only mode on
/// Claude 4.5 and earlier) or adaptive thinking (`thinking: { type:
/// "adaptive" }` plus `output_config.effort`, the current mode on the 4.6
/// generation and later — `type: "enabled"` is deprecated there and returns
/// 400 on 4.7+). The harness maps its reasoning effort onto the right knob
/// for the model; `default` (no effort) omits the field entirely.
#[derive(serde::Serialize)]
#[serde(untagged)]
enum ThinkingConfig {
    Manual {
        #[serde(rename = "type")]
        kind: String,
        budget_tokens: u32,
    },
    Adaptive {
        #[serde(rename = "type")]
        kind: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_config: Option<OutputConfig>,
    },
}

/// `output_config` for adaptive thinking — the reasoning depth the model
/// applies to the whole response (thinking included).
#[derive(serde::Serialize)]
struct OutputConfig {
    effort: String,
}

/// Minimum output headroom reserved ABOVE the thinking budget so the final
/// answer always fits: thinking tokens count toward `max_tokens` and the API
/// returns 400 when `budget_tokens >= max_tokens`.
const THINKING_OUTPUT_HEADROOM: u32 = 1024;

/// Default output ceiling for ADAPTIVE thinking turns.
///
/// Adaptive models size their own thinking WITHIN `max_tokens` (thinking +
/// answer must both fit). The plain 4096 default would squeeze a `high`-
/// effort response, silently capping how much the model can reason — the
/// classic "high feels like low" symptom. Raised only when an adaptive
/// effort is active; an explicit caller max_tokens above the floor is
/// respected verbatim.
const ADAPTIVE_THINKING_MAX_TOKENS: u32 = 16_384;

/// Whether the model accepts adaptive thinking (`thinking.type: "adaptive"`).
///
/// Per Anthropic's current docs, adaptive thinking is the mode for the 4.6
/// generation and later (Sonnet 4.6, Opus 4.6, Opus 4.7, Opus 4.8, Sonnet 5,
/// Opus 5, Fable 5, Mythos 5). Earlier Claude 4 models (Sonnet 4.5, Opus
/// 4.5, Haiku 4.5, Opus 4, Opus 4.1, Sonnet 4) and Claude 3.x support only
/// manual extended thinking — `type: "adaptive"` returns 400 there.
fn model_supports_adaptive_thinking(model: &str) -> bool {
    let m = model.to_ascii_lowercase().replace('.', "-");
    m.contains("sonnet-4-6")
        || m.contains("opus-4-6")
        || m.contains("opus-4-7")
        || m.contains("opus-4-8")
        || m.contains("sonnet-5")
        || m.contains("opus-5")
        || m.contains("fable-5")
        || m.contains("mythos-5")
}

/// Manual extended-thinking budget for a reasoning effort (Claude 4.5 and
/// earlier). Returns `None` for levels the API has no budget for.
fn budget_for_effort(effort: &str) -> Option<u32> {
    match effort {
        "minimal" => Some(1024),
        "low" => Some(2048),
        "medium" => Some(8192),
        "high" => Some(24_576),
        "max" => Some(32_768),
        _ => None,
    }
}

/// Map a reasoning effort onto the model's thinking configuration.
///
/// Newer models (4.6+) get adaptive thinking with `output_config.effort`;
/// older models get manual extended thinking with a token budget. Returns
/// `None` for the default (no effort) or for levels a model cannot express
/// (`minimal`/`none` — Claude has no such effort, so the model default is
/// used instead of guessing).
fn thinking_for_effort(model: &str, effort: &str) -> Option<ThinkingConfig> {
    if model_supports_adaptive_thinking(model) {
        return match effort {
            "low" | "medium" | "high" | "xhigh" | "max" => Some(ThinkingConfig::Adaptive {
                kind: "adaptive".to_string(),
                output_config: Some(OutputConfig {
                    effort: effort.to_string(),
                }),
            }),
            _ => None,
        };
    }
    budget_for_effort(effort).map(|budget_tokens| ThinkingConfig::Manual {
        kind: "enabled".to_string(),
        budget_tokens,
    })
}

/// Resolve the effective `max_tokens` for a request.
///
/// Manual extended thinking requires `max_tokens > budget_tokens` — the
/// budget counts toward the turn's output ceiling and the API rejects the
/// request with 400 when it leaves no room for the answer. When a manual
/// budget is active the floor is raised to `budget + headroom`, so an
/// explicit caller max_tokens is only respected verbatim when it already
/// clears budget + headroom; a smaller explicit value is raised to the
/// floor (the 1024 headroom guarantees the answer always fits).
fn effective_max_tokens(user_max: Option<u32>, thinking: &Option<ThinkingConfig>) -> u32 {
    match thinking {
        Some(ThinkingConfig::Manual { budget_tokens, .. }) => user_max
            .unwrap_or(0)
            .max(*budget_tokens + THINKING_OUTPUT_HEADROOM),
        Some(ThinkingConfig::Adaptive { .. }) => {
            user_max.unwrap_or(0).max(ADAPTIVE_THINKING_MAX_TOKENS)
        }
        _ => user_max.unwrap_or(4096),
    }
}

/// Predict the wire output ceiling with the same thinking policy as builders.
pub(crate) fn request_max_tokens(params: &Parameters, model: &str) -> u32 {
    let thinking = params
        .reasoning_effort
        .as_deref()
        .and_then(|effort| thinking_for_effort(model, effort));
    effective_max_tokens(params.max_tokens, &thinking)
}

/// Top-level automatic prompt-cache control sent on Claude requests.
///
/// With a single `cache_control` at the request root, the API applies the
/// cache breakpoint to the last cacheable block and moves it forward as the
/// conversation grows — the prefix (tools → system → messages) is read from
/// cache on subsequent requests at 10% of the base input price. See the
/// Anthropic prompt-caching docs (automatic caching).
#[derive(serde::Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
}

fn cache_control_for(params: &Parameters) -> Option<CacheControl> {
    if !params.prompt_cache.enabled {
        return None;
    }
    Some(CacheControl {
        kind: "ephemeral".to_string(),
        ttl: if params.prompt_cache.one_hour {
            Some("1h".to_string())
        } else {
            None
        },
    })
}

#[derive(serde::Serialize)]
struct MessageRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata>,
}

// Response types

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct MessageResponse {
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(serde::Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct Usage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

// Helpers

fn convert_stop(value: &serde_json::Value) -> Option<Vec<String>> {
    match value {
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        serde_json::Value::Array(arr) => {
            let strs: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if strs.is_empty() { None } else { Some(strs) }
        }
        _ => None,
    }
}

fn convert_tool_choice(value: &serde_json::Value) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::String(s) => match s.as_str() {
            "auto" | "any" | "none" => Some(serde_json::json!({"type": s})),
            _ => None,
        },
        v => Some(v.clone()),
    }
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        })
        .collect()
}

fn build_messages(prompt: &str) -> Vec<MessageContent> {
    vec![MessageContent {
        role: "user".to_string(),
        content: prompt.to_string(),
    }]
}

fn build_request(
    model: String,
    messages: Vec<MessageContent>,
    params: &Parameters,
    stream: bool,
    system_prompt: Option<&str>,
) -> MessageRequest {
    let stop_sequences = params.stop.as_ref().and_then(convert_stop);
    // Inline mode: no native `tools` on the wire — the model writes tool
    // calls as JSON in its text response and the harness parses them. The
    // two delivery paths are mutually exclusive at the request level.
    let native_tools = params.tool_call_mode == ToolCallMode::Native;
    let tools = if native_tools {
        params.tools.as_ref().map(|t| build_tools(t))
    } else {
        None
    };
    let tool_choice = if native_tools {
        params.tool_choice.as_ref().and_then(convert_tool_choice)
    } else {
        None
    };
    let thinking = params
        .reasoning_effort
        .as_deref()
        .and_then(|effort| thinking_for_effort(&model, effort));
    let metadata = params.user.as_ref().map(|uid| Metadata {
        user_id: Some(uid.clone()),
    });

    MessageRequest {
        model,
        max_tokens: effective_max_tokens(params.max_tokens, &thinking),
        messages,
        system: system_prompt.map(String::from),
        stream: if stream { Some(true) } else { None },
        temperature: params.temperature,
        top_p: params.top_p,
        stop_sequences,
        tools,
        tool_choice,
        thinking,
        cache_control: cache_control_for(params),
        metadata,
    }
}

fn extract_response_text(response: &MessageResponse) -> Result<String, ConnectorError> {
    let texts: Vec<&str> = response
        .content
        .iter()
        .filter(|b| b.kind == "text")
        .filter_map(|b| b.text.as_deref())
        .collect();
    if texts.is_empty() {
        return Err(ConnectorError::NoContent);
    }
    Ok(texts.concat())
}

struct RequestContext {
    api_key: String,
    request: MessageRequest,
}

fn prepare_request(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    stream: bool,
    service: Option<&str>,
) -> Result<RequestContext, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;

    let messages = build_messages(prompt);
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());
    let request = build_request(model, messages, params, stream, system_prompt);

    Ok(RequestContext { api_key, request })
}

// Public API

pub async fn chat(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatOutput, ConnectorError> {
    let ctx = prepare_request(config, params, prompt, system_prompt, false, service)?;
    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/messages");

    let mut headers: Vec<(&str, String)> = vec![
        ("x-api-key", ctx.api_key.clone()),
        ("anthropic-version", "2023-06-01".to_string()),
    ];
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    let response_text = send_request(config, &url, &ctx.request, &headers).await?;
    let chat_response: MessageResponse = serde_json::from_str(&response_text)?;
    let message = extract_response_text(&chat_response)?;
    Ok(ChatOutput {
        raw: response_text,
        message,
    })
}

/// Convert our structured [`ApiChatMessage`] array to `Claude` message format
/// with content blocks (text, `tool_use`, `tool_result`).
fn convert_to_claude_messages(history: &[ApiChatMessage]) -> Vec<ClaudeMessage> {
    let mut out = Vec::with_capacity(history.len());
    for msg in history {
        match msg.role.as_str() {
            "assistant" if msg.tool_calls.is_some() => {
                let mut content = Vec::new();
                // Claude extended thinking: replay the thinking blocks that
                // preceded the tool_use in the original response, VERBATIM
                // (the Anthropic API validates the signature cryptographically
                // and rejects modified or missing blocks with 400). Only the
                // thinking-enabled Claude path sets them.
                if let Some(blocks) = &msg.thinking_blocks {
                    for block in blocks {
                        content.push(ClaudeContentBlock::Thinking {
                            kind: "thinking".to_string(),
                            thinking: block.thinking.clone(),
                            signature: block.signature.clone(),
                        });
                    }
                }
                if let Some(ref text) = msg.content
                    && !text.is_empty()
                {
                    content.push(ClaudeContentBlock::Text {
                        kind: "text".to_string(),
                        text: text.clone(),
                    });
                }
                if let Some(ref tcs) = msg.tool_calls {
                    for tc in tcs {
                        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| {
                                serde_json::Value::Object(serde_json::Map::default())
                            });
                        content.push(ClaudeContentBlock::ToolUse {
                            kind: "tool_use".to_string(),
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input: args,
                        });
                    }
                }
                if content.is_empty() {
                    content.push(ClaudeContentBlock::Text {
                        kind: "text".to_string(),
                        text: String::new(),
                    });
                }
                out.push(ClaudeMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
            "tool" => {
                let text = msg.content.clone().unwrap_or_default();
                let content = match &msg.images {
                    // Multimodal tool result: text + inline images travel as
                    // a block array (the Anthropic shape for screenshots).
                    Some(images) if !images.is_empty() => {
                        let mut blocks = Vec::with_capacity(images.len() + 1);
                        if !text.is_empty() {
                            blocks.push(ClaudeContentBlock::Text {
                                kind: "text".to_string(),
                                text,
                            });
                        }
                        for image in images {
                            blocks.push(claude_image_block(image));
                        }
                        vec![ClaudeContentBlock::ToolResultWithImages {
                            kind: "tool_result".to_string(),
                            tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                            content: blocks,
                        }]
                    }
                    // Plain text result keeps the original string shape.
                    _ => vec![ClaudeContentBlock::ToolResult {
                        kind: "tool_result".to_string(),
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: text,
                    }],
                };
                out.push(ClaudeMessage {
                    role: "user".to_string(),
                    content,
                });
            }
            _ => {
                // user or plain assistant messages
                let mut content = Vec::new();
                if let Some(text) = msg.content.as_deref().filter(|t| !t.is_empty()) {
                    content.push(ClaudeContentBlock::Text {
                        kind: "text".to_string(),
                        text: text.to_string(),
                    });
                }
                if let Some(images) = &msg.images {
                    for image in images {
                        content.push(claude_image_block(image));
                    }
                }
                if content.is_empty() {
                    content.push(ClaudeContentBlock::Text {
                        kind: "text".to_string(),
                        text: String::new(),
                    });
                }
                out.push(ClaudeMessage {
                    role: msg.role.clone(),
                    content,
                });
            }
        }
    }
    out
}

/// Accumulated tool use from Claude's streaming `SSE` events.
#[derive(Default, Debug)]
struct ClaudePendingToolUse {
    id: String,
    name: String,
    input_json: String,
}

/// Emit the turn's accumulated Claude thinking blocks as a single chunk.
///
/// The blocks (summary text + signature) are delivered verbatim so the
/// harness can replay them on the follow-up request — the Anthropic API
/// validates the signature and requires the exact blocks that preceded the
/// `tool_use`. Returns `None` when the turn had no thinking blocks.
fn take_thinking_chunk(
    pending_thinking: &mut Vec<ClaudeThinkingBlock>,
    last_raw: &Option<String>,
) -> Option<StreamChunk> {
    if pending_thinking.is_empty() {
        return None;
    }
    Some(StreamChunk {
        raw: last_raw.clone().unwrap_or_default(),
        token: String::new(),
        reasoning: String::new(),
        thinking_blocks: Some(std::mem::take(pending_thinking)),
        finish_reason: None,
        tool_call: None,
        reset: false,
    })
}

/// Flush accumulated Claude tool uses as structured [`StreamChunk`] items
/// carrying the provider's native call verbatim (no inline-JSON round trip:
/// the harness extractor validates the structured fields directly).
fn flush_claude_tool_uses(
    pending: &mut Vec<ClaudePendingToolUse>,
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
                arguments: tc.input_json,
                thought_signature: String::new(),
            }),
            reset: false,
        });
    }
    out
}

/// A Claude request struct that supports the full messages array with
/// content blocks (`text`, `tool_use`, `tool_result`).
#[derive(serde::Serialize)]
struct ClaudeMessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata>,
}

pub async fn chat_stream_with_messages(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: &[ApiChatMessage],
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());

    let claude_messages = convert_to_claude_messages(messages);

    let stop_sequences = params.stop.as_ref().and_then(convert_stop);
    // Inline mode: no native `tools` on the wire (see `build_request`).
    let native_tools = params.tool_call_mode == ToolCallMode::Native;
    let tools = if native_tools {
        params.tools.as_ref().map(|t| build_tools(t))
    } else {
        None
    };
    let tool_choice = if native_tools {
        params.tool_choice.as_ref().and_then(convert_tool_choice)
    } else {
        None
    };
    let thinking = params
        .reasoning_effort
        .as_deref()
        .and_then(|effort| thinking_for_effort(&model, effort));
    let metadata = params.user.as_ref().map(|uid| Metadata {
        user_id: Some(uid.clone()),
    });

    let request = ClaudeMessagesRequest {
        model,
        max_tokens: effective_max_tokens(params.max_tokens, &thinking),
        messages: claude_messages,
        system: Some(system.to_string()),
        stream: Some(true),
        temperature: params.temperature,
        top_p: params.top_p,
        stop_sequences,
        tools,
        tool_choice,
        thinking,
        cache_control: cache_control_for(params),
        metadata,
    };

    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/messages");

    let mut headers: Vec<(&str, String)> = vec![
        ("x-api-key", api_key.clone()),
        ("anthropic-version", "2023-06-01".to_string()),
    ];
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    log::debug!("chat_stream_with_messages (Claude): sending request to {url}");
    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&request)?;
        let request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        )
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01");
        let request_builder = apply_session_headers(request_builder, params.session_id.as_deref());
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
            parse_sse_stream_with_tools,
        )
    } else {
        let response = send_request_stream(config, &url, &request, &headers).await?;
        parse_sse_stream_with_tools(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Parse a Claude SSE response into stream chunks, accumulating `tool_use`
/// blocks and thinking blocks (the `stream_chat_with_messages` path — the
/// agent loop). The thinking blocks that preceded a tool call are REQUIRED
/// on the follow-up request, so they are emitted verbatim (signature
/// included) before the tool-call chunks.
fn parse_sse_stream_with_tools(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    Box::pin(stream! {
            let mut response = response;
            let mut buf = SseBuffer::new();
            // Accumulate tool_use blocks across streaming events.
            let mut pending_tool_uses: Vec<ClaudePendingToolUse> = Vec::new();
            // Track the current content block index for tool_use accumulation.
            let mut current_block_index: Option<usize> = None;
            let mut last_raw: Option<String>;
            // Claude extended-thinking capture: thinking blocks (summary text
            // + signature) accumulate here and are emitted as a single chunk
            // at turn end so the harness can replay them verbatim on the
            // follow-up request (the API requires the exact blocks that
            // preceded the tool_use, signature included).
            let mut pending_thinking: Vec<ClaudeThinkingBlock> = Vec::new();
            // Index of the currently open thinking block (None = none open).
            let mut thinking_block_index: Option<usize> = None;
            let mut current_thinking: Option<ClaudeThinkingBlock> = None;

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
                            yield Err(ConnectorError::Network(e.to_string()));
                            return;
                        }
                        Err(_) => {
                            yield Err(ConnectorError::Network(format!(
                                "stream timed out after {}s",
                                SSE_CHUNK_TIMEOUT.as_secs()
                            )));
                            return;
                        }
                    };
                for data in frames {
                    match serde_json::from_str::<serde_json::Value>(&data) {
                        Ok(v) => {
                            last_raw = Some(data.clone());
                            let kind = v["type"].as_str().unwrap_or("");
                            match kind {
                                "content_block_start" => {
                                    let block = &v["content_block"];
                                    let block_type = block["type"].as_str().unwrap_or("");
                                    let idx = v["index"].as_u64().unwrap_or(0) as usize;
                                    if block_type == "tool_use" {
                                        let id = block["id"].as_str().unwrap_or("").to_string();
                                        let name = block["name"].as_str().unwrap_or("").to_string();
                                        if idx >= pending_tool_uses.len() {
                                            pending_tool_uses.resize_with(idx + 1, Default::default);
                                        }
                                        pending_tool_uses[idx] = ClaudePendingToolUse {
                                            id,
                                            name,
                                            input_json: String::new(),
                                        };
                                        current_block_index = Some(idx);
                                    } else if block_type == "thinking" {
                                        // Open a thinking block for verbatim
                                        // replay capture. The block's initial
                                        // text arrives HERE (the rest via
                                        // `thinking_delta`, the signature via
                                        // `signature_delta`).
                                        let initial = block["thinking"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        thinking_block_index = Some(idx);
                                        current_thinking = Some(ClaudeThinkingBlock {
                                            thinking: initial.clone(),
                                            signature: block["signature"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_string(),
                                        });
                                        // Stream the initial thinking text to
                                        // the TUI as reasoning too — the block
                                        // start carries the FIRST chunk of the
                                        // summary, without it the TUI would
                                        // miss it (only `thinking_delta`
                                        // chunks would stream).
                                        if !initial.is_empty() {
                                            yield Ok(StreamChunk {
                                                raw: data.clone(),
                                                token: String::new(),
                                                reasoning: initial,
                                                finish_reason: None,
                                                thinking_blocks: None,
                                                tool_call: None,
                                                reset: false,
                                            });
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    let delta = &v["delta"];
                                    let delta_type = delta["type"].as_str().unwrap_or("");
                                    match delta_type {
                                        "text_delta" => {
                                            let token = delta["text"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_owned();
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
                                        "thinking_delta" => {
                                            let text = delta["thinking"]
                                                .as_str()
                                                .unwrap_or("")
                                                .to_owned();
                                            if !text.is_empty() {
                                                if let Some(t) = current_thinking.as_mut() {
                                                    t.thinking.push_str(&text);
                                                }
                                                // Stream the thinking summary
                                                // to the TUI as reasoning.
                                                yield Ok(StreamChunk {
                                                    raw: data.clone(),
                                                    token: String::new(),
                                                    reasoning: text,
                                                    finish_reason: None,
                                                    thinking_blocks: None,
                                                    tool_call: None,
                                                    reset: false,
                                                });
                                            }
                                        }
                                        "signature_delta" => {
                                            if let Some(t) = current_thinking.as_mut()
                                                && let Some(sig) = delta["signature"].as_str()
                                            {
                                                t.signature = sig.to_string();
                                            }
                                        }
                                        "input_json_delta" => {
                                            if let Some(idx) = current_block_index
                                                && idx < pending_tool_uses.len()
                                                && let Some(partial) = delta["partial_json"].as_str()
                                            {
                                                pending_tool_uses[idx].input_json.push_str(partial);
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                "content_block_stop" => {
                                    // Close an open thinking block (its
                                    // thinking_delta/signature_delta filled it).
                                    if let Some(idx) = thinking_block_index
                                        && v["index"].as_u64().map(|i| i as usize) == Some(idx)
                                        && let Some(t) = current_thinking.take()
                                    {
                                        pending_thinking.push(t);
                                        thinking_block_index = None;
                                    }
                                    current_block_index = None;
                                }
                                "message_delta" => {
                                    // Canonicalize the raw stop_reason
                                    // (`max_tokens` → `length`, ...) — the
                                    // harness decides on the canonical
                                    // vocabulary only.
                                    let finish_reason = normalize_finish_reason(
                                        v["delta"]["stop_reason"].as_str().map(String::from),
                                    );
                                    let should_stop = finish_reason.is_some();

                                    if should_stop && !pending_tool_uses.is_empty() {
                                        // The turn ends with a tool call: the
                                        // thinking blocks that preceded it are
                                        // REQUIRED on the follow-up request —
                                        // emit them (verbatim, signature
                                        // included) before the tool-call
                                        // chunks so the harness can replay.
                                        if let Some(chunk) =
                                            take_thinking_chunk(&mut pending_thinking, &last_raw)
                                        {
                                            yield Ok(chunk);
                                        }
                                        for chunk in flush_claude_tool_uses(&mut pending_tool_uses, &mut last_raw) {
                                            yield Ok(chunk);
                                        }
                                        return;
                                    }

                                    if should_stop {
                                        yield Ok(StreamChunk {
                                            raw: data,
                                            token: String::new(),
                                            reasoning: String::new(),
                                            finish_reason,
                                            thinking_blocks: None,
                                            tool_call: None,
                                            reset: false,
                                        });
                                        return;
                                    }
                                }
                                "message_stop" => {
                                    // Flush any remaining tool uses (and the
                                    // turn's thinking blocks, which are only
                                    // required when a tool_use accompanied
                                    // them).
                                    if !pending_tool_uses.is_empty() {
                                        if let Some(chunk) =
                                            take_thinking_chunk(&mut pending_thinking, &last_raw)
                                        {
                                            yield Ok(chunk);
                                        }
                                        for chunk in flush_claude_tool_uses(&mut pending_tool_uses, &mut last_raw) {
                                            yield Ok(chunk);
                                        }
                                    }
                                    return;
                                }
                                "error" => {
                                    let msg = v["error"]["message"]
                                        .as_str()
                                        .unwrap_or("unknown error");
                                    yield Err(ConnectorError::Network(msg.to_string()));
                                    return;
                                }
                                _ => {}
                            }
                        }
                        Err(e) => {
                            yield Err(ConnectorError::Deserialization(e.to_string()));
                            return;
                        }
                    }
                }
                if ended {
                    break;
                }
            }
    })
}

pub async fn chat_stream(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let ctx = prepare_request(config, params, prompt, system_prompt, true, service)?;
    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/messages");

    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&ctx.request)?;
        let request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        )
        .header("x-api-key", ctx.api_key)
        .header("anthropic-version", "2023-06-01");
        let request_builder = apply_session_headers(request_builder, params.session_id.as_deref());
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
            parse_sse_stream,
        )
    } else {
        let mut headers: Vec<(&str, String)> = vec![
            ("x-api-key", ctx.api_key.clone()),
            ("anthropic-version", "2023-06-01".to_string()),
        ];
        if let Some(sid) = params.session_id.as_deref() {
            headers.push(("x-session-id", sid.to_string()));
            headers.push(("x-session-affinity", sid.to_string()));
        }
        let response = send_request_stream(config, &url, &ctx.request, &headers).await?;
        parse_sse_stream(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Parse a Claude SSE response into stream chunks (text, reasoning,
/// thinking blocks, finish reason — no tool accumulation; the
/// `stream_chat`/`stream_chat_with_system` path).
fn parse_sse_stream(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    Box::pin(stream! {
            let mut response = response;
            let mut buf = SseBuffer::new();
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
                            yield Err(ConnectorError::Network(e.to_string()));
                            return;
                        }
                        Err(_) => {
                            yield Err(ConnectorError::Network(format!(
                                "stream timed out after {}s",
                                SSE_CHUNK_TIMEOUT.as_secs()
                            )));
                            return;
                        }
                    };
                for data in frames {
                    match serde_json::from_str::<serde_json::Value>(&data) {
                        Ok(v) => {
                            let kind = v["type"].as_str().unwrap_or("");
                            match kind {
                                "content_block_delta"
                                    if v["delta"]["type"] == "text_delta" =>
                                {
                                    let token = v["delta"]["text"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_owned();
                                    yield Ok(StreamChunk {
                                        raw: data,
                                        token,
                                        reasoning: String::new(),
                                        finish_reason: None,
                                        thinking_blocks: None,
                                        tool_call: None,
                                        reset: false,
                                    });
                                }
                                "message_delta" => {
                                    // Canonicalize the raw stop_reason
                                    // (`max_tokens` → `length`, ...) — the
                                    // harness decides on the canonical
                                    // vocabulary only.
                                    let finish_reason = normalize_finish_reason(
                                        v["delta"]["stop_reason"].as_str().map(String::from),
                                    );
                                    yield Ok(StreamChunk {
                                        raw: data,
                                        token: String::new(),
                                        reasoning: String::new(),
                                        finish_reason,
                                        thinking_blocks: None,
                                        tool_call: None,
                                        reset: false,
                                    });
                                }
                                "message_stop" => return,
                                "error" => {
                                    let msg = v["error"]["message"]
                                        .as_str()
                                        .unwrap_or("unknown error");
                                    yield Err(ConnectorError::Network(msg.to_string()));
                                    return;
                                }
                                _ => {}
                            }
                        }
                        Err(e) => {
                            yield Err(ConnectorError::Deserialization(e.to_string()));
                            return;
                        }
                    }
                }
                if ended {
                    break;
                }
            }
    })
}

#[derive(serde::Deserialize)]
struct ClaudeModelEntry {
    id: String,
}

#[derive(serde::Deserialize)]
struct ClaudeListModelsResponse {
    data: Vec<ClaudeModelEntry>,
}

/// Fetch the list of available models from the Anthropic Claude API.
///
/// Sends a GET to `{base_url}/models` and returns a [`LsOutput`].
pub async fn list_models(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Result<LsOutput, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;

    let base_url = params.base_url.as_deref().unwrap_or(config.base_url);
    let url = format!("{base_url}/models");

    let mut headers: Vec<(&str, String)> = vec![
        ("x-api-key", api_key.clone()),
        ("anthropic-version", "2023-06-01".to_string()),
    ];
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    let response_text = send_get_request(config, &url, &headers).await?;

    let list: ClaudeListModelsResponse = serde_json::from_str(&response_text)?;
    let models: Vec<ModelInfo> = list
        .data
        .into_iter()
        .map(|entry| ModelInfo { id: entry.id })
        .collect();

    Ok(LsOutput::new(response_text, models))
}
