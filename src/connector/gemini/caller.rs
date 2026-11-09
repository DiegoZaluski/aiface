use super::super::common::{
    SSE_CHUNK_TIMEOUT, SseBuffer, apply_provider_headers, apply_session_headers, retry_mid_stream,
    send_get_request, send_request, send_request_stream, send_with_retry, shared_client,
};
use super::super::error::ConnectorError;
use super::super::output::{
    ChatOutput, ChatStream, LsOutput, ModelInfo, StreamChunk, normalize_finish_reason,
};
use super::super::params::{ChatMessage, Parameters, ToolCallMode, ToolDefinition};
use super::super::provider::{ProviderConfig, get_api_key};
use crate::connector::NativeToolCall;

use async_stream::stream;
use std::collections::HashMap;
use std::pin::Pin;
use tokio_stream::Stream;

use super::API_VERSION;

/// A single Gemini `Part` — one of `text`, `functionCall`, or `functionResponse`.
/// Serialized untagged so exactly one key is emitted.
///
/// A `functionCall` part MAY carry a sibling `thoughtSignature` (Gemini 3.x
/// thinking models): it is metadata attached at the PART level (the API
/// rejects `thoughtSignature` INSIDE the `functionCall` object — verified
/// empirically with a 400). It is emitted verbatim when re-sending the call
/// in the conversation history; `skip_serializing_if` keeps it off the wire
/// for non-thinking calls and for every other provider.
#[derive(serde::Serialize)]
#[serde(untagged)]
enum Part {
    Text {
        text: String,
    },
    /// Inline base64 image (multimodal delivery: screenshot tool results).
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineDataPart,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPart,
        #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePart,
    },
}

/// Gemini `inlineData` part (request side): base64 media with a MIME type.
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct InlineDataPart {
    mime_type: String,
    data: String,
}

impl Part {
    fn text(s: impl Into<String>) -> Self {
        Part::Text { text: s.into() }
    }

    fn inline_data(mime_type: &str, data: &str) -> Self {
        Part::InlineData {
            inline_data: InlineDataPart {
                mime_type: mime_type.to_string(),
                data: data.to_string(),
            },
        }
    }

    fn function_call(fc: FunctionCallPart) -> Self {
        Part::FunctionCall {
            function_call: fc,
            thought_signature: None,
        }
    }

    fn function_call_with_signature(fc: FunctionCallPart, signature: String) -> Self {
        Part::FunctionCall {
            function_call: fc,
            thought_signature: Some(signature),
        }
    }

    fn function_response(fr: FunctionResponsePart) -> Self {
        Part::FunctionResponse {
            function_response: fr,
        }
    }
}

/// Gemini `functionCall` part (request + response).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct FunctionCallPart {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    args: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

/// Gemini `functionResponse` part (request only).
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FunctionResponsePart {
    name: String,
    response: serde_json::Value,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct Tool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<FunctionDeclaration>,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct FunctionCallingConfig {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(serde::Serialize)]
struct ToolConfig {
    #[serde(rename = "functionCallingConfig")]
    function_calling_config: FunctionCallingConfig,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
}

/// Gemini thinking knob — `generationConfig.thinkingConfig`.
///
/// `thinkingLevel` (Gemini 3.x) accepts `"minimal" | "low" | "medium" |
/// "high"`; the harness maps its reasoning effort onto these.
///
/// `includeThoughts` asks the API to RETURN the model's reasoning summary in
/// the response. Without it Gemini 3 streams only the final answer plus a
/// `thoughtSignature` — the thinking text never reaches the TUI (the model
/// thinks server-side, billed via `thoughtsTokenCount`, but there is nothing
/// to display). Best-effort: even when set, the summary may be absent for
/// turns where the model did not reason enough.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ThinkingConfig {
    thinking_level: String,
    include_thoughts: bool,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
}

// Response types (shared by both streaming and non-streaming responses)

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    function_call: Option<FunctionCallPart>,
    /// Marks a THINKING part: its `text` is the model's internal reasoning
    /// summary, never the final answer. Surfaced as `reasoning` (so the TUI
    /// shows it as thinking) instead of being emitted as a response token.
    #[serde(default)]
    thought: bool,
    /// Sibling `thoughtSignature` of a native `functionCall` part (Gemini 3.x
    /// thinking models). Captured so the harness can replay it verbatim on
    /// the follow-up request — the API rejects the call otherwise.
    #[serde(default)]
    thought_signature: Option<String>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResponseContent {
    parts: Vec<ResponsePart>,
    #[serde(default)]
    role: Option<String>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: ResponseContent,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default)]
    usage_metadata: Option<UsageMetadata>,
}

#[allow(dead_code)]
#[allow(clippy::struct_field_names)]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    total_token_count: u32,
}

// SSE streaming types — same shape as non-streaming response

// Embedding types

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest {
    content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentResponse {
    embedding: EmbeddingValues,
}

// Helpers

/// Gemini request headers: API key plus cache-affinity session headers when
/// a session id is configured.
pub(crate) fn gemini_headers(api_key: &str, session: Option<&str>) -> Vec<(&'static str, String)> {
    let mut headers = vec![("x-goog-api-key", api_key.to_string())];
    if let Some(sid) = session {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    headers
}

fn build_contents(prompt: &str, system_prompt: Option<&str>) -> (Vec<Content>, Option<Content>) {
    let system_instruction = system_prompt.map(|sys| Content {
        role: None,
        parts: vec![Part::text(sys)],
    });

    let contents = vec![Content {
        role: Some("user".to_string()),
        parts: vec![Part::text(prompt)],
    }];

    (contents, system_instruction)
}

fn build_generation_config(params: &Parameters) -> Option<GenerationConfig> {
    let response_mime_type = params.response_format.as_ref().and_then(|rf| {
        if rf.kind == "json_object" {
            Some("application/json".to_string())
        } else {
            None
        }
    });

    let stop_sequences = params.stop.as_ref().and_then(|v| match v {
        serde_json::Value::String(s) => Some(vec![s.clone()]),
        serde_json::Value::Array(arr) => {
            let strs: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if strs.is_empty() { None } else { Some(strs) }
        }
        _ => None,
    });

    let thinking_config = params.reasoning_effort.as_ref().and_then(|effort| {
        // Gemini 3.x accepts minimal/low/medium/high; anything else maps to
        // the model default (omit the knob).
        let level = match effort.as_str() {
            "minimal" | "low" | "medium" | "high" => effort.as_str(),
            _ => return None,
        };
        Some(ThinkingConfig {
            thinking_level: level.to_string(),
            // Always surface the reasoning summary — the user asked for a
            // thinking level, so the TUI should be able to show it.
            include_thoughts: true,
        })
    });

    let has_any = params.max_tokens.is_some()
        || params.temperature.is_some()
        || params.top_p.is_some()
        || stop_sequences.is_some()
        || params.frequency_penalty.is_some()
        || params.presence_penalty.is_some()
        || params.seed.is_some()
        || response_mime_type.is_some()
        || params.logprobs.is_some()
        || params.top_logprobs.is_some()
        || thinking_config.is_some();

    if !has_any {
        return None;
    }

    Some(GenerationConfig {
        max_output_tokens: params.max_tokens,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: None,
        stop_sequences,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        seed: params.seed,
        response_mime_type,
        response_logprobs: params.logprobs,
        logprobs: params.top_logprobs,
        thinking_config,
    })
}

/// JSON Schema keywords the Gemini API accepts inside
/// `functionDeclarations[].parameters`. Everything else is stripped by
/// [`sanitize_schema`] — the Gemini backend rejects unknown fields with
/// HTTP 400 (`Unknown name ... at 'tools[0].function_declarations[...]'`).
const GEMINI_SCHEMA_KEYWORDS: &[&str] = &[
    "type",
    "description",
    "format",
    "nullable",
    "required",
    "properties",
    "items",
    "enum",
    "maxItems",
    "minItems",
    "propertyOrdering",
];

/// Merge a branch property into the accumulated union map.
///
/// When the same key appears in several `oneOf`/`anyOf` branches (typical
/// for a discriminated union tag, e.g. `type` with a different `const` per
/// branch), the `enum` lists are UNIONED instead of overwriting — otherwise
/// the last branch silently drops the other valid values.
fn merge_branch_prop(
    props: &mut serde_json::Map<String, serde_json::Value>,
    key: &String,
    value: &serde_json::Value,
) {
    if let Some(existing) = props.get_mut(key) {
        // Union enums when both sides carry one.
        if let (
            Some(serde_json::Value::Array(existing_enum)),
            Some(serde_json::Value::Array(new_enum)),
        ) = (existing.get("enum"), value.get("enum"))
        {
            let mut merged = existing_enum.clone();
            for item in new_enum {
                if !merged.contains(item) {
                    merged.push(item.clone());
                }
            }
            existing["enum"] = serde_json::Value::Array(merged);
            return;
        }
    }
    props.insert(key.clone(), value.clone());
}

/// Sanitize a JSON Schema for the Gemini API.
///
/// Gemini supports only a small subset of JSON Schema in function
/// declarations, and rejects anything else with HTTP 400. Transformations:
/// - `const: v` → `enum: [v]` (semantically equivalent — the harness uses
///   `const` as a discriminated-union tag, e.g. the plan tools);
/// - `oneOf`/`anyOf`/`allOf` branches are merged into the parent: a union
///   of their `properties` and `required` (Gemini cannot express unions);
/// - every keyword outside [`GEMINI_SCHEMA_KEYWORDS`] is dropped recursively
///   (`additionalProperties`, `$schema`, `$ref`, `default`, `pattern`, …).
///
/// The extractor in the harness keeps the ORIGINAL schema (with `const` /
/// `oneOf`) for validating tool calls — only the request payload is
/// sanitized, so extraction fidelity is unaffected.
fn sanitize_schema(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(sanitize_schema).collect())
        }
        serde_json::Value::Object(map) => sanitize_schema_object(map),
        other => other.clone(),
    }
}

/// Object branch of [`sanitize_schema`] — see its docs for the contract.
fn sanitize_schema_object(map: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    // Union of properties/required across oneOf/anyOf/allOf branches.
    // Branches are sanitized first so nested `const` tags become `enum`
    // before merging.
    let mut branch_props: serde_json::Map<String, serde_json::Value> = Default::default();
    let mut branch_required: Vec<serde_json::Value> = Vec::new();
    for union_key in ["oneOf", "anyOf", "allOf"] {
        if let Some(serde_json::Value::Array(branches)) = map.get(union_key) {
            for branch in branches {
                let sanitized = sanitize_schema(branch);
                let serde_json::Value::Object(b) = sanitized else {
                    continue;
                };
                if let Some(props) = b.get("properties").and_then(|p| p.as_object()) {
                    for (k, v) in props {
                        merge_branch_prop(&mut branch_props, k, v);
                    }
                }
                if let Some(required) = b.get("required").and_then(|r| r.as_array()) {
                    for item in required {
                        if !branch_required.contains(item) {
                            branch_required.push(item.clone());
                        }
                    }
                }
            }
        }
    }

    let mut out: serde_json::Map<String, serde_json::Value> = Default::default();
    for (key, value) in map {
        match key.as_str() {
            "const" => {
                // Gemini function-declaration enums are STRING arrays; a
                // numeric/boolean const (possible from MCP schemas) would be
                // rejected. Only string consts are expressible as a single
                // value enum — anything else is dropped (its `type` stays).
                if let serde_json::Value::String(s) = value {
                    out.insert("enum".to_string(), serde_json::json!([s]));
                }
            }
            "oneOf" | "anyOf" | "allOf" => {
                // Handled above — flattened into the parent schema.
            }
            "properties" => {
                let mut props = serde_json::Map::new();
                if let Some(obj) = value.as_object() {
                    for (k, v) in obj {
                        props.insert(k.clone(), sanitize_schema(v));
                    }
                }
                // Branch properties fill in gaps; the parent's own wins.
                for (k, v) in &branch_props {
                    props.entry(k.clone()).or_insert_with(|| v.clone());
                }
                out.insert("properties".to_string(), serde_json::Value::Object(props));
            }
            "required" => {
                let mut req: Vec<serde_json::Value> = Vec::new();
                if let Some(arr) = value.as_array() {
                    for item in arr {
                        if !req.contains(item) {
                            req.push(item.clone());
                        }
                    }
                }
                for item in &branch_required {
                    if !req.contains(item) {
                        req.push(item.clone());
                    }
                }
                if !req.is_empty() {
                    out.insert("required".to_string(), serde_json::Value::Array(req));
                }
            }
            "items" => {
                out.insert("items".to_string(), sanitize_schema(value));
            }
            key if GEMINI_SCHEMA_KEYWORDS.contains(&key) => {
                out.insert(key.to_string(), value.clone());
            }
            _ => {
                // Drop unsupported keywords (additionalProperties, $schema,
                // $ref, default, pattern, minLength, maxLength, …).
            }
        }
    }
    // When the parent schema had no `properties`/`required` of its own (the
    // union lived entirely inside oneOf), hoist the merged branches up.
    if !out.contains_key("properties") && !branch_props.is_empty() {
        out.insert(
            "properties".to_string(),
            serde_json::Value::Object(branch_props),
        );
    }
    if !out.contains_key("required") && !branch_required.is_empty() {
        out.insert(
            "required".to_string(),
            serde_json::Value::Array(branch_required),
        );
    }
    serde_json::Value::Object(out)
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<Tool> {
    if tools.is_empty() {
        return vec![];
    }
    let declarations: Vec<FunctionDeclaration> = tools
        .iter()
        .map(|tool| FunctionDeclaration {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.as_ref().map(sanitize_schema),
        })
        .collect();
    vec![Tool {
        function_declarations: declarations,
    }]
}

/// Effective tool config for a Gemini request.
///
/// When an explicit `tool_choice` was set, it wins verbatim. Otherwise,
/// whenever tools are declared, the config is forced to `mode: "AUTO"`.
/// This is REQUIRED for Gemini 3.x models: without a `toolConfig`, they
/// obey the harness prompt's inline-JSON instruction (`TOOL_FORMAT`) and
/// emit tool calls as JSON TEXT mixed with prose (`Olá!...{"name":
/// "find_glob", ...}`) — garbage the extractor must fish out of, with
/// stray fragments (`}`) leaking to the user. With the toolConfig present,
/// the model expresses tool calls as native `functionCall` parts (clean,
/// structured) while still deciding freely whether to call one.
fn build_effective_tool_config(params: &Parameters) -> Option<ToolConfig> {
    if let Some(tc) = params.tool_choice.as_ref() {
        return build_tool_config(tc);
    }
    let has_tools = params.tools.as_ref().is_some_and(|t| !t.is_empty());
    has_tools.then(|| ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode: "AUTO".to_string(),
            allowed_function_names: None,
        },
    })
}

fn build_tool_config(tool_choice: &serde_json::Value) -> Option<ToolConfig> {
    match tool_choice {
        serde_json::Value::String(s) => {
            let mode = s.to_uppercase();
            if mode == "AUTO" || mode == "ANY" || mode == "NONE" {
                Some(ToolConfig {
                    function_calling_config: FunctionCallingConfig {
                        mode,
                        allowed_function_names: None,
                    },
                })
            } else {
                None
            }
        }
        serde_json::Value::Object(obj) => {
            let mode = obj
                .get("type")
                .and_then(|v| v.as_str())
                .map_or_else(|| "ANY".to_string(), str::to_uppercase);
            let names = obj
                .get("function")
                .and_then(|v| v.get("name"))
                .and_then(|v| v.as_str())
                .map(|n| vec![n.to_string()]);
            Some(ToolConfig {
                function_calling_config: FunctionCallingConfig {
                    mode,
                    allowed_function_names: names,
                },
            })
        }
        _ => None,
    }
}

fn extract_response_text(response: &GenerateContentResponse) -> Result<String, ConnectorError> {
    let candidate = response
        .candidates
        .first()
        .ok_or(ConnectorError::NoChoices)?;
    // A thinking model prepends `thought: true` parts (internal reasoning)
    // before the answer — never surface those as the response text.
    let texts: Vec<&str> = candidate
        .content
        .parts
        .iter()
        .filter(|p| !p.thought)
        .filter_map(|p| p.text.as_deref())
        .filter(|t| !t.is_empty())
        .collect();
    if texts.is_empty() {
        return Err(ConnectorError::NoContent);
    }
    Ok(texts.concat())
}

struct RequestContext {
    api_key: String,
    request: GenerateContentRequest,
}

/// Effective Gemini API version for a request.
///
/// Thinking-config requests (`generationConfig.thinkingConfig.thinkingLevel`)
/// must target `v1beta` — Google gates thinking configuration to the beta
/// surface: the stable `v1` endpoint rejects `thinkingConfig` with HTTP 400,
/// and every official doc/example routes it through `/v1beta`. Everything
/// else keeps the stable [`API_VERSION`].
pub(crate) fn api_version_for(params: &Parameters) -> &'static str {
    let has_thinking_level = params
        .reasoning_effort
        .as_deref()
        .is_some_and(|e| matches!(e, "minimal" | "low" | "medium" | "high"));
    if has_thinking_level {
        "v1beta"
    } else {
        API_VERSION
    }
}

/// Resolve the effective Gemini API base URL.
///
/// A user-supplied `params.base_url` (e.g. a proxy or test server) wins
/// verbatim. Otherwise the provider's registered stable URL is used with
/// its version segment re-written to the effective version — the INTERNAL
/// [`API_VERSION`] switch (stable `v1` by default) or `v1beta` for requests
/// that carry a thinking level (see [`api_version_for`]).
pub(crate) fn resolve_base_url(config: &ProviderConfig, params: &Parameters) -> String {
    if let Some(custom) = params.base_url.as_deref() {
        return custom.to_string();
    }
    let default = config.base_url; // e.g. "https://generativelanguage.googleapis.com/v1"
    let prefix = default
        .rsplit_once('/')
        .map_or(default, |(prefix, _)| prefix);
    format!("{prefix}/{}", api_version_for(params))
}

fn prepare_request(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<RequestContext, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;

    let (contents, system_instruction) = build_contents(prompt, system_prompt);
    let generation_config = build_generation_config(params);
    // Inline mode: no native `tools` on the wire — the model writes tool
    // calls as JSON into its text response and the harness parses them.
    // Without `tools` (and `toolConfig`) the API can never emit a
    // `functionCall` part, so the two delivery paths never cross.
    let native_tools = params.tool_call_mode == ToolCallMode::Native;
    let tools = if native_tools {
        params.tools.as_ref().map(|t| build_tools(t))
    } else {
        None
    };
    let tool_config = if native_tools {
        build_effective_tool_config(params)
    } else {
        None
    };

    let request = GenerateContentRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
        tool_config,
    };

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
    let ctx = prepare_request(config, params, prompt, system_prompt, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());
    let base_url = resolve_base_url(config, params);
    let url = format!("{base_url}/models/{model}:generateContent");

    let response_text = send_request(
        config,
        &url,
        &ctx.request,
        &gemini_headers(ctx.api_key.as_str(), params.session_id.as_deref()),
    )
    .await?;
    let chat_response: GenerateContentResponse = serde_json::from_str(&response_text)?;
    let message = extract_response_text(&chat_response)?;
    Ok(ChatOutput {
        raw: response_text.clone(),
        message,
        // Image-capable Gemini models answer with interleaved text/image
        // parts; a text-only model parses to an empty vec here.
        images: super::images::extract_output_images(&response_text),
    })
}

pub async fn chat_stream(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let ctx = prepare_request(config, params, prompt, system_prompt, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| config.default_model.to_string());
    let base_url = resolve_base_url(config, params);
    // `alt=sse` is REQUIRED on BOTH v1 and v1beta: without it the Gemini
    // API does not emit SSE frames (no `data: ` prefixes) — it returns a
    // raw JSON array/NDJSON body, which the SseBuffer parser would silently
    // produce zero frames from (empty stream, no tokens, no error).
    let url = format!("{base_url}/models/{model}:streamGenerateContent?alt=sse");

    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&ctx.request)?;
        let request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        )
        .header("x-goog-api-key", ctx.api_key);
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
        let response = send_request_stream(
            config,
            &url,
            &ctx.request,
            &gemini_headers(ctx.api_key.as_str(), params.session_id.as_deref()),
        )
        .await?;
        parse_sse_stream(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Parse a Gemini `?alt=sse` response body into stream chunks (text,
/// reasoning, finish reason and native `functionCall` parts). Shared by the
/// retried streaming paths.
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
                    match serde_json::from_str::<GenerateContentResponse>(&data) {
                        Ok(ccr) => {
                            // Canonicalize the raw enum (`MAX_TOKENS` →
                            // `length`, `SAFETY` → `content_filter`, ...).
                            let finish_reason = normalize_finish_reason(
                                ccr.candidates
                                    .first()
                                    .and_then(|c| c.finish_reason.as_deref())
                                    .map(String::from),
                            );
                            let should_stop = finish_reason.is_some();
                            // A thinking model streams its reasoning as
                            // `thought: true` parts (text = the internal
                            // summary) ahead of the final answer. Separate
                            // them: thoughts → reasoning, text → token.
                            if let Some(candidate) = ccr.candidates.first() {
                                for part in &candidate.content.parts {
                                    let Some(text) = part.text
                                        .as_deref()
                                        .filter(|t| !t.is_empty())
                                    else {
                                        continue;
                                    };
                                    if part.thought {
                                        yield Ok(StreamChunk {
                                            raw: data.clone(),
                                            token: String::new(),
                                            reasoning: text.to_string(),
                                            finish_reason: None,
                                            thinking_blocks: None,
                                            tool_call: None,
                                            reset: false,
                                        });
                                    } else {
                                        yield Ok(StreamChunk {
                                            raw: data.clone(),
                                            token: text.to_string(),
                                            reasoning: String::new(),
                                            finish_reason: None,
                                            thinking_blocks: None,
                                            tool_call: None,
                                            reset: false,
                                        });
                                    }
                                }
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

/// Convert our structured [`ChatMessage`] array to Gemini `contents[]`.
///
/// Gemini has no `system`/`assistant`/`tool` roles — only `user` and
/// `model`, with tool work carried by `functionCall`/`functionResponse`
/// parts. Mapping rules:
/// - `system` messages are skipped (the system prompt travels separately
///   in `systemInstruction`);
/// - `user`/plain `assistant` become `user`/`model` text parts;
/// - `assistant` with `tool_calls` becomes a `model` content with
///   `functionCall` parts;
/// - `tool` results become a `user` content with a `functionResponse` part
///   whose `name` is resolved from the tool_call_id seen on a preceding
///   assistant `tool_calls`;
/// - consecutive contents mapping to the same role are merged (Gemini
///   rejects repeated roles).
fn convert_messages(messages: &[ChatMessage]) -> Vec<Content> {
    // Map tool_call_id → function name from assistant tool_calls so tool
    // results can build a functionResponse with the right `name`.
    let mut name_by_id: HashMap<&str, &str> = HashMap::new();
    for msg in messages {
        if msg.role == "assistant"
            && let Some(tcs) = &msg.tool_calls
        {
            for tc in tcs {
                name_by_id.insert(tc.id.as_str(), tc.function.name.as_str());
            }
        }
    }

    let mut contents: Vec<Content> = Vec::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => continue, // travels via systemInstruction
            "assistant" => "model",
            _ => "user", // "user" and "tool" both map to user
        };

        let mut parts = Vec::new();
        // Tool results are NOT emitted as text — they travel exclusively as
        // a `functionResponse` part (the raw JSON content would otherwise
        // be duplicated into a text part).
        if msg.role != "tool"
            && let Some(text) = msg.content.as_deref().filter(|t| !t.is_empty())
        {
            parts.push(Part::text(text));
        }
        // Attached images (multimodal delivery) become sibling `inlineData`
        // parts — for tool results they ride in the same user content right
        // after the `functionResponse` part, which is where Gemini requires
        // binary media to live.
        for image in msg.images.as_ref().into_iter().flatten() {
            parts.push(Part::inline_data(&image.media_type, &image.data));
        }
        if let Some(tcs) = &msg.tool_calls {
            for tc in tcs {
                let args = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                let fc = FunctionCallPart {
                    name: tc.function.name.clone(),
                    args: Some(args),
                    id: Some(tc.id.clone()),
                };
                // A Gemini 3.x thought signature must be re-attached as the
                // SIBLING of the functionCall part (not inside it) — the API
                // rejects the replay with HTTP 400 otherwise.
                match tc.thought_signature.as_deref().filter(|s| !s.is_empty()) {
                    Some(sig) => {
                        parts.push(Part::function_call_with_signature(fc, sig.to_string()));
                    }
                    None => parts.push(Part::function_call(fc)),
                }
            }
        }
        if msg.role == "tool" {
            let name = msg
                .tool_call_id
                .as_deref()
                .and_then(|id| name_by_id.get(id).copied())
                .unwrap_or("");
            // Gemini requires `response` to be an OBJECT. When the tool
            // result is valid JSON use it directly; otherwise (plain text
            // like bash/grep output) wrap it so the data is never dropped.
            let response = match msg.content.as_deref() {
                Some(c) => match serde_json::from_str::<serde_json::Value>(c) {
                    Ok(v @ serde_json::Value::Object(_)) => v,
                    Ok(v) => serde_json::json!({ "result": v }),
                    Err(_) => serde_json::json!({ "result": c }),
                },
                None => serde_json::Value::Object(Default::default()),
            };
            parts.push(Part::function_response(FunctionResponsePart {
                name: name.to_string(),
                response,
            }));
        }
        if parts.is_empty() {
            continue;
        }

        // Merge with the previous content when it shares the same role —
        // Gemini rejects consecutive same-role contents (e.g. two tool
        // results in a row, or assistant text followed by tool_calls).
        match contents.last_mut() {
            Some(prev) if prev.role.as_deref() == Some(role) => {
                prev.parts.extend(parts);
            }
            _ => contents.push(Content {
                role: Some(role.to_string()),
                parts,
            }),
        }
    }
    contents
}

/// Convert a native Gemini `functionCall` part into a structured
/// [`NativeToolCall`] the harness extractor validates directly — no
/// inline-JSON round trip. The sibling `thoughtSignature` (Gemini 3.x)
/// travels verbatim so it survives validation and can be replayed on the
/// follow-up request.
fn native_tool_call(fc: &FunctionCallPart, signature: &str) -> NativeToolCall {
    let args = fc
        .args
        .clone()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    NativeToolCall {
        id: fc.id.clone().unwrap_or_default(),
        name: fc.name.clone(),
        arguments: args.to_string(),
        thought_signature: signature.to_string(),
    }
}

/// Stream a chat completion with a full messages array, converting the
/// OpenAI-style history to Gemini `contents[]` with native function-calling
/// parts (`functionCall`/`functionResponse`).
pub async fn chat_stream_with_messages(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: &[ChatMessage],
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

    let contents = convert_messages(messages);
    // Inline mode: no native `tools` on the wire (see `prepare_request`).
    let native_tools = params.tool_call_mode == ToolCallMode::Native;
    let request = GenerateContentRequest {
        contents,
        system_instruction: Some(Content {
            role: None,
            parts: vec![Part::text(system)],
        }),
        generation_config: build_generation_config(params),
        tools: if native_tools {
            params.tools.as_ref().map(|t| build_tools(t))
        } else {
            None
        },
        tool_config: if native_tools {
            build_effective_tool_config(params)
        } else {
            None
        },
    };

    let base_url = resolve_base_url(config, params);
    // `alt=sse` is REQUIRED on both v1 and v1beta — see `chat_stream` for
    // why: without it Gemini returns a raw JSON body the SSE parser
    // silently drops (empty stream, no tokens, no error).
    let url = format!("{base_url}/models/{model}:streamGenerateContent?alt=sse");

    let stream = if params.retry_enabled {
        let json_body = serde_json::to_string(&request)?;
        let request_builder = apply_provider_headers(
            shared_client()
                .post(&url)
                .header("Content-Type", "application/json"),
            config,
        )
        .header("x-goog-api-key", api_key);
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
            parse_sse_stream_messages,
        )
    } else {
        let response = send_request_stream(
            config,
            &url,
            &request,
            &gemini_headers(api_key.as_str(), params.session_id.as_deref()),
        )
        .await?;
        parse_sse_stream_messages(response)
    };
    Ok(ChatStream::new(stream, config.family))
}

/// Parse a Gemini `?alt=sse` response into stream chunks with native
/// `functionCall` accumulation (the `stream_chat_with_messages` path — the
/// agent loop). At turn end the accumulated calls are flushed with a
/// `tool_calls` finish reason; calls left over when the stream ends without
/// a stop signal are flushed too, so they are never silently dropped.
fn parse_sse_stream_messages(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    Box::pin(stream! {
            let mut response = response;
            let mut buf = SseBuffer::new();
            // Accumulate complete functionCall parts; Gemini emits them
            // atomically (args included), so a simple collection suffices.
            // The sibling `thoughtSignature` (Gemini 3.x thinking models)
            // rides along so it can be replayed on the follow-up request.
            let mut pending_tool_calls: Vec<(FunctionCallPart, String)> = Vec::new();
            let mut last_raw: Option<String> = None;
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
                    match serde_json::from_str::<GenerateContentResponse>(&data) {
                        Ok(ccr) => {
                            last_raw = Some(data.clone());
                            if let Some(candidate) = ccr.candidates.first() {
                                for part in &candidate.content.parts {
                                    if let Some(text) = part.text.as_deref().filter(|t| !t.is_empty()) {
                                        // `thought: true` parts are the model's
                                        // internal reasoning — surface as
                                        // `reasoning`, never as a response token.
                                        if part.thought {
                                            yield Ok(StreamChunk {
                                                raw: data.clone(),
                                                token: String::new(),
                                                reasoning: text.to_string(),
                                                finish_reason: None,
                                                thinking_blocks: None,
                                                tool_call: None,
                                                reset: false,
                                            });
                                        } else {
                                            yield Ok(StreamChunk {
                                                raw: data.clone(),
                                                token: text.to_string(),
                                                reasoning: String::new(),
                                                finish_reason: None,
                                                thinking_blocks: None,
                                                tool_call: None,
                                                reset: false,
                                            });
                                        }
                                    }
                                    if let Some(fc) = &part.function_call {
                                        pending_tool_calls.push((
                                            fc.clone(),
                                            part.thought_signature.clone().unwrap_or_default(),
                                        ));
                                    }
                                }
                                let finish_reason =
                                    normalize_finish_reason(candidate.finish_reason.clone());
                                let should_stop = finish_reason.is_some();
                                if should_stop && !pending_tool_calls.is_empty() {
                                    let raw = last_raw.take().unwrap_or_default();
                                    for (fc, sig) in pending_tool_calls.drain(..) {
                                        yield Ok(StreamChunk {
                                            raw: raw.clone(),
                                            token: String::new(),
                                            reasoning: String::new(),
                                            finish_reason: Some("tool_calls".to_string()),
                                            thinking_blocks: None,
                                            tool_call: Some(native_tool_call(&fc, &sig)),
                                            reset: false,
                                        });
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
            // Stream ended without a stop signal: flush any accumulated
            // tool calls so they are not silently dropped.
            if !pending_tool_calls.is_empty() {
                let raw = last_raw.take().unwrap_or_default();
                for (fc, sig) in pending_tool_calls.drain(..) {
                    yield Ok(StreamChunk {
                        raw: raw.clone(),
                        token: String::new(),
                        reasoning: String::new(),
                        finish_reason: Some("tool_calls".to_string()),
                        thinking_blocks: None,
                        tool_call: Some(native_tool_call(&fc, &sig)),
                        reset: false,
                    });
                }
            }
    })
}

pub async fn embed(
    config: &ProviderConfig,
    params: &Parameters,
    input: &str,
    service: Option<&str>,
) -> Result<Vec<f32>, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| "gemini-embedding-001".to_string());

    let request = EmbedContentRequest {
        content: Content {
            role: None,
            parts: vec![Part::text(input)],
        },
        output_dimensionality: None,
    };
    let base_url = resolve_base_url(config, params);
    let url = format!("{base_url}/models/{model}:embedContent");

    let response_text = send_request(
        config,
        &url,
        &request,
        &gemini_headers(api_key.as_str(), params.session_id.as_deref()),
    )
    .await?;
    let embed_response: EmbedContentResponse = serde_json::from_str(&response_text)?;
    Ok(embed_response.embedding.values)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModelEntry {
    name: String,
}

#[derive(serde::Deserialize)]
struct GeminiListModelsResponse {
    models: Vec<GeminiModelEntry>,
}

/// Fetch the list of available models from the Gemini provider.
///
/// Sends a GET to `{base_url}/models`, strips the `models/` prefix from
/// each entry's `name` field, and returns a [`LsOutput`].
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

    let base_url = resolve_base_url(config, params);
    let url = format!("{base_url}/models");

    let response_text = send_get_request(
        config,
        &url,
        &gemini_headers(api_key.as_str(), params.session_id.as_deref()),
    )
    .await?;

    let list: GeminiListModelsResponse = serde_json::from_str(&response_text)?;
    let models: Vec<ModelInfo> = list
        .models
        .into_iter()
        .map(|entry| ModelInfo {
            id: entry
                .name
                .strip_prefix("models/")
                .unwrap_or(&entry.name)
                .to_string(),
        })
        .collect();

    Ok(LsOutput::new(response_text, models))
}

#[cfg(test)]
mod tests {
    use super::sanitize_schema;

    #[test]
    fn const_becomes_enum() {
        let schema = serde_json::json!({
            "type": "string",
            "const": "Add"
        });
        assert_eq!(
            sanitize_schema(&schema),
            serde_json::json!({ "type": "string", "enum": ["Add"] })
        );
    }

    #[test]
    fn one_of_merged_with_required_union() {
        let schema = serde_json::json!({
            "type": "object",
            "description": "The mutation action to perform",
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": { "type": "string", "const": "Add" },
                        "group": { "type": "string" }
                    },
                    "required": ["type", "group"]
                },
                {
                    "type": "object",
                    "properties": {
                        "type": { "type": "string", "const": "Remove" },
                        "id": { "type": "string" }
                    },
                    "required": ["type", "id"]
                }
            ]
        });
        let out = sanitize_schema(&schema);
        assert_eq!(out["type"], "object");
        assert!(out.get("oneOf").is_none(), "oneOf must be removed");
        // The merged schema keeps type + group + id, with the discriminator
        // collected as a UNION of every branch's const.
        let props = out["properties"].as_object().unwrap();
        assert_eq!(props.len(), 3, "all branch properties merged");
        let mut enums: Vec<&str> = props["type"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        enums.sort_unstable();
        assert_eq!(enums, vec!["Add", "Remove"]);
        // Required is the union of both branches.
        let required = out["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("type")));
        assert!(required.contains(&serde_json::json!("group")));
        assert!(required.contains(&serde_json::json!("id")));
    }

    #[test]
    fn unsupported_keywords_are_dropped() {
        let schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "^/"
                }
            }
        });
        let out = sanitize_schema(&schema);
        assert!(out.get("additionalProperties").is_none());
        assert!(out.get("$schema").is_none());
        assert_eq!(
            out["properties"]["path"],
            serde_json::json!({ "type": "string" }),
            "minLength/pattern stripped from nested properties"
        );
    }

    #[test]
    fn all_of_merged_like_one_of() {
        let schema = serde_json::json!({
            "type": "object",
            "allOf": [
                {
                    "type": "object",
                    "properties": { "a": { "type": "string" } },
                    "required": ["a"]
                },
                {
                    "type": "object",
                    "properties": { "b": { "type": "integer" } }
                }
            ]
        });
        let out = sanitize_schema(&schema);
        assert!(out.get("allOf").is_none(), "allOf must be removed");
        let props = out["properties"].as_object().unwrap();
        assert_eq!(props.len(), 2, "both allOf branches merged");
        assert_eq!(out["required"], serde_json::json!(["a"]));
    }

    #[test]
    fn non_string_const_is_dropped_not_enumed() {
        // Numeric/boolean consts cannot be expressed as a string enum in
        // Gemini function declarations — they must be dropped, not converted.
        let schema = serde_json::json!({
            "type": "integer",
            "const": 5
        });
        assert_eq!(
            sanitize_schema(&schema),
            serde_json::json!({ "type": "integer" })
        );
    }

    #[test]
    fn arrays_are_recursed() {
        // Tuple-form items: array schemas must be recursed so a const inside
        // an element is still converted.
        let schema = serde_json::json!([
            { "type": "string", "const": "x" },
            { "type": "integer" }
        ]);
        assert_eq!(
            sanitize_schema(&schema),
            serde_json::json!([
                { "type": "string", "enum": ["x"] },
                { "type": "integer" }
            ])
        );
    }

    #[test]
    fn effective_tool_config_forces_auto_with_tools() {
        use crate::connector::params::{Parameters, ToolFunction};
        let p = Parameters {
            tools: Some(vec![crate::connector::ToolDefinition::new(
                ToolFunction::new("glob").with_description("d"),
            )]),
            ..Default::default()
        };
        let cfg = super::build_effective_tool_config(&p).expect("tools present → toolConfig");
        assert_eq!(
            cfg.function_calling_config.mode, "AUTO",
            "tools without an explicit choice must force native function calling"
        );
        assert!(cfg.function_calling_config.allowed_function_names.is_none());
    }

    #[test]
    fn effective_tool_config_absent_without_tools() {
        let p = crate::connector::params::Parameters::default();
        assert!(super::build_effective_tool_config(&p).is_none());
    }

    #[test]
    fn effective_tool_config_respects_explicit_choice() {
        use crate::connector::params::{Parameters, ToolFunction};
        let p = Parameters {
            tools: Some(vec![crate::connector::ToolDefinition::new(
                ToolFunction::new("glob").with_description("d"),
            )]),
            tool_choice: Some(serde_json::json!("ANY")),
            ..Default::default()
        };
        let cfg = super::build_effective_tool_config(&p).expect("explicit choice wins");
        assert_eq!(cfg.function_calling_config.mode, "ANY");
    }

    #[test]
    fn deep_nesting_arrays_and_objects() {
        let schema = serde_json::json!({
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "const": "x" }
                }
            }
        });
        assert_eq!(
            sanitize_schema(&schema),
            serde_json::json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "enum": ["x"] }
                    }
                }
            })
        );
    }
}
