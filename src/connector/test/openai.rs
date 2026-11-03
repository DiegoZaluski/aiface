//! Tests for the OpenAI-native Responses API caller (`/v1/responses`).
//!
//! Covers: text and reasoning-summary streaming, native function-call items,
//! the request body shape (input items, `store: false`, `reasoning`, tools),
//! non-streaming chat, and `response.incomplete` → `finish_reason: "length"`.
use super::super::{Connector, StreamChunk, user_message};
use super::common::{mock_server, openai_connector};
use tokio_stream::StreamExt;

/// A minimal, realistic Responses SSE stream ending in `[DONE]`.
fn text_stream(text: &str) -> String {
    format!(
        "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\",\"status\":\"in_progress\"}}}}\n\n\
         data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}}}\n\n\
         data: {{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{text}\"}}\n\n\
         data: {{\"type\":\"response.output_text.done\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"text\":\"{text}\"}}\n\n\
         data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}}}\n\n\
         data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_1\",\"status\":\"completed\"}}}}\n\n\
         data: [DONE]\n\n"
    )
}

/// Streaming text deltas are concatenated into tokens.
#[tokio::test]
async fn streams_text_deltas() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n\
               data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n\
               data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n\
               data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hello world");
}

/// Reasoning summary deltas stream as `StreamChunk::reasoning` (the "Thought"
/// block), never as text.
#[tokio::test]
async fn streams_reasoning_summary() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"output_index\":0,\"content_index\":0,\"summary_index\":0,\"delta\":\"I need to\"}\n\n\
               data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"rs_1\",\"output_index\":0,\"content_index\":0,\"summary_index\":0,\"delta\":\" think.\"}\n\n\
               data: {\"type\":\"response.output_text.delta\",\"delta\":\"Answer\"}\n\n\
               data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n\
               data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut reasoning = String::new();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        reasoning.push_str(chunk.reasoning());
        text.push_str(chunk.token());
    }
    assert_eq!(reasoning, "I need to think.");
    assert_eq!(text, "Answer");
}

/// A native function-call item surfaces as a `tool_call` chunk with
/// `finish_reason: "tool_calls"` and the `call_id` preserved for the
/// follow-up `function_call_output`.
#[tokio::test]
async fn streams_function_call() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\"}}\n\n\
               data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"location\\\":\\\"Paris\\\"}\"}\n\n\
               data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}\n\n\
               data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}\n\n\
               data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n\
               data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut saw_tool_call = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(tc) = chunk.tool_call() {
            saw_tool_call = true;
            assert_eq!(tc.id, "call_1");
            assert_eq!(tc.name, "get_weather");
            assert_eq!(tc.arguments, "{\"location\":\"Paris\"}");
            assert_eq!(chunk.finish_reason(), Some("tool_calls"));
        }
    }
    assert!(saw_tool_call, "a native function call must be emitted");
}

/// The Responses request body uses `input` items (not `messages`), is
/// stateless (`store: false`), and carries `reasoning` with a summary when an
/// effort is chosen.
#[tokio::test]
async fn request_body_shape_and_reasoning() {
    let (port, captured, _raw, handle) = mock_server(&text_stream("ok"), 200);
    let c = openai_connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("messages").is_none(),
        "no chat-completions messages"
    );
    assert_eq!(json["store"], false, "stateless responses");
    assert_eq!(json["reasoning"]["effort"], "high");
    assert_eq!(json["reasoning"]["summary"], "auto");

    let input = json["input"].as_array().unwrap();
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[0]["content"][0]["text"], "sys");
    assert_eq!(input[1]["role"], "user");
}

/// The prompt-cache controls are only on the wire when set: the affinity
/// key and the extended retention are top-level Responses fields, and a
/// default connector sends neither.
#[tokio::test]
async fn prompt_cache_key_and_retention_only_when_set() {
    let (port, captured, _raw, handle) = mock_server(&text_stream("ok"), 200);
    let c = openai_connector(port)
        .with_prompt_cache_key("sess-123")
        .with_prompt_cache_retention("24h");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["prompt_cache_key"], "sess-123");
    assert_eq!(json["prompt_cache_retention"], "24h");

    // Default: neither field is sent.
    let (port, captured, _raw, handle) = mock_server(&text_stream("ok"), 200);
    let c = openai_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("prompt_cache_key").is_none());
    assert!(json.get("prompt_cache_retention").is_none());
}

/// In inline mode the request must NOT carry native `tools`.
#[tokio::test]
async fn inline_mode_omits_tools() {
    use super::super::{ToolCallMode, ToolDefinition, ToolFunction};

    let tool = ToolDefinition::new(
        ToolFunction::new("read_file")
            .with_description("Read a file")
            .with_parameters(serde_json::json!({"type": "object"})),
    );
    let (port, captured, _raw, handle) = mock_server(&text_stream("ok"), 200);
    let c = openai_connector(port)
        .with_tools(vec![tool])
        .with_tool_call_mode(ToolCallMode::Inline);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("tools").is_none(), "inline mode sends no tools");
    assert!(json.get("tool_choice").is_none());
}

/// A non-streaming Responses response is parsed from `output` items.
#[tokio::test]
async fn non_streaming_chat_parses_output() {
    let body = r#"{"id":"resp_1","output":[{"type":"reasoning","summary":[]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world","annotations":[]}]}]}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let out = openai_connector(port).chat("hi").await.unwrap();
    handle.join().unwrap();
    assert_eq!(out.message(), "Hello world");
    assert_eq!(out.raw(), body);
}

/// `response.incomplete` with `max_output_tokens` maps to
/// `finish_reason: "length"` so the harness continues the truncated turn.
#[tokio::test]
async fn incomplete_maps_to_length() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n\
               data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_1\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n\
               data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port);
    let stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let chunks: Vec<StreamChunk> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(chunks[0].token(), "partial");
    assert!(chunks.iter().any(|c| c.finish_reason() == Some("length")));
}

/// Missing details must not turn an explicitly incomplete response into success.
#[tokio::test]
async fn incomplete_without_details_is_never_normal_completion() {
    let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n\
               data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_1\",\"status\":\"incomplete\"}}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let stream = openai_connector(port).stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(
        chunks
            .iter()
            .any(|c| c.finish_reason() == Some("incomplete"))
    );
    assert!(!chunks.iter().any(|c| c.finish_reason() == Some("stop")));
}

/// A Responses `error` event surfaces as a stream error.
#[tokio::test]
async fn error_event_surfaces() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"error\",\"error\":{\"message\":\"boom\",\"type\":\"invalid_request_error\"}}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port).with_retry(false);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            saw_error = true;
            assert!(e.to_string().contains("boom"), "got: {e}");
            break;
        }
    }
    assert!(saw_error, "an error event must surface as a stream error");
}

/// A `response.failed` event nests the error under `response.error`; the
/// parser must read it from there (not a top-level `error`).
#[tokio::test]
async fn response_failed_event_surfaces_nested_error() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"kaput\"}}}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port).with_retry(false);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            saw_error = true;
            assert!(e.to_string().contains("kaput"), "got: {e}");
            break;
        }
    }
    assert!(
        saw_error,
        "a response.failed event must surface as an error"
    );
}

/// A flat `error` event (message/code at the top level) surfaces its message.
#[tokio::test]
async fn flat_error_event_surfaces_message() {
    let sse = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
               data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"flat boom\"}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = openai_connector(port).with_retry(false);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            saw_error = true;
            assert!(e.to_string().contains("flat boom"), "got: {e}");
            break;
        }
    }
    assert!(saw_error, "a flat error event must surface its message");
}

/// A streaming `response.completed` frame's nested usage is extractable via
/// `Connector::token_usage`.
#[tokio::test]
async fn token_usage_from_completed_event() {
    let raw = r#"{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":100,"output_tokens":50,"input_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":30}}}}"#;
    let c = Connector::new("openai").unwrap();
    let usage = c.token_usage(raw).unwrap();
    assert_eq!(usage.input_tokens, 20);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.cache_read_input_tokens, 80);
    assert_eq!(usage.reasoning_tokens, 30);
    assert_eq!(usage.total_input_tokens(), 100);
}
