//! Tests for `Connector::stream_chat()` and `Connector::stream_chat_with_system()`.
//!
//! Covers: receiving all SSE chunks, system prompt inclusion in the request body,
//! HTTP error propagation before stream start, and stream termination without
//! a `[DONE]` signal.
use super::super::{Connector, StreamChunk, user_message};
use super::common::{connector, deepseek_connector, kimi_connector, mock_server};
use tokio_stream::StreamExt;

/// Ensures all SSE data chunks are concatenated correctly.
#[tokio::test]
async fn receives_all_chunks() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hello world");

    let raw = stream.raw().await.unwrap();
    assert!(
        raw.contains("world"),
        "last frame should contain 'world', got: {raw}"
    );
}

/// Ensures `.finish_reason()` returns `Some("stop")` on the last content chunk.
#[tokio::test]
async fn finish_reason_on_last_chunk() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token(), "Hello");
    assert_eq!(first.finish_reason(), None);

    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.token(), "");
    assert_eq!(second.finish_reason(), Some("stop"));
}

/// Ensures `.raw()` on a finished stream returns the last SSE frame with metadata.
#[tokio::test]
async fn raw_last_frame_with_usage() {
    let last_frame = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
    let sse = format!(
        "\
data: {{\"choices\":[{{\"delta\":{{\"content\":\"Hello\"}}}}]}}\n\n\
data: {last_frame}\n\n\
data: [DONE]\n\n"
    );
    let (port, _body, _raw, handle) = mock_server(&sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    while stream.next().await.is_some() {}
    let raw = stream.raw().await.unwrap();
    assert_eq!(raw, last_frame);
}

/// Ensures the system prompt is included in the request body when using
/// `stream_chat_with_system()`.
#[tokio::test]
async fn with_system_prompt_in_body() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c
        .stream_chat_with_system("user text", "system text")
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let msgs = json["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "system text");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "user text");
}

/// Ensures an HTTP error on stream start propagates as an `Err` (not a stream).
#[tokio::test]
async fn http_error_propagates() {
    let (port, _body, _raw, handle) = mock_server("Internal Server Error", 500);
    // Single-shot: this test asserts the FIRST attempt's error surfaces
    // (retry is covered by connector::test::retry).
    let c = connector(port).with_retry(false);
    let result = c.stream_chat("hi").await;
    handle.join().unwrap();
    match result {
        Err(e) => assert!(e.to_string().contains("500")),
        Ok(_) => panic!("expected error, got stream"),
    }
}

/// Ensures a stream that ends without a `[DONE]` signal produces an error.
#[tokio::test]
async fn terminated_without_done() {
    let (port, _body, _raw, handle) = mock_server(
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        200,
    );
    // Single-shot: a truncated stream must surface `StreamTerminated` from
    // the first attempt (the retry path would re-request and mask it).
    let c = connector(port).with_retry(false);
    let stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let results: Vec<Result<StreamChunk, _>> = stream.collect().await;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].as_ref().unwrap().token(), "partial");
    assert!(
        results[1]
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("Stream terminated")
    );
}

// The LLM summarizer must NEVER see the agent loop's tool definitions: a
// connector with tools configured that calls `stream_chat_with_system_no_tools`
// sends a request WITHOUT the `tools`/`tool_choice` keys — so the model
// answers with prose instead of a tool call.
#[tokio::test]
async fn stream_with_system_no_tools_omits_tools_from_request() {
    use super::super::ToolDefinition;
    use super::super::ToolFunction;

    let tool = ToolDefinition::new(
        ToolFunction::new("read_file")
            .with_description("Read a file")
            .with_parameters(serde_json::json!({"type":"object"})),
    );
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"summary\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port)
        .with_tools(vec![tool])
        .with_tool_choice(serde_json::json!("auto"));
    let mut stream = c
        .stream_chat_with_system_no_tools("summarize", "You are a summarizer")
        .await
        .unwrap();
    handle.join().unwrap();

    // The request body must carry the system prompt but NO tools.
    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let msgs = json["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "You are a summarizer");
    assert!(
        json.get("tools").is_none(),
        "the summarizer request must not expose tool schemas: {json}"
    );
    assert!(json.get("tool_choice").is_none());

    // The stream still delivers the model's prose.
    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "summary");
}

/// A reasoning effort is sent as the OpenAI top-level `reasoning_effort`.
#[tokio::test]
async fn reasoning_effort_sets_openai_field() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port).with_reasoning_effort("low");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["reasoning_effort"], "low",
        "effort must be sent top-level, got: {body}"
    );
}

/// DeepSeek's thinking mode requires the explicit `thinking` toggle next to
/// `reasoning_effort` — several OpenAI-compatible DeepSeek deployments
/// ignore `reasoning_effort` alone and never enter thinking mode.
#[tokio::test]
async fn deepseek_effort_sends_thinking_toggle() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = deepseek_connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["reasoning_effort"], "high");
    assert_eq!(
        json["thinking"]["type"], "enabled",
        "deepseek must carry the thinking toggle, got: {body}"
    );
}

/// No effort → no toggle either (the toggle only accompanies a reasoning
/// effort).
#[tokio::test]
async fn deepseek_omits_toggle_without_effort() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = deepseek_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("thinking").is_none(), "no effort → no toggle");
}

/// z.ai (GLM hybrid-thinking) uses the same `thinking` wire shape as DeepSeek
/// and likewise ignores `reasoning_effort` without it.
#[tokio::test]
async fn zai_effort_sends_thinking_toggle() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = Connector::new("zai")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-zai-test")
        .with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["thinking"]["type"], "enabled",
        "zai must carry the thinking toggle, got: {body}"
    );
    assert!(json.get("chat_template_kwargs").is_none());
}

/// NVIDIA NIM's hybrid-thinking models (DeepSeek/Qwen on
/// integrate.api.nvidia.com) ignore `reasoning_effort` unless thinking mode
/// is opted in via `chat_template_kwargs`.
#[tokio::test]
async fn nvidia_effort_sends_chat_template_thinking() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = Connector::new("nvidia")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("nvapi-test")
        .with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["chat_template_kwargs"]["thinking"], true,
        "nvidia must opt into NIM thinking mode, got: {body}"
    );
    assert!(
        json.get("thinking").is_none(),
        "nvidia must not get the deepseek toggle"
    );
}

/// The NIM opt-in must never leak to the deepseek provider.
#[tokio::test]
async fn deepseek_omits_chat_template_kwargs() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = deepseek_connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("chat_template_kwargs").is_none());
}

/// No effort → no NIM opt-in either.
#[tokio::test]
async fn nvidia_omits_opt_in_without_effort() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = Connector::new("nvidia")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("nvapi-test");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("chat_template_kwargs").is_none(),
        "no effort → no opt-in"
    );
}

/// Other OpenAI-compatible providers must NEVER receive the DeepSeek-only
/// `thinking` toggle — OpenAI rejects unknown top-level fields with 400.
#[tokio::test]
async fn non_deepseek_omits_thinking_toggle() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("thinking").is_none(), "non-deepseek: no toggle");
}

// ── Kimi (Moonshot) ──────────────────────────────────────────────────────
// Docs: platform.kimi.ai/docs/api/overview (base URL + MOONSHOT_API_KEY
// Bearer auth), /docs/guide/use-reasoning-effort (K3 top-level
// `reasoning_effort`, no `thinking` toggle), /docs/guide/context-caching
// (`prompt_cache_options`, implicit cache writes, per-choice usage).

/// Kimi K3 configures reasoning with the OpenAI-style top-level
/// `reasoning_effort` (`low`/`high`/`max`) — it must NOT receive the
/// DeepSeek-style `thinking` toggle (the docs warn against sending
/// `thinking` to K3).
#[tokio::test]
async fn kimi_effort_uses_top_level_field_without_toggle() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = kimi_connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["reasoning_effort"], "high",
        "kimi sends effort top-level, got: {body}"
    );
    assert!(
        json.get("thinking").is_none(),
        "kimi must NOT get the deepseek toggle: {body}"
    );
}

/// Kimi's prompt-cache policy rides `prompt_cache_options` on EVERY kimi
/// request (implicit mode, cheap `5m` TTL): automatic prefix caching bills
/// cache hits at 1/10 of the input price, and omitting the field leaves the
/// same behavior — but the explicit field pins the intent and documents the
/// connector's economy choice. Other providers must never see it.
#[tokio::test]
async fn kimi_sends_implicit_prompt_cache_options() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = kimi_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["prompt_cache_options"],
        serde_json::json!({"mode": "implicit", "ttl": "5m"}),
        "kimi must pin implicit 5m caching, got: {body}"
    );
}

/// The Kimi-only `prompt_cache_options` must never leak to any other
/// OpenAI-compatible provider (unknown top-level field → 400).
#[tokio::test]
async fn non_kimi_omits_prompt_cache_options() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    for (name, make) in [
        (
            "groq",
            Box::new(|p: u16| connector(p)) as Box<dyn Fn(u16) -> Connector>,
        ),
        ("deepseek", Box::new(deepseek_connector)),
    ] {
        let (port, captured, _raw, handle) = mock_server(sse, 200);
        let c = make(port);
        let mut stream = c
            .stream_chat_with_messages("sys", &[user_message("hi")])
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        handle.join().unwrap();

        let body = captured.lock().unwrap().take().unwrap();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            json.get("prompt_cache_options").is_none(),
            "{name} must not receive kimi's prompt_cache_options: {json}"
        );
    }
}

/// Inline tool-call mode: the request must NOT carry the native `tools`
/// array (nor `tool_choice`) — the model writes tool calls as JSON into its
/// text response and the harness parses them. Without `tools` on the wire
/// the API can never produce structured tool calls, so the native and
/// inline delivery paths are mutually exclusive at the request level.
#[tokio::test]
async fn inline_mode_omits_tools_from_request() {
    use super::super::{ToolCallMode, ToolDefinition, ToolFunction};

    let tool = ToolDefinition::new(
        ToolFunction::new("read_file")
            .with_description("Read a file")
            .with_parameters(serde_json::json!({"type": "object"})),
    );
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port)
        .with_tools(vec![tool])
        .with_tool_choice(serde_json::json!("auto"))
        .with_tool_call_mode(ToolCallMode::Inline);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("tools").is_none(),
        "inline mode must not send native tools: {json}"
    );
    assert!(
        json.get("tool_choice").is_none(),
        "inline mode must not send tool_choice: {json}"
    );

    // The same connector in the default Native mode DOES send them.
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port)
        .with_tools(vec![ToolDefinition::new(ToolFunction::new("read_file"))])
        .with_tool_call_mode(ToolCallMode::Native);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();
    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("tools").is_some(),
        "native mode sends the tools array: {json}"
    );
}

/// No effort → no `reasoning_effort` field in the payload.
#[tokio::test]
async fn request_omits_reasoning_effort_without_setting() {
    let sse = "\
data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n\
data: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("reasoning_effort").is_none(),
        "no effort → no reasoning_effort field"
    );
}

/// Streaming requests must ask the server for token accounting
/// (`stream_options: {"include_usage": true}`); non-streaming requests must
/// NOT carry the field (strict servers reject it there).
#[tokio::test]
async fn stream_request_includes_usage_options() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    while stream.next().await.is_some() {}

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["stream"], true);
    assert_eq!(
        json["stream_options"],
        serde_json::json!({"include_usage": true})
    );
}

#[tokio::test]
async fn non_stream_request_omits_stream_options() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#,
        200,
    );
    let _ = connector(port).chat("hi").await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("stream").is_none());
    assert!(json.get("stream_options").is_none());
}

/// Real-world include_usage wire format: the usage arrives in a SEPARATE
/// frame AFTER finish_reason, with empty `choices`, right before [DONE].
/// `ChatStream::raw()` must surface THAT frame (it carries usage).
#[tokio::test]
async fn usage_frame_after_finish_reason_is_last_raw() {
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"content":"ok"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        r#"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":8}}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    while let Some(chunk) = stream.next().await {
        chunk.unwrap();
    }

    let raw = stream.raw().await.unwrap();
    assert!(
        raw.contains("prompt_tokens"),
        "last frame should be the usage-only frame, got: {raw}"
    );
    assert_eq!(c.token_usage(raw).unwrap().cache_read_input_tokens, 8);
}

/// A stream that ends (server closes) right after finish_reason — without a
/// usage frame and without [DONE] — must still complete cleanly.
#[tokio::test]
async fn stream_end_without_done_after_finish_is_ok() {
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    while let Some(chunk) = stream.next().await {
        chunk.unwrap_or_else(|e| panic!("no error expected after finish_reason: {e}"));
    }
}

/// Session affinity headers must ALSO ride the retry-enabled streaming path
/// (the production default), not just the non-retry branch.
#[tokio::test]
async fn session_headers_on_streaming_path() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n";
    let (port, _body, raw, handle) = mock_server(sse, 200);
    let c = connector(port).with_session_id("sess-stream-1");
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    while stream.next().await.is_some() {}

    let raw = raw.lock().unwrap().take().unwrap();
    assert!(raw.contains("x-session-id: sess-stream-1"), "raw: {raw}");
    assert!(
        raw.contains("x-session-affinity: sess-stream-1"),
        "raw: {raw}"
    );
}
