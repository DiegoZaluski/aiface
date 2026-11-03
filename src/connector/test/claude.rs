use super::super::{Connector, ToolDefinition, ToolFunction};
use super::common::{ENV_LOCK, EnvGuard, claude_connector, claude_connector_no_key, mock_server};
use tokio_stream::StreamExt;

#[tokio::test]
async fn successful_chat() {
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Hello world"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":20}}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let result = claude_connector(port).chat("hello").await;
    handle.join().unwrap();
    let output = result.unwrap();
    assert_eq!(output.message(), "Hello world");
}

#[tokio::test]
async fn raw_json() {
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn"}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let result = claude_connector(port).chat("hello").await;
    handle.join().unwrap();
    let output = result.unwrap();
    assert_eq!(output.message(), "Hi");
    assert_eq!(output.raw(), body);
}

#[tokio::test]
async fn system_prompt_as_top_level_field() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let result = claude_connector(port)
        .chat_with_system("user text", "system text")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["system"], "system text");
    assert_eq!(json["max_tokens"], 4096);
    let msgs = json["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["content"], "user text");
}

#[tokio::test]
async fn params_serialized() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_model("claude-opus-4-8")
        .with_max_tokens(1024)
        .with_temperature(0.7)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["model"], "claude-opus-4-8");
    assert_eq!(json["max_tokens"], 1024);
    assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
}

#[tokio::test]
async fn with_stop_sequences() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_stop(serde_json::json!(["END", "STOP"]))
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["stop_sequences"], serde_json::json!(["END", "STOP"]));
}

#[tokio::test]
async fn tools_serialized() {
    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({"type":"object"})),
    );
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_tools(vec![tool])
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["tools"][0]["name"], "get_weather");
    assert_eq!(json["tools"][0]["description"], "Get the weather");
    assert_eq!(
        json["tools"][0]["input_schema"],
        serde_json::json!({"type":"object"})
    );
}

#[tokio::test]
async fn streaming_receives_tokens() {
    let sse = "\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n\
data: {\"type\":\"message_stop\"}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = claude_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hello world");
}

#[tokio::test]
async fn streaming_finish_reason() {
    let sse = "\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":15}}\n\n\
data: {\"type\":\"message_stop\"}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = claude_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.token(), "Hello");
    assert_eq!(first.finish_reason(), None);

    let second = stream.next().await.unwrap().unwrap();
    assert_eq!(second.token(), "");
    // Anthropic's raw `end_turn` is canonicalized to the OpenAI-style
    // `stop` (see `normalize_finish_reason`).
    assert_eq!(second.finish_reason(), Some("stop"));
}

#[tokio::test]
async fn raw_last_frame_with_usage() {
    let last_frame = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}"#;
    let sse = format!(
        "\
data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"Hi\"}}}}\n\n\
data: {last_frame}\n\n\
data: {{\"type\":\"message_stop\"}}\n\n"
    );
    let (port, _body, _raw, handle) = mock_server(&sse, 200);
    let c = claude_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    while stream.next().await.is_some() {}
    let raw = stream.raw().await.unwrap();
    assert_eq!(raw, last_frame);
}

#[tokio::test]
async fn http_401() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}"#,
        401,
    );
    let err = claude_connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    let msg = err.to_string();
    assert!(msg.contains("401"), "status code missing from: {msg}");
}

#[tokio::test]
async fn malformed_json() {
    let (port, _body, _raw, handle) = mock_server("not-json", 200);
    let err = claude_connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert!(err.to_string().contains("expected") || err.to_string().contains("invalid"));
}

#[tokio::test]
async fn empty_content() {
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[],"stop_reason":"end_turn"}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let err = claude_connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert_eq!(err.to_string(), "No content in response");
}

#[tokio::test]
async fn http_500() {
    let (port, _body, _raw, handle) = mock_server("Internal Server Error", 500);
    let err = claude_connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert!(err.to_string().contains("500"));
}

#[tokio::test]
async fn network_error() {
    let err = claude_connector(0).chat("hello").await.unwrap_err();
    assert!(
        err.to_string().contains("error")
            || err.to_string().contains("refused")
            || err.to_string().contains("reset"),
        "transport error expected, got: {}",
        err
    );
}

#[tokio::test]
async fn missing_api_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("ANTHROPIC_API_KEY");
    // Isolate from any key stored in the OS keyring under the cosh service.
    let err = Connector::new("claude")
        .unwrap()
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("API key not set for provider: claude"),
        "got: {}",
        err
    );
}

#[tokio::test]
async fn api_key_env_fallback() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("ANTHROPIC_API_KEY", "sk-ant-from-env");
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let result = claude_connector_no_key(port)
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());
}

#[tokio::test]
async fn model_fallback() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port).chat("hello").await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["model"], "claude-sonnet-4-6");
}

#[tokio::test]
async fn streaming_http_error_propagates() {
    let (port, _body, _raw, handle) = mock_server("Internal Server Error", 500);
    // Single-shot: this test asserts the FIRST attempt's error surfaces
    // (retry is covered by connector::test::retry).
    let c = claude_connector(port).with_retry(false);
    let result = c.stream_chat("hi").await;
    handle.join().unwrap();
    match result {
        Err(e) => assert!(e.to_string().contains("500")),
        Ok(_) => panic!("expected error, got stream"),
    }
}

#[tokio::test]
async fn user_id_in_metadata() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_user("user-123")
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["metadata"]["user_id"], "user-123");
}

/// A reasoning effort on a 4.6+ model (the default `claude-sonnet-4-6`)
/// enables ADAPTIVE thinking with `output_config.effort` — `type:"enabled"`
/// is deprecated there (400 on 4.7+).
#[tokio::test]
async fn reasoning_effort_enables_adaptive_thinking_on_46() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_reasoning_effort("high")
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["thinking"]["type"], "adaptive");
    assert_eq!(json["thinking"]["output_config"]["effort"], "high");
    assert!(
        json["thinking"].get("budget_tokens").is_none(),
        "adaptive thinking carries no token budget"
    );
    // Adaptive thinking sizes itself within max_tokens: the plain 4096
    // default would squeeze a high-effort turn (thinking + answer must both
    // fit) — the floor is raised so the effort is actually spendable.
    assert_eq!(json["max_tokens"], 16_384);
}

/// An explicit max_tokens above the adaptive floor is respected verbatim.
#[tokio::test]
async fn adaptive_thinking_respects_explicit_max_tokens() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_reasoning_effort("high")
        .with_max_tokens(32_000)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["max_tokens"], 32_000);
}

/// On Claude 4.5 and earlier the reasoning effort maps onto MANUAL extended
/// thinking (`type: "enabled"` + `budget_tokens`) — the only mode those
/// models support (`type: "adaptive"` returns 400 there).
#[tokio::test]
async fn reasoning_effort_on_old_model_uses_manual_thinking() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_model("claude-sonnet-4-5")
        .with_reasoning_effort("high")
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["thinking"]["type"], "enabled");
    assert_eq!(json["thinking"]["budget_tokens"], 24_576);
}

/// Manual extended thinking requires `max_tokens > budget_tokens` (thinking
/// tokens count toward the turn's output ceiling; the API returns 400
/// otherwise). The floor is raised to `budget + headroom` when the caller did
/// not set a larger max_tokens.
#[tokio::test]
async fn effective_output_reservation_matches_captured_wire_limits() {
    for (model, effort, requested, expected) in [
        ("claude-sonnet-4-5", "high", 2_000, 25_600),
        ("claude-sonnet-4-5", "low", 2_000, 3_072),
        ("claude-sonnet-4-6", "high", 2_000, 16_384),
        ("claude-sonnet-4-6", "high", 32_000, 32_000),
        ("claude-sonnet-4-6", "none", 2_000, 2_000),
    ] {
        let (port, captured, _raw, handle) = mock_server(
            r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
            200,
        );
        let connector = claude_connector(port)
            .with_model(model)
            .with_reasoning_effort(effort)
            .with_max_tokens(requested);
        assert_eq!(connector.effective_max_tokens(), Some(expected));
        connector.chat("hello").await.unwrap();
        handle.join().unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&captured.lock().unwrap().take().unwrap()).unwrap();
        assert_eq!(body["max_tokens"], expected);
    }
}

#[tokio::test]
async fn manual_thinking_raises_max_tokens_above_budget() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_model("claude-sonnet-4-5")
        .with_reasoning_effort("high")
        .with_max_tokens(2000)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    // budget 24576 + 1024 headroom — the explicit 2000 is overridden.
    assert_eq!(json["max_tokens"], 24_576 + 1024);
}

/// An explicit max_tokens that already clears the budget is respected
/// verbatim.
#[tokio::test]
async fn explicit_max_tokens_above_budget_is_respected() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_model("claude-sonnet-4-5")
        .with_reasoning_effort("high")
        .with_max_tokens(30_000)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["max_tokens"], 30_000);
}

/// No effort → no `thinking` block (Claude's default behavior).
#[tokio::test]
async fn request_omits_thinking_without_effort() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port).chat("hello").await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("thinking").is_none(),
        "no effort → no thinking block"
    );
}

/// Inline tool-call mode: no native `tools` (nor `tool_choice`) on the wire
/// — the model writes tool calls as JSON into its text response and the
/// harness parses them. The native and inline delivery paths are mutually
/// exclusive at the request level.
#[tokio::test]
async fn inline_mode_omits_tools_from_request() {
    use super::super::{ToolCallMode, ToolDefinition, ToolFunction};

    let tool = ToolDefinition::new(
        ToolFunction::new("read_file")
            .with_description("Read a file")
            .with_parameters(serde_json::json!({"type": "object"})),
    );
    let sse = "\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"ok\"}}\n\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n\
data: {\"type\":\"message_stop\"}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = claude_connector(port)
        .with_tools(vec![tool])
        .with_tool_call_mode(ToolCallMode::Inline);
    let mut stream = c
        .stream_chat_with_messages("sys", &[super::super::user_message("hi")])
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
}

/// A stream that ends with a tool_use must emit the turn's thinking blocks
/// (summary text + signature, captured via `thinking_delta` and
/// `signature_delta`) as a single chunk BEFORE the tool-call chunks — the
/// harness replays them verbatim on the follow-up request (the Anthropic API
/// validates the signature and rejects missing blocks with 400). The
/// thinking deltas are ALSO streamed as `reasoning` for the TUI.
#[tokio::test]
async fn streaming_thinking_blocks_survive_tool_use_turn() {
    let sse = "\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"Let me reason\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" about it more\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_abc123\"}}\n\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"bash\",\"input\":{}}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":50}}\n\n\
data: {\"type\":\"message_stop\"}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = claude_connector(port);
    let stream = c
        .stream_chat_with_messages("sys", &[super::super::user_message("hi")])
        .await
        .unwrap();
    handle.join().unwrap();

    let chunks: Vec<_> = stream.collect().await;
    let chunks: Vec<_> = chunks.into_iter().map(|c| c.unwrap()).collect();

    // Streaming order: reasoning chunks first (the live TUI feed), then the
    // single verbatim thinking-block chunk, then the tool-call token.
    let thinking_pos = chunks
        .iter()
        .position(|c| c.thinking_blocks().is_some())
        .expect("turn with tool use must carry its thinking blocks");
    let tool_pos = chunks
        .iter()
        .position(|c| c.finish_reason() == Some("tool_calls"))
        .expect("the tool-call token");
    assert!(
        thinking_pos < tool_pos,
        "the thinking blocks must precede the tool-call token"
    );

    let thinking_chunk = &chunks[thinking_pos];
    let blocks = thinking_chunk.thinking_blocks().unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(
        blocks[0].thinking, "Let me reason about it more",
        "thinking_delta deltas must be accumulated into the block"
    );
    assert_eq!(blocks[0].signature, "sig_abc123");
    assert!(thinking_chunk.token().is_empty());
    assert_eq!(thinking_chunk.finish_reason(), None);

    // The tool-use chunk carries the accumulated args as a native call.
    let tool_chunk = &chunks[tool_pos];
    let call = tool_chunk
        .tool_call()
        .expect("native tool call on the chunk");
    assert!(
        tool_chunk.token().is_empty(),
        "no synthetic JSON token anymore"
    );
    assert_eq!(call.name, "bash");
    assert_eq!(call.arguments, r#"{"cmd":"ls"}"#);
    assert_eq!(call.id, "toolu_1");
}

/// The thinking deltas stream as `reasoning` chunks so the TUI shows the
/// model thinking live — but never as visible text tokens.
#[tokio::test]
async fn streaming_thinking_deltas_are_reasoning_not_tokens() {
    let sse = "\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"think\"}}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\" harder\"}}\n\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer\"}}\n\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n\
data: {\"type\":\"message_stop\"}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = claude_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[super::super::user_message("hi")])
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    let mut reasoning = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        tokens.push_str(chunk.token());
        reasoning.push_str(chunk.reasoning());
    }
    assert_eq!(
        tokens, "Answer",
        "thinking must never leak into visible tokens"
    );
    assert_eq!(reasoning, "think harder");
}

/// A previous assistant tool-call turn with thinking blocks must replay them
/// VERBATIM (text + signature) as `thinking` content blocks at the start of
/// the assistant message — the API rejects modified or missing blocks with
/// 400.
#[tokio::test]
async fn assistant_tool_turn_replays_thinking_blocks_verbatim() {
    use super::super::{
        ClaudeThinkingBlock, ToolCallFunctionMsg, ToolCallMsg, assistant_tool_call_message,
    };
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );

    let mut call = assistant_tool_call_message(vec![ToolCallMsg {
        id: "toolu_1".to_string(),
        kind: "function".to_string(),
        function: ToolCallFunctionMsg {
            name: "bash".to_string(),
            arguments: r#"{"cmd":"ls"}"#.to_string(),
        },
        thought_signature: None,
    }]);
    call.thinking_blocks = Some(vec![ClaudeThinkingBlock {
        thinking: "Let me reason about it more".to_string(),
        signature: "sig_abc123".to_string(),
    }]);
    let messages = vec![super::super::user_message("hi"), call];

    let mut stream = claude_connector(port)
        .stream_chat_with_messages("sys", &messages)
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let assistant = json["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .expect("assistant message");
    let content = assistant["content"].as_array().unwrap();
    assert_eq!(content.len(), 2, "thinking block + tool_use");
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me reason about it more");
    assert_eq!(content[0]["signature"], "sig_abc123");
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["id"], "toolu_1");
}

/// Assistant turns WITHOUT thinking blocks must not carry a `thinking`
/// content block — the wire format stays clean for non-reasoning turns.
#[tokio::test]
async fn assistant_tool_turn_without_blocks_has_no_thinking_content() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let messages = vec![
        super::super::user_message("hi"),
        super::super::assistant_tool_call_message(vec![super::super::ToolCallMsg {
            id: "toolu_1".to_string(),
            kind: "function".to_string(),
            function: super::super::ToolCallFunctionMsg {
                name: "bash".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]),
    ];
    let mut stream = claude_connector(port)
        .stream_chat_with_messages("sys", &messages)
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let content = json["messages"][1]["content"].as_array().unwrap();
    assert_eq!(content.len(), 1, "only the tool_use block");
    assert_eq!(content[0]["type"], "tool_use");
}

#[tokio::test]
async fn prompt_cache_control_sent_by_default() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":10,"output_tokens":2}}"#,
        200,
    );
    let _ = claude_connector(port).chat("hello").await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["cache_control"],
        serde_json::json!({"type": "ephemeral"})
    );
}

#[tokio::test]
async fn prompt_cache_disabled_omits_field() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_prompt_cache(false)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("cache_control").is_none());
}

#[tokio::test]
async fn prompt_cache_ttl_1h_serialized() {
    let (port, captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_prompt_cache_ttl_1h(true)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["cache_control"],
        serde_json::json!({"type": "ephemeral", "ttl": "1h"})
    );
}

#[tokio::test]
async fn token_usage_reads_cache_fields() {
    let (port, _captured, _raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":50,"output_tokens":10,"cache_read_input_tokens":1800,"cache_creation_input_tokens":248}}"#,
        200,
    );
    let connector = claude_connector(port);
    let out = connector.chat("hello").await.unwrap();
    handle.join().unwrap();

    assert_eq!(
        connector.token_usage(out.raw()),
        Some(super::super::TokenUsage {
            input_tokens: 50,
            output_tokens: 10,
            cache_creation_input_tokens: 248,
            cache_read_input_tokens: 1800,
            reasoning_tokens: 0,
            reported_cost: None,
            reported_cost_credits: None,
        })
    );
    // The extractor also works standalone on a minimal usage object.
    assert!(
        connector
            .token_usage(r#"{"usage":{"input_tokens":1}}"#)
            .is_some()
    );
}

#[tokio::test]
async fn session_headers_sent_when_configured() {
    let (port, _body, raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port)
        .with_session_id("sess-abc123")
        .chat("hello")
        .await;
    handle.join().unwrap();

    let raw = raw.lock().unwrap().take().unwrap();
    assert!(raw.contains("x-session-id: sess-abc123"), "raw: {raw}");
    assert!(
        raw.contains("x-session-affinity: sess-abc123"),
        "raw: {raw}"
    );
}

#[tokio::test]
async fn session_headers_absent_by_default() {
    let (port, _body, raw, handle) = mock_server(
        r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}"#,
        200,
    );
    let _ = claude_connector(port).chat("hello").await;
    handle.join().unwrap();

    let raw = raw.lock().unwrap().take().unwrap();
    assert!(!raw.contains("x-session-id"), "raw: {raw}");
}

#[test]
fn is_local_parses_host_component_only() {
    let local = |base: &str| {
        crate::connector::Connector::new("claude")
            .unwrap()
            .with_base_url(base)
            .is_local()
    };
    // loopback hosts
    assert!(local("http://localhost:11434"));
    assert!(local("http://127.0.0.1:8080/v1"));
    assert!(local("http://127.0.0.2:9")); // whole 127.0.0.0/8
    assert!(local("http://[::1]:9000"));
    assert!(local("http://my.localhost"));
    // remote hosts, even when the URL MENTIONS a loopback address
    assert!(!local("https://api.anthropic.com"));
    assert!(!local("https://example.com/v1?mirror=127.0.0.1"));
    assert!(!local("http://localhost.evil.com")); // host is localhost.evil.com
    assert!(!local("not a url at all"));
}
