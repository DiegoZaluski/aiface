//! Tests for the Gemini provider, focused on the native tool-calling path
//! (`stream_chat_with_messages`) and the v1/v1beta API-version handling.
use super::super::{
    ToolCallFunctionMsg, ToolCallMsg, ToolDefinition, ToolFunction, assistant_tool_call_message,
    tool_result_message, user_message,
};
use super::common::{gemini_connector, mock_server};
use tokio_stream::StreamExt;

/// The internal API_VERSION switch must always resolve to a real Gemini
/// API surface. It is asserted loosely (rather than pinned to "v1") so a
/// developer flipping the switch to "v1beta" to test experimental features
/// does not break the test suite.
#[test]
fn provider_api_version_is_valid() {
    let version = crate::connector::gemini::API_VERSION;
    assert!(
        version == "v1" || version == "v1beta",
        "internal API_VERSION switch must be one of v1/v1beta, got {version}"
    );
}

/// The REAL Gemini API separates SSE events with CRLF CRLF ("\r\n\r\n")
/// — NOT the LF LF ("\n\n") the old mock used. This body was captured
/// verbatim from a live `streamGenerateContent?alt=sse` call and is the
/// regression guard for the silent-failure bug: a parser that only split
/// on "\n\n" buffered the whole body and emitted zero frames (no tokens,
/// no error — the loop "starts and stops with nothing on screen").
#[tokio::test]
async fn stream_parses_real_gemini_crlf_wire_format() {
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"index\":0}],",
        "\"usageMetadata\":{\"promptTokenCount\":2}}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" there\"}],\"role\":\"model\"},\"index\":0}],",
        "\"usageMetadata\":{\"promptTokenCount\":2}}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\",\"thoughtSignature\":\"abc\"}],\"role\":\"model\"},",
        "\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":2}}\r\n\r\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    let mut last_fr = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        tokens.push_str(chunk.token());
        if let Some(fr) = chunk.finish_reason() {
            last_fr = Some(fr.to_string());
        }
    }
    assert_eq!(tokens, "Hi there");
    // Gemini's raw `STOP` is canonicalized to the OpenAI-style `stop`
    // (see `normalize_finish_reason`).
    assert_eq!(last_fr.as_deref(), Some("stop"));
}

/// A stream that ends WITHOUT a trailing blank line after the final
/// `data:` frame must still deliver that frame — `SseBuffer::flush()`
/// recovers it at EOF (the SSE spec dispatches pending data on end of
/// stream). Guards the consumer-level wiring of the flush path in the
/// Gemini streaming loops: the final token and finish chunk must survive
/// even when the server closes the connection right after the last
/// `data:` line.
#[tokio::test]
async fn stream_delivers_final_frame_without_trailing_separator() {
    // NOTE: the last frame has NO trailing "\r\n\r\n" — the connection
    // closes right after the final `data:` line.
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" there\"}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}]}",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    let mut last_fr = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        tokens.push_str(chunk.token());
        if let Some(fr) = chunk.finish_reason() {
            last_fr = Some(fr.to_string());
        }
    }
    assert_eq!(tokens, "Hi there");
    assert_eq!(last_fr.as_deref(), Some("stop"));
}

/// A thinking model streams `thought: true` parts (internal reasoning)
/// BEFORE the final text answer. Those parts must surface as `reasoning`
/// (for the TUI's thinking view) — NEVER as response tokens, which would
/// leak the model's chain-of-thought into the visible answer.
#[tokio::test]
async fn streaming_thought_parts_are_reasoning_not_tokens() {
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Let me reason step by step\",\"thought\":true}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"The answer is 42\"}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}]}\r\n\r\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
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
        tokens, "The answer is 42",
        "thought text must never leak into the response tokens"
    );
    assert_eq!(
        reasoning, "Let me reason step by step",
        "thought text must stream as reasoning for the TUI"
    );
}

/// The same thought/token separation must hold on the prompt-based
/// `stream_chat` path (used by `stream_chat_with_system` and the
/// summarizer's no-tools call).
#[tokio::test]
async fn stream_chat_separates_thought_from_token() {
    let sse = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"think\",\"thought\":true}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"out loud\"}],\"role\":\"model\"},\"index\":0}]}\r\n\r\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\",\"index\":0}]}\r\n\r\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    let mut reasoning = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        tokens.push_str(chunk.token());
        reasoning.push_str(chunk.reasoning());
    }
    assert_eq!(tokens, "out loud");
    assert_eq!(reasoning, "think");
}

/// Non-streaming `chat` must skip `thought` parts too: the first part of a
/// thinking response is the reasoning summary, not the answer — returning it
/// as the message would hand the user the chain-of-thought.
#[tokio::test]
async fn chat_skips_thought_parts_in_response_text() {
    let body = concat!(
        "{\"candidates\":[{\"content\":{\"parts\":[",
        "{\"text\":\"internal reasoning\",\"thought\":true},",
        "{\"text\":\"real answer\"}],\"role\":\"model\"},\"index\":0}],",
        "\"finishReason\":\"STOP\"}",
    );
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let c = gemini_connector(port);
    let out = c.chat("hi").await.unwrap();
    handle.join().unwrap();
    assert_eq!(
        out.message, "real answer",
        "the visible answer must exclude the reasoning summary"
    );
}

/// A plain-text tool result (not valid JSON) must still be preserved —
/// wrapped as `{"result": ...}` — never dropped to an empty object.
#[tokio::test]
async fn tool_result_plain_text_is_wrapped_not_dropped() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);

    let messages = vec![
        user_message("run bash"),
        assistant_tool_call_message(vec![ToolCallMsg {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunctionMsg {
                name: "bash".to_string(),
                arguments: "{}".to_string(),
            },
            thought_signature: None,
        }]),
        // Plain-text output, e.g. bash stdout — NOT valid JSON.
        tool_result_message("call_1", "hello from bash\nline two"),
    ];
    let mut stream = c.stream_chat_with_messages("sys", &messages).await.unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let response = &json["contents"][2]["parts"][0]["functionResponse"]["response"];
    assert_eq!(
        response,
        &serde_json::json!({"result": "hello from bash\nline two"}),
        "plain-text tool results must be wrapped, not dropped"
    );
}

/// `stream_chat_with_messages` must build a Gemini `contents[]` request from
/// the OpenAI-style message array: system travels in `systemInstruction`,
/// assistant tool_calls become `functionCall` parts, tool results become
/// `functionResponse` parts in a user content.
#[tokio::test]
async fn stream_with_messages_builds_gemini_contents() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})),
    );
    let c = gemini_connector(port).with_tools(vec![tool]);

    let messages = vec![
        user_message("What's the weather in SP?"),
        assistant_tool_call_message(vec![ToolCallMsg {
            id: "call_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunctionMsg {
                name: "get_weather".to_string(),
                arguments: r#"{"city":"SP"}"#.to_string(),
            },
            thought_signature: None,
        }]),
        tool_result_message("call_1", r#"{"temp":22}"#),
    ];
    let mut stream = c
        .stream_chat_with_messages("You are a helper", &messages)
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();

    // System prompt travels at the top level, not as a message role.
    assert_eq!(
        json["systemInstruction"]["parts"][0]["text"],
        "You are a helper"
    );

    let contents = json["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3, "user / model / user");
    assert_eq!(contents[0]["role"], "user");
    assert_eq!(contents[0]["parts"][0]["text"], "What's the weather in SP?");
    assert_eq!(contents[1]["role"], "model");
    assert_eq!(
        contents[1]["parts"][0]["functionCall"]["name"],
        "get_weather"
    );
    assert_eq!(
        contents[1]["parts"][0]["functionCall"]["args"],
        serde_json::json!({"city":"SP"})
    );
    // No thought signature was attached to the call — none must be emitted.
    assert!(
        contents[1]["parts"][0].get("thoughtSignature").is_none(),
        "calls without a signature must not emit thoughtSignature"
    );
    assert_eq!(contents[2]["role"], "user");
    assert_eq!(
        contents[2]["parts"][0]["functionResponse"]["name"],
        "get_weather"
    );
    assert_eq!(
        contents[2]["parts"][0]["functionResponse"]["response"],
        serde_json::json!({"temp":22})
    );
    // Tools are declared via functionDeclarations.
    assert_eq!(
        json["tools"][0]["functionDeclarations"][0]["name"],
        "get_weather"
    );
}

/// A Gemini 3.x thought signature attached to a native `functionCall` must
/// be replayed as a SIBLING of the `functionCall` part (NOT inside it — the
/// API rejects `thoughtSignature` inside `functionCall` with 400, verified
/// empirically). Regression guard for the "Function call is missing a
/// thought_signature" HTTP 400.
#[tokio::test]
async fn stream_with_messages_replays_thought_signature_sibling() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);

    let messages = vec![
        user_message("read the file"),
        assistant_tool_call_message(vec![ToolCallMsg {
            id: "fc_1".to_string(),
            kind: "function".to_string(),
            function: ToolCallFunctionMsg {
                name: "fs_read".to_string(),
                arguments: r#"{"targets":[{"path":"TODO.md"}]}"#.to_string(),
            },
            thought_signature: Some("sig_abc123".to_string()),
        }]),
        tool_result_message("fc_1", r#"{"result":"content"}"#),
    ];
    let mut stream = c.stream_chat_with_messages("sys", &messages).await.unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let part = &json["contents"][1]["parts"][0];
    assert_eq!(part["functionCall"]["name"], "fs_read");
    assert_eq!(
        part["thoughtSignature"], "sig_abc123",
        "thoughtSignature must be replayed as a SIBLING of functionCall"
    );
    // And it must NOT be inside the functionCall object.
    assert!(
        part["functionCall"].get("thoughtSignature").is_none(),
        "thoughtSignature must not be nested inside functionCall"
    );
}

/// A `functionCall` part arriving with a sibling `thoughtSignature` in the
/// STREAM must surface it on the structured tool call, so the harness
/// extractor can carry it into the next request's history.
#[tokio::test]
async fn stream_with_messages_surfaces_thought_signature_in_token() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"fs_read\",\"args\":{\"targets\":[{\"path\":\"TODO.md\"}]},\"id\":\"fc_1\"},\"thoughtSignature\":\"sig_xyz\"}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let stream = c
        .stream_chat_with_messages("sys", &[user_message("read it")])
        .await
        .unwrap();
    handle.join().unwrap();

    let chunks: Vec<_> = stream.collect().await;
    assert_eq!(chunks.len(), 1, "only the native tool-call chunk");
    let chunk = chunks[0].as_ref().unwrap();
    let call = chunk.tool_call().expect("native tool call on the chunk");
    assert!(chunk.token().is_empty(), "no synthetic JSON token anymore");
    assert_eq!(call.name, "fs_read");
    assert_eq!(call.id, "fc_1");
    assert_eq!(
        call.thought_signature, "sig_xyz",
        "the streamed thought signature must survive on the native call"
    );
    assert_eq!(
        call.arguments, r#"{"targets":[{"path":"TODO.md"}]}"#,
        "the provider's args JSON travels verbatim"
    );
}

/// Consecutive messages that map to the same Gemini role must be merged
/// into a single content (Gemini rejects repeated roles).
#[tokio::test]
async fn stream_with_messages_merges_consecutive_same_role() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);

    // Two tool results in a row (parallel calls) both map to "user".
    let messages = vec![
        user_message("run both"),
        assistant_tool_call_message(vec![
            ToolCallMsg {
                id: "call_1".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunctionMsg {
                    name: "tool_a".to_string(),
                    arguments: "{}".to_string(),
                },
                thought_signature: None,
            },
            ToolCallMsg {
                id: "call_2".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunctionMsg {
                    name: "tool_b".to_string(),
                    arguments: "{}".to_string(),
                },
                thought_signature: None,
            },
        ]),
        tool_result_message("call_1", r#"{"a":1}"#),
        tool_result_message("call_2", r#"{"b":2}"#),
    ];
    let mut stream = c.stream_chat_with_messages("sys", &messages).await.unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let contents = json["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3, "user / model / merged user");
    assert_eq!(contents[2]["role"], "user");
    let parts = contents[2]["parts"].as_array().unwrap();
    assert_eq!(
        parts.len(),
        2,
        "both functionResponses merged into one content"
    );
    assert_eq!(parts[0]["functionResponse"]["name"], "tool_a");
    assert_eq!(parts[1]["functionResponse"]["name"], "tool_b");
}

/// Streaming must emit text tokens from SSE frames.
#[tokio::test]
async fn stream_with_messages_emits_text_tokens() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hello world");
}

/// A `functionCall` part in the stream must be flushed as a structured
/// native tool call (the format the harness extractor validates directly),
/// carrying the call id for round-tripping.
#[tokio::test]
async fn stream_with_messages_flushes_tool_calls() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"SP\"},\"id\":\"fc_1\"}}]}}]}\n\n\
data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let stream = c
        .stream_chat_with_messages("sys", &[user_message("weather?")])
        .await
        .unwrap();
    handle.join().unwrap();

    let chunks: Vec<_> = stream.collect().await;
    assert_eq!(chunks.len(), 1, "only the native tool-call chunk");
    let chunk = chunks[0].as_ref().unwrap();
    assert_eq!(chunk.finish_reason(), Some("tool_calls"));
    let call = chunk.tool_call().expect("native tool call on the chunk");
    assert!(chunk.token().is_empty(), "no synthetic JSON token anymore");
    assert_eq!(call.name, "get_weather");
    assert_eq!(call.arguments, r#"{"city":"SP"}"#);
    assert_eq!(call.id, "fc_1");
    assert!(
        call.thought_signature.is_empty(),
        "a call without a signature must not carry it"
    );
}

/// Non-streaming chat on v1 must still work (regression guard for the
/// base-url rewrite).
#[tokio::test]
async fn chat_still_works() {
    let body = r#"{"candidates":[{"content":{"parts":[{"text":"Hello world"}]}}]}"#;
    let (port, _body, _raw, handle) = mock_server(body, 200);
    let result = gemini_connector(port).chat("hello").await;
    handle.join().unwrap();
    assert_eq!(result.unwrap().message(), "Hello world");
}

/// Gemini's streaming endpoint only emits SSE `data:` frames when the
/// request carries `?alt=sse` — without it the API returns a raw JSON body
/// the SSE parser silently drops (empty stream, no tokens, no error). This
/// guards both streaming entry points against regressing to that silent
/// failure.
#[tokio::test]
async fn stream_requests_use_alt_sse() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, _body, raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);

    // stream_chat (chat_stream)
    let mut stream = c.stream_chat("hi").await.unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();
    let req = raw.lock().unwrap().take().unwrap();
    assert!(
        req.contains("streamGenerateContent?alt=sse"),
        "chat_stream request must ask for SSE: {req}"
    );

    // stream_chat_with_messages (agent-loop path)
    let (port, _body, raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();
    let req = raw.lock().unwrap().take().unwrap();
    assert!(
        req.contains("streamGenerateContent?alt=sse"),
        "chat_stream_with_messages request must ask for SSE: {req}"
    );
}

/// With tools registered, the request MUST carry `toolConfig` with mode
/// "AUTO". Without it, Gemini 3.x models obey the prompt's inline-JSON
/// instruction (`TOOL_FORMAT`) and emit tool calls as raw TEXT — the JSON
/// split across frames, leaking stray fragments (a lone `}`) to the user
/// while the extractor still dispatches the call. Regression guard for the
/// "called a tool without being asked + garbage output" bug.
#[tokio::test]
async fn request_forces_native_function_calling_config() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})),
    );
    let c = gemini_connector(port).with_tools(vec![tool]);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["toolConfig"]["functionCallingConfig"]["mode"], "AUTO",
        "tools present must force native function calling, got: {body}"
    );
}

/// Inline tool-call mode: no `tools` (nor `toolConfig`) on the wire — the
/// model writes tool calls as JSON into its text response and the harness
/// parses them. Without `tools`/`toolConfig` the API can never emit a
/// `functionCall` part, so the native and inline delivery paths never cross.
#[tokio::test]
async fn inline_mode_omits_tools_and_tool_config() {
    use super::super::ToolCallMode;

    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})),
    );
    let c = gemini_connector(port)
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
    assert!(json.get("tools").is_none(), "inline mode: no tools: {body}");
    assert!(
        json.get("toolConfig").is_none(),
        "inline mode: no toolConfig: {body}"
    );
}

/// Without tools there must be no `toolConfig` at all.
#[tokio::test]
async fn request_omits_tool_config_without_tools() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(json.get("toolConfig").is_none(), "no tools → no toolConfig");
}

/// Tool schemas with JSON Schema keywords the Gemini API rejects
/// (`const`, `oneOf`, `additionalProperties`) must be sanitized before
/// serialization — the 400 `Unknown name "const"` regression guard.
/// The payload sent to the API must only contain supported keywords.
#[tokio::test]
async fn tool_schema_with_const_and_one_of_is_sanitized() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    // Mirrors the real plan_todo_write tool schema (const + oneOf) plus
    // the additionalProperties:false pattern from skills tools.
    let tool = ToolDefinition::new(
        ToolFunction::new("plan_todo_write")
            .with_description("Mutate the TODO list")
            .with_parameters(serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "action": {
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
                    }
                },
                "required": ["action"]
            })),
    );
    let c = gemini_connector(port).with_tools(vec![tool]);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let param = &json["tools"][0]["functionDeclarations"][0]["parameters"];

    // Unsupported keywords must be gone from the whole payload.
    let raw = body.as_str();
    assert!(
        !raw.contains("\"const\"")
            && !raw.contains("oneOf")
            && !raw.contains("additionalProperties"),
        "payload still contains Gemini-rejected keywords: {raw}"
    );

    // The discriminator becomes an enum, branches merge into one object.
    let action = &param["properties"]["action"];
    assert_eq!(action["type"], "object");
    assert!(action.get("oneOf").is_none());
    assert_eq!(param["type"], "object");
}

/// A reasoning effort maps onto Gemini 3.x `thinkingConfig.thinkingLevel`.
#[tokio::test]
async fn reasoning_effort_sets_thinking_level() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port).with_reasoning_effort("high");
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        json["generationConfig"]["thinkingConfig"]["thinkingLevel"], "high",
        "reasoning effort must map onto thinkingLevel, got: {body}"
    );
    // The reasoning summary must be requested too — without includeThoughts
    // Gemini 3 never returns the thinking text, so the TUI would show no
    // Think block even though the model reasoned server-side.
    assert_eq!(
        json["generationConfig"]["thinkingConfig"]["includeThoughts"], true,
        "thinking config must request thought summaries, got: {body}"
    );
}

/// No reasoning effort → no `thinkingConfig` at all (model default).
#[tokio::test]
async fn request_omits_thinking_config_without_effort() {
    let sse = "\
data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = gemini_connector(port);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("hi")])
        .await
        .unwrap();
    while stream.next().await.is_some() {}
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("generationConfig").is_none()
            || json["generationConfig"].get("thinkingConfig").is_none(),
        "no effort → no thinkingConfig"
    );
}

/// Thinking-config requests MUST be routed to the `v1beta` API surface: the
/// stable `v1` generateContent endpoint rejects `thinkingConfig` with HTTP
/// 400, and every official Google doc/example sends thinking configuration
/// through `/v1beta`. A request WITHOUT a thinking level keeps the stable
/// version.
#[test]
fn api_version_routes_thinking_to_beta() {
    use crate::connector::gemini::{api_version_for, resolve_base_url};
    use crate::connector::params::Parameters;

    // With a reasoning effort → v1beta.
    for effort in ["minimal", "low", "medium", "high"] {
        let params = Parameters {
            reasoning_effort: Some(effort.to_string()),
            ..Parameters::default()
        };
        assert_eq!(
            api_version_for(&params),
            "v1beta",
            "thinking level '{effort}' must route to v1beta"
        );
    }
    // No effort → stable v1.
    let params = Parameters::default();
    assert_eq!(api_version_for(&params), "v1");

    // The full base-URL rewrite: the version segment of the registered
    // stable URL is replaced with the effective version.
    let config = crate::connector::provider::ProviderConfig {
        name: "gemini",
        family: crate::connector::provider::Family::Gemini,
        base_url: "https://generativelanguage.googleapis.com/v1",
        default_model: "gemini-3-pro-preview",
        needs_extra_headers: false,
        local: false,
    };
    let thinking = Parameters {
        reasoning_effort: Some("high".to_string()),
        ..Parameters::default()
    };
    assert_eq!(
        resolve_base_url(&config, &thinking),
        "https://generativelanguage.googleapis.com/v1beta",
        "thinking requests must hit /v1beta/models/...:streamGenerateContent"
    );
    assert_eq!(
        resolve_base_url(&config, &Parameters::default()),
        "https://generativelanguage.googleapis.com/v1"
    );
    // A user-supplied base_url wins verbatim (proxies / test servers).
    let custom = Parameters {
        base_url: Some("http://127.0.0.1:9999/custom".to_string()),
        reasoning_effort: Some("high".to_string()),
        ..Parameters::default()
    };
    assert_eq!(
        resolve_base_url(&config, &custom),
        "http://127.0.0.1:9999/custom"
    );
}
