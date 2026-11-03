//! Demonstrate the `connector` module: provider registry, the Connector
//! builder, message/tool payloads, error classification, model discovery,
//! and best-effort live calls when an API key is available.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example client
//! ```

use aiface::connector::{
    ChatMessage, Connector, ConnectorError, ResponseFormat, ToolDefinition, ToolFunction,
    assistant_tool_call_message, discover_context_window, effective_context_window,
    model_reasoning, system_message, tool_result_message, user_message,
};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), ConnectorError> {
    // ── 1. Provider registry ─────────────────────────────────────────────────
    println!("== 1. provider registry ==");
    println!(
        "  {} known providers",
        aiface::connector::known_providers().count()
    );
    for (name, env) in aiface::connector::known_providers_with_env().take(6) {
        println!("  {name:<12} key env: {env}");
    }
    println!(
        "  ollama env var: {:?}",
        aiface::connector::get_provider_env_var("ollama")
    );
    println!(
        "  has_api_key(openai): {}",
        aiface::connector::has_api_key("openai")
    );
    println!(
        "  detect_provider: {:?}",
        aiface::connector::detect_provider()
    );
    println!();

    // ── 2. Builder + introspection (no network) ──────────────────────────────
    println!("== 2. Connector builder ==");
    let connector = Connector::new("openai")?
        .with_model("gpt-4o")
        .with_temperature(0.7)
        .with_max_tokens(128);
    println!("  provider: {:?}", connector.provider_name());
    println!("  model override: {:?}", connector.model());
    println!("  effective model: {:?}", connector.effective_model());
    println!("  has_tools: {}", connector.has_tools());

    let ollama = Connector::new("ollama")?;
    println!("  ollama is_local: {}", ollama.is_local());
    println!("  openai is_local: {}", connector.is_local());

    // with_tools registers native tool definitions.
    let weather = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the current weather for a city")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            })),
    );
    let tooled = connector.clone().with_tools(vec![weather]);
    println!("  has_tools after with_tools: {}", tooled.has_tools());
    println!();

    // ── 3. Message payloads + serialization ──────────────────────────────────
    println!("== 3. ChatMessage payloads ==");
    let messages: Vec<ChatMessage> = vec![
        system_message("You are a helpful assistant."),
        user_message("What is the weather in Paris?"),
        assistant_tool_call_message(vec![aiface::connector::ToolCallMsg {
            id: "call_1".into(),
            kind: "function".into(),
            function: aiface::connector::ToolCallFunctionMsg {
                name: "get_weather".into(),
                arguments: r#"{"city": "Paris"}"#.into(),
            },
            thought_signature: None,
        }]),
        tool_result_message("call_1", "16°C, clear"),
    ];
    for m in &messages {
        println!("  {}", serde_json::to_string(m).unwrap());
    }
    // ResponseFormat serializes as {"type": "json_object"}.
    let fmt = ResponseFormat::json_object();
    println!(
        "  response_format: {}",
        serde_json::to_string(&fmt).unwrap()
    );
    println!();

    // ── 4. Error classification (deterministic) ──────────────────────────────
    println!("== 4. ConnectorError classification ==");
    let ctx = ConnectorError::classify_http(
        400,
        "This model's maximum context length is 128000 tokens. However, your messages resulted in 150000 tokens.".into(),
    );
    match ctx {
        ConnectorError::ContextWindowExceeded {
            status,
            window_tokens,
            ..
        } => {
            println!("  context overflow: HTTP {status}, window ≈ {window_tokens:?} tokens");
        }
        other => println!("  unexpected: {other}"),
    }
    let plain = ConnectorError::classify_http(401, "invalid api key".into());
    println!("  401 invalid key -> HttpError: {plain}");
    let sse = ConnectorError::classify_http(
        200,
        "This model's maximum context length is 64000 tokens.".into(),
    );
    println!(
        "  200 SSE frame with overflow body -> is_context_window: {}",
        sse.is_context_window()
    );
    println!();

    // ── 5. Model discovery (static table, offline) ───────────────────────────
    // gpt-4o and claude-sonnet-4-6 are in the static table, so both calls
    // resolve instantly with zero network traffic.
    println!("== 5. discovery ==");
    for model in ["gpt-4o", "claude-sonnet-4-6"] {
        let window = discover_context_window(model, None).await;
        let reasoning = model_reasoning(model, None).map(|r| r.supported);
        println!(
            "  {model:<18} window {:<8} effective ~{:<8} reasoning: {reasoning:?}",
            window.map(|w| w.to_string()).unwrap_or_else(|| "?".into()),
            window
                .map(effective_context_window)
                .map(|w| w.to_string())
                .unwrap_or_else(|| "?".into()),
        );
    }
    println!();

    // ── 6. Best-effort live calls (need an API key) ──────────────────────────
    println!("== 6. live calls (best-effort) ==");
    match Connector::new("openai") {
        Ok(live) => {
            match live.chat("Say hi in one word").await {
                Ok(reply) => println!("  chat: {}", reply.message()),
                Err(err) => println!("  chat skipped: {err}"),
            }
            match live.list_models().await {
                Ok(out) => println!("  list_models: {} models", out.models().len()),
                Err(err) => println!("  list_models skipped: {err}"),
            }
            match live.stream_chat("Say hi in one word").await {
                Ok(mut stream) => {
                    let mut text = String::new();
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(chunk) => text.push_str(chunk.token()),
                            Err(err) => {
                                println!("  stream error mid-way: {err}");
                                break;
                            }
                        }
                    }
                    println!("  stream: {text}");
                }
                Err(err) => println!("  stream skipped: {err}"),
            }
        }
        Err(err) => println!("  Connector::new failed: {err}"),
    }
    println!();

    // ── 7. tokens(): usage extraction from a raw response ────────────────────
    println!("== 7. token accounting ==");
    let raw = r#"{"id":"chatcmpl-1","choices":[{"message":{"content":"hi"}}],"usage":{"prompt_tokens":9,"completion_tokens":7}}"#;
    println!(
        "  completion_tokens from raw: {:?}",
        Connector::new("openai")?.tokens(raw)
    );
    println!();
    Ok(())
}
