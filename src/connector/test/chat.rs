//! Tests for `Connector::chat()` and `Connector::chat_with_system()`.
//!
//! Covers: error propagation (unknown provider, missing API key, HTTP errors,
//! malformed JSON, empty choices, null content, network error), successful
//! response, request body composition (system prompt, params, additional fields,
//! model fallback, tools), extra headers (openrouter), and API key resolution.

use super::super::{Connector, ResponseFormat, ToolDefinition, ToolFunction};
use super::common::{ENV_LOCK, EnvGuard, connector, connector_no_key, mock_server};

/// Ensures `Connector::new("invalid")` returns `UnknownProvider`.
#[tokio::test]
async fn unknown_provider() {
    let err = Connector::new("nonexistent-provider").unwrap_err();
    assert_eq!(err.to_string(), "Unknown provider: nonexistent-provider");
}

/// Ensures a missing API key returns a descriptive error for `chat()`.
#[tokio::test]
async fn missing_api_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    // Isolate from any key stored in the OS keyring under the cosh service.
    let err = Connector::new("openai")
        .unwrap()
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("API key not set for provider: openai"),
        "got: {msg}"
    );
}

/// Ensures a 401 response propagates the error body.
#[tokio::test]
async fn http_401() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#,
        401,
    );
    let err = connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    let msg = err.to_string();
    assert!(msg.contains("401"), "status code missing from: {msg}");
    assert!(
        msg.contains("Incorrect API key"),
        "server detail missing from: {msg}"
    );
}

/// Ensures a 500 response propagates the status code.
#[tokio::test]
async fn http_500() {
    let (port, _body, _raw, handle) = mock_server("Internal Server Error", 500);
    let err = connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert!(err.to_string().contains("500"));
}

/// Ensures a malformed JSON response returns a deserialization error.
#[tokio::test]
async fn malformed_json() {
    let (port, _body, _raw, handle) = mock_server("not-json-at-all", 200);
    let err = connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    let msg = err.to_string();
    assert!(
        msg.contains("expected") || msg.contains("invalid"),
        "deserialization error expected, got: {msg}"
    );
}

/// Ensures an empty choices array returns `No choices in response`.
#[tokio::test]
async fn empty_choices() {
    let (port, _body, _raw, handle) = mock_server(r#"{"choices":[]}"#, 200);
    let err = connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert_eq!(err.to_string(), "No choices in response");
}

/// Ensures a choice with null content returns `No content in response`.
#[tokio::test]
async fn null_content() {
    let (port, _body, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":null}}]}"#, 200);
    let err = connector(port).chat("hello").await.unwrap_err();
    handle.join().unwrap();
    assert_eq!(err.to_string(), "No content in response");
}

/// Ensures a network-level error (unreachable port) is caught.
#[tokio::test]
async fn network_error() {
    let err = Connector::new("openai")
        .unwrap()
        .with_base_url("http://127.0.0.1:1/v1")
        .with_api_key("sk-test")
        .chat("hello")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("error") || msg.contains("refused") || msg.contains("reset"),
        "transport error expected, got: {msg}"
    );
}

/// Ensures a valid response returns the assistant content via `.message()`.
#[tokio::test]
async fn successful_response() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"choices":[{"message":{"content":"Hello world"}}]}"#,
        200,
    );
    let result = connector(port)
        .chat_with_system("hello", "Be concise")
        .await;
    handle.join().unwrap();
    assert_eq!(result.unwrap().message(), "Hello world");
}

/// Ensures `.raw()` returns the exact JSON from the API (unmodified).
#[tokio::test]
async fn raw_json() {
    let raw_body = r#"{"choices":[{"message":{"content":"Hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;
    let (port, _body, _raw, handle) = mock_server(raw_body, 200);
    let result = connector(port).chat("hello").await;
    handle.join().unwrap();
    let output = result.unwrap();
    assert_eq!(output.message(), "Hi");
    assert_eq!(output.raw(), raw_body);
}

/// Ensures the system prompt appears in the request body as a system-role message.
#[tokio::test]
async fn system_prompt_in_body() {
    let (port, captured, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let result = connector(port)
        .chat_with_system("user text", "system text")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let msgs = json["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "system text");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"], "user text");
}

/// Ensures params (model, max_tokens, temperature, seed) appear in the request body.
#[tokio::test]
async fn params_serialized() {
    let (port, captured, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let _ = connector(port)
        .with_model("gpt-4")
        .with_max_tokens(500)
        .with_temperature(0.7)
        .with_seed(42)
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["model"], "gpt-4");
    assert_eq!(json["max_tokens"], 500);
    assert!((json["temperature"].as_f64().unwrap() - 0.7).abs() < 1e-6);
    assert_eq!(json["seed"], 42);
}

/// Ensures additional fields (top_p, stop, frequency_penalty, presence_penalty,
/// response_format, logprobs, top_logprobs, user) appear in the request body.
#[tokio::test]
async fn params_additional_fields() {
    let (port, captured, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let _ = connector(port)
        .with_top_p(0.9)
        .with_stop(serde_json::json!(["END", "STOP"]))
        .with_frequency_penalty(0.5)
        .with_presence_penalty(-0.5)
        .with_response_format(ResponseFormat::json_object())
        .with_logprobs(true)
        .with_top_logprobs(5)
        .with_user("test-user")
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!((json["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(json["stop"], serde_json::json!(["END", "STOP"]));
    assert!((json["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert!((json["presence_penalty"].as_f64().unwrap() - (-0.5)).abs() < 1e-6);
    assert_eq!(json["response_format"]["type"], "json_object");
    assert!(json["logprobs"].as_bool().unwrap());
    assert_eq!(json["top_logprobs"], 5);
    assert_eq!(json["user"], "test-user");
}

/// Ensures `with_tools` and `with_tool_choice` appear in the request body.
#[tokio::test]
async fn tools_serialized() {
    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({"type":"object"})),
    );
    let (port, captured, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let _ = connector(port)
        .with_tools(vec![tool])
        .with_tool_choice(serde_json::json!("auto"))
        .chat("hello")
        .await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["tools"][0]["type"], "function");
    assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(json["tool_choice"], "auto");
}

/// Ensures the default model is sent when no model is explicitly set.
#[tokio::test]
async fn model_fallback() {
    let (port, captured, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let _ = connector(port).chat("hello").await;
    handle.join().unwrap();

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["model"], "llama-3.3-70b-versatile");
}

/// Ensures the OpenRouter provider sends the expected extra headers.
#[tokio::test]
async fn openrouter_extra_headers() {
    let (port, _body, raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let c = Connector::new("openrouter")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test");
    let _ = c.chat("hello").await;
    handle.join().unwrap();

    let req = raw.lock().unwrap().take().unwrap().to_lowercase();
    assert!(req.contains("http-referer:"), "missing http-referer header");
    assert!(req.contains("x-title:"), "missing x-title header");
}

/// Ensures `Connector::chat()` falls back to the provider's env var when no
/// key is stored (keyring isolation via a non-existent service).
#[tokio::test]
async fn api_key_env_fallback() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("GROQ_API_KEY", "sk-from-env");
    let (port, _body, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"ok"}}]}"#, 200);
    let result = connector_no_key(port)
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());
}
