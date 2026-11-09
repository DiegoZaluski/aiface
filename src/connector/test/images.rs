//! Integration tests for `Connector::generate_image()`.
//!
//! Each family gets a mock-server round trip asserting the WIRE contract
//! (URL path, request-body fields, response parsing) plus the error and
//! no-surface paths.

use super::super::Connector;
use super::common::{ENV_LOCK, EnvGuard, mock_server};

/// The OpenAI-shaped surface: request hits `/images/generations`, the
/// configured knobs serialize, and b64 bytes come back decoded.
#[tokio::test]
async fn openai_generates_and_parses_b64() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    // "hello" in standard base64.
    let (port, body, _raw, handle) = mock_server(
        r#"{"created":1713833628,"data":[{"b64_json":"aGVsbG8=","revised_prompt":"a better prompt"}],"usage":{"total_tokens":10,"input_tokens":5,"output_tokens":5}}"#,
        200,
    );
    let out = Connector::new("openai")
        .unwrap()
        .with_model("gpt-image-1")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .generate_image("a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    let sent: serde_json::Value = serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(sent["model"], "gpt-image-1");
    assert_eq!(sent["prompt"], "a red panda");
    // GPT Image models reject response_format — it must be absent.
    assert!(sent.get("response_format").is_none());

    let img = out.first().unwrap();
    assert_eq!(img.bytes(), b"hello");
    assert_eq!(img.mime_type(), "image/png");
    assert_eq!(img.revised_prompt(), Some("a better prompt"));
}

/// DALL·E models get `response_format: "b64_json"` requested explicitly —
/// their default is an expiring URL.
#[tokio::test]
async fn dalle_requests_b64_explicitly() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let (port, body, _raw, handle) = mock_server(
        r#"{"created":1,"data":[{"b64_json":"aGVsbG8="}]}"#,
        200,
    );
    Connector::new("openai")
        .unwrap()
        .with_model("dall-e-3")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .generate_image("a cat")
        .await
        .unwrap();
    handle.join().unwrap();

    let sent: serde_json::Value = serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(sent["response_format"], "b64_json");
}

/// `with_image_count` / `with_image_size` / `with_image_quality` serialize
/// verbatim on the OpenAI-shaped surface.
#[tokio::test]
async fn image_knobs_serialize() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let (port, body, _raw, handle) = mock_server(r#"{"data":[{"b64_json":"aGVsbG8="}]}"#, 200);
    Connector::new("openai")
        .unwrap()
        .with_model("gpt-image-1")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .with_image_count(2)
        .with_image_size("1536x1024")
        .with_image_quality("high")
        .generate_image("a cat")
        .await
        .unwrap();
    handle.join().unwrap();

    let sent: serde_json::Value = serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(sent["n"], 2);
    assert_eq!(sent["size"], "1536x1024");
    assert_eq!(sent["quality"], "high");
}

/// Gemini: request carries `responseModalities: ["TEXT","IMAGE"]` and the
/// response's `inlineData` part comes back decoded.
#[tokio::test]
async fn gemini_generates_and_parses_inline_data() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("GEMINI_API_KEY");
    let (port, body, _raw, handle) = mock_server(
        r#"{"candidates":[{"finishReason":"STOP","content":{"parts":[
            {"text":"here it is"},
            {"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
        ]}}]}"#,
        200,
    );
    let out = Connector::new("gemini")
        .unwrap()
        .with_model("gemini-2.5-flash-image")
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("test-key")
        .with_retry(false)
        .with_image_size("16:9")
        .generate_image("a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    let sent: serde_json::Value = serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(
        sent["generationConfig"]["responseModalities"],
        serde_json::json!(["TEXT", "IMAGE"])
    );
    // image_size carries the aspect ratio on this surface.
    assert_eq!(
        sent["generationConfig"]["imageConfig"]["aspectRatio"],
        "16:9"
    );

    let img = out.first().unwrap();
    assert_eq!(img.bytes(), b"hello");
    assert_eq!(img.mime_type(), "image/png");
}

/// A Gemini safety-block (no candidates, promptFeedback.blockReason)
/// yields an EMPTY output — a normal answer, not an error.
#[tokio::test]
async fn gemini_blocked_prompt_is_empty_output() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("GEMINI_API_KEY");
    let (port, _body, _raw, handle) =
        mock_server(r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#, 200);
    let out = Connector::new("gemini")
        .unwrap()
        .with_model("gemini-2.5-flash-image")
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("test-key")
        .with_retry(false)
        .generate_image("disallowed")
        .await
        .unwrap();
    handle.join().unwrap();
    assert!(
        out.images().is_empty(),
        "a blocked generation must yield no images"
    );
}

/// Venice rides the OpenAI-shaped surface (its gateway implements
/// `/images/generations`): the request goes to the provider's base URL
/// and `variants` is NOT sent — the count knob is OpenAI's `n`, which
/// Venice rejects (it serves one image per call).
#[tokio::test]
async fn venice_generates_through_openai_surface() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let (port, body, _raw, handle) = mock_server(
        r#"{"created":1,"data":[{"b64_json":"aGVsbG8="}]}"#,
        200,
    );
    let out = Connector::new("venice")
        .unwrap()
        .with_model("z-image-turbo")
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_image("a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    let sent: serde_json::Value = serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(sent["model"], "z-image-turbo");
    assert!(sent.get("n").is_none());

    let img = out.first().unwrap();
    assert_eq!(img.bytes(), b"hello");
    // The raw response is preserved for fields this type does not model
    // (Venice's cost object lives there).
    assert!(img.raw().contains("data"));
}

/// HTTP errors propagate with the provider's body.
#[tokio::test]
async fn http_error_propagates() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let (port, _body, _raw, handle) =
        mock_server(r#"{"error":{"message":"billing blocked","type":"invalid_request_error"}}"#, 400);
    let err = Connector::new("openai")
        .unwrap()
        .with_model("gpt-image-1")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .generate_image("a cat")
        .await
        .unwrap_err();
    handle.join().unwrap();
    let msg = err.to_string();
    assert!(msg.contains("400") && msg.contains("billing blocked"), "got: {msg}");
}

/// Providers with no image surface fail BEFORE any request is sent.
/// Local model servers are deliberately NOT in this list: a
/// `with_base_url`-pointed server may implement the OpenAI-shaped route
/// (e.g. a self-hosted NIM), so the family's surface is attempted.
#[tokio::test]
async fn no_surface_providers_fail_fast() {
    for name in ["claude", "copilot"] {
        let connector = Connector::new(name).unwrap().with_api_key("test");
        let err = connector.generate_image("a cat").await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("image generation"),
            "{name} should report no image surface, got: {msg}"
        );
    }
}

/// A missing API key is reported before any request for cloud providers.
#[tokio::test]
async fn missing_api_key_is_reported() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let err = Connector::new("openai")
        .unwrap()
        .with_service_keyring("cosh-tests-no-key")
        .generate_image("a cat")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("API key not set for provider: openai"),
        "got: {msg}"
    );
}
