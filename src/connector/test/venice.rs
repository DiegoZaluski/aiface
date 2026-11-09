//! Tests for the Venice.ai provider (`venice`): registry wiring, the shared
//! OpenAI-compatible request path, and the REAL billed cost Venice reports
//! in a top-level `cost: {usd, diem}` object on every inference response.
//!
//! Wire shapes come from docs.venice.ai (Chat Completions is a drop-in
//! replacement for the OpenAI endpoint; `venice_parameters` stays OUT of
//! the default request — the connector sends only standard OpenAI fields).

use super::super::{Connector, get_provider, get_provider_env_var};
use super::common::{ENV_LOCK, EnvGuard, mock_server};

const CHAT_OK: &str = r#"{"choices":[{"message":{"content":"ok"}}]}"#;

/// Registry sanity: `venice` is an OpenAI-compatible cloud provider with the
/// documented base URL, a current price/performance default model, and its
/// documented env var (`VENICE_API_KEY`).
#[test]
fn venice_registry_entry() {
    let cfg = get_provider("venice").unwrap();
    assert_eq!(cfg.base_url, "https://api.venice.ai/api/v1");
    assert_eq!(cfg.default_model, "z-ai-glm-5-3-flash");
    assert!(!cfg.local);
    assert!(!cfg.needs_extra_headers);

    assert_eq!(get_provider_env_var("venice"), Some("VENICE_API_KEY"));
    assert!(super::super::supports_cost_reporting("venice"));
}

/// Chat goes out as a standard OpenAI request: Bearer key, the default model
/// resolved from the registry, and NO `venice_parameters` block (the
/// connector sends standard fields only — Venice's uncensored system prompt
/// stays on its documented default).
#[tokio::test]
async fn venice_chat_sends_standard_openai_request() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let (port, body, raw, handle) = mock_server(CHAT_OK, 200);
    let result = Connector::new("venice")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-venice-test")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());

    let req = raw.lock().unwrap().take().unwrap().to_lowercase();
    assert!(
        req.contains("authorization: bearer sk-venice-test"),
        "VENICE_API_KEY bearer header missing from request"
    );
    let body = body.lock().unwrap().take().unwrap();
    assert!(
        body.contains("\"model\":\"z-ai-glm-5-3-flash\""),
        "registry default model missing from request: {body}"
    );
    assert!(
        !body.contains("venice_parameters"),
        "venice_parameters must not ride a default request: {body}"
    );
}

/// Without a key anywhere the provider fails fast with the plain
/// missing-key error — never a confusing transport failure.
#[tokio::test]
async fn venice_requires_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let err = Connector::new("venice")
        .unwrap()
        .with_base_url("http://127.0.0.1:1/v1")
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("API key not set for provider: venice"),
        "got: {err}"
    );
}

/// Non-streaming responses carry the REAL billed cost in a top-level
/// `cost: {usd, diem}` object (docs.venice.ai Chat Completions response
/// schema). The amount must surface through `ChatOutput::reported_cost()`
/// and `token_usage` — the TUI usage panel's source.
#[tokio::test]
async fn venice_chat_reports_real_cost() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let body = r#"{"id":"chatcmpl-venice","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"cost":{"usd":0.00042,"diem":0.00042}}"#;
    let (port, _req, _raw, handle) = mock_server(body, 200);
    let out = Connector::new("venice")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-venice-test")
        .chat("hello")
        .await
        .unwrap();
    handle.join().unwrap();
    assert!((out.reported_cost().unwrap() - 0.000_42).abs() < 1e-12);

    let usage = Connector::new("venice")
        .unwrap()
        .token_usage(out.raw())
        .unwrap();
    assert_eq!(usage.reported_cost, Some(0.000_42));
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
}

/// Streaming: the cost rides the final usage frame's top-level `cost`
/// object. The stream's `reported_cost()` must pick it up the same way
/// Zen/Go's cost trailer does.
#[tokio::test]
async fn venice_stream_reports_real_cost() {
    use tokio_stream::StreamExt as _;
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let sse = "\
data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi!\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":6,\"total_tokens\":14},\"cost\":{\"usd\":0.00021,\"diem\":0.00021}}\n\n\
data: [DONE]\n\n";
    let (port, _req, _raw, handle) = mock_server(sse, 200);
    let mut stream = Connector::new("venice")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-venice-test")
        .stream_chat("hello")
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hi!");
    assert_eq!(stream.reported_cost().await, Some(0.000_21));
    let usage = stream.usage().await.unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 6);
    assert_eq!(usage.reported_cost, Some(0.000_21));
}

/// Static discovery covers the gateway's ids without any network call.
/// The context windows resolve through `discover_context_window`'s static
/// table — which returns BEFORE any network I/O for known models, so a
/// plain tokio test with no mock server proves the table path.
#[tokio::test]
async fn venice_models_resolve_windows_statically() {
    for (model, window) in [
        ("z-ai-glm-5-3-flash", 1_048_576),
        ("e2ee-glm-5-3-flash", 1_048_576),
        ("z-ai-glm-5-3", 1_000_000),
        ("claude-haiku-5-5", 1_000_000),
        ("gemini-3-6-flash", 1_000_000),
        ("openai-gpt-oss-120b", 128_000),
    ] {
        assert_eq!(
            super::super::discovery::discover_context_window(model, None).await,
            Some(window),
            "static window mismatch for {model}"
        );
    }
}

/// Reasoning capability resolves statically too, from the live catalog's
/// `model_spec.capabilities` (`supportsReasoningEffort` / effort options).
#[test]
fn venice_models_resolve_reasoning_statically() {
    for (model, efforts) in [
        (
            "openai-gpt-oss-120b",
            Some(vec![
                "none".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ]),
        ),
        (
            "claude-sonnet-5-5",
            Some(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "xhigh".to_string(),
                "max".to_string(),
            ]),
        ),
        ("z-ai-glm-5-3-flash", None),
    ] {
        let reasoning =
            super::super::discovery::model_reasoning(model, None).unwrap_or_else(|| {
                panic!("model {model} must resolve its reasoning statically")
            });
        assert!(
            reasoning.supported,
            "{model} must be marked reasoning-capable"
        );
        assert_eq!(reasoning.efforts, efforts, "effort set mismatch for {model}");
    }
    // The non-reasoning gateway line must NOT offer a reasoning knob.
    // (Bare, provider-agnostic ids like `mistral-small-…` are deliberately
    // absent from the static tables — they resolve through the catalog.)
    for model in ["venice-uncensored-1-2", "gemma-4-uncensored"] {
        let reasoning = super::super::discovery::model_reasoning(model, None)
            .unwrap_or_else(|| panic!("model {model} must resolve its reasoning statically"));
        assert!(
            !reasoning.supported,
            "{model} must be marked non-reasoning"
        );
    }
}
