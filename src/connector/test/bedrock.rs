//! Tests for the `bedrock` provider — the OpenAI-compatible family entry for
//! Amazon Bedrock's `/openai/v1` endpoints.
//!
//! Covers: provider registry wiring, the `AWS_BEARER_TOKEN_BEDROCK` auth
//! path, region resolution from `AWS_REGION`/`AWS_DEFAULT_REGION`, the AWS
//! error-body classification (both flat and `Output`-wrapped shapes, with
//! transient `ThrottlingException` marking), and the `list_models`/`embed`
//! gates for the endpoint surfaces Bedrock's OpenAI-compatible API does not
//! implement.

use super::super::retry::is_retryable_error;
use super::super::{Connector, StreamChunk, get_provider, user_message};
use super::common::{ENV_LOCK, EnvGuard, mock_server};
use tokio_stream::StreamExt;

/// A bedrock connector pointed at a mock server (explicit base URL, like
/// every other family test — the mock URL wins over region resolution).
fn bedrock_connector(port: u16) -> Connector {
    Connector::new("bedrock")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/openai/v1"))
        .with_api_key("aws-bearer-test")
}

/// The registry entry exists, is OpenAI-compatible, cloud, and points at the
/// documented `bedrock-runtime` endpoint shape with a working default model.
#[test]
fn bedrock_provider_registry() {
    let cfg = get_provider("bedrock").expect("bedrock registered");
    assert_eq!(cfg.name, "bedrock");
    assert!(
        cfg.base_url
            .starts_with("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"),
        "registry URL pins the bedrock-runtime OpenAI endpoint, got: {}",
        cfg.base_url
    );
    assert!(!cfg.default_model.is_empty());
    assert!(
        cfg.base_url.contains("openai/v1"),
        "the OpenAI-compatible path must be part of the base URL"
    );
}

/// The Bedrock bearer token resolves through the provider's env var (the
/// same keyring-then-environment path every cloud provider uses).
#[tokio::test]
async fn bedrock_key_resolves_from_env() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("AWS_BEARER_TOKEN_BEDROCK", "aws-bearer-test");
    assert_eq!(
        super::super::get_api_key("bedrock", None).as_deref(),
        Some("aws-bearer-test")
    );
    assert!(super::super::has_api_key("bedrock"));
    assert_eq!(
        super::super::get_provider_env_var("bedrock"),
        Some("AWS_BEARER_TOKEN_BEDROCK")
    );
}

/// An empty token export is not a credential (same placeholder filter as
/// every other provider).
#[tokio::test]
async fn bedrock_empty_token_is_not_a_credential() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("AWS_BEARER_TOKEN_BEDROCK", "");
    assert_eq!(super::super::get_api_key("bedrock", None), None);
}

/// The bearer token rides the shared `Authorization` header path.
#[tokio::test]
async fn bedrock_request_carries_bearer_auth() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
    let (port, _body, raw, handle) = mock_server(sse, 200);
    let c = bedrock_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    while stream.next().await.is_some() {}

    let raw = raw.lock().unwrap().take().unwrap();
    assert!(
        raw.to_lowercase()
            .contains("authorization: bearer aws-bearer-test"),
        "bedrock auth must ride the shared bearer path, raw: {raw}"
    );
    // The OpenAI-compatible request shape reaches the endpoint unmodified.
    assert!(raw.contains("/openai/v1/chat/completions"), "raw: {raw}");
}

/// The default model is the documented Chat Completions model on
/// bedrock-runtime.
#[tokio::test]
async fn bedrock_default_model_reaches_the_wire() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let cfg = get_provider("bedrock").unwrap();
    let c = Connector::new("bedrock")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/openai/v1"))
        .with_api_key("aws-bearer-test")
        .with_model(cfg.default_model);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    while stream.next().await.is_some() {}

    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["model"], "openai.gpt-oss-120b-1:0");
}

/// Streaming works through the shared family path.
#[tokio::test]
async fn bedrock_streaming() {
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{"content":" world"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        r#"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = bedrock_connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hello world");
}

/// Native tool calling works through the shared family path.
#[tokio::test]
async fn bedrock_streaming_tool_calls() {
    use super::super::{ToolCallMode, ToolDefinition, ToolFunction};

    let tool = ToolDefinition::new(
        ToolFunction::new("get_weather")
            .with_description("Get the weather")
            .with_parameters(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            })),
    );
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_b1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"São Paulo\"}"}}]}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );
    let (port, captured, _raw, handle) = mock_server(sse, 200);
    let c = bedrock_connector(port)
        .with_tools(vec![tool])
        .with_tool_choice(serde_json::json!("auto"))
        .with_tool_call_mode(ToolCallMode::Native);
    let mut stream = c
        .stream_chat_with_messages("sys", &[user_message("weather?")])
        .await
        .unwrap();
    handle.join().unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }
    let tool_call = chunks
        .iter()
        .find_map(|c| c.tool_call.as_ref())
        .expect("native tool call must surface");
    assert_eq!(tool_call.name, "get_weather");
    assert_eq!(tool_call.id, "call_b1");
    assert_eq!(tool_call.arguments, r#"{"city":"São Paulo"}"#);

    // The tools array rode the request.
    let body = captured.lock().unwrap().take().unwrap();
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        json.get("tools").is_some(),
        "native mode sends tools: {json}"
    );
}

/// An AWS-shaped error body (`{"__type": ..., "message": ...}`) classifies
/// as an HTTP error with the AWS message — not as a parse failure.
#[tokio::test]
async fn bedrock_aws_error_body_classifies() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"__type":"AccessDeniedException","message":"User is not authorized to perform: bedrock:InvokeModel"}"#,
        200,
    );
    let c = bedrock_connector(port).with_retry(false);
    let result = c.chat("hi").await;
    handle.join().unwrap();
    let err = result.expect_err("AWS error body must fail the call");
    assert!(
        err.to_string().contains("not authorized"),
        "the AWS message must surface: {err}"
    );
    assert!(!is_retryable_error(&err), "AccessDenied is deterministic");
}

/// The `Output`-wrapped AWS error shape classifies the same way.
#[tokio::test]
async fn bedrock_wrapped_aws_error_body_classifies() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"Output":{"__type":"com.amazonaws.bedrock#ValidationException","message":"Model ID not recognized"}}"#,
        200,
    );
    let c = bedrock_connector(port).with_retry(false);
    let result = c.chat("hi").await;
    handle.join().unwrap();
    let err = result.expect_err("wrapped AWS error body must fail the call");
    assert!(
        err.to_string().contains("Model ID not recognized"),
        "the AWS message must surface: {err}"
    );
}

/// `ThrottlingException` (the Bedrock analogue of a rate limit) is marked
/// transient and therefore retried by the middleware.
#[tokio::test]
async fn bedrock_throttling_is_transient() {
    let (port, _body, _raw, handle) = mock_server(
        r#"{"__type":"ThrottlingException","message":"Rate exceeded"}"#,
        200,
    );
    let c = bedrock_connector(port).with_retry(false);
    let result = c.chat("hi").await;
    handle.join().unwrap();
    let err = result.expect_err("throttling must fail the single-shot call");
    assert!(
        err.to_string().contains("Rate exceeded"),
        "the AWS message must surface: {err}"
    );
    assert!(
        is_retryable_error(&err),
        "ThrottlingException must be retryable: {err}"
    );
}

/// Streaming: an AWS throttling frame mid-stream surfaces as a retryable
/// error. The frame is a VALID JSON body that simply does not match the
/// OpenAI chunk schema (`choices` empty, `__type` present) — exactly what
/// the Bedrock endpoints emit as an error event.
#[tokio::test]
async fn bedrock_throttling_mid_stream_is_transient() {
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"content":"partial"}}]}"#,
        "\n\n",
        r#"data: {"__type":"ThrottlingException","message":"Rate exceeded"}"#,
        "\n\n",
    );
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let c = bedrock_connector(port).with_retry(false);
    let stream = c.stream_chat("hi").await.unwrap();
    handle.join().unwrap();
    let results: Vec<Result<StreamChunk, _>> = stream.collect().await;
    let last = results.last().expect("at least one item");
    let err = last.as_ref().expect_err("throttling must fail the stream");
    assert!(
        err.to_string().contains("Rate exceeded"),
        "the AWS message must surface: {err}"
    );
    assert!(
        is_retryable_error(err),
        "mid-stream ThrottlingException must be retryable: {err}"
    );
}

/// `list_models` fails up front against bedrock-runtime (the endpoint has no
/// `GET /models`), with a message pointing at the Mantle catalog. No explicit
/// base URL — the gate keys off the default endpoint path.
#[tokio::test]
async fn bedrock_list_models_gate() {
    let c = Connector::new("bedrock")
        .unwrap()
        .with_api_key("aws-bearer-test");
    let err = c.list_models().await.expect_err("runtime has no /models");
    let msg = err.to_string();
    assert!(
        msg.contains("bedrock-runtime"),
        "the gate must explain the runtime limitation: {msg}"
    );
    assert!(
        msg.contains("bedrock-mantle"),
        "the gate must point at the Mantle catalog: {msg}"
    );
}

/// An explicit base URL pointing at the Mantle endpoint opts back into the
/// shared `GET /models` path.
#[tokio::test]
async fn bedrock_list_models_with_explicit_mantle_url() {
    let (port, _body, raw, handle) =
        mock_server(r#"{"data":[{"id":"openai.gpt-oss-120b-1:0"}]}"#, 200);
    let c = Connector::new("bedrock")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("aws-bearer-test");
    let result = c.list_models().await;
    handle.join().unwrap();
    let out = result.expect("explicit Mantle URL opts into /models");
    assert!(
        out.models()
            .iter()
            .any(|m| m.id() == "openai.gpt-oss-120b-1:0"),
        "the catalog must parse: {:?}",
        out.models()
    );
    let raw = raw.lock().unwrap().take().unwrap();
    assert!(raw.contains("/v1/models"), "raw: {raw}");
}

/// `embed` fails up front against bedrock (no OpenAI-compatible embedding
/// surface), with a message explaining the Invoke-only reality. No explicit
/// base URL — the gate keys off the default endpoint path.
#[tokio::test]
async fn bedrock_embed_gate() {
    let c = Connector::new("bedrock")
        .unwrap()
        .with_api_key("aws-bearer-test");
    let err = c.embed("text").await.expect_err("no embedding surface");
    let msg = err.to_string();
    assert!(
        msg.contains("embedding") && msg.contains("bedrock"),
        "the gate must explain the limitation: {msg}"
    );
}

/// An explicit base URL (a compatible proxy serving embeddings) opts back
/// into the shared embedding path.
#[tokio::test]
async fn bedrock_embed_with_explicit_proxy_url() {
    let (port, _body, _raw, handle) = mock_server(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#, 200);
    let c = Connector::new("bedrock")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("aws-bearer-test");
    let result = c.embed("text").await;
    handle.join().unwrap();
    let embedding = result.expect("explicit proxy URL opts into embeddings");
    assert_eq!(embedding.len(), 3);
}

/// Region resolution: `AWS_REGION` re-points the registry's default region;
/// `AWS_DEFAULT_REGION` is the fallback; no variable (or the default region
/// itself) keeps the registry URL. Other providers are never rewritten.
///
/// Runs under the env-mutation lock: `resolved_provider_base_url` reads
/// process-wide AWS environment variables, so without the lock this test
/// would race any other test that touches AWS env state.
#[tokio::test]
async fn resolved_provider_base_url_follows_aws_region() {
    let _lock = ENV_LOCK.lock().await;
    let default = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1";

    // No region exported → registry default.
    let _r = EnvGuard::remove("AWS_REGION");
    let _d = EnvGuard::remove("AWS_DEFAULT_REGION");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        default
    );

    // AWS_REGION wins.
    let _eu = EnvGuard::set("AWS_REGION", "eu-west-1");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"
    );
    // Blank AWS_REGION is not a region; AWS_DEFAULT_REGION takes over.
    let _blank = EnvGuard::set("AWS_REGION", "  ");
    let _sp = EnvGuard::set("AWS_DEFAULT_REGION", "sa-east-1");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        "https://bedrock-runtime.sa-east-1.amazonaws.com/openai/v1"
    );
    // The default region itself needs no rewrite.
    let _same = EnvGuard::set("AWS_REGION", "us-east-1");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        default
    );

    // SECURITY: the region is spliced into the endpoint HOST that carries
    // the bearer token. Anything outside the AWS region grammar — path
    // injection, userinfo, dots, uppercase — falls back to the registry
    // default instead of silently re-pointing the endpoint. The fallback
    // variable must be OUT of the way here, or a hostile AWS_REGION would
    // (correctly) hand resolution to the still-valid AWS_DEFAULT_REGION.
    let _d_clear = EnvGuard::remove("AWS_DEFAULT_REGION");
    for hostile in [
        "evil.com/x#",
        "https://evil.com",
        "us-east-1@evil.com",
        "EVIL",
        "eu..west-1",
        "eu_west-1",
        "a-1",
    ] {
        let _g = EnvGuard::set("AWS_REGION", hostile);
        assert_eq!(
            super::super::resolved_provider_base_url("bedrock").as_ref(),
            default,
            "hostile region {hostile:?} must not re-point the endpoint"
        );
    }

    // Exotic-but-real AWS shapes parse: GovCloud and the EU sovereign cloud.
    let _gov = EnvGuard::set("AWS_REGION", "us-gov-west-1");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        "https://bedrock-runtime.us-gov-west-1.amazonaws.com/openai/v1"
    );
    let _eusc = EnvGuard::set("AWS_REGION", "eusc-de-east-1");
    assert_eq!(
        super::super::resolved_provider_base_url("bedrock").as_ref(),
        "https://bedrock-runtime.eusc-de-east-1.amazonaws.com/openai/v1"
    );

    // Other providers are returned verbatim regardless of AWS env.
    assert_eq!(
        super::super::resolved_provider_base_url("groq").as_ref(),
        get_provider("groq").unwrap().base_url
    );
}
