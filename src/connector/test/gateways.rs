//! Tests for the documented OpenCode gateways: `zen`
//! ([`OPENCODE_ZEN_PROVIDER`]) and `opencode-go`
//! ([`OPENCODE_GO_PROVIDER`]).
//!
//! Covers: key-based chat on both gateways (with cosh's honest User-Agent)
//! and the Go shared-account key fallback (`OPENCODE_GO_API_KEY` →
//! `OPENCODE_API_KEY`). Both gateways always require an API key.

use super::super::{
    Connector, OPENCODE_GO_PROVIDER, OPENCODE_ZEN_PROVIDER, get_provider, get_provider_env_var,
};
use super::common::{ENV_LOCK, EnvGuard, mock_server};

const CHAT_OK: &str = r#"{"choices":[{"message":{"content":"ok"}}]}"#;

/// Registry sanity: both gateways are OpenAI-compatible cloud providers with
/// the documented base URLs, and resolve their documented env vars.
#[test]
fn gateway_registry_entries() {
    let zen = get_provider(OPENCODE_ZEN_PROVIDER).unwrap();
    assert_eq!(zen.base_url, "https://opencode.ai/zen/v1");
    assert_eq!(zen.default_model, "gpt-5.6-luna");
    assert!(!zen.local);

    let go = get_provider(OPENCODE_GO_PROVIDER).unwrap();
    assert_eq!(go.base_url, "https://opencode.ai/zen/go/v1");
    assert_eq!(go.default_model, "glm-5.3-flash");
    assert!(!go.local);

    assert_eq!(
        get_provider_env_var(OPENCODE_ZEN_PROVIDER),
        Some("OPENCODE_API_KEY")
    );
    assert_eq!(
        get_provider_env_var(OPENCODE_GO_PROVIDER),
        Some("OPENCODE_GO_API_KEY")
    );
}

/// Zen gateway chat goes out under the account key and carries cosh's
/// honest User-Agent.
#[tokio::test]
async fn zen_gateway_sends_account_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_API_KEY");
    let (port, _body, raw, handle) = mock_server(CHAT_OK, 200);
    let result = Connector::new(OPENCODE_ZEN_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-zen-account")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());

    let req = raw.lock().unwrap().take().unwrap().to_lowercase();
    assert!(
        req.contains("authorization: bearer sk-zen-account"),
        "account key missing from request"
    );
    assert!(
        req.contains("user-agent: cosh/"),
        "co-sh User-Agent missing from request"
    );
}

/// Without a key the documented Zen gateway fails with the plain
/// missing-key error.
#[tokio::test]
async fn zen_gateway_requires_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_API_KEY");
    let err = Connector::new(OPENCODE_ZEN_PROVIDER)
        .unwrap()
        .with_base_url("http://127.0.0.1:1/v1")
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("API key not set for provider: zen"),
        "got: {err}"
    );
}

/// Go gateway chat works with an explicit key and the honest User-Agent.
#[tokio::test]
async fn go_gateway_sends_account_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_GO_API_KEY");
    let _guard2 = EnvGuard::remove("OPENCODE_API_KEY");
    let (port, _body, raw, handle) = mock_server(CHAT_OK, 200);
    let result = Connector::new(OPENCODE_GO_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-go-account")
        .with_model("kimi-k3")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());

    let req = raw.lock().unwrap().take().unwrap().to_lowercase();
    assert!(
        req.contains("authorization: bearer sk-go-account"),
        "account key missing from request"
    );
    assert!(
        req.contains("user-agent: cosh/"),
        "co-sh User-Agent missing from request"
    );
}

/// The Go subscription is managed through the Zen account:
/// `OPENCODE_API_KEY` is accepted when `OPENCODE_GO_API_KEY` is unset.
#[tokio::test]
async fn go_gateway_falls_back_to_zen_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_GO_API_KEY");
    let _guard2 = EnvGuard::set("OPENCODE_API_KEY", "sk-shared-zen-key");
    let (port, _body, raw, handle) = mock_server(CHAT_OK, 200);
    let result = Connector::new(OPENCODE_GO_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok(), "shared Zen key must unlock Go");
    assert!(
        raw.lock()
            .unwrap()
            .take()
            .unwrap()
            .to_lowercase()
            .contains("authorization: bearer sk-shared-zen-key")
    );
}

/// Go's own env var wins over the shared Zen fallback.
#[tokio::test]
async fn go_gateway_prefers_its_own_env_var() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::set("OPENCODE_GO_API_KEY", "sk-go-own-key");
    let _guard2 = EnvGuard::set("OPENCODE_API_KEY", "sk-shared-zen-key");
    let (port, _body, raw, handle) = mock_server(CHAT_OK, 200);
    let result = Connector::new(OPENCODE_GO_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await;
    handle.join().unwrap();
    assert!(result.is_ok());
    assert!(
        raw.lock()
            .unwrap()
            .take()
            .unwrap()
            .to_lowercase()
            .contains("authorization: bearer sk-go-own-key")
    );
}

/// Without a key anywhere the Go gateway fails fast with the missing-key
/// error.
#[tokio::test]
async fn go_gateway_requires_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_GO_API_KEY");
    let _guard2 = EnvGuard::remove("OPENCODE_API_KEY");
    let err = Connector::new(OPENCODE_GO_PROVIDER)
        .unwrap()
        .with_base_url("http://127.0.0.1:1/v1")
        .with_service_keyring("cosh-tests-no-key")
        .chat("hello")
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("API key not set for provider: opencode-go"),
        "got: {err}"
    );
}

/// REAL cost in the streamed response: the Zen/Go gateways end the SSE
/// stream with a usage frame followed by the non-standard
/// `{"choices":[],"cost":"0"}` trailer (exact wire format from
/// anomalyco/opencode#42918). The REAL billed amount must land both in
/// `reported_cost()` and in `usage()` — the TUI usage panel's source — so
/// the panel shows what the gateway actually charged instead of the
/// price-table estimate.
#[tokio::test]
async fn go_gateway_stream_reports_real_cost() {
    use tokio_stream::StreamExt as _;
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_GO_API_KEY");
    let _guard2 = EnvGuard::remove("OPENCODE_API_KEY");
    let sse = "\
data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"glm-5.3-flash\",\"choices\":[]}\n\n\
data: {\"id\":\"\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi!\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"gen-...\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"glm-5.3-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":6,\"total_tokens\":14}}\n\n\
data: {\"choices\":[],\"cost\":\"0\"}\n\n\
data: [DONE]\n\n";
    let (port, _body, _raw, handle) = mock_server(sse, 200);
    let mut stream = Connector::new(OPENCODE_GO_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-go-account")
        .stream_chat("hello")
        .await
        .unwrap();
    handle.join().unwrap();

    let mut tokens = String::new();
    while let Some(chunk) = stream.next().await {
        tokens.push_str(chunk.unwrap().token());
    }
    assert_eq!(tokens, "Hi!");

    assert_eq!(stream.reported_cost().await, Some(0.0));
    let usage = stream.usage().await.unwrap();
    assert_eq!(usage.input_tokens, 8);
    assert_eq!(usage.output_tokens, 6);
    assert_eq!(usage.reported_cost, Some(0.0));
}

/// Non-streaming responses carry the same top-level `cost` — the
/// [`ChatOutput`] accessor surfaces the REAL billed amount there too.
#[tokio::test]
async fn zen_gateway_chat_reports_real_cost() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENCODE_API_KEY");
    let body = r#"{"id":"chatcmpl-x","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":8,"completion_tokens":6},"cost":"0.0031"}"#;
    let (port, _req, _raw, handle) = mock_server(body, 200);
    let out = Connector::new(OPENCODE_ZEN_PROVIDER)
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-zen-account")
        .chat("hello")
        .await
        .unwrap();
    handle.join().unwrap();
    assert!((out.reported_cost().unwrap() - 0.0031).abs() < 1e-12);

    // The usage channel carries it as well: `token_usage` on the raw body
    // stamps the REAL amount, which `effective_cost` prefers over the
    // price-table estimate.
    let c = Connector::new(OPENCODE_ZEN_PROVIDER).unwrap();
    let usage = c.token_usage(out.raw()).unwrap();
    assert_eq!(usage.reported_cost, Some(0.0031));
}
