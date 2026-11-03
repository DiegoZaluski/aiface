//! Tests for `Connector::embed()`.
//!
//! Covers: error propagation (unknown provider, missing API key, HTTP error,
//! malformed JSON, empty data) and a successful embedding response.

use super::super::Connector;
use super::common::{ENV_LOCK, EnvGuard, connector, mock_server};

/// Ensures `Connector::embed()` returns `UnknownProvider` for invalid names.
#[tokio::test]
async fn unknown_provider() {
    let err = Connector::new("nonexistent-provider").unwrap_err();
    assert_eq!(err.to_string(), "Unknown provider: nonexistent-provider");
}

/// Ensures a missing API key returns a descriptive error for `embed()`.
#[tokio::test]
async fn missing_api_key() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    // Isolate from any key stored in the OS keyring under the cosh service.
    let err = Connector::new("openai")
        .unwrap()
        .with_service_keyring("cosh-tests-no-key")
        .embed("text")
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("API key not set for provider: openai"),
        "got: {msg}"
    );
}

/// Ensures an HTTP error response propagates the status code.
#[tokio::test]
async fn http_error() {
    let (port, _body, _raw, handle) = mock_server("error", 500);
    let err = connector(port).embed("text").await.unwrap_err();
    handle.join().unwrap();
    assert!(err.to_string().contains("500"));
}

/// Ensures a malformed JSON response returns a deserialization error.
#[tokio::test]
async fn malformed_json() {
    let (port, _body, _raw, handle) = mock_server("not-json", 200);
    let err = connector(port).embed("text").await.unwrap_err();
    handle.join().unwrap();
    let msg = err.to_string();
    assert!(
        msg.contains("expected") || msg.contains("invalid"),
        "deserialization error expected, got: {msg}"
    );
}

/// Ensures an empty data array returns `No embeddings in response`.
#[tokio::test]
async fn empty_data() {
    let (port, _body, _raw, handle) = mock_server(r#"{"data":[]}"#, 200);
    let err = connector(port).embed("text").await.unwrap_err();
    handle.join().unwrap();
    assert_eq!(err.to_string(), "No embeddings in response");
}

/// Ensures a valid embedding response returns the correct vector.
#[tokio::test]
async fn successful() {
    let (port, _body, _raw, handle) = mock_server(r#"{"data":[{"embedding":[0.1,0.2,0.3]}]}"#, 200);
    let result = connector(port).embed("text").await;
    handle.join().unwrap();
    assert_eq!(result.unwrap(), vec![0.1, 0.2, 0.3]);
}
