//! Direct model catalog for the Copilot family.
//!
//! The shared runtime's `models.list` RPC answers an EMPTY catalog for
//! Free-tier accounts: it reads the `api.individual.githubcopilot.com`
//! agent-mode catalog, which is only provisioned for paid seats — while
//! the same account chats fine through the runtime (verified live). The
//! real picker catalog lives on the SHARED chat API
//! (`api.githubcopilot.com/models`), which the runtime never lists. This
//! module reproduces the two-step chain every approved Copilot client
//! uses:
//!
//! 1. exchange the GitHub OAuth token for a short-lived Copilot token at
//!    `api.github.com/copilot_internal/v2/token` — the gate REJECTS
//!    requests without an `Editor-Version` header (403 "only use approved
//!    clients");
//! 2. list `models` on the endpoint the grant advertises, which requires
//!    BOTH a known `Editor-Version` AND a registered
//!    `Copilot-Integration-Id` (cosh declares the identity of the SDK it
//!    actually embeds: `copilot-sdk`).
//!
//! Models whose `policy.state` is not `enabled` are dropped: `disabled`
//! entries demand explicit terms acceptance (the same
//! `model_terms_required` gate other providers surface) and would fail on
//! first use; entries without a policy are plumbing/internal models that
//! are not part of the picker contract.

use crate::connector::error::ConnectorError;
use crate::connector::oauth::token::Secret;
use crate::connector::output::{LsOutput, ModelInfo};
use serde::Deserialize;
use std::sync::OnceLock;
use std::time::Duration;

/// The account-level GitHub token exchange endpoint.
const EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// Catalog host used when the grant carries no `endpoints.api`.
const FALLBACK_CATALOG_HOST: &str = "https://api.githubcopilot.com";

/// The honest editor identity. The exchange accepts any `name/version`
/// shape (verified live, including this one); the shared catalog requires
/// a REGISTERED identity, which is why the integration id below — not
/// this header — carries the catalog request.
const EDITOR_VERSION: &str = concat!("cosh/", env!("CARGO_PKG_VERSION"));

/// The integration id of the SDK cosh embeds. The shared catalog rejects
/// unknown integration ids (400 "unknown Copilot-Integration-Id");
/// `copilot-sdk` is registered and yields the full picker catalog.
const COPILOT_INTEGRATION_ID: &str = "copilot-sdk";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

fn client() -> Result<&'static reqwest::Client, ConnectorError> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client);
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| ConnectorError::Network(format!("HTTP client init failed: {e}")))?;
    let _ = CLIENT.set(client);
    CLIENT
        .get()
        .ok_or_else(|| ConnectorError::Network("HTTP client initialization failed".into()))
}

/// The short-lived Copilot grant minted from the account's GitHub token.
#[derive(Debug)]
struct ExchangeGrant {
    token: Secret,
    /// Model-catalog base URL the grant advertises (https only — anything
    /// else falls back to the shared host).
    api: Option<String>,
}

#[derive(Deserialize)]
struct ExchangeBody {
    token: Secret,
    #[serde(default)]
    endpoints: Option<ExchangeEndpoints>,
}

#[derive(Deserialize)]
struct ExchangeEndpoints {
    #[serde(default)]
    api: Option<String>,
}

#[derive(Deserialize)]
struct CatalogBody {
    #[serde(default)]
    data: Vec<CatalogEntry>,
}

#[derive(Deserialize)]
struct CatalogEntry {
    id: String,
    #[serde(default)]
    policy: Option<CatalogPolicy>,
}

#[derive(Deserialize)]
struct CatalogPolicy {
    #[serde(default)]
    state: Option<String>,
}

/// Exchange the account's GitHub token for a Copilot grant.
///
/// The gate requires an `Editor-Version` header; a bare Bearer request
/// answers 403 ("Please only use approved clients for Copilot").
async fn exchange_at(url: &str, github_token: &str) -> Result<ExchangeGrant, ConnectorError> {
    let response = client()?
        .get(url)
        .header("authorization", format!("Bearer {github_token}"))
        .header("accept", "application/json")
        .header("user-agent", EDITOR_VERSION)
        .header("editor-version", EDITOR_VERSION)
        .send()
        .await
        .map_err(|e| ConnectorError::Network(format!("Copilot token exchange failed: {e}")))?;
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|e| ConnectorError::Network(format!("Copilot token exchange read failed: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(ConnectorError::classify_http(status, body));
    }
    let parsed: ExchangeBody = serde_json::from_str(&body)
        .map_err(|_| ConnectorError::Deserialization("Copilot token exchange body".into()))?;
    Ok(ExchangeGrant {
        token: parsed.token,
        api: parsed.endpoints.and_then(|e| e.api),
    })
}

/// List the shared chat catalog with the grant's token.
///
/// Returns the raw body and the ids whose `policy.state` is `enabled`.
async fn catalog_at(
    url: &str,
    copilot_token: &str,
) -> Result<(String, Vec<String>), ConnectorError> {
    let response = client()?
        .get(url)
        .header("authorization", format!("Bearer {copilot_token}"))
        .header("accept", "application/json")
        .header("editor-version", EDITOR_VERSION)
        .header("copilot-integration-id", COPILOT_INTEGRATION_ID)
        .header("user-agent", EDITOR_VERSION)
        .send()
        .await
        .map_err(|e| ConnectorError::Network(format!("Copilot models request failed: {e}")))?;
    let status = response.status().as_u16();
    let body = response
        .text()
        .await
        .map_err(|e| ConnectorError::Network(format!("Copilot models read failed: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(ConnectorError::classify_http(status, body));
    }
    let parsed: CatalogBody = serde_json::from_str(&body)
        .map_err(|_| ConnectorError::Deserialization("Copilot models body".into()))?;
    let ids = parsed
        .data
        .into_iter()
        .filter(|entry| {
            entry
                .policy
                .as_ref()
                .and_then(|p| p.state.as_deref())
                .is_some_and(|state| state == "enabled")
        })
        .map(|entry| entry.id)
        .collect();
    Ok((body, ids))
}

/// List the Copilot models available to the authenticated account.
///
/// Two network steps (exchange → shared catalog); the catalog URL comes
/// from the grant when it advertises one. The raw catalog body is
/// preserved verbatim in [`LsOutput::raw`] for callers that need the full
/// metadata (vendor, capabilities, policy terms).
pub(crate) async fn list_catalog(github_token: &str) -> Result<LsOutput, ConnectorError> {
    let grant = exchange_at(EXCHANGE_URL, github_token).await?;
    let host = grant
        .api
        .as_deref()
        .filter(|url| url.starts_with("https://"))
        .unwrap_or(FALLBACK_CATALOG_HOST);
    let (raw, ids) = catalog_at(&format!("{host}/models"), grant.token.expose()).await?;
    let models = ids.into_iter().map(|id| ModelInfo { id }).collect();
    Ok(LsOutput::new(raw, models))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::oauth::test::support::serve;

    #[tokio::test]
    async fn only_enabled_models_survive_the_policy_filter() {
        let exchange_body = r#"{"token":"copilot-grant-token","expires_at":123,
            "endpoints":{"api":"https://api.example"},"sku":"free_limited_copilot"}"#;
        let catalog_body = r#"{"object":"list","data":[
            {"id":"gpt-4.1","policy":{"state":"enabled"}},
            {"id":"claude-opus-5","policy":{"state":"disabled",
             "terms":"Enable access…"}},
            {"id":"text-embedding-ada-002"}]}"#;
        let (url, server) = serve(vec![
            (200, exchange_body.to_string()),
            (200, catalog_body.to_string()),
        ])
        .await;
        let base = url.as_str().trim_end_matches('/');
        let grant = exchange_at(&format!("{base}/token"), "gh-test")
            .await
            .unwrap();
        let (raw, ids) = catalog_at(&format!("{base}/models"), grant.token.expose())
            .await
            .unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 2);
        let exchange_head = &requests[0].head;
        assert!(
            exchange_head
                .to_ascii_lowercase()
                .contains("editor-version: cosh/"),
            "the exchange must carry the editor identity header"
        );
        assert!(
            exchange_head
                .to_ascii_lowercase()
                .contains("authorization: bearer gh-test"),
            "the exchange must carry the account token"
        );
        let catalog_head = &requests[1].head;
        assert!(
            catalog_head
                .to_ascii_lowercase()
                .contains("copilot-integration-id: copilot-sdk"),
            "the catalog request must carry the embedded SDK's integration id"
        );
        assert_eq!(ids, vec!["gpt-4.1".to_string()]);
        assert!(raw.contains("claude-opus-5"), "raw body preserved verbatim");
    }

    #[tokio::test]
    async fn exchange_rejection_maps_to_an_http_error() {
        let (url, server) = serve(vec![(
            403,
            r#"{"message":"Please only use approved clients for Copilot."}"#.to_string(),
        )])
        .await;
        let base = url.as_str().trim_end_matches('/');
        let error = exchange_at(&format!("{base}/token"), "gh-test")
            .await
            .unwrap_err();
        match error {
            ConnectorError::HttpError { status, body, .. } => {
                assert_eq!(status, 403);
                assert!(body.contains("approved clients"));
            }
            other => panic!("expected HttpError, got {other:?}"),
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn exchange_body_without_a_token_is_a_deserialization_error() {
        let (url, server) = serve(vec![(200, r#"{"chat_enabled":true}"#.to_string())]).await;
        let base = url.as_str().trim_end_matches('/');
        let error = exchange_at(&format!("{base}/token"), "gh-test")
            .await
            .unwrap_err();
        assert!(matches!(error, ConnectorError::Deserialization(_)));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn catalog_rejection_maps_to_an_http_error() {
        let (url, server) = serve(vec![(
            400,
            "bad request: unknown Copilot-Integration-Id".to_string(),
        )])
        .await;
        let base = url.as_str().trim_end_matches('/');
        let error = catalog_at(&format!("{base}/models"), "grant-token")
            .await
            .unwrap_err();
        match error {
            ConnectorError::HttpError { status, body, .. } => {
                assert_eq!(status, 400);
                assert!(body.contains("unknown Copilot-Integration-Id"));
            }
            other => panic!("expected HttpError, got {other:?}"),
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn non_https_catalog_host_is_rejected() {
        let exchange_body = r#"{"token":"grant","endpoints":{"api":"http://evil.example"}}"#;
        let (url, server) = serve(vec![(200, exchange_body.to_string())]).await;
        let base = url.as_str().trim_end_matches('/');
        let grant = exchange_at(&format!("{base}/token"), "gh-test")
            .await
            .unwrap();
        assert_eq!(grant.api.as_deref(), Some("http://evil.example"));
        // The caller-level gate (`list_catalog`) only accepts https hosts;
        // assert the filter logic directly here.
        let accepted = grant
            .api
            .as_deref()
            .filter(|url| url.starts_with("https://"));
        assert!(accepted.is_none(), "non-https grant hosts must be rejected");
        server.await.unwrap();
    }
}
