//! GitHub Copilot OAuth device flow ([RFC 8628]) over the generic device
//! grant (the same shared machinery as the generic OAuth module).
//!
//! GitHub's device endpoints are the standard-conformant ones (form-encoded,
//! `authorization_pending`/`slow_down`/`access_denied` error codes), so this
//! adapter is a thin configuration layer: it pins the well-known Copilot CLI
//! client id and the github.com endpoints into an [`OAuthProviderConfig`]
//! and reuses the hardened transport, polling backoff and token model
//! unchanged. Storage stays with the consumer (see the oauth module docs).
//!
//! The resulting token is a GitHub OAuth token carrying the user's Copilot
//! entitlement — exactly the `GITHUB_TOKEN` credential the Copilot family
//! resolves (see `connector/provider.rs`). Scopes stay empty on purpose:
//! the Copilot application grants what it grants, and extra scope asks
//! would only widen the prompt the user approves.
//!
//! [RFC 8628]: https://www.rfc-editor.org/rfc/rfc8628.html

// Keep the import list aligned with what the adapter actually uses: the
// generic device grant, the validated config view and the shared token
// model. Nothing here touches storage — that stays with the consumer.
use crate::connector::oauth::device::DeviceAuthorization;
use crate::connector::oauth::{OAuthProviderConfig, Result, TokenSet, device};
use url::Url;

/// GitHub's device authorization endpoint (RFC 8628 §3.1).
pub const DEFAULT_DEVICE_URL: &str = "https://github.com/login/device/code";
/// GitHub's token endpoint (device poll + refresh).
pub const DEFAULT_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
/// The public client id of GitHub's own Copilot CLI OAuth application —
/// public by design (native app, no client secret), embedded in every
/// Copilot CLI distribution. Overridable for fixtures and future apps.
pub const DEFAULT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// Endpoint/client configuration for the Copilot device flow. Every field
/// has a production default; tests override the URLs with loopback fixtures
/// (which also requires [`Self::allow_http_loopback`]).
#[derive(Clone)]
pub struct CopilotOAuthConfig {
    pub device_authorization_url: Url,
    pub token_url: Url,
    pub client_id: String,
    /// Only enable for local HTTP fixtures; never permits remote HTTP.
    pub allow_http_loopback: bool,
}
impl std::fmt::Debug for CopilotOAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopilotOAuthConfig").finish_non_exhaustive()
    }
}
impl Default for CopilotOAuthConfig {
    fn default() -> Self {
        Self {
            // Safe invariant: compile-time HTTPS literals whose parseability
            // and endpoint validation the config validator exercises.
            device_authorization_url: Url::parse(DEFAULT_DEVICE_URL).expect("static HTTPS URL"),
            token_url: Url::parse(DEFAULT_TOKEN_URL).expect("static HTTPS URL"),
            client_id: DEFAULT_CLIENT_ID.into(),
            allow_http_loopback: false,
        }
    }
}

impl CopilotOAuthConfig {
    /// The generic module's validated config view. `authorization_url` is
    /// required by [`OAuthProviderConfig::new`] but unused by the device
    /// grant — it points at GitHub's web authorize page for completeness.
    fn provider_config(&self) -> OAuthProviderConfig {
        let mut provider = OAuthProviderConfig::new(
            // Same host as the device endpoint; never fetched by this flow.
            Url::parse("https://github.com/login/oauth/authorize").expect("static HTTPS URL"),
            self.token_url.clone(),
            self.client_id.clone(),
        );
        provider.device_authorization_url = Some(self.device_authorization_url.clone());
        provider.allow_http_loopback = self.allow_http_loopback;
        provider
    }

    /// Start the device transaction: returns the user code + verification
    /// URL for the UI and the pollable transaction for [`poll_and_exchange`].
    pub async fn begin(&self) -> Result<DeviceAuthorization> {
        device::begin_device_authorization(&self.provider_config()).await
    }
}

/// Poll GitHub until the user approves (or the transaction expires/denies).
/// Thin over [`device::poll_device_token`]. GitHub DEVIATES from the RFC:
/// the token endpoint answers the pending and slow_down states with HTTP
/// 200 and the `error` member in the body (instead of the 4xx the RFC
/// sketches) — the generic loop classifies the body FIRST precisely for
/// this. The terminations (`access_denied`, expiry) are standard.
pub async fn poll_and_exchange(auth: DeviceAuthorization) -> Result<TokenSet> {
    device::poll_device_token(auth).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The adapter must be a pure configuration layer over the generic
    /// device grant: the wire flow hits GitHub's standard endpoints with
    /// the configured client id, then exchanges the device code for a
    /// token body the generic model accepts.
    #[tokio::test]
    async fn device_flow_uses_github_wire_format_and_generic_polling() {
        let (url, server) = crate::connector::oauth::test::support::serve(vec![
            (
                200,
                r#"{"device_code":"DEV1234","user_code":"ABCD-1234","verification_uri":"https://github.com/login/device","expires_in":30,"interval":1}"#.into(),
            ),
            (
                200,
                r#"{"access_token":"gho_test","token_type":"bearer","expires_in":2839971400}"#.into(),
            ),
        ])
        .await;
        let config = CopilotOAuthConfig {
            device_authorization_url: url.join("device/code").unwrap(),
            token_url: url.join("token").unwrap(),
            allow_http_loopback: true,
            ..CopilotOAuthConfig::default()
        };

        let auth = config.begin().await.unwrap();
        assert_eq!(auth.user_code, "ABCD-1234");
        let tokens = tokio::time::timeout(Duration::from_secs(10), poll_and_exchange(auth))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tokens.access_token().expose(), "gho_test");

        let requests = server.await.unwrap();
        assert!(
            requests[0].head.starts_with("POST /device/code HTTP/1.1"),
            "the device transaction starts at GitHub's RFC 8628 endpoint"
        );
        let start_body = form_body(&requests[0].body);
        let client_id = start_body
            .iter()
            .find(|(k, _)| k == "client_id")
            .map(|(_, v)| v.as_str());
        assert_eq!(
            client_id,
            Some(DEFAULT_CLIENT_ID),
            "the Copilot CLI public client id must travel in the form body"
        );
        assert!(
            requests[1].head.starts_with("POST /token HTTP/1.1"),
            "the poll exchanges the device code at the token endpoint"
        );
        let poll_body = form_body(&requests[1].body);
        let device_code = poll_body
            .iter()
            .find(|(k, _)| k == "device_code")
            .map(|(_, v)| v.as_str());
        assert_eq!(
            device_code,
            Some("DEV1234"),
            "the poll must carry the device code from the transaction"
        );
        let grant_type = poll_body
            .iter()
            .find(|(k, _)| k == "grant_type")
            .map(|(_, v)| v.as_str());
        assert_eq!(
            grant_type,
            Some("urn:ietf:params:oauth:grant-type:device_code"),
        );
    }

    /// The production defaults must be the real GitHub endpoints and the
    /// Copilot CLI client id — an adapter pointed elsewhere by accident
    /// would fail closed (endpoint validation) or at the provider.
    #[test]
    fn production_defaults_are_github_endpoints() {
        let config = CopilotOAuthConfig::default();
        assert_eq!(config.device_authorization_url.as_str(), DEFAULT_DEVICE_URL);
        assert_eq!(config.token_url.as_str(), DEFAULT_TOKEN_URL);
        assert_eq!(config.client_id, DEFAULT_CLIENT_ID);
        assert!(
            !config.allow_http_loopback,
            "remote HTTP must stay forbidden by default"
        );
    }

    /// Form bodies are `application/x-www-form-urlencoded` from the generic
    /// transport; decode into pairs for assertions.
    fn form_body(raw: &[u8]) -> Vec<(String, String)> {
        url::form_urlencoded::parse(raw)
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect()
    }
}
