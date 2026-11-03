//! Charm Hyper device-authentication adapter.
//!
//! Charm Hyper does NOT publish a standard OAuth contract for third-party
//! CLIs: the flow implemented by Charm's own CLI (crush,
//! `internal/oauth/hyper/device.go`) is a proprietary device-authorization
//! protocol over `https://hyper.charm.land`:
//!
//! | Step | Endpoint | Shape |
//! |------|----------|-------|
//! | Start | `POST /device/auth` | JSON `{"device_name": ...}` → `device_code`, `user_code`, `verification_url`, `expires_in` |
//! | Poll | `GET /device/auth/{device_code}` | JSON; `authorization_pending` until approved, then `refresh_token` |
//! | Exchange | `POST /token/exchange` | JSON `{"refresh_token": ...}` → access token |
//!
//! Note the differences from RFC 8628: JSON bodies instead of form encoding,
//! GET-based polling with the device code in the path, and the poll returns a
//! *refresh* token that must be exchanged separately. This module therefore
//! does NOT reuse the generic device flow — it is a provider-specific adapter
//! built on the same hardened HTTP transport and the same token model.
//!
//! Because there is no public third-party contract, the base URL is
//! configurable (Charm may change the protocol; the crush CLI itself honors a
//! `HYPER_URL` override).

use crate::connector::oauth::{
    OAuthError, Result, Secret, TokenRefresher, TokenSet,
    config::{validate_endpoint, verification_url},
    device::{backoff, device_deadline, wait_to_poll},
    http,
};
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    pin::Pin,
    time::{Duration, Instant, SystemTime},
};
use url::Url;
pub const DEFAULT_BASE_URL: &str = "https://hyper.charm.land";
const POLL_INTERVAL: Duration = Duration::from_secs(5);
#[derive(Clone)]
pub struct CharmOAuthConfig {
    pub base_url: Url,
    pub device_name: String,
    /// Only enable for local HTTP fixtures; never permits remote HTTP.
    pub allow_http_loopback: bool,
}
impl std::fmt::Debug for CharmOAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharmOAuthConfig").finish_non_exhaustive()
    }
}
impl Default for CharmOAuthConfig {
    fn default() -> Self {
        Self {
            // Safe invariant: DEFAULT_BASE_URL is a compile-time HTTPS literal
            // whose parseability is exercised by `config_and_refresher_enforce_https`.
            base_url: Url::parse(DEFAULT_BASE_URL).expect("static HTTPS URL"),
            device_name: "cosh".into(),
            allow_http_loopback: false,
        }
    }
}
impl CharmOAuthConfig {
    pub fn validate(&self) -> Result<()> {
        validate_endpoint(&self.base_url, self.allow_http_loopback)?;
        if self.base_url.query().is_some() || self.device_name.is_empty() {
            return Err(OAuthError::InvalidConfig(
                "Charm base URL cannot contain a query; device name must be nonempty".into(),
            ));
        }
        Ok(())
    }
    fn endpoint(&self, path: &str) -> Result<Url> {
        self.validate()?;
        self.base_url
            .join(path)
            .map_err(|_| OAuthError::InvalidConfig("invalid Charm endpoint".into()))
    }
}
pub struct CharmDeviceAuthorization {
    device_code: Secret,
    pub user_code: String,
    pub verification_url: Url,
    deadline: Instant,
    config: CharmOAuthConfig,
}
impl std::fmt::Debug for CharmDeviceAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharmDeviceAuthorization")
            .finish_non_exhaustive()
    }
}
impl CharmDeviceAuthorization {
    /// Absolute instant at which this device transaction expires — for a
    /// countdown UI, parallel to [`crate::connector::oauth::TokenSession::expires_at`].
    pub fn deadline(&self) -> Instant {
        self.deadline
    }
}
#[derive(Deserialize)]
struct DeviceAuthStart {
    device_code: Secret,
    user_code: String,
    verification_url: String,
    expires_in: u64,
}
#[derive(Deserialize)]
struct DeviceAuthPoll {
    #[serde(default)]
    refresh_token: Option<Secret>,
    #[serde(default)]
    error: Option<Secret>,
    #[serde(default)]
    error_description: Option<Secret>,
}
#[derive(Deserialize)]
struct TokenExchangeResponse {
    access_token: Secret,
    #[serde(default)]
    refresh_token: Option<Secret>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    expires_at: Option<i64>,
}
pub async fn begin(config: &CharmOAuthConfig) -> Result<CharmDeviceAuthorization> {
    config.validate()?;
    let started = Instant::now();
    #[derive(Serialize)]
    struct Start<'a> {
        device_name: &'a str,
    }
    let (status, bytes) = http::post_json(
        &config.endpoint("/device/auth")?,
        &Start {
            device_name: &config.device_name,
        },
        config.allow_http_loopback,
    )
    .await?;
    if !(200..300).contains(&status) {
        return Err(charm_error(status, &bytes));
    }
    let body: DeviceAuthStart = http::parse_json(&bytes)?;
    if body.device_code.expose().is_empty() || body.user_code.is_empty() {
        return Err(OAuthError::MalformedResponse(
            "missing Charm device or user code".into(),
        ));
    }
    Ok(CharmDeviceAuthorization {
        device_code: body.device_code,
        user_code: body.user_code,
        verification_url: verification_url(&body.verification_url, config.allow_http_loopback)?,
        deadline: device_deadline(started, body.expires_in)?,
        config: config.clone(),
    })
}
pub async fn poll_and_exchange(auth: CharmDeviceAuthorization) -> Result<TokenSet> {
    let refresh_token = poll_for_refresh_token(&auth).await?;
    // The exchange helper preserves R1 even on this initial authorization.
    // The exchange is bounded by the same device deadline as the poll: a slow
    // provider must not extend the transaction past its advertised lifetime.
    match tokio::time::timeout_at(
        auth.deadline.into(),
        exchange_refresh_token(&auth.config, &refresh_token),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(OAuthError::DeviceExpired),
    }
}
async fn poll_for_refresh_token(auth: &CharmDeviceAuthorization) -> Result<Secret> {
    let mut interval = POLL_INTERVAL;
    let mut endpoint = auth.config.endpoint("/device/auth/")?;
    // The proprietary protocol requires the secret in the path. Encode one
    // segment, never interpolate a code that could inject a query or path.
    endpoint
        .path_segments_mut()
        .map_err(|_| OAuthError::InvalidConfig("invalid Charm base URL".into()))?
        .pop_if_empty()
        .push(auth.device_code.expose());
    loop {
        wait_to_poll(auth.deadline, interval).await?;
        let response = tokio::time::timeout_at(
            auth.deadline.into(),
            http::get_json(&endpoint, auth.config.allow_http_loopback),
        )
        .await
        .map_err(|_| OAuthError::DeviceExpired)?;
        let (status, bytes) = match response {
            Err(error) if error.is_timeout() => {
                interval = backoff(interval, false)?;
                continue;
            }
            result => result?,
        };
        if !(200..300).contains(&status) {
            let error = charm_error(status, &bytes);
            match error.error_code() {
                Some("authorization_pending") => continue,
                Some("slow_down") => {
                    interval = backoff(interval, true)?;
                    continue;
                }
                Some("access_denied") => return Err(OAuthError::DeviceDenied),
                Some("expired_token") => return Err(OAuthError::DeviceExpired),
                _ => return Err(error),
            }
        }
        let body: DeviceAuthPoll = http::parse_json(&bytes)?;
        match (body.refresh_token, body.error) {
            (Some(token), None) if !token.expose().is_empty() => return Ok(token),
            (None, Some(error)) => match error.expose() {
                "authorization_pending" => {}
                "slow_down" => interval = backoff(interval, true)?,
                "access_denied" => return Err(OAuthError::DeviceDenied),
                "expired_token" => return Err(OAuthError::DeviceExpired),
                _ => {
                    return Err(OAuthError::token_endpoint(
                        status,
                        error.expose().into(),
                        body.error_description.map(|s| s.expose().into()),
                    ));
                }
            },
            _ => {
                return Err(OAuthError::MalformedResponse(
                    "ambiguous or empty Charm polling response".into(),
                ));
            }
        }
    }
}
async fn exchange_refresh_token(
    config: &CharmOAuthConfig,
    refresh_token: &Secret,
) -> Result<TokenSet> {
    config.validate()?;
    #[derive(Serialize)]
    struct Exchange<'a> {
        refresh_token: &'a str,
    }
    let (status, bytes) = http::post_json(
        &config.endpoint("/token/exchange")?,
        &Exchange {
            refresh_token: refresh_token.expose(),
        },
        config.allow_http_loopback,
    )
    .await?;
    if !(200..300).contains(&status) {
        return Err(charm_error(status, &bytes));
    }
    let body: TokenExchangeResponse = http::parse_json(&bytes)?;
    charm_tokens(body, refresh_token)
}
fn charm_tokens(body: TokenExchangeResponse, previous_refresh: &Secret) -> Result<TokenSet> {
    if body.access_token.expose().is_empty() {
        return Err(OAuthError::MalformedResponse(
            "empty Charm access token".into(),
        ));
    }
    // crush oauth.Token.SetExpiresAt: positive relative expiry wins, else
    // positive absolute expiry, else epoch (expired). Overflow is an error.
    let expiry = if let Some(seconds) = body.expires_in.filter(|s| *s > 0) {
        TokenSet::expiry_from_now(seconds as u64)?
    } else if let Some(seconds) = body.expires_at.filter(|s| *s > 0) {
        SystemTime::UNIX_EPOCH
            .checked_add(Duration::from_secs(seconds as u64))
            .ok_or_else(|| OAuthError::MalformedResponse("Charm expiry overflow".into()))?
    } else {
        SystemTime::UNIX_EPOCH
    };
    let mut tokens = TokenSet::new(
        body.access_token,
        body.refresh_token.filter(|s| !s.expose().is_empty()),
        Some(expiry),
        vec![],
    );
    tokens.inherit_refresh(Some(previous_refresh));
    tokens.scopes_omitted();
    Ok(tokens)
}
fn charm_error(status: u16, bytes: &[u8]) -> OAuthError {
    let error = http::classify_error(status, bytes);
    // Only structured machine codes establish revocation. HTTP 400/401/403
    // alone may mean invalid_client, invalid_request, or proxy denial.
    //
    // Deliberate divergence from crush's `TokenExchangeError.IsRefreshTokenRevoked`
    // (internal/oauth/token.go), which substring-matches `revoked`/`invalid_grant`
    // anywhere in the raw body. Any field there — including an attacker-influenced
    // `error_description` — would flip the verdict, so we require the parsed
    // `error` code instead. Being stricter than the reference CLI only ever
    // costs an extra user reauthorization, never a wrongfully accepted token.
    match error.error_code() {
        Some("invalid_grant" | "refresh_token_revoked" | "token_revoked" | "revoked") => {
            OAuthError::RefreshTokenInvalid
        }
        _ => error,
    }
}
pub struct CharmRefresher {
    config: CharmOAuthConfig,
}
impl CharmRefresher {
    pub fn new(config: CharmOAuthConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }
}
impl TokenRefresher for CharmRefresher {
    fn refresh<'a>(
        &'a self,
        refresh_token: &'a Secret,
    ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>> {
        Box::pin(async move { exchange_refresh_token(&self.config, refresh_token).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crush_expiry_fallback_and_unknown_expired_policy() {
        for fields in [
            "",
            ",\"expires_in\":0",
            ",\"expires_in\":-1,\"expires_at\":0",
        ] {
            let raw = format!("{{\"access_token\":\"A\"{fields}}}");
            let tokens = charm_tokens(
                http::parse_json(raw.as_bytes()).unwrap(),
                &Secret::new("R1"),
            )
            .unwrap();
            assert!(tokens.is_expired());
            assert_eq!(tokens.refresh_token().unwrap().expose(), "R1");
        }
        for relative in [0, -1] {
            let raw = format!(
                "{{\"access_token\":\"A\",\"expires_in\":{relative},\"expires_at\":2000000000}}"
            );
            let tokens = charm_tokens(
                http::parse_json(raw.as_bytes()).unwrap(),
                &Secret::new("R1"),
            )
            .unwrap();
            assert_eq!(tokens.to_stored().expires_at_unix, Some(2000000000));
        }
        let body =
            http::parse_json(br#"{"access_token":"A","expires_in":3600,"expires_at":1}"#).unwrap();
        assert!(!charm_tokens(body, &Secret::new("R1")).unwrap().is_expired());
        let body =
            http::parse_json(br#"{"access_token":"A","expires_in":9223372036854775807}"#).unwrap();
        assert!(charm_tokens(body, &Secret::new("R1")).is_err());
    }
    #[test]
    fn errors_are_classified_by_structured_body_not_http_status_or_prose() {
        for status in [400, 401, 403, 500] {
            assert!(matches!(
                charm_error(status, br#"{"error":"invalid_grant"}"#),
                OAuthError::RefreshTokenInvalid
            ));
            let error = charm_error(status,br#"{"error":"invalid_client","error_description":"not revoked: invalid_grant appears here"}"#);
            assert_eq!(error.error_code(), Some("invalid_client"));
            assert!(!error.requires_reauthorization());
            assert!(matches!(
                charm_error(status, b"revoked"),
                OAuthError::UnexpectedHttpStatus { .. }
            ));
        }
    }
    #[tokio::test]
    async fn poll_and_exchange_preserves_initial_refresh_and_matches_crush_wire_format() {
        let (url,server) = crate::connector::oauth::test::support::serve(vec![
            (200,r#"{"device_code":"device/?secret","user_code":"USER","verification_url":"https://hyper.charm.land/verify","expires_in":30}"#.into()),
            (200,r#"{"refresh_token":"R1"}"#.into()),
            (200,r#"{"access_token":"A1","expires_in":3600}"#.into()),
        ]).await;
        let config = CharmOAuthConfig {
            base_url: url,
            device_name: "cosh-fixture".into(),
            allow_http_loopback: true,
        };
        let auth = begin(&config).await.unwrap();
        let tokens = tokio::time::timeout(Duration::from_secs(10), poll_and_exchange(auth))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tokens.refresh_token().unwrap().expose(), "R1");
        let requests = server.await.unwrap();
        assert!(requests[0].head.starts_with("POST /device/auth HTTP/1.1"));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&requests[0].body).unwrap()["device_name"],
            "cosh-fixture"
        );
        assert!(
            requests[1]
                .head
                .starts_with("GET /device/auth/device%2F%3Fsecret HTTP/1.1")
        );
        assert!(
            requests[2]
                .head
                .starts_with("POST /token/exchange HTTP/1.1")
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&requests[2].body).unwrap()["refresh_token"],
            "R1"
        );
    }
    /// A poll response with neither `refresh_token` nor `error` cannot mean
    /// "keep waiting": treating it as pending would spin until the device
    /// deadline on a provider that already finished. Reject the response
    /// instead of continuing to poll (#24).
    #[tokio::test]
    async fn poll_response_without_token_or_error_stops_polling() {
        let (url, server) = crate::connector::oauth::test::support::serve(vec![
            (
                200,
                r#"{"device_code":"D","user_code":"USER","verification_url":"https://hyper.charm.land/verify","expires_in":30}"#
                    .into(),
            ),
            (200, r#"{"user_id":"u","organization_id":"o"}"#.into()),
        ])
        .await;
        let config = CharmOAuthConfig {
            base_url: url,
            device_name: "cosh-fixture".into(),
            allow_http_loopback: true,
        };
        let auth = begin(&config).await.unwrap();
        let error = tokio::time::timeout(Duration::from_secs(20), poll_and_exchange(auth))
            .await
            .expect("polling must stop on an ambiguous response, not run to the deadline")
            .unwrap_err();
        assert!(
            matches!(error, OAuthError::MalformedResponse(_)),
            "{error:?}"
        );
        // Exactly two requests: start + one poll. The fixture closes after
        // the second, so a third accept would hang the join.
        assert_eq!(server.await.unwrap().len(), 2);
    }

    /// A response carrying both a token and an error is equally unusable and
    /// must not be resolved in either direction (#24).
    #[tokio::test]
    async fn poll_response_with_both_token_and_error_is_malformed() {
        let (url, server) = crate::connector::oauth::test::support::serve(vec![
            (
                200,
                r#"{"device_code":"D","user_code":"USER","verification_url":"https://hyper.charm.land/verify","expires_in":30}"#
                    .into(),
            ),
            (200, r#"{"refresh_token":"R1","error":"access_denied"}"#.into()),
        ])
        .await;
        let config = CharmOAuthConfig {
            base_url: url,
            device_name: "cosh-fixture".into(),
            allow_http_loopback: true,
        };
        let auth = begin(&config).await.unwrap();
        let error = tokio::time::timeout(Duration::from_secs(20), poll_and_exchange(auth))
            .await
            .expect("polling must stop on a self-contradictory response")
            .unwrap_err();
        assert!(
            matches!(error, OAuthError::MalformedResponse(_)),
            "{error:?}"
        );
        assert_eq!(server.await.unwrap().len(), 2);
    }

    #[test]
    fn config_and_refresher_enforce_https() {
        let mut config = CharmOAuthConfig {
            base_url: Url::parse("http://evil.example").unwrap(),
            ..Default::default()
        };
        assert!(CharmRefresher::new(config.clone()).is_err());
        config.allow_http_loopback = true;
        assert!(config.validate().is_err());
        config.base_url = Url::parse("http://[::1]:1234").unwrap();
        assert!(config.validate().is_ok());
        config.allow_http_loopback = false;
        assert!(config.validate().is_err());
    }
}
