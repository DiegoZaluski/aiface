//! RFC 8628 device grant. Expiry is required, polling is deadline bounded,
//! slow_down adds five seconds, and transport timeouts double the interval.
use super::{
    config::{OAuthProviderConfig, verification_url},
    error::{OAuthError, Result},
    http,
    token::{Secret, TokenSet},
};
use serde::Deserialize;
use std::time::{Duration, Instant};
use url::Url;

pub struct DeviceAuthorization {
    device_code: Secret,
    pub user_code: String,
    pub verification_uri: Url,
    pub verification_uri_complete: Option<Url>,
    interval: Duration,
    deadline: Instant,
    config: OAuthProviderConfig,
}
impl std::fmt::Debug for DeviceAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceAuthorization")
            .field("interval", &self.interval)
            .finish_non_exhaustive()
    }
}
#[derive(Deserialize)]
struct DeviceAuthorizationResponse {
    device_code: Secret,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    #[serde(default)]
    interval: Option<u64>,
}
pub async fn begin_device_authorization(
    config: &OAuthProviderConfig,
) -> Result<DeviceAuthorization> {
    config.validate()?;
    let endpoint = config
        .device_authorization_url
        .as_ref()
        .ok_or_else(|| OAuthError::InvalidConfig("missing device authorization endpoint".into()))?;
    let started = Instant::now();
    let scope = config.scope_param();
    let mut form = vec![("client_id", config.client_id.as_str())];
    if let Some(scope) = &scope {
        form.push(("scope", scope));
    }
    let (status, bytes) = http::post_form(endpoint, &form, config.allow_http_loopback).await?;
    if !(200..300).contains(&status) {
        return Err(http::classify_error(status, &bytes));
    }
    let body: DeviceAuthorizationResponse = http::parse_json(&bytes)?;
    if body.device_code.expose().is_empty() || body.user_code.is_empty() {
        return Err(OAuthError::MalformedResponse(
            "missing device or user code".into(),
        ));
    }
    let deadline = device_deadline(started, body.expires_in)?;
    let interval = Duration::from_secs(body.interval.unwrap_or(5).max(1));
    Ok(DeviceAuthorization {
        device_code: body.device_code,
        user_code: body.user_code,
        verification_uri: verification_url(&body.verification_uri, config.allow_http_loopback)?,
        verification_uri_complete: body
            .verification_uri_complete
            .map(|u| verification_url(&u, config.allow_http_loopback))
            .transpose()?,
        interval,
        deadline,
        config: config.clone(),
    })
}
pub(crate) fn device_deadline(started: Instant, seconds: u64) -> Result<Instant> {
    let deadline = started
        .checked_add(Duration::from_secs(seconds))
        .ok_or_else(|| OAuthError::MalformedResponse("device lifetime overflow".into()))?;
    if Instant::now() >= deadline {
        return Err(OAuthError::DeviceExpired);
    }
    Ok(deadline)
}
pub(crate) async fn wait_to_poll(deadline: Instant, interval: Duration) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(OAuthError::DeviceExpired);
    }
    tokio::time::sleep(interval.min(remaining)).await;
    if Instant::now() >= deadline {
        return Err(OAuthError::DeviceExpired);
    }
    Ok(())
}
pub(crate) fn backoff(interval: Duration, slow_down: bool) -> Result<Duration> {
    (if slow_down {
        interval.checked_add(Duration::from_secs(5))
    } else {
        interval.checked_mul(2)
    })
    .ok_or_else(|| OAuthError::MalformedResponse("poll interval overflow".into()))
}
pub async fn poll_device_token(auth: DeviceAuthorization) -> Result<TokenSet> {
    let mut interval = auth.interval;
    loop {
        wait_to_poll(auth.deadline, interval).await?;
        let form = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ("device_code", auth.device_code.expose()),
            ("client_id", auth.config.client_id.as_str()),
        ];
        let response = tokio::time::timeout_at(
            auth.deadline.into(),
            http::post_form(
                &auth.config.token_url,
                &form,
                auth.config.allow_http_loopback,
            ),
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
        // The BODY is authoritative for 2xx too: several servers (GitHub
        // included) answer the pending and slow-down states with HTTP 200
        // and an `error` member, instead of the 4xx the RFC sketches — so
        // classifying the body FIRST is what keeps the loop alive. A token
        // body carries no `error` member, so the classify pass yields no
        // code and falls through to the token parse.
        let error = http::classify_error(status, &bytes);
        match error.error_code() {
            Some("authorization_pending") => {}
            Some("slow_down") => interval = backoff(interval, true)?,
            Some("access_denied") => return Err(OAuthError::DeviceDenied),
            Some("expired_token") => return Err(OAuthError::DeviceExpired),
            // Any other protocol error (2xx or not) is terminal…
            Some(_) => return Err(error),
            // …and no error member means either a token body (2xx) or a
            // bare HTTP failure (non-2xx) — the status decides.
            None if (200..300).contains(&status) => {
                let mut tokens = http::into_token_set(http::parse_json(&bytes)?)?;
                tokens.inherit_scopes(&auth.config.scopes);
                return Ok(tokens);
            }
            None => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polling_backoff_is_persistent_and_checked() {
        let interval = backoff(Duration::from_secs(5), true).unwrap();
        assert_eq!(interval, Duration::from_secs(10));
        assert_eq!(backoff(interval, true).unwrap(), Duration::from_secs(15));
        assert_eq!(backoff(interval, false).unwrap(), Duration::from_secs(20));
        assert!(backoff(Duration::MAX, true).is_err());
        assert!(backoff(Duration::MAX, false).is_err());
        assert!(device_deadline(Instant::now(), u64::MAX).is_err());
    }
    #[tokio::test]
    async fn device_lifetime_required_and_verification_uri_checked() {
        for suffix in [
            "",
            ",\"expires_in\":0",
            ",\"expires_in\":18446744073709551615",
        ] {
            let body = format!(
                "{{\"device_code\":\"D\",\"user_code\":\"U\",\"verification_uri\":\"https://provider.example/verify\"{suffix}}}"
            );
            let (url, server) = super::super::test::support::serve(vec![(200, body)]).await;
            let mut config = super::super::test::support::config(url.clone());
            config.device_authorization_url = Some(url);
            assert!(begin_device_authorization(&config).await.is_err());
            server.await.unwrap();
        }
        for url in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "http://evil.example",
            "https://user:password@provider.example/",
            "https://provider.example/verify#frag",
        ] {
            assert!(verification_url(url, false).is_err(), "{url}");
        }
        // Loopback HTTP is the one documented exception, and only when the
        // caller has explicitly opted in for local fixtures.
        assert!(verification_url("http://127.0.0.1:9999/verify", false).is_err());
        assert!(verification_url("http://127.0.0.1:9999/verify", true).is_ok());
        assert!(verification_url("http://[::1]:9999/verify", true).is_ok());
        assert!(verification_url("http://localhost:9999/verify", true).is_err());
    }

    /// `verification_uri_complete` is a second browser-bound URL and must be
    /// held to the same standard as `verification_uri` (#23).
    #[tokio::test]
    async fn verification_uri_complete_is_validated_like_the_primary_uri() {
        for complete in [
            "javascript:alert(1)",
            "http://evil.example/verify?user_code=USER",
            "https://user:password@provider.example/verify",
        ] {
            let body = format!(
                r#"{{"device_code":"D","user_code":"U","verification_uri":"https://provider.example/verify","verification_uri_complete":"{complete}","expires_in":900}}"#
            );
            let (url, server) = super::super::test::support::serve(vec![(200, body)]).await;
            let mut config = super::super::test::support::config(url.clone());
            config.device_authorization_url = Some(url);
            assert!(
                begin_device_authorization(&config).await.is_err(),
                "{complete}"
            );
            server.await.unwrap();
        }
    }
    #[tokio::test]
    async fn never_polls_after_expiry_or_waits_beyond_deadline_for_http() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/token", listener.local_addr().unwrap())).unwrap();
        let make_auth = |interval| DeviceAuthorization {
            device_code: Secret::new("D"),
            user_code: "U".into(),
            verification_uri: Url::parse("https://provider.example/verify").unwrap(),
            verification_uri_complete: None,
            interval,
            deadline: Instant::now()
                .checked_add(Duration::from_millis(100))
                .unwrap(),
            config: super::super::test::support::config(url.clone()),
        };
        assert!(matches!(
            poll_device_token(make_auth(Duration::from_secs(5))).await,
            Err(OAuthError::DeviceExpired)
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), listener.accept())
                .await
                .is_err()
        );
        let started = Instant::now();
        assert!(matches!(
            poll_device_token(make_auth(Duration::from_millis(1))).await,
            Err(OAuthError::DeviceExpired)
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
        // The server never replied; the device deadline still capped the request.
        assert!(
            tokio::time::timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_ok()
        );
    }

    /// REGRESSION (GitHub's device flow): the token endpoint answers the
    /// PENDING state with HTTP 200 and `{"error":"authorization_pending"}`
    /// in the body — not the 4xx the RFC sketches. Classifying only on the
    /// status killed the loop on the first poll (before the user had even
    /// approved in the browser), so the flow reported failure while GitHub
    /// showed a completed authorization. The BODY must be authoritative
    /// for 2xx too: pending keeps polling, the token body resolves.
    #[tokio::test]
    async fn two_hundred_with_error_body_keeps_polling_until_the_token() {
        let pending = r#"{"error":"authorization_pending","error_description":"The authorization request is still pending."}"#.to_string();
        let slow_down =
            r#"{"error":"slow_down","error_description":"Too many requests.","interval":1}"#
                .to_string();
        let token =
            r#"{"access_token":"gho_test","token_type":"bearer","scope":"read:user"}"#.to_string();
        let (url, server) = super::super::test::support::serve(vec![
            (200, pending),
            (200, slow_down),
            (200, token),
        ])
        .await;
        let auth = DeviceAuthorization {
            device_code: Secret::new("D"),
            user_code: "U".into(),
            verification_uri: Url::parse("https://provider.example/verify").unwrap(),
            verification_uri_complete: None,
            interval: Duration::from_millis(1),
            deadline: Instant::now().checked_add(Duration::from_secs(10)).unwrap(),
            config: super::super::test::support::config(url),
        };
        let tokens = poll_device_token(auth).await.unwrap();
        assert_eq!(tokens.access_token().expose(), "gho_test");
        let requests = server.await.unwrap();
        assert_eq!(
            requests.len(),
            3,
            "pending and slow_down must keep the loop alive"
        );
        for request in &requests {
            assert!(
                request
                    .body
                    .windows(b"device_code=D".len())
                    .any(|w| w == b"device_code=D"),
                "every poll must carry the fixture's device code grant"
            );
        }
    }
}
