//! OAuth transport: no redirects/retries, explicit deadlines and streaming
//! caps. Module-owned request/response buffers zeroize on their final drop.
//! HTTP/TLS/parser internals can still make copies outside this module.
use super::{
    config::validate_endpoint,
    error::{OAuthError, Result},
    token::{Secret, TokenSet},
};
use hyper::body::Bytes;
use serde::{Deserialize, Serialize};
use std::{sync::OnceLock, time::Duration};
use zeroize::Zeroizing;

pub(crate) const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
fn client() -> Result<&'static reqwest::Client> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client);
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let _ = CLIENT.set(client);
    CLIENT
        .get()
        .ok_or_else(|| OAuthError::InvalidConfig("HTTP client initialization failed".into()))
}
#[derive(Deserialize)]
struct OAuthErrorBody {
    error: Secret,
    #[serde(default)]
    error_description: Option<Secret>,
}
#[derive(Deserialize)]
pub(crate) struct TokenResponseBody {
    pub access_token: Secret,
    pub token_type: Secret,
    #[serde(default)]
    pub expires_in: Option<u64>,
    #[serde(default)]
    pub refresh_token: Option<Secret>,
    #[serde(default)]
    pub scope: Option<String>,
}
struct WipeBody(Zeroizing<Vec<u8>>);
impl AsRef<[u8]> for WipeBody {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl std::io::Write for WipeBody {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() > MAX_RESPONSE_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("OAuth request exceeds size cap"));
        }
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
async fn send(request: reqwest::RequestBuilder) -> Result<(u16, Zeroizing<Vec<u8>>)> {
    let response = request
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await?;
    let status = response.status().as_u16();
    Ok((status, read_capped(response).await?))
}
pub(crate) async fn post_form(
    url: &url::Url,
    form: &[(&str, &str)],
    allow_loopback: bool,
) -> Result<(u16, Zeroizing<Vec<u8>>)> {
    validate_endpoint(url, allow_loopback)?;
    let capacity = form
        .iter()
        .try_fold(0usize, |n, (k, v)| {
            n.checked_add(k.len())?.checked_add(v.len())?.checked_add(2)
        })
        .and_then(|n| n.checked_mul(3))
        .filter(|n| *n <= MAX_RESPONSE_BYTES)
        .ok_or_else(|| OAuthError::InvalidConfig("OAuth request exceeds size cap".into()))?;
    let mut encoded = Zeroizing::new(String::with_capacity(capacity));
    url::form_urlencoded::Serializer::new(&mut *encoded).extend_pairs(form.iter().copied());
    // Move the allocation into reqwest's shared body owner; no plaintext copy.
    let bytes = std::mem::take(&mut *encoded).into_bytes();
    send(
        client()?
            .post(url.clone())
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(Bytes::from_owner(WipeBody(Zeroizing::new(bytes)))),
    )
    .await
}
pub(crate) async fn post_token_form(
    url: &url::Url,
    form: &[(&str, &str)],
    allow_loopback: bool,
) -> Result<TokenResponseBody> {
    let (status, bytes) = post_form(url, form, allow_loopback).await?;
    if !(200..300).contains(&status) {
        return Err(classify_error(status, &bytes));
    }
    let body = parse_json(&bytes)?;
    validate_token_body(&body)?;
    Ok(body)
}
/// POST a JSON body to a provider-specific endpoint. Consumed by the
/// JSON-based proprietary device-protocol adapter (a single file inside its
/// family directory); unreachable from the generic flows.
pub(crate) async fn post_json<T: Serialize + ?Sized>(
    url: &url::Url,
    body: &T,
    allow_loopback: bool,
) -> Result<(u16, Zeroizing<Vec<u8>>)> {
    validate_endpoint(url, allow_loopback)?;
    let mut encoded = WipeBody(Zeroizing::new(Vec::with_capacity(MAX_RESPONSE_BYTES)));
    serde_json::to_writer(&mut encoded, body).map_err(|_| {
        OAuthError::InvalidConfig("cannot encode OAuth JSON request within size cap".into())
    })?;
    send(
        client()?
            .post(url.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(Bytes::from_owner(encoded)),
    )
    .await
}
pub(crate) async fn get_json(
    url: &url::Url,
    allow_loopback: bool,
) -> Result<(u16, Zeroizing<Vec<u8>>)> {
    validate_endpoint(url, allow_loopback)?;
    send(client()?.get(url.clone())).await
}
async fn read_capped(mut response: reqwest::Response) -> Result<Zeroizing<Vec<u8>>> {
    fn too_large() -> OAuthError {
        OAuthError::MalformedResponse("response exceeds 64 KiB cap".into())
    }
    if response
        .content_length()
        .is_some_and(|len| len > MAX_RESPONSE_BYTES as u64)
    {
        return Err(too_large());
    }
    let mut body = Zeroizing::new(Vec::with_capacity(MAX_RESPONSE_BYTES));
    while let Some(chunk) = response.chunk().await? {
        if chunk.len() > MAX_RESPONSE_BYTES - body.len() {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}
pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    // serde's error can quote attacker-controlled values. Never forward it.
    serde_json::from_slice(bytes)
        .map_err(|_| OAuthError::MalformedResponse("invalid JSON or field type".into()))
}
pub(crate) fn classify_error(status: u16, bytes: &[u8]) -> OAuthError {
    match serde_json::from_slice::<OAuthErrorBody>(bytes) {
        Ok(body) => OAuthError::token_endpoint(
            status,
            body.error.expose().into(),
            body.error_description.map(|s| s.expose().into()),
        ),
        Err(_) => OAuthError::UnexpectedHttpStatus { status },
    }
}
pub(crate) fn validate_token_body(body: &TokenResponseBody) -> Result<()> {
    if body.access_token.expose().is_empty()
        || body
            .refresh_token
            .as_ref()
            .is_some_and(|s| s.expose().is_empty())
    {
        return Err(OAuthError::MalformedResponse("empty token".into()));
    }
    if !body.token_type.expose().eq_ignore_ascii_case("bearer") {
        return Err(OAuthError::MalformedResponse(
            "unsupported token type; Bearer required".into(),
        ));
    }
    Ok(())
}
pub(crate) fn into_token_set(body: TokenResponseBody) -> Result<TokenSet> {
    validate_token_body(&body)?;
    let expiry = body.expires_in.map(TokenSet::expiry_from_now).transpose()?;
    let omitted = body.scope.is_none();
    let scopes = body
        .scope
        .map(|s| {
            s.split(' ')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let mut tokens = TokenSet::new(body.access_token, body.refresh_token, expiry, scopes);
    if omitted {
        tokens.scopes_omitted();
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    #[test]
    fn lifetime_zero_absent_overflow_and_invalid_types_are_distinct() {
        for (field, expected) in [
            ("", "unknown"),
            (",\"expires_in\":0", "expired"),
            (",\"expires_in\":18446744073709551615", "error"),
            (",\"expires_in\":-1", "error"),
            (",\"expires_in\":\"secret\"", "error"),
        ] {
            let raw = format!("{{\"access_token\":\"a\",\"token_type\":\"Bearer\"{field}}}");
            let tokens = parse_json(&raw.into_bytes()).and_then(into_token_set);
            match expected {
                "unknown" => assert!(tokens.unwrap().expires_at().is_none()),
                "expired" => assert!(tokens.unwrap().is_expired()),
                _ => assert!(tokens.is_err()),
            }
        }
    }
    #[tokio::test]
    async fn chunked_body_is_rejected_before_end_of_stream() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = url::Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            super::super::test::support::read_request(&mut stream).await;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            for _ in 0..17 {
                if stream.write_all(b"1000\r\n").await.is_err() {
                    return;
                }
                if stream.write_all(&[b'x'; 4096]).await.is_err() {
                    return;
                }
                if stream.write_all(b"\r\n").await.is_err() {
                    return;
                }
            }
            // Deliberately never terminate the HTTP body. collect()/bytes()
            // would hang; streaming cap must reject without waiting for EOF.
            std::future::pending::<()>().await;
        });
        let error = tokio::time::timeout(Duration::from_secs(2), get_json(&url, true))
            .await
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, OAuthError::MalformedResponse(_)));
        server.abort();
    }
    #[tokio::test]
    async fn declared_oversize_body_is_rejected_without_reading_it() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = url::Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            super::super::test::support::read_request(&mut stream).await;
            // Content-Length above the cap, but the body never arrives.
            // The cap must be enforced from the header alone.
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 65537\r\n\r\n")
                .await
                .unwrap();
            std::future::pending::<()>().await;
        });
        let error = tokio::time::timeout(Duration::from_secs(2), get_json(&url, true))
            .await
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, OAuthError::MalformedResponse(_)));
        server.abort();
    }
    #[tokio::test]
    async fn transport_conversion_strips_urls_and_source_chains() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let raw = reqwest::Client::new()
            .get(format!("http://{addr}/device/auth/TOP_SECRET"))
            .send()
            .await
            .unwrap_err();
        assert!(raw.url().is_some());
        let err = OAuthError::from(raw);
        assert!(!format!("{err:?} {err}").contains("TOP_SECRET"));
        assert!(std::error::Error::source(&err).is_none());
    }
    #[tokio::test]
    async fn oauth_client_does_not_follow_redirects() {
        let (url, server) = super::super::test::support::serve(vec![(302, "ignored".into())]).await;
        let (status, _) = post_form(&url, &[("refresh_token", "R1")], true)
            .await
            .unwrap();
        assert_eq!(status, 302);
        assert_eq!(server.await.unwrap().len(), 1);
    }
    #[test]
    fn provider_errors_and_token_types_are_never_formatted() {
        let error = OAuthError::token_endpoint(
            400,
            "SECRET\nCODE".into(),
            Some("\u{1b}[31mSECRET_DESCRIPTION".repeat(1000)),
        );
        assert_eq!(error.error_code(), Some("invalid_provider_error"));
        assert!(!format!("{error:?} {error}").contains("SECRET"));
        if let OAuthError::TokenEndpoint {
            description: Some(description),
            ..
        } = error
        {
            assert!(description.chars().count() <= 512);
            assert!(!description.chars().any(char::is_control));
        } else {
            panic!("missing description");
        }
        let body: TokenResponseBody =
            parse_json(br#"{"access_token":"A","token_type":"SECRET_TOKEN_TYPE"}"#).unwrap();
        let error = validate_token_body(&body).unwrap_err();
        assert!(!format!("{error:?} {error}").contains("SECRET_TOKEN_TYPE"));
    }
}
