//! Authorization Code + PKCE flow with a loopback redirect receiver.
//!
//! Orchestration (RFC 8252 §4.1, RFC 7636, RFC 9700):
//!
//! 1. [`begin_authorization`] binds `127.0.0.1:0` (ephemeral port), generates
//!    a fresh PKCE pair and CSRF `state`, and returns a
//!    [`PendingAuthorization`] carrying the URL to open in the system
//!    browser.
//! 2. [`PendingAuthorization::wait`] serves exactly one transaction on that
//!    listener: incidental requests (favicon, wrong path, mismatched `state`)
//!    get answers but never consume the transaction; the first request with a
//!    matching `state` ends it — with either the code or the provider's
//!    `error` parameters.
//! 3. [`AuthorizationCode::exchange`] POSTs the code + verifier to the token
//!    endpoint and yields a validated [`TokenSet`].
//!
//! [`authorize`] composes the three steps for the common case.

use std::net::SocketAddr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::{Notify, mpsc};
use url::Url;

use super::config::OAuthProviderConfig;
use super::error::{OAuthError, Result};
use super::http;
use super::pkce::{PkcePair, generate_state};
use super::token::{Secret, TokenSet};

/// Path the loopback receiver answers on. Fixed and documented so providers
/// can pre-register `http://127.0.0.1/callback`-style redirect patterns.
pub const CALLBACK_PATH: &str = "/callback";

/// Default deadline for the whole interactive transaction.
pub const DEFAULT_FLOW_TIMEOUT: Duration = Duration::from_secs(300);

/// Static success page served to the browser after a valid callback. No
/// tokens, no scripts, no reflection of request data; `no-store` /
/// `no-referrer` keep the callback URL out of caches and referer headers.
const SUCCESS_HTML: &str = concat!(
    "<!doctype html><html><head><meta charset=\"utf-8\">",
    "<title>Authorization complete</title></head><body>",
    "<p>Authorization complete. You can close this tab and return to the terminal.</p>",
    "</body></html>"
);

const ERROR_HTML: &str = concat!(
    "<!doctype html><html><head><meta charset=\"utf-8\">",
    "<title>Authorization failed</title></head><body>",
    "<p>Authorization failed. You can close this tab and return to the terminal.</p>",
    "</body></html>"
);

/// A bound loopback listener plus everything the browser needs to reach the
/// authorization endpoint. Drop it to abort the flow; the verifier and state
/// zeroize themselves.
pub struct PendingAuthorization {
    listener: Option<TcpListener>,
    config: OAuthProviderConfig,
    pkce: PkcePair,
    state: Secret,
    authorization_url: Url,
    redirect_uri: String,
    cancel: Arc<Cancellation>,
    timeout: Duration,
    deadline: Instant,
}

impl std::fmt::Debug for PendingAuthorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingAuthorization")
            .field("redirect_uri", &self.redirect_uri)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// The authorization code returned by the provider, bound to the transaction
/// that produced it. The verifier lives here and nowhere else, so a code can
/// never be exchanged against the wrong PKCE pair.
pub struct AuthorizationCode {
    code: Secret,
    pkce: PkcePair,
    redirect_uri: String,
    config: OAuthProviderConfig,
}

/// Open the interactive flow: bind the loopback listener, build the
/// authorization URL, return the pending transaction.
///
/// The URL is handed back to the caller, which decides how to reach the
/// browser: [`PendingAuthorization::open_in_browser`] for a best-effort
/// system-browser launch, and/or display the URL for manual visit (mandatory
/// fallback for headless/SSH sessions).
pub fn begin_authorization(config: &OAuthProviderConfig) -> Result<PendingAuthorization> {
    begin_authorization_with_timeout(config, DEFAULT_FLOW_TIMEOUT)
}

/// Same as [`begin_authorization`] with a deadline measured from creation.
/// Requires a Tokio runtime with IO and time enabled.
pub fn begin_authorization_with_timeout(
    config: &OAuthProviderConfig,
    timeout: Duration,
) -> Result<PendingAuthorization> {
    config.validate()?;
    // Requires a Tokio runtime with IO and time enabled. Check presence so
    // an ordinary synchronous caller receives an error instead of a panic.
    tokio::runtime::Handle::try_current().map_err(|_| {
        OAuthError::InvalidConfig(
            "authorization requires a Tokio runtime with IO and time enabled".into(),
        )
    })?;
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| OAuthError::InvalidConfig("authorization timeout overflow".into()))?;

    // Bind on the ephemeral port and KEEP the socket: reserving-then-closing
    // would race another process for the port we just advertised.
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .map_err(OAuthError::LoopbackBind)?;
    listener
        .set_nonblocking(true)
        .map_err(OAuthError::LoopbackBind)?;
    let listener = TcpListener::from_std(listener).map_err(OAuthError::LoopbackBind)?;
    let addr: SocketAddr = listener.local_addr().map_err(OAuthError::LoopbackBind)?;
    let redirect_uri = format!("http://{addr}{}", config.callback_path);

    let pkce = PkcePair::generate()?;
    let state = generate_state()?;

    let mut authorization_url = config.authorization_url.clone();
    {
        let mut q = authorization_url.query_pairs_mut();
        q.append_pair("response_type", "code");
        q.append_pair("client_id", &config.client_id);
        q.append_pair("redirect_uri", &redirect_uri);
        if let Some(scope) = config.scope_param() {
            q.append_pair("scope", &scope);
        }
        q.append_pair("state", state.expose());
        q.append_pair("code_challenge", pkce.challenge());
        q.append_pair("code_challenge_method", "S256");
        for (k, v) in &config.extra_authorize_params {
            q.append_pair(k, v);
        }
    }

    Ok(PendingAuthorization {
        listener: Some(listener),
        config: config.clone(),
        pkce,
        state,
        authorization_url,
        redirect_uri,
        cancel: Arc::new(Cancellation::default()),
        timeout,
        deadline,
    })
}

impl PendingAuthorization {
    /// The URL the user's browser must visit. Present it to the user even
    /// when also attempting [`Self::open_in_browser`]: the launch is
    /// best-effort and silently impossible on headless systems.
    pub fn authorization_url(&self) -> &Url {
        &self.authorization_url
    }

    /// The loopback redirect URI baked into the authorization URL — useful
    /// when the provider requires pre-registration and the caller wants to
    /// show it.
    pub fn redirect_uri(&self) -> &str {
        &self.redirect_uri
    }

    /// Best-effort launch of the system browser at the authorization URL.
    /// Returns after spawning a detached launcher thread. Only spawn failures
    /// are reported; launch failures are ignored. The user can still open
    /// [`Self::authorization_url`] by hand.
    pub fn open_in_browser(&self) -> Result<()> {
        // Hardened launcher: http(s) schemes only. Run off the async
        // executor — some platforms' launch paths block.
        let url = zeroize::Zeroizing::new(self.authorization_url.to_string());
        std::thread::Builder::new()
            .name("oauth-browser".into())
            .spawn(move || {
                let _ = webbrowser::open(&url);
            })
            .map_err(|_| OAuthError::UrlBuild("cannot spawn browser launcher".into()))?;
        Ok(())
    }

    /// A handle that aborts [`Self::wait`] when triggered.
    pub fn cancel_handle(&self) -> CancelHandle {
        CancelHandle {
            inner: Arc::clone(&self.cancel),
        }
    }

    /// Wait for the provider to redirect the browser back, validating the
    /// callback against this transaction's `state`. Consumes the listener.
    pub async fn wait(mut self) -> Result<AuthorizationCode> {
        if self.cancel.cancelled.load(Ordering::Acquire) {
            return Err(OAuthError::Cancelled);
        }
        let listener = self.listener.take().ok_or(OAuthError::Cancelled)?;
        let authority = listener
            .local_addr()
            .map_err(OAuthError::LoopbackBind)?
            .to_string();
        let context = Arc::new(CallbackContext {
            state: Secret::new(self.state.expose()),
            path: self.config.callback_path.clone(),
            authority,
            issuer: self.config.expected_issuer.clone(),
        });
        let (tx, mut rx) = mpsc::channel(1);
        // The accept loop is owned by this future (not detached). Dropping
        // wait drops the listener and JoinSet, which aborts all connections.
        let server = async move {
            let mut connections = tokio::task::JoinSet::new();
            // The only way this loop exits is a failed accept (the branch
            // guard keeps it running otherwise). A terminal accept error —
            // fd exhaustion, EMFILE... — is captured and returned so the
            // select below can surface it as a transport failure instead of
            // masquerading as a user cancellation.
            let mut accept_error: Option<std::io::Error> = None;
            loop {
                tokio::select! {
                    biased;
                    _ = connections.join_next(), if !connections.is_empty() => {},
                    accepted = listener.accept(), if accept_error.is_none() && connections.len() < 8 => {
                        let (stream, _) = match accepted {
                            Ok(pair) => pair,
                            Err(e) => {
                                accept_error = Some(e);
                                break;
                            }
                        };
                        let context = context.clone();
                        let tx = tx.clone();
                        connections.spawn(async move {
                            let (callback_tx, mut callback_rx) = mpsc::channel(1);
                            let svc = service_fn(move |req| {
                                let result = handle_callback(req, &context, &callback_tx);
                                std::future::ready(Ok::<_, std::convert::Infallible>(result))
                            });
                            let mut builder = http1::Builder::new();
                            builder.timer(hyper_util::rt::TokioTimer::new())
                                .header_read_timeout(Duration::from_secs(5))
                                .max_buf_size(16 * 1024).keep_alive(false);
                            let _ = tokio::time::timeout(Duration::from_secs(5),
                                builder.serve_connection(TokioIo::new(stream), svc)).await;
                            // Flush the static browser page before publishing the
                            // outcome; otherwise wait could abort this connection
                            // between validation and its response write.
                            if let Ok(outcome) = callback_rx.try_recv() { let _ = tx.try_send(outcome); }
                        });
                    }
                }
            }
            if let Some(e) = accept_error {
                return Err(OAuthError::LoopbackBind(e));
            }
            // Unreachable: the loop only exits through the break above.
            Ok(())
        };
        let outcome = tokio::select! {
            biased;
            _ = self.cancel.wait() => return Err(OAuthError::Cancelled),
            _ = tokio::time::sleep_until(self.deadline.into()) => return Err(OAuthError::Timeout),
            result = rx.recv() => result.ok_or(OAuthError::Cancelled)?,
            server_result = server => {
                return Err(server_result.err().unwrap_or(OAuthError::Cancelled))
            },
        };
        match outcome {
            CallbackOutcome::Code(code) => Ok(AuthorizationCode {
                code,
                pkce: self.pkce,
                redirect_uri: self.redirect_uri,
                config: self.config,
            }),
            CallbackOutcome::Denied { error, description } => {
                Err(OAuthError::authorization_denied(error, description))
            }
        }
    }
}

#[derive(Default)]
struct Cancellation {
    cancelled: AtomicBool,
    notify: Notify,
}
impl Cancellation {
    async fn wait(&self) {
        // Register before checking the persistent flag: no lost wakeup in
        // the race between checking and notify_waiters().
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if !self.cancelled.load(Ordering::Acquire) {
            notified.await;
        }
    }
}
/// Cloneable, persistent abort switch; cancellation before wait is retained.
#[derive(Clone)]
pub struct CancelHandle {
    inner: Arc<Cancellation>,
}
impl CancelHandle {
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Release);
        self.inner.notify.notify_waiters();
    }
}
enum CallbackOutcome {
    Code(Secret),
    Denied {
        error: String,
        description: Option<String>,
    },
}
struct CallbackContext {
    state: Secret,
    path: String,
    authority: String,
    issuer: Option<String>,
}
fn handle_callback<B>(
    req: Request<B>,
    context: &CallbackContext,
    tx: &mpsc::Sender<CallbackOutcome>,
) -> Response<Full<Bytes>> {
    let bad = || html_response(StatusCode::BAD_REQUEST, ERROR_HTML);
    if req.method() != hyper::Method::GET {
        return html_response(StatusCode::METHOD_NOT_ALLOWED, ERROR_HTML);
    }
    if req.uri().path() != context.path {
        return html_response(StatusCode::NOT_FOUND, ERROR_HTML);
    }
    let hosts: Vec<_> = req.headers().get_all(hyper::header::HOST).iter().collect();
    if hosts.len() != 1
        || hosts[0].to_str().ok() != Some(context.authority.as_str())
        || req
            .uri()
            .authority()
            .is_some_and(|a| a.as_str() != context.authority)
        || req.uri().scheme_str().is_some_and(|s| s != "http")
    {
        return bad();
    }
    // Bounded by Hyper's header cap. Wipe decoded callback secrets on drop.
    let params = zeroize::Zeroizing::new(
        url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect::<Vec<_>>(),
    );
    let values = |key: &str| {
        params
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
    };
    if values("state") != [context.state.expose()] {
        return bad();
    }
    let issuers = values("iss");
    if let Some(expected) = &context.issuer {
        if issuers != [expected.as_str()] {
            return bad();
        }
    } else if !issuers.is_empty() {
        // Strict policy: unsolicited issuer information is not trusted.
        return bad();
    }
    let codes = values("code");
    let errors = values("error");
    let descriptions = values("error_description");
    if descriptions.len() > 1 || values("error_uri").len() > 1 {
        return bad();
    }
    let (outcome, page) = match (codes.as_slice(), errors.as_slice()) {
        ([code], [])
            if !code.is_empty() && descriptions.is_empty() && values("error_uri").is_empty() =>
        {
            (CallbackOutcome::Code(Secret::new(*code)), SUCCESS_HTML)
        }
        ([], [error]) if !error.is_empty() => (
            CallbackOutcome::Denied {
                error: (*error).into(),
                description: descriptions.first().map(|s| (*s).into()),
            },
            ERROR_HTML,
        ),
        _ => return bad(),
    };
    if tx.try_send(outcome).is_err() {
        return bad();
    }
    html_response(StatusCode::OK, page)
}
fn html_response(status: StatusCode, body: &'static str) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from_static(body.as_bytes())));
    *response.status_mut() = status;
    for (name, value) in [
        ("content-type", "text/html; charset=utf-8"),
        ("cache-control", "no-store"),
        ("referrer-policy", "no-referrer"),
        ("content-security-policy", "default-src 'none'"),
        ("connection", "close"),
    ] {
        response
            .headers_mut()
            .insert(name, hyper::header::HeaderValue::from_static(value));
    }
    response
}

impl AuthorizationCode {
    /// Exchange the code for tokens at the provider's token endpoint
    /// (RFC 6749 §4.1.3): form body with `grant_type=authorization_code`,
    /// the code, the exact same `redirect_uri` used at authorization time
    /// (required by OAuth 2.0 servers even though the 2.1 draft drops it),
    /// and the PKCE verifier (RFC 7636 §4.5).
    pub async fn exchange(self) -> Result<TokenSet> {
        let form = [
            ("grant_type", "authorization_code"),
            ("code", self.code.expose()),
            ("redirect_uri", &self.redirect_uri),
            ("client_id", &self.config.client_id),
            ("code_verifier", self.pkce.verifier().expose()),
        ];
        let body = http::post_token_form(
            &self.config.token_url,
            &form,
            self.config.allow_http_loopback,
        )
        .await?;
        let mut tokens = http::into_token_set(body)?;
        tokens.inherit_scopes(&self.config.scopes);
        Ok(tokens)
    }
}

/// One-call convenience: begin the flow, optionally open the browser, wait
/// for the callback, exchange the code.
///
/// `open_browser` controls the best-effort browser launch; callers that want
/// a manual fallback receive the URL through `on_url` before the launcher
/// starts. Treat the transaction URL as sensitive; do not log it.
///
/// # Example
///
/// ```no_run
/// use aiface::connector::oauth::{authorize, OAuthProviderConfig};
/// use url::Url;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut config = OAuthProviderConfig::new(
///     Url::parse("https://provider.example/oauth/authorize")?,
///     Url::parse("https://provider.example/oauth/token")?,
///     "my-client-id",
/// );
/// config.scopes = vec!["read".into(), "write".into()];
/// config.expected_issuer = Some("https://provider.example".into());
/// config.validate()?;
///
/// let tokens = authorize(&config, true, |url| {
///     // Manual fallback when the browser cannot open. Sensitive: never
///     // send to logs in production.
///     println!("Open: {url}");
/// })
/// .await?;
/// println!("Access token ready: {}", !tokens.access_token().expose().is_empty());
/// # Ok(())
/// # }
/// ```
pub async fn authorize(
    config: &OAuthProviderConfig,
    open_browser: bool,
    on_url: impl FnOnce(&Url),
) -> Result<TokenSet> {
    let pending = begin_authorization(config)?;
    on_url(pending.authorization_url());
    if open_browser {
        // Best-effort: a failed launch must not kill the flow — the caller
        // may still show `pending.authorization_url()` before calling wait.
        let _ = pending.open_in_browser();
    }
    pending.wait().await?.exchange().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::oauth::config::{ClientAuth, OAuthProviderConfig};
    use std::sync::Mutex;
    use tokio::io::AsyncWriteExt;

    /// Minimal fake authorization server: serves nothing (the test parses the
    /// authorization URL without ever fetching it) and answers the token
    /// exchange over a raw loopback TCP socket with a canned JSON response.
    /// Loopback HTTP is accepted by config validation only for IP literals.
    async fn fake_token_endpoint(
        expected_challenge: Arc<Mutex<String>>,
        expected_redirect: Arc<Mutex<String>>,
    ) -> (Url, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        let url = Url::parse(&format!("http://{addr}/token")).unwrap();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            // Read the request head + body (form body is small; read until we
            // have the full headers and the announced Content-Length).
            let recorded = super::super::test::support::read_request(&mut stream).await;
            let request = format!(
                "{}\r\n\r\n{}",
                recorded.head,
                String::from_utf8(recorded.body).unwrap()
            );
            assert!(
                request.starts_with("POST /token HTTP/1.1"),
                "got: {request}"
            );
            assert!(
                request.contains("content-type: application/x-www-form-urlencoded"),
                "token exchange must be form-urlencoded: {request}"
            );
            let body = request.split("\r\n\r\n").nth(1).unwrap_or("").to_owned();
            let params: std::collections::HashMap<String, String> =
                url::form_urlencoded::parse(body.as_bytes())
                    .map(|(k, v)| (k.into_owned(), v.into_owned()))
                    .collect();
            assert_eq!(params.get("grant_type").unwrap(), "authorization_code");
            assert_eq!(params.get("code").unwrap(), "test-code");
            assert_eq!(params.get("client_id").unwrap(), "test-client");
            // PKCE: the verifier must be present and the S256 of it must
            // match the challenge the test captured from the auth URL.
            let verifier = params.get("code_verifier").unwrap().clone();
            let expected = expected_challenge.lock().unwrap().clone();
            use base64::Engine;
            use sha2::Digest;
            let digest = sha2::Sha256::digest(verifier.as_bytes());
            let actual = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
            assert_eq!(actual, expected, "PKCE S256 mismatch");
            // redirect_uri must round-trip exactly (RFC 6749 §4.1.3).
            let redirect = params.get("redirect_uri").unwrap();
            assert_eq!(redirect, &*expected_redirect.lock().unwrap());

            let response_body = r#"{"access_token":"test-access","token_type":"Bearer","expires_in":3600,"refresh_token":"test-refresh","scope":"a b"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\ncache-control: no-store\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        (url, handle)
    }

    fn test_config(token_url: Url) -> OAuthProviderConfig {
        OAuthProviderConfig {
            authorization_url: Url::parse("https://provider.example/authorize").unwrap(),
            token_url,
            device_authorization_url: None,
            client_id: "test-client".into(),
            client_auth: ClientAuth::None,
            scopes: vec!["a".into(), "b".into()],
            extra_authorize_params: vec![],
            expected_issuer: None,
            callback_path: CALLBACK_PATH.into(),
            allow_http_loopback: true,
        }
    }

    /// Full Authorization Code + PKCE round trip against a fake token
    /// endpoint, driving the loopback receiver with a real HTTP client as
    /// the "browser".
    #[tokio::test]
    async fn end_to_end_code_pkce_flow() {
        let expected_challenge = Arc::new(Mutex::new(String::new()));
        let expected_redirect = Arc::new(Mutex::new(String::new()));
        let (token_url, server) =
            fake_token_endpoint(Arc::clone(&expected_challenge), expected_redirect.clone()).await;
        let config = test_config(token_url);

        let pending = begin_authorization(&config).unwrap();

        // The authorization URL must carry every mandated parameter.
        let url = pending.authorization_url().clone();
        let params: std::collections::HashMap<_, _> = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(params.get("response_type").unwrap(), "code");
        assert_eq!(params.get("client_id").unwrap(), "test-client");
        assert_eq!(params.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(params.get("scope").unwrap(), "a b");
        let challenge = params.get("code_challenge").unwrap().clone();
        let state = params.get("state").unwrap().clone();
        let redirect_uri = params.get("redirect_uri").unwrap().clone();
        *expected_challenge.lock().unwrap() = challenge;
        *expected_redirect.lock().unwrap() = redirect_uri.clone();

        // The listener only accepts connections once `wait()` starts polling
        // it, so the wait runs concurrently with the simulated browser.
        let wait_task = tokio::spawn(pending.wait());

        // Play the browser: first an incidental probe (favicon), then a
        // wrong-state request — neither may consume the transaction — and
        // finally the legitimate redirect.
        let browser = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let origin = redirect_uri.trim_end_matches("/callback");
        let favicon = browser
            .get(format!("{origin}/favicon.ico"))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(favicon, reqwest::StatusCode::NOT_FOUND);

        let wrong_state = browser
            .get(format!("{redirect_uri}?state=wrong&code=evil"))
            .send()
            .await
            .unwrap();
        assert_eq!(wrong_state.status(), reqwest::StatusCode::BAD_REQUEST);

        let legit = browser
            .get(format!("{redirect_uri}?state={state}&code=test-code"))
            .send()
            .await
            .unwrap();
        assert_eq!(legit.status(), reqwest::StatusCode::OK);
        assert_eq!(
            legit.headers().get("cache-control").unwrap(),
            "no-store",
            "callback responses must never be cached"
        );

        let code = wait_task.await.unwrap().unwrap();
        let tokens = code.exchange().await.unwrap();
        server.await.unwrap();
        assert_eq!(tokens.access_token().expose(), "test-access");
        assert_eq!(tokens.refresh_token().unwrap().expose(), "test-refresh");
        assert_eq!(tokens.scopes(), &["a", "b"]);
        assert!(tokens.access_token_valid(Duration::from_secs(60)));
    }

    /// A provider denial (`error=access_denied`) with a valid state must end
    /// the transaction with `AuthorizationDenied`, not hang.
    #[tokio::test]
    async fn provider_denial_is_surfaced() {
        let (token_url, _server_unused) = fake_token_endpoint(
            Arc::new(Mutex::new(String::new())),
            Arc::new(Mutex::new(String::new())),
        )
        .await;
        let config = test_config(token_url);
        let pending = begin_authorization(&config).unwrap();
        let url = pending.authorization_url().clone();
        let params: std::collections::HashMap<_, _> = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let state = params.get("state").unwrap().clone();
        let redirect_uri = pending.redirect_uri().to_owned();

        // Same as the success-path test: the listener only accepts once
        // `wait()` polls it, so wait runs concurrently with the browser.
        let wait_task = tokio::spawn(pending.wait());

        let browser = reqwest::Client::new();
        let denied = browser
            .get(format!(
                "{redirect_uri}?state={state}&error=access_denied&error_description=nope"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(denied.status(), reqwest::StatusCode::OK);
        assert!(
            denied
                .text()
                .await
                .unwrap()
                .contains("Authorization failed")
        );

        let err = match wait_task.await.unwrap() {
            Ok(_) => panic!("expected the denial to surface, got an authorization code"),
            Err(err) => err,
        };
        match err {
            OAuthError::AuthorizationDenied {
                code, description, ..
            } => {
                assert_eq!(code, "access_denied");
                assert_eq!(description.as_deref(), Some("nope"));
            }
            other => panic!("expected AuthorizationDenied, got {other:?}"),
        }
    }

    /// The one-call entry point: `on_url` must fire with the full
    /// authorization URL *before* any waiting begins, so a headless caller can
    /// display it, and the composed flow must still complete the exchange.
    #[tokio::test]
    async fn authorize_exposes_url_before_waiting_and_completes_the_flow() {
        let expected_challenge = Arc::new(Mutex::new(String::new()));
        let expected_redirect = Arc::new(Mutex::new(String::new()));
        let (token_url, server) =
            fake_token_endpoint(Arc::clone(&expected_challenge), expected_redirect.clone()).await;
        let config = test_config(token_url);

        let (url_tx, url_rx) = tokio::sync::oneshot::channel();
        let flow = tokio::spawn(async move {
            authorize(&config, false, move |url| {
                let _ = url_tx.send(url.clone());
            })
            .await
        });

        let url = url_rx
            .await
            .expect("on_url must receive the authorization URL");
        let params: std::collections::HashMap<_, _> = url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        assert_eq!(params.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(params.get("client_id").unwrap(), "test-client");
        *expected_challenge.lock().unwrap() = params["code_challenge"].clone();
        *expected_redirect.lock().unwrap() = params["redirect_uri"].clone();

        let redirect_uri = &params["redirect_uri"];
        let code = reqwest::Client::new()
            .get(format!(
                "{redirect_uri}?state={}&code=test-code",
                params["state"]
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(code.status(), StatusCode::OK);

        let tokens = flow.await.unwrap().unwrap();
        server.await.unwrap();
        assert_eq!(tokens.access_token().expose(), "test-access");
        assert_eq!(tokens.refresh_token().unwrap().expose(), "test-refresh");
        assert_eq!(tokens.scopes(), &["a", "b"]);
    }

    /// `expected_issuer` is the RFC 9207 / RFC 9700 §4.4.2 mix-up defense, so
    /// a malformed or non-TLS value must be rejected at configuration time
    /// rather than silently disabling the check at the callback.
    #[test]
    fn expected_issuer_must_be_absolute_https_and_queryless() {
        let mut config = test_config(Url::parse("https://provider.example/token").unwrap());
        config.expected_issuer = Some("https://provider.example".into());
        assert!(config.validate().is_ok());
        for issuer in [
            "provider.example",
            "/relative/issuer",
            "http://provider.example",
            "http://127.0.0.1:9999",
            "https://user:secret@provider.example",
            "https://provider.example/issuer?tenant=1",
            "https://provider.example#fragment",
            "",
        ] {
            config.expected_issuer = Some(issuer.into());
            assert!(
                matches!(config.validate(), Err(OAuthError::InvalidConfig(_))),
                "{issuer:?} must be rejected"
            );
        }
    }

    /// Config validation: https required for remote hosts; loopback http
    /// literals accepted; `localhost` (resolution-dependent) rejected.
    #[test]
    fn scheme_policy() {
        let https = test_config(Url::parse("https://provider.example/token").unwrap());
        assert!(https.validate().is_ok());

        let loopback = test_config(Url::parse("http://127.0.0.1:9999/token").unwrap());
        assert!(loopback.validate().is_ok());

        let remote_http = test_config(Url::parse("http://evil.example/token").unwrap());
        assert!(matches!(
            remote_http.validate(),
            Err(OAuthError::InvalidConfig(_))
        ));

        let localhost = test_config(Url::parse("http://localhost:9999/token").unwrap());
        assert!(matches!(
            localhost.validate(),
            Err(OAuthError::InvalidConfig(_))
        ));
    }
    #[test]
    fn callback_rejects_cardinality_method_host_and_issuer_without_consuming() {
        let context = CallbackContext {
            state: Secret::new("STATE"),
            path: "/callback".into(),
            authority: "127.0.0.1:1234".into(),
            issuer: Some("https://provider.example".into()),
        };
        let (tx, mut rx) = mpsc::channel(1);
        for query in [
            "state=STATE&state=STATE&code=C&iss=https://provider.example",
            "state=wrong&code=C&iss=https://provider.example",
            "state=STATE&code=C&code=D&iss=https://provider.example",
            "state=STATE&code=C&error=access_denied&iss=https://provider.example",
            "state=STATE&error=access_denied&error=invalid_request&iss=https://provider.example",
            "state=STATE&code=&iss=https://provider.example",
            "state=STATE&error=&iss=https://provider.example",
            "state=STATE&code=C",
            "state=STATE&code=C&iss=https://attacker.example",
            "state=STATE&code=C&iss=https://provider.example/&iss=https://provider.example",
            "state=STATE&code=C&iss=https://provider.example/",
            "state=STATE&error=access_denied&iss=https://attacker.example",
        ] {
            let req = Request::builder()
                .uri(format!("/callback?{query}"))
                .header("host", &context.authority)
                .body(())
                .unwrap();
            assert_eq!(
                handle_callback(req, &context, &tx).status(),
                StatusCode::BAD_REQUEST,
                "{query}"
            );
            assert!(rx.try_recv().is_err());
        }
        let valid = "/callback?state=STATE&code=C&iss=https://provider.example";
        for (method, host, uri) in [
            ("POST", "127.0.0.1:1234", valid),
            ("GET", "evil.example", valid),
            (
                "GET",
                "127.0.0.1:1234",
                "http://evil.example/callback?state=STATE&code=C&iss=https://provider.example",
            ),
        ] {
            let req = Request::builder()
                .method(method)
                .uri(uri)
                .header("host", host)
                .body(())
                .unwrap();
            assert!(!handle_callback(req, &context, &tx).status().is_success());
            assert!(rx.try_recv().is_err());
        }
        let req = Request::builder()
            .uri(valid)
            .header("host", &context.authority)
            .body(())
            .unwrap();
        assert_eq!(handle_callback(req, &context, &tx).status(), StatusCode::OK);
        assert!(matches!(rx.try_recv().unwrap(), CallbackOutcome::Code(_)));
    }
    #[tokio::test]
    async fn dropping_wait_future_closes_listener_and_connections() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let config = test_config(Url::parse("https://provider.example/token").unwrap());
        let pending = begin_authorization(&config).unwrap();
        let redirect = Url::parse(pending.redirect_uri()).unwrap();
        let addr = format!("127.0.0.1:{}", redirect.port().unwrap());
        let mut connection = tokio::net::TcpStream::connect(&addr).await.unwrap();
        connection
            .write_all(b"GET /callback HTTP/1.1\r\nHost: ")
            .await
            .unwrap();
        let mut waiting = Box::pin(pending.wait());
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut waiting)
                .await
                .is_err()
        );
        drop(waiting);
        assert!(
            tokio::net::TcpStream::connect(&addr).await.is_err(),
            "listener survived future drop"
        );
        let mut buf = [0; 1];
        let read = tokio::time::timeout(Duration::from_secs(1), connection.read(&mut buf))
            .await
            .unwrap();
        assert!(
            matches!(read, Ok(0) | Err(_)),
            "accepted connection survived future drop"
        );
    }
    #[tokio::test]
    async fn cancellation_before_wait_is_persistent_and_timeout_overflow_errors() {
        let config = test_config(Url::parse("https://provider.example/token").unwrap());
        let pending = begin_authorization(&config).unwrap();
        pending.cancel_handle().cancel();
        assert!(matches!(pending.wait().await, Err(OAuthError::Cancelled)));
        assert!(begin_authorization_with_timeout(&config, Duration::MAX).is_err());
    }
    #[test]
    fn config_rejects_reserved_parameters_and_unsafe_urls_at_refresher_boundary() {
        for raw in [
            "http://evil.example/token",
            "https://user:secret@provider.example/token",
            "https://provider.example/token#secret",
        ] {
            let config = test_config(Url::parse(raw).unwrap());
            assert!(super::super::StandardRefresher::new(config).is_err());
        }
        let mut config = test_config(Url::parse("http://[::1]:1234/token").unwrap());
        assert!(config.validate().is_ok());
        config.allow_http_loopback = false;
        assert!(config.validate().is_err());
        config.token_url = Url::parse("https://provider.example/token").unwrap();
        for parameter in [
            "state",
            "client_id",
            "redirect_uri",
            "code_challenge",
            "scope",
        ] {
            config.extra_authorize_params = vec![(parameter.into(), "override".into())];
            assert!(config.validate().is_err());
        }
        config.extra_authorize_params.clear();
        config.authorization_url.set_query(Some("state=override"));
        assert!(config.validate().is_err());
    }
}
