//! Cancellation-safe, single-flight refresh with atomic rotation publication.
//! A spawned operation owns each renewal; dropping any caller cannot cancel
//! it. Every waiter gets its result. A failed flight latches by default, so a
//! possibly-rotated refresh token is never replayed blindly; reconstruct the
//! session after recovery/reauthorization. The one exception is a failure that
//! provably never reached the provider — see `TokenSession::latches`.
use super::{
    config::OAuthProviderConfig,
    error::{OAuthError, Result, TransportFailure},
    http,
    token::{DEFAULT_REFRESH_SKEW, Secret, StoredTokenSet, TokenSet},
};
use std::{future::Future, pin::Pin, sync::Arc, time::Duration};
use tokio::sync::{Mutex, watch};

/// Hard ceiling for one renewal attempt, whichever [`TokenRefresher`] runs.
///
/// Deliberately conservative and intentionally NOT configurable: a refresh
/// that outlives it gets [`OAuthError::RefreshOutcomeUnknown`] and latches —
/// the session cannot tell whether the server rotated the token. Lengthening
/// this window trades latched sessions for replay risk (RFC 9700 §4.14.2), so
/// a consumer needing a longer window should own that trade-off inside its
/// own refresher, not here.
const REFRESH_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(60);

pub trait TokenRefresher: Send + Sync {
    /// Exchange the current refresh token for a fresh token set.
    /// Implementations must map a rejected refresh token to
    /// [`OAuthError::RefreshTokenInvalid`].
    fn refresh<'a>(
        &'a self,
        refresh_token: &'a Secret,
    ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>>;
}

pub struct StandardRefresher {
    config: OAuthProviderConfig,
}
impl StandardRefresher {
    pub fn new(config: OAuthProviderConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }
}
impl TokenRefresher for StandardRefresher {
    fn refresh<'a>(
        &'a self,
        refresh_token: &'a Secret,
    ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>> {
        Box::pin(async move {
            let form = [
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.expose()),
                ("client_id", self.config.client_id.as_str()),
            ];
            match http::post_token_form(
                &self.config.token_url,
                &form,
                self.config.allow_http_loopback,
            )
            .await
            {
                Ok(body) => http::into_token_set(body),
                Err(e) if e.error_code() == Some("invalid_grant") => {
                    Err(OAuthError::RefreshTokenInvalid)
                }
                Err(e) => Err(e),
            }
        })
    }
}
type FlightResult = std::result::Result<Arc<TokenSet>, Arc<OAuthError>>;
struct Flight {
    result: watch::Sender<Option<FlightResult>>,
}
#[derive(Clone)]
pub struct TokenSession {
    inner: Arc<Inner>,
}
struct Inner {
    state: watch::Sender<Arc<TokenSet>>,
    flight: Mutex<Option<Arc<Flight>>>,
    refresher: Box<dyn TokenRefresher>,
    skew: Duration,
}
impl TokenSession {
    /// Own one token set for its whole lifetime: automatic renewal with
    /// single-flight, terminal latch on ambiguous refresh outcomes, and
    /// observability for consumer-owned persistence.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use aiface::connector::oauth::{OAuthProviderConfig, StandardRefresher, TokenSession};
    /// use url::Url;
    ///
    /// # async fn example(tokens: aiface::connector::oauth::TokenSet) -> Result<(), Box<dyn std::error::Error>> {
    /// let config = OAuthProviderConfig::new(
    ///     Url::parse("https://provider.example/oauth/authorize")?,
    ///     Url::parse("https://provider.example/oauth/token")?,
    ///     "my-client-id",
    /// );
    /// let refresher = StandardRefresher::new(config)?;
    /// let session = TokenSession::new(tokens, Box::new(refresher));
    ///
    /// // Always-current access token; renews automatically when expiring.
    /// let access = session.valid_access_token().await?;
    ///
    /// // After a proactive 401 from the resource server:
    /// let fresh = session.refresh_after_rejection(&access).await?;
    ///
    /// // Persistence is the consumer's job (keyring, encrypted store, ...):
    /// let stored = session.snapshot(); // keep this, restore via TokenSet::from_stored
    /// let _ = session.subscribe();     // push notifications on every renewal
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(tokens: TokenSet, refresher: Box<dyn TokenRefresher>) -> Self {
        Self::with_skew(tokens, refresher, DEFAULT_REFRESH_SKEW)
    }
    pub fn with_skew(tokens: TokenSet, refresher: Box<dyn TokenRefresher>, skew: Duration) -> Self {
        let (state, _) = watch::channel(Arc::new(tokens));
        Self {
            inner: Arc::new(Inner {
                state,
                flight: Mutex::new(None),
                refresher,
                skew,
            }),
        }
    }
    pub async fn valid_access_token(&self) -> Result<Secret> {
        self.access_token(None).await
    }

    /// Force a renewal after a resource-server invalid_token/401 response,
    /// including tokens with unknown expiry. Supply the token actually rejected
    /// so a late 401 for an old token cannot refresh a newly rotated session.
    pub async fn refresh_after_rejection(&self, rejected: &Secret) -> Result<Secret> {
        self.access_token(Some(rejected.expose())).await
    }
    /// Whether a failed flight must stay latched, i.e. whether replaying the
    /// refresh token is a reuse-detection hazard.
    ///
    /// Latching is the default because most failures leave the outcome
    /// genuinely unknown: a request timeout after the bytes were transmitted
    /// may still have rotated the token server-side, and RFC 9700 §4.14.2
    /// treats a replayed rotated token as grounds for revoking the whole
    /// family. Terminal failures (`RefreshTokenInvalid`,
    /// `RefreshOutcomeUnknown`) latch for the same reason.
    ///
    /// The single exception is a connect failure: no connection was ever
    /// established, so no request reached the authorization server and nothing
    /// could have rotated. Retrying is provably safe, and without the
    /// exception a single offline moment would latch the session permanently —
    /// the caller would keep replaying a stale error with no way to clear it.
    fn latches(error: &OAuthError) -> bool {
        !matches!(
            error,
            OAuthError::Transport(TransportFailure { connect: true, .. })
        )
    }
    async fn access_token(&self, rejected: Option<&str>) -> Result<Secret> {
        // What to do with the flight currently occupying the slot. Decided
        // while the slot is locked; the returned value owns no borrow of it.
        enum Disposition {
            /// A renewal is in flight: wait on it.
            Wait(Arc<Flight>),
            /// Re-evaluate the current state (no flight, a completed one, or
            /// one that failed without ever reaching the provider).
            Reevaluate,
        }
        let flight = {
            let mut slot = self.inner.flight.lock().await;
            let disposition = match slot.clone() {
                Some(flight) => match flight.result.borrow().as_ref() {
                    Some(Err(error)) if !Self::latches(error) => Disposition::Reevaluate,
                    Some(Err(error)) => return Err(OAuthError::Shared(error.clone())),
                    None => Disposition::Wait(Arc::clone(&flight)),
                    Some(Ok(_)) => Disposition::Reevaluate,
                },
                None => Disposition::Reevaluate,
            };
            if let Disposition::Wait(pending) = disposition {
                pending
            } else {
                let current = self.inner.state.borrow().clone();
                let rejected_current = rejected == Some(current.access_token().expose());
                if !rejected_current && current.access_token_valid(self.inner.skew) {
                    return Ok(Secret::new(current.access_token().expose()));
                }
                let refresh = current
                    .refresh_token()
                    .ok_or(OAuthError::ReauthorizationRequired)?;
                let refresh = Secret::new(refresh.expose());
                let flight = Arc::new(Flight {
                    result: watch::channel(None).0,
                });
                *slot = Some(flight.clone());
                // No await between storing the flight and spawning its owner.
                let inner = self.inner.clone();
                let worker_inner = inner.clone();
                let worker = tokio::spawn(async move {
                    tokio::time::timeout(
                        REFRESH_ATTEMPT_TIMEOUT,
                        worker_inner.refresher.refresh(&refresh),
                    )
                    .await
                    .unwrap_or(Err(OAuthError::RefreshOutcomeUnknown))
                });
                let completed = flight.clone();
                tokio::spawn(async move {
                    let result = match worker.await {
                        Ok(Ok(mut renewed)) => {
                            renewed.inherit_refresh(current.refresh_token());
                            renewed.inherit_scopes(current.scopes());
                            let expired =
                                renewed.is_expired() || renewed.access_token().expose().is_empty();
                            let renewed = Arc::new(renewed);
                            // Always save rotation, even if A2 is already expired.
                            // send_replace persists without subscribed receivers.
                            inner.state.send_replace(renewed.clone());
                            if expired {
                                Err(Arc::new(OAuthError::ReauthorizationRequired))
                            } else {
                                Ok(renewed)
                            }
                        }
                        Ok(Err(error)) => Err(Arc::new(error)),
                        Err(_) => Err(Arc::new(OAuthError::RefreshOutcomeUnknown)),
                    };
                    completed.result.send_replace(Some(result));
                });
                flight
            }
        };
        let mut receiver = flight.result.subscribe();
        loop {
            let result = receiver.borrow_and_update().clone();
            if let Some(result) = result {
                return result
                    .map(|tokens| Secret::new(tokens.access_token().expose()))
                    .map_err(OAuthError::Shared);
            }
            receiver
                .changed()
                .await
                .map_err(|_| OAuthError::RefreshOutcomeUnknown)?;
        }
    }
    /// Explicit snapshot for consumer-owned secure persistence.
    pub fn snapshot(&self) -> StoredTokenSet {
        self.inner.state.borrow().to_stored()
    }
    /// Observe renewals independently of request callers (including cancelled
    /// callers). Persist the latest snapshot in the consumer's storage layer.
    /// This coalesces updates; in-memory publication cannot guarantee durability
    /// across process crashes. Holding a receiver/Arc also extends secret lifetime.
    pub fn subscribe(&self) -> watch::Receiver<Arc<TokenSet>> {
        self.inner.state.subscribe()
    }
    pub fn expires_at(&self) -> Option<std::time::SystemTime> {
        self.inner.state.borrow().expires_at()
    }
}
impl std::fmt::Debug for TokenSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenSession")
            .field("expires_at", &self.expires_at())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingRefresher {
        calls: Arc<AtomicUsize>,
        rotate: bool,
    }

    impl TokenRefresher for CountingRefresher {
        fn refresh<'a>(
            &'a self,
            _refresh_token: &'a Secret,
        ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                Ok(TokenSet::new(
                    Secret::new("fresh-access"),
                    if self.rotate {
                        Some(Secret::new("rotated-refresh"))
                    } else {
                        None
                    },
                    Some(std::time::SystemTime::now() + Duration::from_secs(3600)),
                    vec![],
                ))
            })
        }
    }

    fn expired_session(refresher: CountingRefresher) -> TokenSession {
        let expired = TokenSet::new(
            Secret::new("stale-access"),
            Some(Secret::new("current-refresh")),
            Some(std::time::SystemTime::now() - Duration::from_secs(10)),
            vec![],
        );
        TokenSession::new(expired, Box::new(refresher))
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_renewal() {
        let calls = Arc::new(AtomicUsize::new(0));
        let session = expired_session(CountingRefresher {
            calls: Arc::clone(&calls),
            rotate: false,
        });

        let mut handles = vec![];
        for _ in 0..8 {
            let s = session.clone();
            handles.push(tokio::spawn(async move { s.valid_access_token().await }));
        }
        for h in handles {
            let token = h.await.unwrap().unwrap();
            assert_eq!(token.expose(), "fresh-access");
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "refresh must be single-flight"
        );
    }

    #[tokio::test]
    async fn refresh_token_preserved_when_not_rotated() {
        let session = expired_session(CountingRefresher {
            calls: Arc::new(AtomicUsize::new(0)),
            rotate: false,
        });
        let _ = session.valid_access_token().await.unwrap();
        let stored = session.snapshot();
        assert_eq!(stored.refresh_token.as_deref(), Some("current-refresh"));
    }

    #[tokio::test]
    async fn refresh_token_replaced_when_rotated() {
        let session = expired_session(CountingRefresher {
            calls: Arc::new(AtomicUsize::new(0)),
            rotate: true,
        });
        let _ = session.valid_access_token().await.unwrap();
        let stored = session.snapshot();
        assert_eq!(stored.refresh_token.as_deref(), Some("rotated-refresh"));
    }

    #[tokio::test]
    async fn missing_refresh_token_requires_reauthorization() {
        let expired = TokenSet::new(
            Secret::new("stale"),
            None,
            Some(std::time::SystemTime::now() - Duration::from_secs(10)),
            vec![],
        );
        let session = TokenSession::new(
            expired,
            Box::new(CountingRefresher {
                calls: Arc::new(AtomicUsize::new(0)),
                rotate: false,
            }),
        );
        let err = session.valid_access_token().await.unwrap_err();
        assert!(matches!(err, OAuthError::ReauthorizationRequired));
        assert!(err.requires_reauthorization());
    }

    #[tokio::test]
    async fn valid_token_skips_refresh() {
        let calls = Arc::new(AtomicUsize::new(0));
        let fresh = TokenSet::new(
            Secret::new("good"),
            Some(Secret::new("rt")),
            Some(std::time::SystemTime::now() + Duration::from_secs(3600)),
            vec![],
        );
        let session = TokenSession::new(
            fresh,
            Box::new(CountingRefresher {
                calls: Arc::clone(&calls),
                rotate: false,
            }),
        );
        let token = session.valid_access_token().await.unwrap();
        assert_eq!(token.expose(), "good");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    struct GatedRefresher {
        calls: Arc<AtomicUsize>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        fail: bool,
    }
    impl TokenRefresher for GatedRefresher {
        fn refresh<'a>(
            &'a self,
            refresh: &'a Secret,
        ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>> {
            Box::pin(async move {
                assert_eq!(refresh.expose(), "R1");
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.started.notify_one();
                self.release.notified().await;
                if self.fail {
                    return Err(OAuthError::RefreshTokenInvalid);
                }
                let mut tokens = TokenSet::new(
                    Secret::new("A2"),
                    Some(Secret::new("R2")),
                    Some(
                        std::time::SystemTime::now()
                            .checked_add(Duration::from_secs(30))
                            .unwrap(),
                    ),
                    vec![],
                );
                tokens.scopes_omitted();
                Ok(tokens)
            })
        }
    }
    #[tokio::test]
    async fn cancelled_refresh_caller_does_not_lose_rotation_or_start_second_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let tokens = TokenSet::new(
            Secret::new("A1"),
            Some(Secret::new("R1")),
            Some(std::time::SystemTime::UNIX_EPOCH),
            vec!["scope:old".into()],
        );
        let session = TokenSession::new(
            tokens,
            Box::new(GatedRefresher {
                calls: calls.clone(),
                started: started.clone(),
                release: release.clone(),
                fail: false,
            }),
        );
        let mut updates = session.subscribe();
        let first_session = session.clone();
        let first = tokio::spawn(async move { first_session.valid_access_token().await });
        started.notified().await;
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        // There are no request callers while the provider issues R2.
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), updates.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(session.snapshot().refresh_token.as_deref(), Some("R2"));
        let mut waiters = vec![];
        for _ in 0..8 {
            let session = session.clone();
            waiters.push(tokio::spawn(
                async move { session.valid_access_token().await },
            ));
        }
        for waiter in waiters {
            assert_eq!(waiter.await.unwrap().unwrap().expose(), "A2");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(session.snapshot().scopes, ["scope:old"]);
        // A stale resource-server rejection must not refresh R2 again.
        assert_eq!(
            session
                .refresh_after_rejection(&Secret::new("A1"))
                .await
                .unwrap()
                .expose(),
            "A2"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn terminal_refresh_failure_is_shared_and_latched() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let tokens = TokenSet::new(
            Secret::new("A1"),
            Some(Secret::new("R1")),
            Some(std::time::SystemTime::UNIX_EPOCH),
            vec![],
        );
        let session = TokenSession::new(
            tokens,
            Box::new(GatedRefresher {
                calls: calls.clone(),
                started: started.clone(),
                release: release.clone(),
                fail: true,
            }),
        );
        let mut tasks = vec![];
        for _ in 0..8 {
            let session = session.clone();
            tasks.push(tokio::spawn(
                async move { session.valid_access_token().await },
            ));
        }
        started.notified().await;
        release.notify_one();
        for task in tasks {
            assert!(task.await.unwrap().unwrap_err().requires_reauthorization());
        }
        assert!(
            session
                .valid_access_token()
                .await
                .unwrap_err()
                .requires_reauthorization()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    struct ConnectFlakyRefresher {
        calls: Arc<AtomicUsize>,
    }
    impl TokenRefresher for ConnectFlakyRefresher {
        fn refresh<'a>(
            &'a self,
            _refresh_token: &'a Secret,
        ) -> Pin<Box<dyn Future<Output = Result<TokenSet>> + Send + 'a>> {
            Box::pin(async move {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(OAuthError::Transport(TransportFailure {
                        timeout: false,
                        connect: true,
                    }));
                }
                Ok(TokenSet::new(
                    Secret::new("fresh-access"),
                    None,
                    Some(
                        std::time::SystemTime::now()
                            .checked_add(Duration::from_secs(3600))
                            .unwrap(),
                    ),
                    vec![],
                ))
            })
        }
    }

    /// Latching exists to stop a replay of a possibly-rotated token, so it must
    /// cover every failure whose outcome is unknown — including a timeout after
    /// the request was transmitted, the case a connect error is distinguished
    /// from.
    #[test]
    fn only_failures_that_never_reached_the_provider_are_retryable() {
        for error in [
            OAuthError::RefreshTokenInvalid,
            OAuthError::RefreshOutcomeUnknown,
            OAuthError::ReauthorizationRequired,
            OAuthError::Transport(TransportFailure {
                timeout: true,
                connect: false,
            }),
            OAuthError::Transport(TransportFailure {
                timeout: false,
                connect: false,
            }),
            OAuthError::UnexpectedHttpStatus { status: 500 },
            OAuthError::MalformedResponse("anything".into()),
        ] {
            assert!(TokenSession::latches(&error), "{error:?} must latch");
        }
        assert!(!TokenSession::latches(&OAuthError::Transport(
            TransportFailure {
                timeout: false,
                connect: true,
            }
        )));
    }

    /// A connect failure never reached the provider, so no rotation is
    /// possible and the session must stay usable instead of latching forever.
    #[tokio::test]
    async fn connect_failure_does_not_permanently_latch_the_session() {
        let calls = Arc::new(AtomicUsize::new(0));
        let session = TokenSession::new(
            TokenSet::new(
                Secret::new("stale-access"),
                Some(Secret::new("current-refresh")),
                Some(std::time::SystemTime::UNIX_EPOCH),
                vec![],
            ),
            Box::new(ConnectFlakyRefresher {
                calls: Arc::clone(&calls),
            }),
        );
        let error = session.valid_access_token().await.unwrap_err();
        // Waiters observe the flight's failure wrapped for sharing; the latch
        // decision is made on the unwrapped error the worker stored.
        let OAuthError::Shared(inner) = &error else {
            panic!("{error:?}");
        };
        assert!(
            matches!(
                **inner,
                OAuthError::Transport(TransportFailure { connect: true, .. })
            ),
            "{inner:?}"
        );
        assert!(!error.requires_reauthorization());
        assert_eq!(
            session.valid_access_token().await.unwrap().expose(),
            "fresh-access"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        // The renewed token is now valid, so no further calls are made.
        assert_eq!(
            session.valid_access_token().await.unwrap().expose(),
            "fresh-access"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn standard_refresh_preserves_scopes_and_encodes_borrowed_secret() {
        let (url, server) = super::super::test::support::serve(vec![(
            200,
            r#"{"access_token":"A2","token_type":"Bearer","expires_in":30,"refresh_token":"R2"}"#
                .into(),
        )])
        .await;
        let refresher = StandardRefresher::new(super::super::test::support::config(url)).unwrap();
        let tokens = TokenSet::new(
            Secret::new("A1"),
            Some(Secret::new("R+1&")),
            Some(std::time::SystemTime::UNIX_EPOCH),
            vec!["a".into(), "b".into()],
        );
        let session = TokenSession::new(tokens, Box::new(refresher));
        assert_eq!(session.valid_access_token().await.unwrap().expose(), "A2");
        assert_eq!(session.snapshot().scopes, ["a", "b"]);
        let requests = server.await.unwrap();
        let fields: std::collections::HashMap<_, _> =
            url::form_urlencoded::parse(&requests[0].body).collect();
        assert_eq!(fields["refresh_token"], "R+1&");
        assert_eq!(fields["grant_type"], "refresh_token");
        assert_eq!(fields["client_id"], "test-client");
        assert_eq!(session.valid_access_token().await.unwrap().expose(), "A2");
    }
}
