//! Secret wrappers and the token model.
//!
//! Two layers, deliberately separate:
//!
//! - [`Secret`] / [`TokenSet`]: the *live* in-memory representation. Secrets
//!   are zeroized on drop, redacted from `Debug`, and not `Clone` — copies do
//!   not proliferate by accident.
//! - [`StoredTokenSet`]: the *persistable snapshot* handed to the consumer,
//!   which owns storage (keyring, file, nothing...). It is serializable on
//!   purpose; the consumer decides where it goes.

use std::fmt;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

/// Default proactive renewal margin, separate from actual expiration, absorbing clock skew and the latency of the very request the
/// token is about to authenticate. Not an OAuth requirement — an engineering
/// policy, bounded further to 10% of the observed TTL.
pub const DEFAULT_REFRESH_SKEW: Duration = Duration::from_secs(60);

/// A secret string that zeroizes itself on drop, redacts `Debug`, and cannot
/// be cloned. Access the plaintext only through [`Secret::expose`] at the
/// exact point of use (building an `Authorization` header, a form body...).
pub struct Secret(Zeroizing<String>);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Secret(Zeroizing::new(value.into()))
    }

    /// Borrow the plaintext. Keep the borrow short-lived.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::new)
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Secret::new(value)
    }
}

/// A validated OAuth 2.0 token set (RFC 6749 §5.1), live in memory.
///
/// `access_token` is always present; `refresh_token` is optional per the
/// spec and may be *replaced* on every refresh (rotation, RFC 9700 §4.14.2)
/// — never assume it survives a renewal. `expires_at` is wall-clock because
/// it must be meaningful across process restarts once persisted; expiry
/// checks use [`TokenSet::access_token_valid`] with a skew margin rather
/// than raw comparisons.
///
/// No `Zeroize` derive here: the secret fields are [`Secret`]-wrapped
/// (self-zeroizing via `Zeroizing<String>`), and the non-secret fields
/// (timestamps, scopes) carry no secret material.
pub struct TokenSet {
    access_token: Secret,
    refresh_token: Option<Secret>,
    /// Absolute expiry of the access token. `None` means the provider did
    /// not send `expires_in` — treated as "unknown", never fabricated.
    expires_at: Option<SystemTime>,
    /// Scopes actually granted (may differ from what was requested).
    scopes: Vec<String>,
    scopes_present: bool,
    refresh_margin_cap: Duration,
    monotonic_expiry: Option<std::time::Instant>,
}

impl fmt::Debug for TokenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenSet")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_at", &self.expires_at)
            .field("scopes", &"<redacted>")
            .finish()
    }
}

impl TokenSet {
    pub fn new(
        access_token: Secret,
        refresh_token: Option<Secret>,
        expires_at: Option<SystemTime>,
        scopes: Vec<String>,
    ) -> Self {
        TokenSet {
            access_token,
            refresh_token,
            expires_at,
            scopes,
            scopes_present: true,
            refresh_margin_cap: expires_at
                .and_then(|at| at.duration_since(SystemTime::now()).ok())
                .unwrap_or_default()
                / 10,
            monotonic_expiry: expires_at.map(|at| {
                std::time::Instant::now()
                    .checked_add(at.duration_since(SystemTime::now()).unwrap_or_default())
                    .unwrap_or_else(std::time::Instant::now)
            }),
        }
    }

    pub fn access_token(&self) -> &Secret {
        &self.access_token
    }

    pub fn refresh_token(&self) -> Option<&Secret> {
        self.refresh_token.as_ref()
    }

    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }

    /// Whether the access token is still usable, subtracting `skew` from its
    /// lifetime. A token with unknown expiry (`None`) is treated as valid:
    /// the spec permits omitting `expires_in`, and inventing a lifetime
    /// would be worse.
    pub fn access_token_valid(&self, skew: Duration) -> bool {
        !self.should_refresh(skew)
    }

    /// Unknown expiry is not proof of validity. A resource-server rejection
    /// must trigger TokenSession::refresh_after_rejection.
    pub fn is_expired(&self) -> bool {
        self.should_refresh(Duration::ZERO)
    }

    /// Proactive renewal uses at most 10% of the observed lifetime, preventing
    /// a short-lived token from being immediately refreshed on every call.
    pub fn should_refresh(&self, skew: Duration) -> bool {
        let margin = skew.min(self.refresh_margin_cap);
        self.expires_at
            .is_some_and(|at| at.duration_since(SystemTime::now()).unwrap_or_default() <= margin)
            || self
                .monotonic_expiry
                .is_some_and(|at| at.saturating_duration_since(std::time::Instant::now()) <= margin)
    }

    pub(crate) fn expiry_from_now(seconds: u64) -> super::Result<SystemTime> {
        SystemTime::now()
            .checked_add(Duration::from_secs(seconds))
            .ok_or_else(|| super::OAuthError::MalformedResponse("expires_in overflow".into()))
    }

    pub(crate) fn scopes_omitted(&mut self) {
        self.scopes_present = false;
    }
    pub(crate) fn inherit_scopes(&mut self, scopes: &[String]) {
        if !self.scopes_present {
            self.scopes = scopes.to_vec();
            self.scopes_present = true;
        }
    }
    pub(crate) fn inherit_refresh(&mut self, refresh: Option<&Secret>) {
        if self.refresh_token.is_none() {
            self.refresh_token = refresh.map(|s| Secret::new(s.expose()));
        }
    }

    /// Export a persistable snapshot for the consumer's storage layer.
    pub fn to_stored(&self) -> StoredTokenSet {
        StoredTokenSet {
            access_token: self.access_token.expose().to_owned(),
            refresh_token: self.refresh_token.as_ref().map(|s| s.expose().to_owned()),
            expires_at_unix: self
                .expires_at
                .map(|t| t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default())
                .map(|d| d.as_secs()),
            scopes: self.scopes.clone(),
        }
    }

    /// Rebuild a live token set from a consumer-restored snapshot. Borrows
    /// rather than consumes so the stored snapshot still zeroizes itself on
    /// drop (it implements `Drop` via `ZeroizeOnDrop` and cannot be moved
    /// out of).
    pub fn from_stored(stored: &StoredTokenSet) -> super::Result<Self> {
        let expires_at = stored
            .expires_at_unix
            .map(|secs| {
                SystemTime::UNIX_EPOCH
                    .checked_add(Duration::from_secs(secs))
                    .ok_or_else(|| {
                        super::OAuthError::MalformedResponse("stored expiry overflow".into())
                    })
            })
            .transpose()?;
        if stored.access_token.is_empty()
            || stored.refresh_token.as_ref().is_some_and(String::is_empty)
        {
            return Err(super::OAuthError::MalformedResponse(
                "empty stored token".into(),
            ));
        }
        Ok(Self::new(
            Secret::new(stored.access_token.clone()),
            stored.refresh_token.clone().map(Secret::new),
            expires_at,
            stored.scopes.clone(),
        ))
    }
}

/// Serializable snapshot of a [`TokenSet`], for the consumer's storage.
///
/// This type intentionally *is* `Serialize` — persistence is its whole point
/// — but nothing inside the library ever serializes a live [`TokenSet`]
/// implicitly; the conversion is always an explicit [`TokenSet::to_stored`]
/// call made by the consumer.
#[derive(Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct StoredTokenSet {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Seconds since the Unix epoch; absent when the provider gave no
    /// `expires_in`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at_unix: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
}

impl fmt::Debug for StoredTokenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredTokenSet")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_at_unix", &self.expires_at_unix)
            .field("scopes", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_never_leaks_secret_material() {
        let secret = Secret::new("super-secret-token-value");
        assert_eq!(format!("{secret:?}"), "Secret(<redacted>)");

        let tokens = TokenSet::new(
            Secret::new("access-token-plaintext"),
            Some(Secret::new("refresh-token-plaintext")),
            None,
            vec!["scope:a".into()],
        );
        let dbg = format!("{tokens:?}");
        assert!(!dbg.contains("access-token-plaintext"));
        assert!(!dbg.contains("refresh-token-plaintext"));

        let stored = tokens.to_stored();
        let dbg = format!("{stored:?}");
        assert!(!dbg.contains("access-token-plaintext"));
    }

    #[test]
    fn stored_round_trip_preserves_all_fields() {
        let expiry = SystemTime::UNIX_EPOCH + Duration::from_secs(1_900_000_000);
        let tokens = TokenSet::new(
            Secret::new("at"),
            Some(Secret::new("rt")),
            Some(expiry),
            vec!["a".into(), "b".into()],
        );
        let restored = TokenSet::from_stored(&tokens.to_stored()).unwrap();
        assert_eq!(restored.access_token().expose(), "at");
        assert_eq!(restored.refresh_token().unwrap().expose(), "rt");
        assert_eq!(restored.expires_at(), Some(expiry));
        assert_eq!(restored.scopes(), &["a", "b"]);
    }

    #[test]
    fn validity_respects_skew_and_unknown_expiry() {
        // Unknown expiry => considered valid (no fabricated lifetime).
        let no_expiry = TokenSet::new(Secret::new("x"), None, None, vec![]);
        assert!(no_expiry.access_token_valid(DEFAULT_REFRESH_SKEW));

        let soon = SystemTime::now() + Duration::from_secs(30);
        let expiring = TokenSet::new(Secret::new("x"), None, Some(soon), vec![]);
        assert!(expiring.access_token_valid(DEFAULT_REFRESH_SKEW));
        assert!(expiring.access_token_valid(Duration::from_secs(5)));

        let later = SystemTime::now() + Duration::from_secs(3600);
        let fresh = TokenSet::new(Secret::new("x"), None, Some(later), vec![]);
        assert!(fresh.access_token_valid(DEFAULT_REFRESH_SKEW));
    }
    #[test]
    fn stored_expiry_overflow_errors_without_panicking_or_becoming_unknown() {
        let mut stored = TokenSet::new(Secret::new("A"), None, None, vec![]).to_stored();
        stored.expires_at_unix = Some(u64::MAX);
        assert!(TokenSet::from_stored(&stored).is_err());
        stored.expires_at_unix = Some(0);
        assert!(TokenSet::from_stored(&stored).unwrap().is_expired());
    }
    #[test]
    fn refresh_margin_is_bounded_by_ttl_and_separate_from_expiry() {
        let tokens = TokenSet::new(
            Secret::new("A"),
            None,
            Some(
                SystemTime::now()
                    .checked_add(Duration::from_secs(1))
                    .unwrap(),
            ),
            vec![],
        );
        assert!(!tokens.is_expired());
        assert!(!tokens.should_refresh(Duration::MAX));
        let expired = TokenSet::new(Secret::new("A"), None, Some(SystemTime::UNIX_EPOCH), vec![]);
        assert!(expired.is_expired());
    }
}
