//! OAuth 2.0 for CLI providers.
//!
//! Security profile (researched against current RFCs and BCPs):
//!
//! - **Authorization Code + PKCE** ([RFC 7636](https://www.rfc-editor.org/rfc/rfc7636.html)):
//!   S256 only — no `plain` fallback. 32 bytes of OS entropy per verifier and
//!   per CSRF `state`; no time/seed-derived randomness.
//! - **Native-app redirect** ([RFC 8252](https://www.rfc-editor.org/rfc/rfc8252.html)):
//!   loopback `http://127.0.0.1:{ephemeral}/callback`, system browser instead
//!   of embedded webviews, incidental requests (favicon, wrong path, bad
//!   `state`) never consume the transaction.
//! - **Device Authorization Grant** ([RFC 8628](https://www.rfc-editor.org/rfc/rfc8628.html))
//!   for headless/SSH sessions, with `slow_down` backoff handling.
//! - **Best Current Practice** ([RFC 9700 / BCP 240](https://www.rfc-editor.org/rfc/rfc9700.html)):
//!   no Implicit, no ROPC, exact redirect matching, `state` always on.
//! - **Token handling** ([RFC 6749 §5](https://www.rfc-editor.org/rfc/rfc6749.html#section-5)):
//!   `token_type` must be Bearer, negative `expires_in` rejected, refresh
//!   token rotation respected (replace when re-issued, preserve when omitted),
//!   single-flight refresh so concurrent callers share one renewal.
//!
//! Storage is **delegated to the consumer**: the library hands out
//! [`StoredTokenSet`] snapshots and accepts them back; whether they land in a
//! keyring, a file, or nowhere is the caller's decision. Inside the library,
//! secrets are held in [`Secret`] (zeroized on drop, redacted `Debug`, no
//! `Clone`), never logged or formatted in errors. Protocol-required URL data
//! (authorization state, callback code, verification links, device-code path
//! segments) must travel in URLs by design — never log these URLs.
//! Provider diagnostics are sanitized but redacted by Debug/Display; explicit
//! field access can still reveal text echoed by a provider.
//!
//! Zeroization covers module-owned buffers, not copies made by TLS, HTTP,
//! URL parsers, browser processes, or consumer serialization. Persist snapshots
//! using an OS credential store; never put them in logs or ordinary config.
//! Detached refresh tasks survive caller cancellation, but not process/runtime
//! termination. Subscribe to session changes to persist rotations independently
//! of request tasks. Bearer access tokens are opaque; this is not an OIDC ID-token
//! validator and does not implement DPoP or mTLS.
//!
//! This module is provider-agnostic on purpose: it knows RFC profiles, not
//! vendor names. Provider-specific adapters live inside their own family
//! directories as single files (e.g. a proprietary device-protocol adapter at
//! `openai_compatible/charm_oauth.rs` — the vendor name appears only in that
//! file name, never here) and consume the `pub(crate)` surfaces here.

pub(crate) mod config;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod flow;
pub(crate) mod http;
pub(crate) mod pkce;
pub(crate) mod session;
pub(crate) mod token;

pub use config::{ClientAuth, OAuthProviderConfig};
pub use device::{DeviceAuthorization, begin_device_authorization, poll_device_token};
pub use error::{OAuthError, Result};
pub use flow::{
    AuthorizationCode, CancelHandle, PendingAuthorization, authorize, begin_authorization,
    begin_authorization_with_timeout,
};
pub use flow::{CALLBACK_PATH, DEFAULT_FLOW_TIMEOUT};
pub use session::{StandardRefresher, TokenRefresher, TokenSession};
pub use token::{DEFAULT_REFRESH_SKEW, Secret, StoredTokenSet, TokenSet};

#[cfg(test)]
pub(crate) mod test;
