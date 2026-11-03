//! PKCE (RFC 7636) and CSRF `state` generation.
//!
//! - `code_verifier`: 43–128 chars from the unreserved set. We generate 32
//!   bytes of OS entropy and encode them base64url-no-pad, yielding exactly
//!   43 chars and 256 bits of entropy (RFC 7636 §4.1 recommendation).
//! - `code_challenge`: `BASE64URL-NO-PAD(SHA256(ASCII(verifier)))` — S256
//!   only. The `plain` method is deliberately not implemented (RFC 9700 §2.1,
//!   OAuth 2.1 draft-16 §7.5 remove it).
//! - `state`: same entropy treatment, hex-encoded (alphanumeric — safe in
//!   every query-string dialect). Independent from the verifier: never reuse
//!   one for the other.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

use super::Secret;
use super::error::{OAuthError, Result};

/// PKCE code verifier (secret until the token exchange) + its public S256
/// challenge. The verifier is [`Secret`]-wrapped: it must never appear in
/// logs, and it is zeroized when the pending transaction is dropped.
pub struct PkcePair {
    verifier: Secret,
    challenge: String,
}

impl PkcePair {
    /// Generate a fresh pair from OS entropy. Aborts (no fallback) when the
    /// entropy source is unavailable.
    pub fn generate() -> Result<Self> {
        let mut bytes = zeroize::Zeroizing::new([0u8; 32]);
        getrandom::fill(&mut *bytes).map_err(|_| OAuthError::EntropyUnavailable)?;
        // base64url-no-pad of 32 bytes = 43 chars, all within the RFC 7636
        // §4.1 unreserved charset (A-Z a-z 0-9 - _ — a subset of it).
        let verifier = URL_SAFE_NO_PAD.encode(*bytes);
        Ok(Self::from_verifier(Secret::new(verifier)))
    }

    /// Rebuild a pair from an existing verifier, deriving the S256 challenge.
    /// Production code must use [`PkcePair::generate`]; this constructor
    /// exists so tests can exercise the RFC 7636 Appendix B vector.
    fn from_verifier(verifier: Secret) -> Self {
        let digest = Sha256::digest(verifier.expose().as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(digest);
        PkcePair {
            verifier,
            challenge,
        }
    }

    /// The secret verifier, sent only to the token endpoint.
    pub fn verifier(&self) -> &Secret {
        &self.verifier
    }

    /// The public S256 challenge, sent to the authorization endpoint.
    pub fn challenge(&self) -> &str {
        &self.challenge
    }
}

/// Generate an opaque CSRF `state` token (RFC 6749 §10.12): 32 bytes of OS
/// entropy, hex-encoded (64 lowercase alphanumeric chars). Returned as a
/// [`Secret`] because a leaked live `state` would let an attacker bind their
/// own authorization response to our pending transaction.
pub fn generate_state() -> Result<Secret> {
    let mut bytes = zeroize::Zeroizing::new([0u8; 32]);
    getrandom::fill(&mut *bytes).map_err(|_| OAuthError::EntropyUnavailable)?;
    let mut hex = String::with_capacity(64);
    for b in bytes.iter() {
        use std::fmt::Write;
        let _ = write!(hex, "{b:02x}");
    }
    Ok(Secret::new(hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 7636 Appendix B test vector.
    #[test]
    fn s256_matches_rfc_appendix_b() {
        let pair =
            PkcePair::from_verifier(Secret::new("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"));
        assert_eq!(
            pair.challenge(),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn generated_verifier_is_legal() {
        let pair = PkcePair::generate().unwrap();
        let v = pair.verifier().expose();
        // RFC 7636 §4.1: 43-128 chars, unreserved charset.
        assert_eq!(v.len(), 43);
        assert!(
            v.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '~'))
        );
        // Challenge is 43-char base64url-no-pad (SHA-256 = 32 bytes).
        assert_eq!(pair.challenge().len(), 43);
        assert!(
            pair.challenge()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        );
    }

    #[test]
    fn verifiers_and_states_are_unique() {
        let a = PkcePair::generate().unwrap();
        let b = PkcePair::generate().unwrap();
        assert_ne!(a.verifier().expose(), b.verifier().expose());
        assert_ne!(a.challenge(), b.challenge());

        let s1 = generate_state().unwrap();
        let s2 = generate_state().unwrap();
        assert_ne!(s1.expose(), s2.expose());
        assert_eq!(s1.expose().len(), 64);
        assert!(s1.expose().chars().all(|c| c.is_ascii_hexdigit()));
    }
}
