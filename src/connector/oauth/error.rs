//! Errors never format provider text, URLs, bodies, or underlying transport
//! sources. Sanitized provider diagnostics are available only by explicit
//! field access: a server may echo secrets even in otherwise printable text.
use std::{fmt, sync::Arc};
use thiserror::Error;
pub type Result<T> = std::result::Result<T, OAuthError>;

/// Transport classification with no request data or exposed error chain.
#[derive(Debug)]
pub struct TransportFailure {
    pub timeout: bool,
    pub connect: bool,
}
#[derive(Error)]
#[non_exhaustive]
pub enum OAuthError {
    #[error("invalid OAuth configuration: {0}")]
    InvalidConfig(String),
    #[error("OS entropy unavailable")]
    EntropyUnavailable,
    #[error("failed to build OAuth URL")]
    UrlBuild(String),
    #[error("failed to bind loopback listener")]
    LoopbackBind(std::io::Error),
    #[error("OAuth operation timed out")]
    Timeout,
    #[error("authorization cancelled")]
    Cancelled,
    #[error("authorization denied by provider (diagnostics redacted)")]
    AuthorizationDenied {
        code: String,
        description: Option<String>,
    },
    #[error("OAuth HTTP transport failed")]
    Transport(TransportFailure),
    #[error("OAuth endpoint returned HTTP {status}")]
    UnexpectedHttpStatus { status: u16 },
    #[error("malformed OAuth response: {0}")]
    MalformedResponse(String),
    #[error("OAuth endpoint returned an error (HTTP {http_status}; diagnostics redacted)")]
    TokenEndpoint {
        http_status: u16,
        code: String,
        description: Option<String>,
    },
    #[error("device authorization denied")]
    DeviceDenied,
    #[error("device authorization expired")]
    DeviceExpired,
    #[error("new interactive authorization required")]
    ReauthorizationRequired,
    #[error("refresh token rejected or revoked")]
    RefreshTokenInvalid,
    #[error("refresh outcome unknown; reauthorize to avoid replaying a rotated token")]
    RefreshOutcomeUnknown,
    /// All participants in a refresh flight observe the same failure.
    #[error("{0}")]
    Shared(Arc<OAuthError>),
}
impl fmt::Debug for OAuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl From<reqwest::Error> for OAuthError {
    fn from(err: reqwest::Error) -> Self {
        // reqwest 0.13 documents that errors retain the complete request URL.
        // Strip it before classification, then discard the source chain too.
        let err = err.without_url();
        Self::Transport(TransportFailure {
            timeout: err.is_timeout(),
            connect: err.is_connect(),
        })
    }
}
impl From<Box<reqwest::Error>> for OAuthError {
    fn from(err: Box<reqwest::Error>) -> Self {
        Self::from(*err)
    }
}
impl OAuthError {
    pub(crate) fn authorization_denied(code: String, description: Option<String>) -> Self {
        Self::AuthorizationDenied {
            code: sanitize_code(&code),
            description: description.map(|s| sanitize(&s)),
        }
    }
    pub(crate) fn token_endpoint(
        http_status: u16,
        code: String,
        description: Option<String>,
    ) -> Self {
        Self::TokenEndpoint {
            http_status,
            code: sanitize_code(&code),
            description: description.map(|s| sanitize(&s)),
        }
    }
    pub fn error_code(&self) -> Option<&str> {
        match self {
            Self::AuthorizationDenied { code, .. } | Self::TokenEndpoint { code, .. } => Some(code),
            Self::Shared(err) => err.error_code(),
            _ => None,
        }
    }
    pub fn requires_reauthorization(&self) -> bool {
        match self {
            Self::ReauthorizationRequired
            | Self::RefreshTokenInvalid
            | Self::RefreshOutcomeUnknown => true,
            Self::Shared(err) => err.requires_reauthorization(),
            _ => false,
        }
    }
    pub(crate) fn is_timeout(&self) -> bool {
        matches!(
            self,
            Self::Transport(TransportFailure { timeout: true, .. }) | Self::Timeout
        )
    }
}
fn sanitize(raw: &str) -> String {
    raw.chars()
        .filter(|c| {
            !c.is_control() && !matches!(*c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        })
        .take(512)
        .collect()
}

fn sanitize_code(raw: &str) -> String {
    let sanitized = sanitize(raw);
    // Do not turn an invalid code such as "invalid_\ngrant" into a known
    // terminal machine code while removing controls or truncating it.
    if sanitized == raw {
        sanitized
    } else {
        "invalid_provider_error".into()
    }
}
