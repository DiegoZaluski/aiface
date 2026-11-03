//! Static provider configuration, validated once before any flow starts.

use url::{Host, Url};

use super::error::{OAuthError, Result};

/// How the client authenticates to the token endpoint (RFC 6749 §2.3).
///
/// A distributed CLI is a *public* client: any embedded secret is extractable
/// from the binary and provides no confidentiality (RFC 8252 §8.5). `None` is
/// therefore the default and the only variant; confidential-client support is
/// a deliberate omission until a real provider requires it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClientAuth {
    /// Public client: `client_id` in the form body, no secret.
    #[default]
    None,
}

/// Validated configuration for one OAuth provider.
///
/// Build with [`OAuthProviderConfig::new`] and finish with
/// [`OAuthProviderConfig::validate`], which enforces the invariants the rest
/// of the module relies on (HTTPS for remote endpoints, non-empty client id,
/// parseable URLs).
#[derive(Clone)]
pub struct OAuthProviderConfig {
    /// Authorization endpoint (browser-facing). Must be HTTPS.
    pub authorization_url: Url,
    /// Token endpoint (code exchange + refresh). Must be HTTPS.
    pub token_url: Url,
    /// Device authorization endpoint (RFC 8628 §3.1), when the provider
    /// supports the device grant. Must be HTTPS.
    pub device_authorization_url: Option<Url>,
    /// Registered client id. Public clients have no secret.
    pub client_id: String,
    /// Client authentication method — see [`ClientAuth`].
    pub client_auth: ClientAuth,
    /// Scopes to request. Empty means "provider default".
    pub scopes: Vec<String>,
    /// Extra authorization-endpoint parameters required by the provider
    /// (e.g. `access_type=offline` for Google). Kept out of the generic API
    /// so provider quirks stay data, not code.
    pub extra_authorize_params: Vec<(String, String)>,
    /// RFC 9207 support: when set, every callback must contain exactly one
    /// matching `iss`, including error callbacks. Keep the issuer's exact
    /// spelling (comparison is not URL normalization).
    /// Without this, state binds the transaction only: use a distinct
    /// callback_path for each provider to defend against issuer mix-up.
    pub expected_issuer: Option<String>,
    /// Absolute loopback callback path, e.g. `/oauth/provider/callback`.
    pub callback_path: String,
    /// Explicit exception for local HTTP fixtures only. Never enable for
    /// remote deployments; DNS names including localhost are not allowed.
    pub allow_http_loopback: bool,
}

impl std::fmt::Debug for OAuthProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthProviderConfig")
            .finish_non_exhaustive()
    }
}
impl OAuthProviderConfig {
    /// Public-client defaults; HTTPS enforcement and issuer validation happen
    /// at each flow/refresher boundary. Set expected_issuer when supported.
    pub fn new(authorization_url: Url, token_url: Url, client_id: impl Into<String>) -> Self {
        Self {
            authorization_url,
            token_url,
            client_id: client_id.into(),
            device_authorization_url: None,
            client_auth: ClientAuth::None,
            scopes: vec![],
            extra_authorize_params: vec![],
            expected_issuer: None,
            callback_path: super::flow::CALLBACK_PATH.into(),
            allow_http_loopback: false,
        }
    }

    /// Check every invariant once. Call this before handing the config to any
    /// flow; the flows assume it holds.
    pub fn validate(&self) -> Result<()> {
        if self.client_id.is_empty() {
            return Err(OAuthError::InvalidConfig(
                "client_id must not be empty".into(),
            ));
        }
        for url in [&self.authorization_url, &self.token_url] {
            validate_endpoint(url, self.allow_http_loopback)?;
        }
        if let Some(url) = &self.device_authorization_url {
            validate_endpoint(url, self.allow_http_loopback)?;
        }
        if !self.callback_path.starts_with('/')
            || self.callback_path.starts_with("//")
            || self
                .callback_path
                .chars()
                .any(|c| !c.is_ascii_alphanumeric() && !"/-_".contains(c))
            || self
                .callback_path
                .split('/')
                .any(|part| part == "." || part == "..")
        {
            return Err(OAuthError::InvalidConfig("invalid callback path".into()));
        }
        const RESERVED: &[&str] = &[
            "response_type",
            "client_id",
            "redirect_uri",
            "scope",
            "state",
            "code_challenge",
            "code_challenge_method",
            "iss",
        ];
        let mut seen = std::collections::HashSet::new();
        for (key, _) in self.authorization_url.query_pairs().chain(
            self.extra_authorize_params
                .iter()
                .map(|(k, v)| (k.as_str().into(), v.as_str().into())),
        ) {
            if RESERVED.contains(&key.as_ref()) || !seen.insert(key.into_owned()) {
                return Err(OAuthError::InvalidConfig(
                    "duplicate or reserved authorization parameter".into(),
                ));
            }
        }
        if let Some(issuer) = &self.expected_issuer {
            let parsed = Url::parse(issuer)
                .map_err(|_| OAuthError::InvalidConfig("invalid expected issuer".into()))?;
            validate_endpoint(&parsed, false)?;
            if parsed.query().is_some() {
                return Err(OAuthError::InvalidConfig(
                    "issuer cannot have a query".into(),
                ));
            }
        }
        Ok(())
    }

    /// The scope string for request bodies: space-joined (RFC 6749 §3.3).
    pub fn scope_param(&self) -> Option<String> {
        if self.scopes.is_empty() {
            None
        } else {
            Some(self.scopes.join(" "))
        }
    }
}

/// HTTPS everywhere except explicitly enabled loopback IP fixtures.
pub(crate) fn validate_endpoint(url: &Url, allow_http_loopback: bool) -> Result<()> {
    let loopback = match url.host() {
        Some(Host::Ipv4(ip)) => ip.is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        _ => false,
    };
    if !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || url.host().is_none()
        || !(url.scheme() == "https" || (allow_http_loopback && loopback && url.scheme() == "http"))
    {
        return Err(OAuthError::InvalidConfig("endpoint requires HTTPS, no userinfo or fragment (HTTP only for explicitly enabled loopback IP fixtures)".into()));
    }
    Ok(())
}

pub(crate) fn verification_url(raw: &str, allow_http_loopback: bool) -> Result<Url> {
    let url = Url::parse(raw)
        .map_err(|_| OAuthError::MalformedResponse("invalid verification URL".into()))?;
    validate_endpoint(&url, allow_http_loopback)
        .map_err(|_| OAuthError::MalformedResponse("unsafe verification URL".into()))?;
    Ok(url)
}
