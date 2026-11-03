//! Provider-native cost reporting for the usage dashboard.
//!
//! Spend is never estimated locally: each supported provider is asked
//! directly and is the single source of truth for its own billing.
//!
//! Internal module — the dev-facing API is `Connector::session_cost`,
//! whose docstring carries the canonical Supported/Unsupported provider
//! matrix (mirrored on the re-exported [`supports_cost_reporting`]).
//!
//! Units: Charm bills in Hypercredits; its per-request usage reports BOTH
//! `usd` and `hypercredits`. USD is preferred for the dashboard; the credit
//! balance is surfaced as [`ProviderCost::credit_balance`].

use super::error::ConnectorError;
use super::params::Parameters;
use super::provider::{ProviderConfig, get_api_key};

/// Spend information as reported by the provider itself.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProviderCost {
    /// REAL spend (USD) attributable to this session's requests so far —
    /// the sum of every per-request cost the provider reported (or the
    /// provider-side usage figure for account-spend providers).
    pub session_cost_usd: f64,
    /// Provider-side balance (USD) when the provider exposes one.
    pub balance_usd: Option<f64>,
    /// Provider-side balance in NON-USD units when that is its native
    /// unit (Charm Hypercredits). Informational — the dashboard only
    /// renders `session_cost_usd`.
    pub credit_balance: Option<f64>,
}

/// Structured rejection for providers with no spend-reporting capability.
///
/// Returned instead of a guessed number so callers can `match`/`if` and
/// simply hide the price rows when the provider cannot vouch for them.
#[derive(Debug, Clone, PartialEq)]
pub enum CostError {
    /// The provider has no API for reporting spend. Nothing was
    /// requested over the network.
    CostNotSupported,
    /// The provider supports cost reporting but no API key could be resolved.
    MissingApiKey(String),
    /// The provider's cost endpoint rejected the request or answered
    /// something unusable.
    CostUnavailable(String),
}

impl std::fmt::Display for CostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CostNotSupported => write!(
                f,
                "provider does not report costs; token/cost display is disabled for it"
            ),
            Self::MissingApiKey(p) => {
                write!(f, "API key not set for provider: {p}")
            }
            Self::CostUnavailable(detail) => write!(f, "cost unavailable: {detail}"),
        }
    }
}

impl std::error::Error for CostError {}

impl From<ConnectorError> for CostError {
    fn from(e: ConnectorError) -> Self {
        match e {
            ConnectorError::MissingApiKey(p) => Self::MissingApiKey(p),
            other => Self::CostUnavailable(other.to_string()),
        }
    }
}

/// Cheap, OFFLINE check the TUI runs BEFORE deciding to render any cost
/// row. Supported: `openrouter`, `zen`, `opencode-go`, `charm`, `xai`,
/// `vercel`, `kimi` — every other cloud provider returns `false`.
#[must_use]
pub fn supports_cost_reporting(provider: &str) -> bool {
    matches!(
        provider,
        "openrouter" | "zen" | "opencode-go" | "charm" | "xai" | "vercel" | "kimi"
    )
}

/// Report the REAL session spend using the provider as the source of truth.
///
/// `session_reported_costs` are the per-request costs the provider already
/// reported in each response ([`super::TokenUsage::reported_cost`]); their
/// sum is the session spend. For `openrouter` — and for `vercel` when no
/// per-request cost was captured — the figure comes from the provider's
/// account endpoint instead.
///
/// # Errors
///
/// - [`CostError::CostNotSupported`] when the provider has no spend API
///   (offline check, no network).
/// - [`CostError::MissingApiKey`] when the provider reports spend but no
///   key could be resolved.
/// - [`CostError::CostUnavailable`] when the provider's endpoint fails.
pub async fn provider_session_cost(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    if !supports_cost_reporting(config.name) {
        return Err(CostError::CostNotSupported);
    }
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| CostError::MissingApiKey(config.name.to_string()))?;

    match config.name {
        "openrouter" => openrouter_cost(config, params, &api_key).await,
        "zen" | "opencode-go" => {
            opencode_cost(config, params, &api_key, session_reported_costs).await
        }
        "charm" => charm_cost(config, params, &api_key, session_reported_costs).await,
        "xai" => xai_cost(config, params, &api_key, session_reported_costs).await,
        "vercel" => vercel_cost(config, params, &api_key, session_reported_costs).await,
        "kimi" => kimi_cost(config, params, &api_key, session_reported_costs).await,
        // `supports_cost_reporting` guarantees this arm is unreachable.
        _ => Err(CostError::CostNotSupported),
    }
}

async fn get_json(
    config: &ProviderConfig,
    params: &Parameters,
    url: &str,
    api_key: &str,
) -> Result<serde_json::Value, CostError> {
    let mut headers: Vec<(&str, String)> = vec![("Authorization", format!("Bearer {api_key}"))];
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    let text = super::common::send_get_request(config, url, &headers).await?;
    serde_json::from_str(&text).map_err(|e| CostError::CostUnavailable(e.to_string()))
}

fn base_url<'a>(config: &ProviderConfig, params: &'a Parameters) -> &'a str {
    params.base_url.as_deref().unwrap_or(config.base_url)
}

/// Charm Hyper: remaining Hypercredit balance straight from the account
/// (`GET /credits` → `{"balance": <f64>}`).
///
/// Fallback for when the gateway's streaming usage frames do NOT carry
/// `usage.remaining.hypercredits` (observed live: glm models via the Prism
/// router omit the field). Crush uses the same strategy
/// (`internal/agent/hyper/provider.go` `FetchCredits`). Returns `None` on
/// any failure — a missing balance must never block or error the turn.
pub async fn hyper_credits_balance(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Option<f64> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))?;
    let body = get_json(
        config,
        params,
        &format!("{}/credits", base_url(config, params)),
        &api_key,
    )
    .await
    .ok()?;
    let balance = body.get("balance").and_then(serde_json::Value::as_f64)?;
    log::debug!("[HYPER] credits endpoint fallback: balance={balance}");
    (balance.is_finite() && balance >= 0.0).then_some(balance)
}

fn reported_sum(session_reported_costs: &[f64]) -> f64 {
    session_reported_costs.iter().sum()
}

// OpenRouter: lifetime `usage` (USD) + optional `limit_remaining` from
// `GET /key`; both used verbatim.
async fn openrouter_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
) -> Result<ProviderCost, CostError> {
    let body = get_json(
        config,
        params,
        &format!("{}/key", base_url(config, params)),
        api_key,
    )
    .await?;
    let data = body.get("data").unwrap_or(&body);
    let usage = data
        .get("usage")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| CostError::CostUnavailable("missing `data.usage`".to_string()))?;
    let balance_usd = data
        .get("limit_remaining")
        .and_then(serde_json::Value::as_f64);
    Ok(ProviderCost {
        session_cost_usd: usage,
        balance_usd,
        credit_balance: None,
    })
}

// Zen/Go expose no spend endpoint (anomalyco/opencode#10448) — the
// /models GET only verifies the key so a failure surfaces as
// CostUnavailable instead of a stale silent number.
async fn opencode_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    get_json(
        config,
        params,
        &format!("{}/models", base_url(config, params)),
        api_key,
    )
    .await?;
    Ok(ProviderCost {
        session_cost_usd: reported_sum(session_reported_costs),
        balance_usd: None,
        credit_balance: None,
    })
}

// Charm: summed per-request frames + Hypercredit balance from /credits.
async fn charm_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    let credits = get_json(
        config,
        params,
        &format!("{}/credits", base_url(config, params)),
        api_key,
    )
    .await?;
    Ok(ProviderCost {
        session_cost_usd: reported_sum(session_reported_costs),
        balance_usd: None,
        credit_balance: credits.get("balance").and_then(serde_json::Value::as_f64),
    })
}

// Money fields arrive as JSON strings from some providers ("95.50").
fn f64_field(v: &serde_json::Value, key: &str) -> Option<f64> {
    match v.get(key) {
        Some(serde_json::Value::Number(n)) => n.as_f64(),
        Some(serde_json::Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

// xAI: per-request cost rides `usage.cost_in_usd_ticks` (converted to
// USD by `extract_reported_cost`); /api-key balance is best-effort so
// an outage cannot hide a known session cost.
async fn xai_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    let balance_usd = match get_json(
        config,
        params,
        &format!("{}/api-key", base_url(config, params)),
        api_key,
    )
    .await
    {
        Ok(key_info) => f64_field(&key_info, "remaining_balance").or_else(|| {
            key_info
                .get("api_key")
                .and_then(|k| f64_field(k, "remaining_balance"))
        }),
        Err(_) => None,
    };
    Ok(ProviderCost {
        session_cost_usd: reported_sum(session_reported_costs),
        balance_usd,
        credit_balance: None,
    })
}

// Vercel: per-request `providerMetadata.gateway.cost` when captured,
// else lifetime `total_used` from /credits; `balance` is USD.
async fn vercel_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    let credits = get_json(
        config,
        params,
        &format!("{}/credits", base_url(config, params)),
        api_key,
    )
    .await?;
    let session_cost_usd = if session_reported_costs.is_empty() {
        f64_field(&credits, "total_used")
            .ok_or_else(|| CostError::CostUnavailable("missing `total_used`".to_string()))?
    } else {
        reported_sum(session_reported_costs)
    };
    Ok(ProviderCost {
        session_cost_usd,
        balance_usd: f64_field(&credits, "balance"),
        credit_balance: None,
    })
}

// Kimi: per-request costs are not reported inside the response body (the
// usage object carries token counters only), so the session figure is the
// sum of what the harness captured — which stays empty/unpriced — and the
// account balance comes from the documented billing endpoint
// `GET /v1/users/me/balance` → `data.available_balance` (USD, docs:
// platform.kimi.ai/docs/api/balance). The envelope (`code`/`data`/`status`)
// is Moonshot-specific and read defensively: a missing `data` reports
// CostUnavailable instead of a silently wrong 0.0 balance.
async fn kimi_cost(
    config: &ProviderConfig,
    params: &Parameters,
    api_key: &str,
    session_reported_costs: &[f64],
) -> Result<ProviderCost, CostError> {
    let body = get_json(
        config,
        params,
        &format!("{}/users/me/balance", base_url(config, params)),
        api_key,
    )
    .await?;
    let data = body
        .get("data")
        .ok_or_else(|| CostError::CostUnavailable("missing `data`".to_string()))?;
    Ok(ProviderCost {
        session_cost_usd: reported_sum(session_reported_costs),
        balance_usd: f64_field(data, "available_balance"),
        credit_balance: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::test::common::EnvGuard;

    fn cfg(name: &str) -> ProviderConfig {
        *crate::connector::get_provider(name).unwrap()
    }

    #[test]
    fn support_matrix_matches_module_docs() {
        for provider in [
            "openrouter",
            "zen",
            "opencode-go",
            "charm",
            "xai",
            "vercel",
            "kimi",
        ] {
            assert!(
                supports_cost_reporting(provider),
                "{provider} must report costs"
            );
        }
        for provider in [
            "openai", "claude", "gemini", "groq", "mistral", "deepseek", "together", "ollama",
        ] {
            assert!(
                !supports_cost_reporting(provider),
                "{provider} must NOT report costs"
            );
        }
    }

    #[test]
    fn numeric_or_string_money_fields_parse() {
        let v = serde_json::json!({"balance": "95.50", "total_used": 4.5});
        assert_eq!(f64_field(&v, "balance"), Some(95.5));
        assert_eq!(f64_field(&v, "total_used"), Some(4.5));
        assert_eq!(f64_field(&v, "missing"), None);
    }

    #[tokio::test]
    async fn unsupported_provider_fails_fast_without_network() {
        let err = provider_session_cost(&cfg("openai"), &Parameters::default(), None, &[])
            .await
            .unwrap_err();
        assert_eq!(err, CostError::CostNotSupported);
    }

    #[tokio::test]
    async fn local_provider_is_never_considered_a_cost_source() {
        // Local servers have no billing at all — same structured rejection.
        let err = provider_session_cost(&cfg("ollama"), &Parameters::default(), None, &[])
            .await
            .unwrap_err();
        assert_eq!(err, CostError::CostNotSupported);
    }

    #[tokio::test]
    async fn supported_provider_without_key_reports_missing_api_key() {
        let _guard = EnvGuard::remove("OPENROUTER_API_KEY");
        let err = provider_session_cost(&cfg("openrouter"), &Parameters::default(), None, &[])
            .await
            .unwrap_err();
        assert_eq!(err, CostError::MissingApiKey("openrouter".to_string()));
    }
}
