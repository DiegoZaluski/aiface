mod caller;
pub mod tokens;

pub use caller::{chat, chat_stream, chat_stream_with_messages, embed, list_models};
pub use tokens::{extract_tokens, extract_usage};

// Internal, but visible to the crate's own test suite.
#[cfg(test)]
pub(crate) use caller::{api_version_for, resolve_base_url};

/// Gemini API surface version used for requests WITHOUT a thinking level.
///
/// `v1` is the stable, production-ready API. `v1beta` exposes experimental
/// features. This is an INTERNAL switch — library users cannot select the
/// version through the public `Connector` API. Flip it to `"v1beta"` only
/// to test unstable Gemini features locally; keep it `"v1"` in production.
///
/// Requests that carry `generationConfig.thinkingConfig.thinkingLevel` are
/// routed to `v1beta` regardless (the stable `v1` endpoint rejects thinking
/// configuration with HTTP 400) — see `caller::api_version_for`.
///
/// The provider registry pins the stable `v1` base URL; the caller
/// re-writes the version segment with this constant so the switch is the
/// single source of truth.
pub(crate) const API_VERSION: &str = "v1";
