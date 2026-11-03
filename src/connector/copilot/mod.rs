//! Copilot family — the official GitHub Copilot agent runtime behind the
//! connector's unified API.
//!
//! The `copilot` provider does NOT speak an HTTP chat-completions dialect.
//! It drives GitHub's official Rust SDK ([`github_copilot_sdk`]), which
//! launches the Copilot runtime (the native Rust harness behind Copilot
//! CLI / App / SDK) and talks to it over JSON-RPC. This module is the
//! WRAPPER that standardizes that runtime onto the connector's conventions:
//!
//! - [`chat`] / [`chat_stream`] / [`chat_stream_with_messages`] keep the
//!   exact signatures every other family exposes ([`ChatOutput`],
//!   [`ChatStream`] yielding [`StreamChunk`]);
//! - [`extract_tokens`] / [`extract_usage`] parse the runtime's usage event
//!   into the shared [`TokenUsage`];
//! - parameters lower onto `SessionConfig`: `model`, `reasoning_effort`
//!   and the system prompt (a system-message APPEND — the runtime always
//!   carries its own instructions). `temperature` and the other
//!   OpenAI-style sampling knobs have NO runtime surface for a session and
//!   are ignored;
//! - the whole harness is tool-native: declared tools become runtime tools
//!   and the model's tool calls arrive as native structured events, mapped
//!   to [`NativeToolCall`] like every other family.
//!
//! Credentialing: the `GITHUB_TOKEN` key resolved by the provider registry
//! is passed as the session's `github_token`. The Copilot OAuth device flow
//! (`copilot login`) is NOT auto-started here — an embedded agent must not
//! silently drive a browser login; a missing token fails with
//! [`ConnectorError::MissingApiKey`] like every other provider. Model
//! listing does NOT go through the runtime: [`models::list_catalog`] is a
//! direct HTTP chain (GitHub token exchange, then the shared chat catalog)
//! because the runtime's own `models.list` RPC reads the agent-mode
//! catalog, which stays empty for Free-tier accounts even though the
//! account can chat fine.
//!
//! [`github_copilot_sdk`]: https://docs.rs/github-copilot-sdk

pub mod caller;
pub mod models;
pub mod oauth;
pub mod runtime;
pub mod tokens;

pub use caller::{chat, chat_stream, chat_stream_with_messages, list_models};
pub use tokens::{extract_tokens, extract_usage};
