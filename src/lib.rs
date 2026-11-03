//! # Aiface
//!
//! Unified multi-provider LLM client — a single `Connector` for Claude,
//! OpenAI, Gemini, GitHub Copilot and every OpenAI-compatible backend.
//!
//! Build a `Connector` with a provider name, configure parameters via the
//! builder API, then call [`chat`](connector::Connector::chat),
//! [`stream_chat`](connector::Connector::stream_chat), or
//! [`embed`](connector::Connector::embed).
//!
//! All providers share the same interface — swap the name passed to
//! [`Connector::new`](connector::Connector::new) to switch backends.
//!
//! See the [`docs/`](docs/README.md) directory for the module guides
//! (client, params, output, discovery, provider, error and OAuth).

pub mod connector;
pub mod native_call;

pub use native_call::NativeToolCall;
