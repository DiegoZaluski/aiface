//! # Aiface
//!
//! Unified multi-provider LLM client — a single `Connector` for Claude,
//! OpenAI, Gemini, GitHub Copilot and every OpenAI-compatible backend.
//!
//! Build a `Connector` with a provider name, configure parameters via the
//! builder API, then call [`chat`](connector::Connector::chat),
//! [`stream_chat`](connector::Connector::stream_chat),
//! [`embed`](connector::Connector::embed), or
//! [`decide`](connector::Connector::decide).
//!
//! All providers share the same interface — swap the name passed to
//! [`Connector::new`](connector::Connector::new) to switch backends.
//! `decide` extends the same interface to System One decision models
//! (TypeSafe Jev, Microsoft-Decision-1) AND to wrapped LLMs: typed
//! questions in, probability distributions out, one contract either way.
//!
//! See the [`docs/`](docs/README.md) directory for the module guides
//! (client, decision, params, output, discovery, provider, error and
//! OAuth).

pub mod connector;
pub mod native_call;

pub use native_call::NativeToolCall;
