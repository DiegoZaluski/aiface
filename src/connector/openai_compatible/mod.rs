//! OpenAI-compatible API caller — shared implementation for every provider
//! in the `OpenAICompatible` family.
//!
//! All three operations (`chat`, `chat_stream`, `embed`) follow the same
//! pattern: build a `reqwest` request from the shared `Parameters`, send
//! it to the provider's base URL, and parse the OpenAI-shaped response.

pub mod caller;
pub mod charm_oauth;
pub mod tokens;

pub use caller::{chat, chat_stream, chat_stream_with_messages, embed, list_models};
pub use tokens::{extract_tokens, extract_usage};
