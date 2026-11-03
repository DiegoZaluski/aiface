//! OpenAI-native Responses API caller.
//!
//! The `openai` provider talks to the current `/v1/responses` endpoint (the
//! "Responses API") rather than the legacy `/v1/chat/completions` used by the
//! rest of the `OpenAICompatible` family. The Responses API is what exposes
//! reasoning for OpenAI's reasoning models: reasoning summaries stream as
//! `response.reasoning_summary_text.delta` events and arrive as `reasoning`
//! output items, which the caller surfaces as [`StreamChunk::reasoning`]
//! so the TUI can render a live "Thought" block.
//!
//! `embed` and `list_models` are not part of the Responses API — they keep
//! using the shared `/v1/embeddings` and `/v1/models` endpoints via the
//! `openai_compatible` module.

pub mod caller;
pub mod tokens;

pub use caller::{chat, chat_stream, chat_stream_with_messages};
pub use tokens::{extract_tokens, extract_usage};
