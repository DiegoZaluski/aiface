//! Unified multi-provider LLM client.
//!
//! Build a [`Connector`] with a provider name, configure parameters via the
//! builder API, then call [`chat`](Connector::chat),
//! [`stream_chat`](Connector::stream_chat), or [`embed`](Connector::embed).
//!
//! All providers share the same interface — swap the name passed to
//! [`Connector::new`] to switch backends.

pub(crate) mod claude;
pub(crate) mod common;
pub(crate) mod copilot;
pub(crate) mod cost;
pub mod decision;
pub(crate) mod discovery;
pub(crate) mod gemini;
pub mod oauth;
pub(crate) mod openai;
pub mod openai_compatible;

mod client;
mod error;
mod output;
mod params;
mod provider;

pub mod retry;

pub use cost::{CostError, ProviderCost, provider_session_cost, supports_cost_reporting};

pub use client::Connector;
pub use crate::NativeToolCall;
// The Copilot family's device-flow adapter is consumed by the app's
// sign-in dispatcher (the family itself stays `pub(crate)` — only its
// OAuth surface is public, like `openai_compatible`'s).
pub use copilot::oauth as copilot_oauth;
pub use discovery::{
    ModelReasoning, discover_context_window, effective_context_window, model_reasoning,
    model_reasoning_from_catalog, refresh_pricing_catalog, resolve_reasoning_effort,
};
pub use error::ConnectorError;
pub use decision::{
    DecisionAnswer, DecisionOutput, DecisionQuestion, DecisionRequest, Probabilities,
    ProbabilitySource,
};
pub use output::{ChatOutput, ChatStream, LsOutput, ModelInfo, StreamChunk};
pub use provider::Family;
// Re-exported so `Connector::token_usage` consumers name the type without
// reaching into the private claude module path.
mod usage;
pub use params::{
    ChatMessage, ClaudeThinkingBlock, ImageBlock, ResponseFormat, ToolCallFunctionMsg,
    ToolCallMode, ToolCallMsg, ToolDefinition, ToolFunction, assistant_tool_call_message,
    system_message, tool_result_message, tool_result_message_with_images, user_message,
};
pub use provider::{
    COSH_SERVICE, OPENCODE_GO_PROVIDER, OPENCODE_ZEN_PROVIDER, clear_api_key_cache,
    custom_provider_names, detect_provider, forget_provider_key, get_api_key, get_provider,
    get_provider_env_var, has_api_key, invalidate_api_key, is_local_provider, is_opencode_gateway,
    known_local_providers, known_providers, known_providers_with_env, normalize_local_base_url,
    register_custom_providers, resolved_provider_base_url,
};
pub use usage::TokenUsage;

#[cfg(test)]
mod test;
