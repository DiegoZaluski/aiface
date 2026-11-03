//! A tool call the provider delivered NATIVELY (structured `tool_calls` /
//! `tool_use` / `functionCall` parts). The provider already split the call
//! into structured fields, so no text parsing happens downstream — the
//! record is handed over exactly as the provider streamed it.

/// A tool call the provider delivered NATIVELY (structured `tool_calls` /
/// `tool_use` / `functionCall` parts), validated downstream exactly like an
/// inline-JSON text call.
#[derive(Debug, Clone)]
pub struct NativeToolCall {
    /// Tool call ID from the API's native mechanism.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Raw accumulated `arguments` JSON text as the provider streamed it
    /// (empty when the provider sent no arguments at all — e.g. Ollama).
    pub arguments: String,
    /// Gemini 3.x thought signature (sibling of a native `functionCall`),
    /// replayed verbatim on the follow-up request. Empty for every other
    /// provider.
    pub thought_signature: String,
}
