//! Token usage extraction for the Claude caller.
//!
//! [`extract_tokens`] keeps the historical completion-only count;
//! [`extract_usage`] returns the full [`TokenUsage`], including the
//! prompt-cache fields (`cache_read_input_tokens`,
//! `cache_creation_input_tokens`) used to track cache savings.

use crate::connector::usage::TokenUsage;

fn parse_usage(raw: &str) -> Option<TokenUsage> {
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    // `usage` may sit at the root (full responses), under "message" (the
    // message_start SSE frame nests it as `message.usage`), or at the delta
    // level of a final message_delta frame (`usage.output_tokens` only).
    if let Ok(u) = serde_json::from_value::<TokenUsage>(v["usage"].clone()) {
        return Some(u);
    }
    if let Ok(u) = serde_json::from_value::<TokenUsage>(v["message"]["usage"].clone()) {
        return Some(u);
    }
    None
}

/// Extract completion tokens from a raw Claude API response.
///
/// Looks for `usage.output_tokens` in the JSON response body (root-level
/// `usage`, the nested `message.usage` of a streaming `message_start`
/// frame, or a final `message_delta` frame's `usage`). Returns `None` when
/// the field is absent or zero — zero output means "no completion yet"
/// (e.g. a pre-warm request), which callers treat as no usage.
#[must_use]
pub fn extract_tokens(raw: &str) -> Option<u32> {
    parse_usage(raw).map(|u| u.output_tokens).filter(|n| *n > 0)
}

/// Extract the full token usage (including prompt-cache accounting) from a
/// raw Claude API response or streaming frame. Returns `None` when no usage
/// object is present or the JSON is malformed.
#[must_use]
pub fn extract_usage(raw: &str) -> Option<TokenUsage> {
    parse_usage(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_output_tokens() {
        let json = r#"{"usage":{"input_tokens":10,"output_tokens":20}}"#;
        assert_eq!(extract_tokens(json), Some(20));
    }

    #[test]
    fn missing_usage_returns_none() {
        assert_eq!(extract_tokens(r#"{"id":"msg_1"}"#), None);
    }

    #[test]
    fn invalid_json_returns_none() {
        assert_eq!(extract_tokens("not json"), None);
    }

    #[test]
    fn extracts_cache_fields() {
        let json = r#"{"usage":{"input_tokens":50,"output_tokens":503,"cache_read_input_tokens":1800,"cache_creation_input_tokens":248}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.output_tokens, 503);
        assert_eq!(usage.cache_read_input_tokens, 1800);
        assert_eq!(usage.cache_creation_input_tokens, 248);
        assert_eq!(usage.total_input_tokens(), 2098);
    }

    #[test]
    fn extracts_from_message_start_frame() {
        let frame = r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":20,"output_tokens":1,"cache_read_input_tokens":5000}}}"#;
        let usage = extract_usage(frame).unwrap();
        assert_eq!(usage.cache_read_input_tokens, 5000);
        assert_eq!(extract_tokens(frame), Some(1));
    }

    #[test]
    fn extracts_from_message_delta_frame() {
        let frame = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#;
        assert_eq!(extract_usage(frame).unwrap().output_tokens, 15);
        assert_eq!(extract_tokens(frame), Some(15));
    }

    #[test]
    fn zero_output_tokens_is_none() {
        assert_eq!(
            extract_tokens(r#"{"usage":{"input_tokens":8,"output_tokens":0}}"#),
            None
        );
    }

    #[test]
    fn absent_cache_fields_default_to_zero() {
        let usage = extract_usage(r#"{"usage":{"input_tokens":10,"output_tokens":5}}"#).unwrap();
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.cache_creation_input_tokens, 0);
    }
}
