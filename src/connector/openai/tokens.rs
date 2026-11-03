use crate::connector::usage::TokenUsage;

/// Extract output tokens from a raw OpenAI Responses API response.
///
/// Looks for `usage.output_tokens` (the Responses field name, distinct from
/// the Chat Completions `completion_tokens`). Returns `None` when the field
/// is absent or zero.
#[must_use]
pub fn extract_tokens(raw: &str) -> Option<u32> {
    extract_usage(raw)
        .map(|u| u.output_tokens)
        .filter(|n| *n > 0)
}

/// Extract the full token usage from a raw Responses API response or the
/// final streaming frame, including cached-input and reasoning counters:
/// `usage.input_tokens` / `output_tokens`,
/// `input_tokens_details.cached_tokens` (cache read) and
/// `output_tokens_details.reasoning_tokens`. Returns `None` when no usage
/// object is present or the JSON is malformed.
#[must_use]
pub fn extract_usage(raw: &str) -> Option<TokenUsage> {
    #[derive(serde::Deserialize)]
    struct InputDetails {
        #[serde(default)]
        cached_tokens: Option<u32>,
    }
    #[derive(serde::Deserialize)]
    struct OutputDetails {
        #[serde(default)]
        reasoning_tokens: Option<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Usage {
        #[serde(default)]
        input_tokens: u32,
        #[serde(default)]
        output_tokens: u32,
        #[serde(default)]
        input_tokens_details: Option<InputDetails>,
        #[serde(default)]
        output_tokens_details: Option<OutputDetails>,
    }
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    // Usage sits at the top level of a non-streaming response but is nested
    // under `response.usage` in a streaming `response.completed` event.
    let usage_val = v
        .get("usage")
        .cloned()
        .or_else(|| v.get("response").and_then(|r| r.get("usage")).cloned())?;
    let usage: Usage = serde_json::from_value(usage_val).ok()?;
    // Responses `input_tokens` INCLUDES cached tokens
    // (input_tokens_details.cached_tokens ⊆ input_tokens), unlike Anthropic
    // where input excludes them. Normalize onto the shared TokenUsage
    // contract (input = NOT served from cache) so total_input_tokens() stays
    // correct.
    let cache_read = usage
        .input_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .output_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);
    if usage.input_tokens == 0 && usage.output_tokens == 0 {
        return None;
    }
    Some(TokenUsage {
        input_tokens: usage.input_tokens.saturating_sub(cache_read),
        output_tokens: usage.output_tokens,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: cache_read,
        reasoning_tokens: reasoning,
        reported_cost: None,
        reported_cost_credits: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_output_tokens() {
        let json = r#"{"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}"#;
        assert_eq!(extract_tokens(json), Some(20));
    }

    #[test]
    fn missing_usage_returns_none() {
        assert_eq!(extract_tokens(r#"{"output":[]}"#), None);
    }

    #[test]
    fn invalid_json_returns_none() {
        assert_eq!(extract_tokens("not json"), None);
    }

    #[test]
    fn zero_output_tokens_is_none() {
        let json = r#"{"usage":{"input_tokens":5,"output_tokens":0}}"#;
        assert_eq!(extract_tokens(json), None);
    }

    #[test]
    fn extracts_cache_and_reasoning_details() {
        let json = r#"{"usage":{"input_tokens":100,"output_tokens":50,"input_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":30}}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_read_input_tokens, 80);
        assert_eq!(usage.reasoning_tokens, 30);
        assert_eq!(usage.total_input_tokens(), 100);
    }

    #[test]
    fn null_details_do_not_break_extraction() {
        let json = r#"{"usage":{"input_tokens":10,"output_tokens":2,"input_tokens_details":{"cached_tokens":null}}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn empty_usage_is_none() {
        assert_eq!(extract_usage(r#"{"usage":{}}"#), None);
    }

    #[test]
    fn absent_details_default_to_zero() {
        let json = r#"{"usage":{"input_tokens":10,"output_tokens":2}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
    }
}
