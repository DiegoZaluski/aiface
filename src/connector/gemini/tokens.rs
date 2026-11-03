use crate::connector::usage::TokenUsage;

/// Extract completion tokens from a raw Gemini API response.
///
/// Looks for `usageMetadata.candidatesTokenCount` in the JSON response body
/// (camelCase keys from the Gemini API). Returns `None` when the field is
/// absent or zero (no completion yet).
#[must_use]
pub fn extract_tokens(raw: &str) -> Option<u32> {
    extract_usage(raw)
        .map(|u| u.output_tokens)
        .filter(|n| *n > 0)
}

/// Extract the full token usage from a raw Gemini API response, including
/// cache and reasoning counters when the provider reports them
/// (`cachedContentTokenCount`, `thoughtsTokenCount`). Returns `None` when
/// no `usageMetadata` object is present or the JSON is malformed.
#[must_use]
pub fn extract_usage(raw: &str) -> Option<TokenUsage> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct UsageMetadata {
        #[serde(default)]
        prompt_token_count: u32,
        #[serde(default)]
        candidates_token_count: u32,
        #[serde(default)]
        cached_content_token_count: u32,
        #[serde(default)]
        thoughts_token_count: u32,
    }
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    let meta: UsageMetadata = serde_json::from_value(v["usageMetadata"].clone()).ok()?;
    // Gemini's `promptTokenCount` INCLUDES cached tokens
    // (cachedContentTokenCount ⊆ promptTokenCount), unlike Anthropic where
    // input excludes them. Normalize onto the shared TokenUsage contract
    // (input = NOT served from cache) so total_input_tokens() stays correct.
    if meta.prompt_token_count == 0 && meta.candidates_token_count == 0 {
        return None;
    }
    Some(TokenUsage {
        input_tokens: meta
            .prompt_token_count
            .saturating_sub(meta.cached_content_token_count),
        output_tokens: meta.candidates_token_count,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: meta.cached_content_token_count,
        reasoning_tokens: meta.thoughts_token_count,
        reported_cost: None,
        reported_cost_credits: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_candidates_token_count() {
        let json = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}"#;
        assert_eq!(extract_tokens(json), Some(20));
    }

    #[test]
    fn missing_usage_returns_none() {
        assert_eq!(extract_tokens(r#"{"candidates":[]}"#), None);
    }

    #[test]
    fn invalid_json_returns_none() {
        assert_eq!(extract_tokens("not json"), None);
    }

    #[test]
    fn zero_candidates_is_none() {
        let json = r#"{"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0}}"#;
        assert_eq!(extract_tokens(json), None);
    }

    #[test]
    fn extracts_cache_and_thoughts_counters() {
        let json = r#"{"usageMetadata":{"promptTokenCount":200,"candidatesTokenCount":60,"cachedContentTokenCount":150,"thoughtsTokenCount":40}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.output_tokens, 60);
        assert_eq!(usage.cache_read_input_tokens, 150);
        assert_eq!(usage.reasoning_tokens, 40);
        assert_eq!(usage.total_input_tokens(), 200);
    }

    #[test]
    fn empty_usage_metadata_is_none() {
        assert_eq!(extract_usage(r#"{"usageMetadata":{}}"#), None);
    }

    #[test]
    fn absent_counters_default_to_zero() {
        let json = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
    }
}
