use crate::connector::usage::TokenUsage;

/// Extract completion tokens from a raw `OpenAI`-compatible API response.
///
/// Looks for `usage.completion_tokens` in the JSON response body — the
/// standard `OpenAI` shape that most compatible providers follow. Returns
/// `None` when the field is absent or zero (no completion yet).
#[must_use]
pub fn extract_tokens(raw: &str) -> Option<u32> {
    extract_usage(raw)
        .map(|u| u.output_tokens)
        .filter(|n| *n > 0)
}

/// Extract the full token usage from a raw `OpenAI`-compatible API response
/// or streaming usage frame, including cache and reasoning details when the
/// provider reports them (`prompt_tokens_details.cached_tokens`,
/// `completion_tokens_details.reasoning_tokens`). Returns `None` when no
/// usage object is present or the JSON is malformed.
#[must_use]
pub fn extract_usage(raw: &str) -> Option<TokenUsage> {
    #[derive(serde::Deserialize)]
    struct Details {
        cached_tokens: Option<u32>,
        reasoning_tokens: Option<u32>,
    }
    #[derive(serde::Deserialize)]
    struct Usage {
        #[serde(default)]
        prompt_tokens: u32,
        #[serde(default)]
        completion_tokens: u32,
        #[serde(default)]
        prompt_tokens_details: Option<Details>,
        #[serde(default)]
        completion_tokens_details: Option<Details>,
    }
    let v: serde_json::Value = serde_json::from_str(raw).ok()?;
    // Kimi (docs: platform.kimi.ai/docs/guide/context-caching, streaming) is
    // OpenAI-compatible but historically ALSO duplicated `usage` inside each
    // choice (their forum: "usage information is under choices"). Top-level
    // `usage` is tried first — OpenAI's shape, and Kimi's when
    // `stream_options.include_usage` is set — with the per-choice copy as a
    // fallback so a Kimi stream frame that only carries it there still
    // resolves. Within one choice set only completion/total differ, so any
    // choice works.
    let usage_value = if v.get("usage").is_some_and(|u| u.is_object()) {
        v["usage"].clone()
    } else {
        v["choices"]
            .get(0)
            .and_then(|c| c.get("usage"))
            .filter(|u| u.is_object())
            .cloned()?
    };
    let usage: Usage = serde_json::from_value(usage_value).ok()?;
    // OpenAI's `prompt_tokens` INCLUDES cached tokens
    // (prompt_tokens_details.cached_tokens ⊆ prompt_tokens), unlike
    // Anthropic where input excludes them. Normalize onto the shared
    // TokenUsage contract (input = NOT served from cache) so
    // total_input_tokens() stays correct.
    let cache_read = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens)
        .unwrap_or(0);
    let reasoning = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0);
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return None;
    }
    Some(TokenUsage {
        input_tokens: usage.prompt_tokens.saturating_sub(cache_read),
        output_tokens: usage.completion_tokens,
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
    fn extracts_completion_tokens() {
        let json = r#"{"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;
        assert_eq!(extract_tokens(json), Some(20));
    }

    #[test]
    fn missing_usage_returns_none() {
        assert_eq!(extract_tokens(r#"{"choices":[]}"#), None);
    }

    #[test]
    fn invalid_json_returns_none() {
        assert_eq!(extract_tokens("not json"), None);
    }

    #[test]
    fn zero_completion_tokens_is_none() {
        let json = r#"{"usage":{"prompt_tokens":5,"completion_tokens":0}}"#;
        assert_eq!(extract_tokens(json), None);
    }

    #[test]
    fn extracts_cache_and_reasoning_details() {
        let json = r#"{"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":80},"completion_tokens_details":{"reasoning_tokens":30}}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 20);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_read_input_tokens, 80);
        assert_eq!(usage.reasoning_tokens, 30);
        assert_eq!(usage.total_input_tokens(), 100);
    }

    #[test]
    fn null_details_do_not_break_extraction() {
        let json = r#"{"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":null}}}"#;
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
        let json = r#"{"usage":{"prompt_tokens":10,"completion_tokens":2}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.cache_read_input_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
    }

    /// Kimi (Moonshot) historically duplicated `usage` INSIDE each choice
    /// instead of top level (docs: platform.kimi.ai/docs/guide/utilize-the-
    /// streaming-output-feature-of-kimi-api; Kimi forum thread "API not fully
    /// OpenAI-compatible"). With `stream_options.include_usage` the top-level
    /// copy appears — but a frame that only carries the per-choice copy must
    /// still resolve, otherwise token accounting silently drops to nothing.
    #[test]
    fn kimi_per_choice_usage_falls_back() {
        let json =
            r#"{"choices":[{"delta":{},"usage":{"prompt_tokens":100,"completion_tokens":20}}]}"#;
        let usage = extract_usage(json).expect("per-choice usage must resolve");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 20);
    }

    /// The top-level usage always WINS over a per-choice copy: the OpenAI
    /// shape (and Kimi's final include_usage frame, which this caller always
    /// requests) must not be mis-parsed through the fallback.
    #[test]
    fn top_level_usage_wins_over_choice_copy() {
        let json = r#"{"choices":[{"delta":{},"usage":{"prompt_tokens":1,"completion_tokens":1}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let usage = extract_usage(json).unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    /// A per-choice usage with BOTH counters zero (a degenerate/keep-alive
    /// choice frame) is not a usable accounting record — same None contract
    /// as the empty top-level usage.
    #[test]
    fn kimi_per_choice_empty_usage_is_none() {
        let json = r#"{"choices":[{"delta":{},"usage":{}}]}"#;
        assert_eq!(extract_usage(json), None);
    }

    /// A `usage: null` mid-stream frame (some gateways emit it on every
    /// text delta) must not consume the fallback with a non-object: the
    /// choice copy is then the only signal, and with neither present the
    /// result stays None rather than a deserialization error.
    #[test]
    fn null_usage_frame_without_choice_usage_is_none() {
        let json = r#"{"choices":[{"delta":{"content":"x"}}],"usage":null}"#;
        assert_eq!(extract_usage(json), None);
    }
}
