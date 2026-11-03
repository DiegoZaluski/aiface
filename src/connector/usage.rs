//! Unified token-usage accounting across provider families.
//!
//! [`TokenUsage`] normalizes each provider's usage object onto common
//! fields, including the prompt-cache and reasoning counters that matter
//! for cost tracking:
//!
//! - **Claude**: `usage.input_tokens` / `output_tokens`,
//!   `cache_read_input_tokens` / `cache_creation_input_tokens`.
//! - **OpenAI-compatible**: `usage.prompt_tokens` / `completion_tokens`,
//!   `prompt_tokens_details.cached_tokens` (cache read) and
//!   `completion_tokens_details.reasoning_tokens`.
//! - **Gemini**: `usageMetadata.promptTokenCount` /
//!   `candidatesTokenCount`, `cachedContentTokenCount` (cache read) and
//!   `thoughtsTokenCount` (reasoning).
//!
//! Cache *creation* has no OpenAI/Gemini equivalent, so it stays zero for
//! those families.
//!
//! ## Real reported cost
//!
//! Some providers report what they actually billed inside the response
//! itself (OpenRouter's `usage.cost`; the OpenCode Zen/Go gateways append a
//! final frame with a top-level `cost`; Charm Hyper's `usage.cost.usd` — see
//! `extract_reported_cost`). That REAL amount lands in
//! [`TokenUsage::reported_cost`] and is the only price the usage panel ever
//! shows: there is no local price-table fallback.

/// Full token usage from an LLM response (or streaming usage frame),
/// normalized across provider families.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    /// Input tokens NOT served from or written to the prompt cache.
    #[serde(default)]
    pub input_tokens: u32,
    /// Completion tokens.
    #[serde(default)]
    pub output_tokens: u32,
    /// Tokens written to the prompt cache by this request (Anthropic only;
    /// billed at 1.25x base input price with the 5-minute TTL, 2x with 1h).
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from the prompt cache (billed at ~10% of base input
    /// price on Anthropic and OpenAI-compatible providers with caching
    /// enabled). This is where the caching savings show up.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Reasoning/thinking tokens, when the provider bills them separately
    /// from the visible completion (OpenAI `reasoning_tokens`, Gemini
    /// `thoughtsTokenCount`). Included in `output_tokens`.
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// REAL cost (USD) the provider reported inside the response itself
    /// (`usage.cost` on OpenRouter/Vercel; a top-level `cost` on the final
    /// OpenCode Zen/Go frame; `usage.cost.usd` on Charm Hyper). `None` when
    /// the provider does not report a cost — the record then stays unpriced
    /// and is excluded from dollar totals. There is NO price-table fallback:
    /// the provider is the only source of truth for its own billing.
    #[serde(default)]
    pub reported_cost: Option<f64>,
    /// REAL cost in the provider's NATIVE prepaid unit when that is not USD
    /// (Charm Hyper: `usage.cost.hypercredits`). Parallel to
    /// [`TokenUsage::reported_cost`] — both come from the same
    /// `usage.cost` object in one response. `None` when the provider bills
    /// natively in USD (OpenRouter, xAI, …) or reports nothing.
    #[serde(default)]
    pub reported_cost_credits: Option<f64>,
}

impl TokenUsage {
    /// Total input tokens processed:
    /// `input + cache_creation + cache_read`.
    #[must_use]
    pub fn total_input_tokens(&self) -> u64 {
        u64::from(self.input_tokens)
            + u64::from(self.cache_creation_input_tokens)
            + u64::from(self.cache_read_input_tokens)
    }

    /// Merge per-frame usage extracts from a SINGLE streaming response.
    ///
    /// Providers split usage across SSE frames: Claude reports input/cache
    /// counters in the `message_start` frame and the final output count in
    /// `message_delta`; Gemini streams cumulative `usageMetadata` in every
    /// frame. Within one request every counter only grows, so the merge is
    /// a per-field max — never a sum (summing would double-count the
    /// `message_start` input tokens).
    #[must_use]
    pub fn merge_stream(self, next: TokenUsage) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens.max(next.input_tokens),
            output_tokens: self.output_tokens.max(next.output_tokens),
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .max(next.cache_creation_input_tokens),
            cache_read_input_tokens: self
                .cache_read_input_tokens
                .max(next.cache_read_input_tokens),
            reasoning_tokens: self.reasoning_tokens.max(next.reasoning_tokens),
            // Within one request the reported cost only ever grows (it
            // arrives on the FINAL frame); keep the max seen, preserving a
            // value that only one side carries.
            reported_cost: match (self.reported_cost, next.reported_cost) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, b) => b,
            },
            reported_cost_credits: match (self.reported_cost_credits, next.reported_cost_credits) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (Some(a), None) => Some(a),
                (None, b) => b,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_stream_takes_per_field_max() {
        // Claude: message_start carries input + cache counters (output=1),
        // message_delta carries the final output count.
        let start = TokenUsage {
            input_tokens: 1200,
            output_tokens: 1,
            cache_creation_input_tokens: 400,
            cache_read_input_tokens: 8000,
            reasoning_tokens: 0,
            reported_cost: None,
            reported_cost_credits: None,
        };
        let delta = TokenUsage {
            input_tokens: 0,
            output_tokens: 350,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
            reported_cost: None,
            reported_cost_credits: None,
        };
        let merged = start.merge_stream(delta);
        assert_eq!(merged.input_tokens, 1200);
        assert_eq!(merged.output_tokens, 350);
        assert_eq!(merged.cache_creation_input_tokens, 400);
        assert_eq!(merged.cache_read_input_tokens, 8000);
    }

    /// The reported cost rides the same per-field-max merge: a gateway that
    /// only sends it on the final frame must not lose it, and a later frame
    /// can only raise it.
    #[test]
    fn merge_stream_keeps_reported_cost() {
        let none = TokenUsage::default();
        let mut with_cost = TokenUsage {
            reported_cost: Some(0.42),
            ..TokenUsage::default()
        };
        assert_eq!(none.merge_stream(with_cost).reported_cost, Some(0.42));
        let higher = TokenUsage {
            reported_cost: Some(0.95),
            ..TokenUsage::default()
        };
        with_cost = with_cost.merge_stream(higher);
        assert_eq!(with_cost.reported_cost, Some(0.95));
        // A lower value never regresses the total.
        let lower = TokenUsage {
            reported_cost: Some(0.1),
            ..TokenUsage::default()
        };
        assert_eq!(with_cost.merge_stream(lower).reported_cost, Some(0.95));
    }

    #[test]
    fn merge_stream_accumulates_growing_gemini_counters() {
        let a = TokenUsage {
            input_tokens: 50,
            output_tokens: 20,
            ..TokenUsage::default()
        };
        let b = TokenUsage {
            input_tokens: 200,
            output_tokens: 60,
            cache_read_input_tokens: 150,
            reasoning_tokens: 40,
            ..TokenUsage::default()
        };
        let merged = a.merge_stream(b);
        assert_eq!(merged.total_input_tokens(), 350);
        assert_eq!(merged.output_tokens, 60);
        assert_eq!(merged.reasoning_tokens, 40);
    }
}
