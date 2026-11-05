use super::TokenUsage;
use super::cost::{CostError, ProviderCost, provider_session_cost};
use super::decision::DecisionRequest;
use super::error::ConnectorError;
use super::output::{ChatOutput, ChatStream, LsOutput};
use super::params::{ChatMessage, Parameters, ResponseFormat, ToolCallMode, ToolDefinition};
use super::provider::{Family, ProviderConfig, get_provider};

use super::claude;
use super::copilot;
use super::gemini;
use super::openai;
use super::openai_compatible;

/// A unified client for any LLM provider.
///
/// All providers share the same API — just change the name passed to `new()`.
///
/// ```
/// # use aiface::connector::Connector;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let reply = Connector::new("openai")?
///     .with_model("gpt-4o")
///     .with_temperature(0.7)
///     .with_api_key("sk-...")
///     .chat("Explain Rust ownership")
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct Connector {
    provider: Option<&'static ProviderConfig>, // None before new() succeeds
    pub(crate) params: Parameters,
}

impl Connector {
    pub(crate) fn provider(&self) -> Result<&'static ProviderConfig, ConnectorError> {
        self.provider.ok_or(ConnectorError::NotImplemented(
            "Connector used without calling new()",
        ))
    }

    /// Create a new Connector for the given provider name.
    ///
    /// Supported providers: `claude`, `openai`, `groq`, `mistral`, `together`, `openrouter`,
    /// `xai`, `deepseek`, `perplexity`, `fireworks`, `cohere`, `huggingface`,
    /// `sambanova`, `poe`, `cerebras`, `nvidia`, `anyscale`, `vercel`, `cloudflare`,
    /// `azure`, `ollama`, `lmstudio`, `vllm`, `llamacpp`, `gemini`, `zai`,
    /// `kimi`, `zen`, `opencode-go`.
    ///
    /// # Errors
    ///
    /// Returns `UnknownProvider` if the name is not recognized.
    pub fn new(provider: &str) -> Result<Self, ConnectorError> {
        let cfg = get_provider(provider)
            .ok_or_else(|| ConnectorError::UnknownProvider(provider.to_string()))?;
        Ok(Self {
            provider: Some(cfg),
            params: Parameters::default(),
        })
    }

    /// Set the model to use (e.g., `"gpt-4o"`, `"claude-3-opus"`).
    ///
    /// Each provider has a sensible default — call this only when you need a
    /// different model.
    pub fn with_model(mut self, v: impl Into<String>) -> Self {
        self.params.model = Some(v.into());
        self
    }

    /// Maximum number of tokens to generate.
    pub const fn with_max_tokens(mut self, v: u32) -> Self {
        self.params.max_tokens = Some(v);
        self
    }

    /// Output ceiling the adapter will send, including any reasoning floor.
    /// `None` means the adapter delegates the default limit to the provider.
    /// This is a reservation, not a prediction of actual billed output.
    #[must_use]
    pub fn effective_max_tokens(&self) -> Option<u32> {
        match self.provider?.family {
            Family::Claude => Some(claude::caller::request_max_tokens(
                &self.params,
                self.effective_model()?,
            )),
            _ => self.params.max_tokens,
        }
    }

    /// Sampling temperature (0.0 – 2.0). Higher values make output more random.
    pub const fn with_temperature(mut self, v: f32) -> Self {
        self.params.temperature = Some(v);
        self
    }

    /// Nucleus sampling threshold (0.0 – 1.0).
    pub const fn with_top_p(mut self, v: f32) -> Self {
        self.params.top_p = Some(v);
        self
    }

    /// Sequences where the API will stop generating.
    pub fn with_stop(mut self, v: serde_json::Value) -> Self {
        self.params.stop = Some(v);
        self
    }

    /// Frequency penalty (-2.0 – 2.0). Positive values penalize frequent tokens.
    pub const fn with_frequency_penalty(mut self, v: f32) -> Self {
        self.params.frequency_penalty = Some(v);
        self
    }

    /// Presence penalty (-2.0 – 2.0). Positive values penalize tokens that have appeared.
    pub const fn with_presence_penalty(mut self, v: f32) -> Self {
        self.params.presence_penalty = Some(v);
        self
    }

    /// Seed for deterministic sampling.
    pub const fn with_seed(mut self, v: i64) -> Self {
        self.params.seed = Some(v);
        self
    }

    /// Constrain the model to produce a specific output format (e.g., JSON).
    pub fn with_response_format(mut self, v: ResponseFormat) -> Self {
        self.params.response_format = Some(v);
        self
    }

    /// Start a DECISION request: typed questions in, probability
    /// distributions out — the System One contract.
    ///
    /// The backend is chosen by the connector's provider, invisibly:
    /// - a decision provider (`typesafe`, `azure-decision`, or
    ///   `openrouter`'s System One route) calls the native decisions API —
    ///   calibrated probabilities from a purpose-built scoring model;
    /// - any other provider (all 28 chat backends) routes the same
    ///   questions through the LLM wrapper — the model's SELF-REPORTED
    ///   probabilities, flagged as [`ProbabilitySource::DerivedFromLlm`] in
    ///   the output.
    ///
    /// Both paths return the same [`DecisionOutput`], validated against
    /// the questions declared on the builder.
    ///
    /// ```
    /// # async fn example() -> Result<(), aiface::connector::ConnectorError> {
    /// # use aiface::connector::Connector;
    /// let output = Connector::new("typesafe")?
    ///     .with_api_key("ts-...")
    ///     .decide("Customer says their production service is down.")
    ///     .noul("urgent", "Is this request urgent?")
    ///     .choice("team", "Which team should handle this?", [
    ///         ("billing", "Charges and refunds"),
    ///         ("technical", "Bugs and outages"),
    ///     ])
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn decide(&self, state: impl Into<serde_json::Value>) -> DecisionRequest<'_> {
        DecisionRequest::new(self, state.into())
    }

    /// Whether to return log probabilities of the output tokens.
    pub const fn with_logprobs(mut self, v: bool) -> Self {
        self.params.logprobs = Some(v);
        self
    }

    /// Number of most probable tokens to return log probabilities for.
    pub const fn with_top_logprobs(mut self, v: u32) -> Self {
        self.params.top_logprobs = Some(v);
        self
    }

    /// Tool definitions the model may call.
    pub fn with_tools(mut self, v: Vec<ToolDefinition>) -> Self {
        self.params.tools = Some(v);
        self
    }

    /// Set tool definitions in-place (mutates the existing connector).
    pub fn set_tools(&mut self, v: Vec<ToolDefinition>) {
        self.params.tools = Some(v);
    }

    /// Controls which tool is called (e.g., `"auto"`, `"none"`, or a specific tool).
    pub fn with_tool_choice(mut self, v: serde_json::Value) -> Self {
        self.params.tool_choice = Some(v);
        self
    }

    /// Reasoning effort for models that support it (`"low"`, `"medium"`,
    /// `"high"`).
    ///
    /// Each family maps this onto its native knob: OpenAI-compatible sends
    /// the top-level `reasoning_effort`, Gemini sends
    /// `generationConfig.thinkingConfig.thinkingLevel`, and Claude enables
    /// extended thinking with a matching token budget. Models without a
    /// reasoning knob ignore it.
    pub fn with_reasoning_effort(mut self, v: impl Into<String>) -> Self {
        self.params.reasoning_effort = Some(v.into());
        self
    }

    /// The reasoning effort set via [`with_reasoning_effort`](Self::with_reasoning_effort).
    ///
    /// Lets callers (e.g. the harness) re-apply the same level when the
    /// connector is rebuilt for a fallback provider or model.
    #[must_use]
    pub fn reasoning_effort(&self) -> Option<&str> {
        self.params.reasoning_effort.as_deref()
    }

    /// A unique identifier for the end-user (for monitoring/abuse detection).
    pub fn with_user(mut self, v: impl Into<String>) -> Self {
        self.params.user = Some(v.into());
        self
    }

    /// Override the base URL for the API endpoint.
    pub fn with_base_url(mut self, v: impl Into<String>) -> Self {
        self.params.base_url = Some(v.into());
        self
    }

    /// Set the API key explicitly (otherwise resolved from env vars).
    pub fn with_api_key(mut self, v: impl Into<String>) -> Self {
        self.params.api_key = Some(v.into());
        self
    }

    /// Set an opaque session identifier sent as `x-session-id` /
    /// `x-session-affinity` headers on every request, enabling cache-affinity
    /// routing at gateways/proxies. Pass a non-identifying value (hash or
    /// UUID), never raw user data.
    pub fn with_session_id(mut self, v: impl Into<String>) -> Self {
        self.params.session_id = Some(v.into());
        self
    }

    /// The opaque cache-affinity session id set via
    /// [`with_session_id`](Self::with_session_id).
    ///
    /// Lets callers (e.g. the harness) re-apply the same affinity routing
    /// when a connector is rebuilt for a fallback provider or for
    /// summarization — the value belongs to the SESSION, not the model.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.params.session_id.as_deref()
    }

    pub fn with_service_keyring(mut self, v: impl Into<String>) -> Self {
        self.params.service_keyring = Some(v.into());
        self
    }

    /// Enable/disable Anthropic prompt caching on Claude requests (default:
    /// enabled).
    ///
    /// Enabled requests carry a single top-level `cache_control: {"type":
    // "ephemeral"}` — the API's automatic-caching mode (the same field the
    /// Messages API reference documents at the request root). The cache
    /// breakpoint moves to the last cacheable block on every request, so
    /// the growing conversation prefix is read from cache at 10% of the
    /// base input price. Other providers are unaffected.
    ///
    /// Note: changing `tool_choice`, the thinking configuration, or the
    /// reasoning effort between turns invalidates the cached message
    /// prefix (per Anthropic's invalidation rules) and forces a re-write.
    pub fn with_prompt_cache(mut self, enabled: bool) -> Self {
        self.params.prompt_cache.enabled = enabled;
        self
    }

    /// Use the 1-hour prompt-cache TTL instead of the 5-minute default
    /// (Claude only).
    ///
    /// 1h cache writes cost 2x the base input price (vs 1.25x for 5m), but
    /// survive turn gaps longer than 5 minutes. Cache reads cost the same
    /// 10% either way.
    pub fn with_prompt_cache_ttl_1h(mut self, one_hour: bool) -> Self {
        self.params.prompt_cache.one_hour = one_hour;
        self
    }

    /// The 1-hour prompt-cache TTL state set via
    /// [`with_prompt_cache_ttl_1h`](Self::with_prompt_cache_ttl_1h).
    ///
    /// Lets callers (e.g. the harness) re-apply the same choice when the
    /// connector is rebuilt for a fallback provider or model.
    #[must_use]
    pub fn prompt_cache_ttl_1h(&self) -> bool {
        self.params.prompt_cache.one_hour
    }

    /// Send the OpenAI Responses-API `prompt_cache_key` on requests.
    ///
    /// The key routes a session's requests to the machines holding its
    /// cached prompt prefixes, which directly improves cache hit rates;
    /// GPT-5.6 and later require it for reliable cache-breakpoint matching.
    /// Pass a stable, non-identifying value (a session UUID or hash) shared
    /// by every request of the same conversation. Other providers ignore
    /// the setting entirely.
    pub fn with_prompt_cache_key(mut self, v: impl Into<String>) -> Self {
        self.params.prompt_cache_key = Some(v.into());
        self
    }

    /// The cache-affinity key set via
    /// [`with_prompt_cache_key`](Self::with_prompt_cache_key).
    #[must_use]
    pub fn prompt_cache_key(&self) -> Option<&str> {
        self.params.prompt_cache_key.as_deref()
    }

    /// Extend the OpenAI prompt-cache retention (Responses API only, e.g.
    /// `"24h"`).
    ///
    /// Longer retention keeps cached prefixes alive across hours-long gaps,
    /// so a resumed session hits the cache again instead of reprocessing
    /// the whole context. Unset = the model-dependent default. Only
    /// supported by recent OpenAI models — older ones ignore the field.
    pub fn with_prompt_cache_retention(mut self, v: impl Into<String>) -> Self {
        self.params.openai_cache_retention = Some(v.into());
        self
    }

    /// The retention set via
    /// [`with_prompt_cache_retention`](Self::with_prompt_cache_retention).
    #[must_use]
    pub fn prompt_cache_retention(&self) -> Option<&str> {
        self.params.openai_cache_retention.as_deref()
    }

    /// Send a chat completion request with a user prompt.
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// `Deserialization` if the response is malformed, or `Network` on transport failure.
    pub async fn chat(&self, prompt: &str) -> Result<ChatOutput, ConnectorError> {
        let provider = self.provider()?;
        let service = self.params.service_keyring.as_deref();
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat(provider, &self.params, prompt, None, service).await
            }
            Family::OpenAi => openai::chat(provider, &self.params, prompt, None, service).await,
            Family::Gemini => gemini::chat(provider, &self.params, prompt, None, service).await,
            Family::Claude => claude::chat(provider, &self.params, prompt, None, service).await,
            Family::Copilot => copilot::chat(provider, &self.params, prompt, None, service).await,
            // The System One family's only surface is decisions — no chat.
            Family::SystemOne => Err(ConnectorError::NotImplemented("chat on a decision provider")),
        }
    }

    /// Send a chat completion request with both system and user prompts.
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// `Deserialization` if the response is malformed, or `Network` on transport failure.
    pub async fn chat_with_system(
        &self,
        prompt: &str,
        system: &str,
    ) -> Result<ChatOutput, ConnectorError> {
        let provider = self.provider()?;
        let service = self.params.service_keyring.as_deref();
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat(provider, &self.params, prompt, Some(system), service).await
            }
            Family::OpenAi => {
                openai::chat(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Gemini => {
                gemini::chat(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Claude => {
                claude::chat(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Copilot => {
                copilot::chat(provider, &self.params, prompt, Some(system), service).await
            }
            Family::SystemOne => Err(ConnectorError::NotImplemented("chat on a decision provider")),
        }
    }

    /// Generate an embedding vector for the given input text.
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// `Deserialization` if the response is malformed, or `Network` on transport failure.
    pub async fn embed(&self, input: &str) -> Result<Vec<f32>, ConnectorError> {
        let provider = self.provider()?;
        let service = self.params.service_keyring.as_deref();

        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::embed(provider, &self.params, input, service).await
            }
            // Embeddings are not part of the Responses API — the `openai`
            // provider keeps using the shared `/v1/embeddings` endpoint.
            Family::OpenAi => {
                openai_compatible::embed(provider, &self.params, input, service).await
            }
            Family::Gemini => gemini::embed(provider, &self.params, input, service).await,
            Family::Claude => Err(ConnectorError::NotImplemented("embedding")),
            // The Copilot runtime has no embeddings surface.
            Family::Copilot => Err(ConnectorError::NotImplemented("embedding")),
            // The System One family's only surface is decisions — no embeddings.
            Family::SystemOne => Err(ConnectorError::NotImplemented("embedding")),
        }
    }

    /// Stream a chat completion with a user prompt.
    ///
    /// Returns a [`ChatStream`](crate::connector::output::ChatStream) that yields [`StreamChunk`](crate::connector::output::StreamChunk) items. After the
    /// stream ends, call [`raw`](ChatStream::raw) to get the last SSE frame
    /// (typically containing `usage`, `finish_reason`, etc.).
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// or `Network` on transport failure before the stream starts.
    pub async fn stream_chat(&self, prompt: &str) -> Result<ChatStream, ConnectorError> {
        let provider = self.provider()?;
        // The explicit `service` argument wins if provided, otherwise fall
        // back to the connector's configured keyring service.
        let service = self.params.service_keyring.as_deref();
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat_stream(provider, &self.params, prompt, None, service).await
            }
            Family::OpenAi => {
                openai::chat_stream(provider, &self.params, prompt, None, service).await
            }
            Family::Gemini => {
                gemini::chat_stream(provider, &self.params, prompt, None, service).await
            }
            Family::Claude => {
                claude::chat_stream(provider, &self.params, prompt, None, service).await
            }
            Family::Copilot => {
                copilot::chat_stream(provider, &self.params, prompt, None, service).await
            }
            Family::SystemOne => Err(ConnectorError::NotImplemented("stream_chat on a decision provider")),
        }
    }

    /// Stream a chat completion with both system and user prompts.
    ///
    /// Returns a [`ChatStream`](crate::connector::output::ChatStream) that yields [`StreamChunk`](crate::connector::output::StreamChunk) items. After the
    /// stream ends, call [`raw`](ChatStream::raw) to get the last SSE frame
    /// (typically containing `usage`, `finish_reason`, etc.).
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// or `Network` on transport failure before the stream starts.
    pub async fn stream_chat_with_system(
        &self,
        prompt: &str,
        system: &str,
    ) -> Result<ChatStream, ConnectorError> {
        let provider = self.provider()?;
        let service = self.params.service_keyring.as_deref();
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat_stream(
                    provider,
                    &self.params,
                    prompt,
                    Some(system),
                    service,
                )
                .await
            }
            Family::OpenAi => {
                openai::chat_stream(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Gemini => {
                gemini::chat_stream(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Claude => {
                claude::chat_stream(provider, &self.params, prompt, Some(system), service).await
            }
            Family::Copilot => {
                copilot::chat_stream(provider, &self.params, prompt, Some(system), service).await
            }
            Family::SystemOne => Err(ConnectorError::NotImplemented("stream_chat on a decision provider")),
        }
    }

    /// Stream a chat completion with a system prompt and NO tool definitions.
    ///
    /// Used by the LLM summarizer (a separate agent): it must never see the
    /// agent loop's tool schemas, or it will answer with tool calls instead of
    /// prose. The connector's own `params.tools` is untouched — the request
    /// is built from a cloned params with `tools`/`tool_choice` cleared, so
    /// the shared connector keeps its tools for the main loop.
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// or `Network` on transport failure before the stream starts.
    pub async fn stream_chat_with_system_no_tools(
        &self,
        prompt: &str,
        system: &str,
    ) -> Result<ChatStream, ConnectorError> {
        let provider = self.provider()?;
        let mut params = self.params.clone();
        params.tools = None;
        params.tool_choice = None;
        let service = self.params.service_keyring.as_deref();
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat_stream(provider, &params, prompt, Some(system), service)
                    .await
            }
            Family::OpenAi => {
                openai::chat_stream(provider, &params, prompt, Some(system), service).await
            }
            Family::Gemini => {
                gemini::chat_stream(provider, &params, prompt, Some(system), service).await
            }
            Family::Claude => {
                claude::chat_stream(provider, &params, prompt, Some(system), service).await
            }
            Family::Copilot => {
                copilot::chat_stream(provider, &params, prompt, Some(system), service).await
            }
            Family::SystemOne => Err(ConnectorError::NotImplemented("stream_chat on a decision provider")),
        }
    }

    /// Stream a chat completion with a full messages array (including system,
    /// user, assistant with `tool_calls`, and tool roles).
    ///
    /// The `system` parameter provides the base system prompt (instructions +
    /// tool definitions). The `messages` array contains the conversation
    /// history with proper roles, tool calls, and tool results.
    ///
    /// Use this method instead of [`stream_chat_with_system`](Connector::stream_chat_with_system) when the model
    /// needs to see structured tool call history (`role: "tool"` messages).
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// or `Network` on transport failure before the stream starts.
    pub async fn stream_chat_with_messages(
        &self,
        system: &str,
        messages: &[ChatMessage],
    ) -> Result<ChatStream, ConnectorError> {
        let provider = self.provider()?;
        let params = &self.params;
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::chat_stream_with_messages(
                    provider,
                    params,
                    system,
                    messages,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::OpenAi => {
                openai::chat_stream_with_messages(
                    provider,
                    params,
                    system,
                    messages,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Gemini => {
                gemini::chat_stream_with_messages(
                    provider,
                    params,
                    system,
                    messages,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Claude => {
                claude::chat_stream_with_messages(
                    provider,
                    params,
                    system,
                    messages,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Copilot => {
                copilot::chat_stream_with_messages(
                    provider,
                    params,
                    system,
                    messages,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::SystemOne => Err(ConnectorError::NotImplemented("stream_chat on a decision provider")),
        }
    }

    /// Extract completion tokens from a raw API response (chat or stream).
    ///
    /// Dispatches to the correct provider-family extractor based on the
    /// configured [`Family`](super::provider::Family):
    /// - `OpenAICompatible` → `usage.completion_tokens`
    /// - `Claude` → `usage.output_tokens`
    /// - `Gemini` → `usageMetadata.candidatesTokenCount`
    ///
    /// Returns `None` when the field is absent or the raw JSON is malformed.
    #[must_use]
    pub fn tokens(&self, raw: &str) -> Option<u32> {
        let provider = self.provider().ok()?;
        match provider.family {
            Family::OpenAICompatible => openai_compatible::extract_tokens(raw),
            Family::OpenAi => openai::extract_tokens(raw),
            Family::Claude => claude::extract_tokens(raw),
            Family::Gemini => gemini::extract_tokens(raw),
            Family::Copilot => copilot::extract_tokens(raw),
            // No usage surface on the decisions API.
            Family::SystemOne => None,
        }
    }

    /// Extract the full token usage (including prompt-cache accounting) from
    /// a raw API response.
    ///
    /// Works with a complete response body, a streaming `message_start`
    /// frame (Claude), or the final `message_delta` frame. For Claude the
    /// [`TokenUsage`] carries the cache fields: `cache_read_input_tokens`
    /// (tokens served from cache at 10% of input price) and
    /// `cache_creation_input_tokens` (tokens written to the cache at a 25%
    /// or 100% premium). Total input tokens =
    /// `input_tokens + cache_read + cache_creation`.
    ///
    /// Streaming note: each call extracts the usage of a SINGLE frame —
    /// Claude splits input/cache counters into the `message_start` frame and
    /// the final output count into `message_delta`, so consumers must merge
    /// per-frame results (take input/cache from `message_start`, take the
    /// latest non-zero output from `message_delta`) rather than summing.
    ///
    /// The REAL cost the provider reported inside the response
    /// (`usage.cost`, or the OpenCode Zen/Go top-level `cost`) is stamped
    /// onto [`TokenUsage::reported_cost`] — the provider's actual billed
    /// amount, with no local re-pricing fallback.
    ///
    /// Returns `None` when no usage object is present or the JSON is invalid.
    #[must_use]
    pub fn token_usage(&self, raw: &str) -> Option<TokenUsage> {
        let provider = self.provider().ok()?;
        let mut usage = match provider.family {
            Family::Claude => claude::extract_usage(raw),
            Family::OpenAICompatible => openai_compatible::extract_usage(raw),
            Family::OpenAi => openai::extract_usage(raw),
            Family::Gemini => gemini::extract_usage(raw),
            Family::Copilot => copilot::extract_usage(raw),
            // No usage surface on the decisions API (decide() parses its own).
            Family::SystemOne => None,
        }?;
        usage.reported_cost = super::output::extract_reported_cost(raw).or(usage.reported_cost);
        usage.reported_cost_credits =
            super::output::extract_reported_credits(raw).or(usage.reported_cost_credits);
        Some(usage)
    }

    /// Real session spend, reported by the PROVIDER itself — never estimated.
    ///
    /// Supported:
    ///     `openrouter`,
    ///     `zen`,
    ///     `opencode-go`,
    ///     `charm`,
    ///     `xai`,
    ///     `vercel`,
    ///     `kimi`.
    ///
    /// Unsupported:
    ///     `openai`,
    ///     `claude`,
    ///     `mistral`,
    ///     `fireworks`,
    ///     `deepseek`,
    ///     `groq`,
    ///     `together`,
    ///     `cerebras`,
    ///     `cohere`,
    ///     `perplexity`,
    ///     `sambanova`,
    ///     `poe`,
    ///     `nvidia`,
    ///     `anyscale`,
    ///     `cloudflare`,
    ///     `azure`,
    ///     `gemini`,
    ///     `zai`,
    ///     `huggingface`.
    ///
    /// Unsupported providers expose no spend API usable with a normal
    /// inference key: `openai`/`claude`/`mistral` require an admin key,
    /// `fireworks` a separate `account_id`, `deepseek` reports balance
    /// only; the rest document no spend API at all. All return
    /// [`CostError::CostNotSupported`] — hide the price rows.
    ///
    /// `session_reported_costs` are the per-request costs the provider
    /// already reported in each response ([`TokenUsage::reported_cost`]);
    /// their sum is the session spend. For `openrouter` the figure comes
    /// from the provider's account endpoint instead. `kimi` reports no
    /// per-request cost in its responses, so the session figure stays the
    /// (empty) reported sum and only `balance_usd` carries real data
    /// (account `available_balance` from `GET /users/me/balance`).
    ///
    /// # Errors
    ///
    /// [`CostError::CostNotSupported`] (unsupported provider — hide cost
    /// UI), [`CostError::MissingApiKey`] (no key resolved),
    /// [`CostError::CostUnavailable`] (endpoint failed or unusable answer).
    pub async fn session_cost(
        &self,
        session_reported_costs: &[f64],
    ) -> Result<ProviderCost, CostError> {
        let provider = self.provider()?;
        provider_session_cost(
            provider,
            &self.params,
            self.params.service_keyring.as_deref(),
            session_reported_costs,
        )
        .await
    }

    /// The provider name (e.g. `"openai"`, `"claude"`, `"gemini"`).
    ///
    /// Returns `None` if the connector was not properly initialised via [`new`](Self::new).
    #[must_use]
    pub fn provider_name(&self) -> Option<&'static str> {
        self.provider.map(|p| p.name)
    }

    /// Charm Hyper: remaining Hypercredit balance straight from the account
    /// endpoint (`GET /credits`). Fallback for when the streaming usage
    /// frames omit `usage.remaining.hypercredits`. `None` on any failure —
    /// best-effort, never blocks or errors the turn.
    pub async fn hyper_credits_balance(&self) -> Option<f64> {
        let Ok(provider) = self.provider() else {
            return None;
        };
        super::cost::hyper_credits_balance(
            provider,
            &self.params,
            self.params.service_keyring.as_deref(),
        )
        .await
    }

    /// Choose how tool calls are delivered: native structured parts
    /// ([`ToolCallMode::Native`], the default — the same contract the crush
    /// agent uses) or inline-JSON text the harness parses
    /// ([`ToolCallMode::Inline`]). In `Inline` mode the request carries no
    /// native `tools`, so the two paths never cross: the API cannot produce
    /// structured tool calls, and the harness is the only consumer of the
    /// model's text.
    pub fn with_tool_call_mode(mut self, mode: ToolCallMode) -> Self {
        self.params.tool_call_mode = mode;
        self
    }

    /// The active tool-call delivery mode (defaults to
    /// [`ToolCallMode::Native`]).
    #[must_use]
    pub const fn tool_call_mode(&self) -> ToolCallMode {
        self.params.tool_call_mode
    }

    /// Enable/disable request retry (retryable failures — rate limits,
    /// HTTP 5xx, network errors — are retried with exponential backoff).
    /// Defaults to enabled; tests disable it for single-shot mock servers.
    pub fn with_retry(mut self, enabled: bool) -> Self {
        self.params.retry_enabled = enabled;
        self
    }

    /// Override the maximum number of retries per request (default: 3, i.e.
    /// up to 4 total attempts). `0` disables retries entirely (same as
    /// [`with_retry(false)](Self::with_retry)` but per-request-count).
    pub fn with_max_retries(mut self, n: usize) -> Self {
        self.params.max_retries = Some(n);
        self
    }

    /// Override the retry backoff base delay (default 5s → 10s → 20s).
    /// Primarily for tests: a tiny delay makes retry tests run in
    /// milliseconds instead of sleeping real backoffs. The value is used
    /// VERBATIM — the production equal-jitter spread is not applied.
    pub fn with_retry_delay(mut self, delay: std::time::Duration) -> Self {
        self.params.retry_delay_override = Some(delay);
        self
    }

    /// Whether the provider endpoint runs on this machine (`localhost` /
    /// `127.0.0.1`).
    ///
    /// Local model servers (ollama, lmstudio, vllm, llamacpp) may lack
    /// reliable native function calling, so the harness keeps the legacy
    /// inline-JSON `TOOL_FORMAT` prompt for them; cloud providers get the
    /// native function-calling instruction.
    #[must_use]
    pub fn is_local(&self) -> bool {
        let base_url = self
            .params
            .base_url
            .as_deref()
            .or_else(|| self.provider.map(|p| p.base_url))
            .unwrap_or_default();
        // Parse the URL and compare the HOST component only: substring
        // matching misfires on paths/query strings (e.g.
        // `https://api.example.com/v1?mirror=127.0.0.1`) and on hosts that
        // merely CONTAIN "localhost" in their name.
        url::Url::parse(base_url)
            .ok()
            .and_then(|u| {
                // IPv6 hosts come back bracketed from host_str in some
                // versions; normalize by stripping brackets.
                u.host_str()
                    .map(|h| h.trim_matches(['[', ']']).to_ascii_lowercase())
            })
            .is_some_and(|host| {
                host == "localhost"
                    || host.ends_with(".localhost")
                    // The whole IPv4 loopback block is 127.0.0.0/8.
                    || host
                        .split('.')
                        .try_fold((0u32, 0usize), |(acc, i), octet| {
                            octet
                                .parse::<u8>()
                                .map(|o| ((acc << 8) | u32::from(o), i + 1))
                                .map_err(|_| ())
                        })
                        .is_ok_and(|(ip, octets)| octets == 4 && (ip >> 24) == 127)
                    || host == "::1"
            })
    }

    /// The model override, if one was set via [`with_model`](Self::with_model).
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.params.model.as_deref()
    }

    /// The effective model name: the explicit override if one was set via
    /// [`with_model`](Self::with_model), otherwise the provider's registered
    /// default model.
    #[must_use]
    pub fn effective_model(&self) -> Option<&str> {
        self.params
            .model
            .as_deref()
            .or_else(|| self.provider.map(|p| p.default_model))
    }

    /// Whether native tool definitions are currently registered on this
    /// connector (`set_tools`/`with_tools` was called and the list is
    /// non-empty).
    #[must_use]
    pub fn has_tools(&self) -> bool {
        self.params.tools.as_ref().is_some_and(|t| !t.is_empty())
    }

    /// Fetch the list of available models from the provider.
    ///
    /// Returns a [`LsOutput`] with structured model names ([`models`](LsOutput::models))
    /// and the raw JSON response ([`raw`](LsOutput::raw)).
    ///
    /// # Errors
    ///
    /// Returns `MissingApiKey` if no API key is found, `HttpError` on non-2xx status,
    /// `Deserialization` if the response is malformed, or `Network` on transport failure.
    pub async fn list_models(&self) -> Result<LsOutput, ConnectorError> {
        let provider = self.provider()?;
        match provider.family {
            Family::OpenAICompatible => {
                openai_compatible::list_models(
                    provider,
                    &self.params,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            // Model listing is not part of the Responses API — the `openai`
            // provider keeps using the shared `/v1/models` endpoint.
            Family::OpenAi => {
                openai_compatible::list_models(
                    provider,
                    &self.params,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Gemini => {
                gemini::list_models(
                    provider,
                    &self.params,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Claude => {
                claude::list_models(
                    provider,
                    &self.params,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            Family::Copilot => {
                copilot::list_models(
                    provider,
                    &self.params,
                    self.params.service_keyring.as_deref(),
                )
                .await
            }
            // The decisions API has no model-catalog endpoint.
            Family::SystemOne => Err(ConnectorError::NotImplemented("list_models on a decision provider")),
        }
    }
}
