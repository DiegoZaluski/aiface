# `connector::provider` — registry and API keys

The provider registry: which backends exist, their base URLs, default models,
and how API keys are resolved and cached. Local model servers are flagged and
configured by URL instead of API key.

## The registry

Each provider has a [`ProviderConfig`]:

```rust,ignore
pub struct ProviderConfig {
    pub name: &'static str,
    pub family: Family,              // OpenAICompatible | Gemini | Claude
    pub base_url: &'static str,
    pub default_model: &'static str,
    pub needs_extra_headers: bool,   // e.g. OpenRouter
    pub local: bool,                 // local model server (URL, not API key)
}
```

~30 providers are registered: `openai`, `groq`, `mistral`, `together`,
`openrouter`, `xai`, `deepseek`, `perplexity`, `fireworks`, `cohere`,
`huggingface`, `sambanova`, `poe`, `cerebras`, `nvidia`, `anyscale`,
`vercel`, `cloudflare`, `azure`, `ollama`, `lmstudio`, `vllm`, `llamacpp`,
`gemini`, `claude`, `zai`, `charm`, `opencode`, `kimi`, `venice`. The local
model servers (ollama, lmstudio, vllm, llamacpp, and the extra ones below)
point at `http://localhost:*`.

Local providers (configured by URL, never by API key):

| Provider | Default port |
|---|---|
| ollama | 11434 |
| llamacpp | 8080 |
| lmstudio | 1234 |
| vllm | 8000 |
| llamafile | 8080 |
| koboldcpp | 5001 |
| text-generation-webui | 5000 |
| localai | 8080 |
| jan | 1337 |
| gpt4all | 4891 |
| aphrodite | 2242 |
| sglang | 30000 |
| tabbyapi | 5000 |

## Enumerating providers

```rust,ignore
pub fn known_providers() -> impl Iterator<Item = &'static str>
pub fn known_providers_with_env() -> impl Iterator<Item = (&'static str, &'static str)>
pub fn get_provider_env_var(provider: &str) -> Option<&'static str>
pub fn is_local_provider(provider: &str) -> bool
pub fn known_local_providers() -> impl Iterator<Item = &'static str>
pub fn get_provider(name: &str) -> Option<&'static ProviderConfig>
```

`known_providers_with_env` pairs each provider with its key env var
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …) — handy for UI
rendering. Note: **local providers are excluded** from `known_providers_with_env`
— they have no API key, and keeping them out protects `--check-keyring` and
`detect_provider()` from "detecting" a local server as a configured cloud
provider.

`is_local_provider` / `known_local_providers` classify the registry for the
ADD Provider screen: local providers prompt for a server URL instead of an API
key.

### Normalizing local base URLs

```rust,ignore
pub fn normalize_local_base_url(provider: &str, url: &str) -> String
```

Users type `http://127.0.0.1:8080` (no `/v1`). For OpenAI-compatible local
providers this appends `/v1` internally; URLs that already have a path are
left untouched, and unknown providers are returned verbatim. The raw URL the
user typed is stored in setup.json — normalization is applied at request time,
never persisted, and lives in the TUI layer (not in `with_base_url`), so SDK
tests can keep mocking exact URLs.

## The OpenCode Zen gateway (`opencode`) and its anonymous free tier

`opencode` points at `https://opencode.ai/zen/v1` (default model
`x-preview-f-free`, key env `OPENCODE_API_KEY`). The gateway deliberately
serves an ANONYMOUS free tier: requests carrying the sentinel bearer
[`ZEN_PUBLIC_KEY`] (`"public"`) are answered only for models flagged
`allowAnonymous` server-side and are rate-limited per IP. cosh's rules for
using it — enforced in code, not convention:

- **Account keys always win.** A key resolved from keyring/env (or passed via
  `with_api_key`) means the sentinel is never sent.
- **Strictly opt-in.** The sentinel is only used when the connector flag
  `with_zen_public_tier(true)` is set OR the process-wide switch
  [`set_zen_public_tier_enabled`] was turned on (the TUI flips it from the
  user's persisted answer in setup.json).
- **Free models only.** Anonymous requests are checked against
  [`ZEN_FREE_MODELS`] before any network activity; a paid model fails with
  `ConnectorError::AnonymousModelBlocked`. Anonymous `list_models` returns
  only free models. The list deliberately excludes
  `muse-spark-1.2-contributor-free` (its data policy grants training rights).
- **Honest identification.** Zen requests carry `User-Agent: cosh/<version>`
  ([`ZEN_USER_AGENT`]) and no other client's headers — anonymous traffic keeps
  the stricter public rate-limit bucket instead of imitating the official
  client.

```rust,ignore
pub const ZEN_PROVIDER: &str = "opencode";
pub const ZEN_PUBLIC_KEY: &str = "public";
pub const ZEN_USER_AGENT: &str = "cosh/<version>";
pub const ZEN_FREE_MODELS: &[&str];
pub fn is_zen_free_model(model: &str) -> bool;
pub fn set_zen_public_tier_enabled(enabled: bool);   // process-wide opt-in
pub fn zen_public_tier_enabled() -> bool;
```

## The Venice.ai provider (`venice`) — private OpenAI-compatible gateway

`venice` points at `https://api.venice.ai/api/v1` (docs.venice.ai) — a
drop-in replacement for the OpenAI Chat Completions endpoint with Bearer
auth (`VENICE_API_KEY`). It joins the OpenAI-compatible family through the
standard surfaces: chat (streaming, tool calling, vision, JSON schema),
`/embeddings`, and `GET /models` all match the OpenAI shapes, so the shared
family path serves them unchanged. No `venice_parameters` rides a default
request — the connector sends standard OpenAI fields only, and Venice's
own system-prompt default stays on.

What is Venice-specific, and where it lands:

- **Real billed cost on every response.** Each inference answer carries a
  top-level `cost: {usd, diem}` object (the same trailer shape Zen/Go
  use), parsed into `TokenUsage::reported_cost` by the shared extractor;
  `supports_cost_reporting("venice")` is `true` and `session_cost` sums
  the captured amounts. No billing endpoint is consulted — the documented
  `/billing` surface is beta, and a `balance_usd` of `None` beats a wrong
  zero.
- **Rich model catalog, keyless.** `GET /models` answers WITHOUT auth and
  carries `model_spec` per model: pricing (per-1M token `input`/`output`/
  `cache_input`), `context_length`, capabilities (`supportsReasoning`,
  `supportsReasoningEffort` + effort options), privacy tier (`private` =
  zero data retention, `anonymized` = not guaranteed) and `traits`.
  `list_models` picks the ids up through the shared path.
- **Static discovery entries.** The gateway's ids (vendor-prefixed, some
  with an `e2ee-` prefix for end-to-end-encrypted deployments) resolve
  their context windows and reasoning capabilities from the static tables
  (`KNOWN_CONTEXT_WINDOWS` / `KNOWN_REASONING`) before any catalog
  fallback.
- **Pricing model.** Prepaid credits that never expire, or per-request
  USDC via the x402 protocol. `model_spec.pricing` reports `{usd, diem}`
  (1 Diem = $1/day of staked compute allowance) and already reflects any
  promotional discount active for the calling account.

The default model is the price/performance pick from the live catalog:
`z-ai-glm-5-3-flash` (GLM 5.3 Flash — $0.15 input / $0.50 output per 1M
tokens, ~1M context, code-optimized, `private`). Cheaper alternatives for
bulk workloads: `deepseek-v4-flash-0731` ($0.13/$0.26, 1M) and
`openai-gpt-oss-120b` ($0.07/$0.30, 128k). Cache reads are billed around
1/10 of the input price on most models, so multi-turn sessions amortize
cheaply.

### Image generation on Venice

Venice's gateway also mirrors the OpenAI Images route
(`/images/generations`), so `generate_image` works through the shared
OpenAI-shaped path with `with_model` picking one of the catalog's
`type=image` models (e.g. `z-image-turbo` at $0.01/image). Venice serves
ONE image per call — `with_image_count` is not sent (its `variants`
parameter would ride its native `/image/generate` surface, which the
connector does not use). `with_image_quality` carries the resolution
tier (`"1K"`/`"2K"`/`"4K"`) on quality-tiered models — the billed tier
defaults to the model's `defaultQuality` when unset.

### Images in chat answers (multi-modal output)

Chat responses can EMBED images in the answer itself, per family:

- **Gemini** — image-capable models (`gemini-2.5-flash-image`, …)
  answer `generateContent` with interleaved text/`inlineData` parts;
  every image arrives decoded on `ChatOutput::images` (and
  `ChatStream::images` for streams, accumulated in wire order and
  deduped — the final frame re-sends earlier parts). A withheld
  generation (`finishReason != "STOP"`) contributes nothing; the
  answer's text survives.
- **OpenRouter-style gateways** — the `message.images[]`
  normalization (data-URI `image_url.url`) is parsed for the whole
  `OpenAICompatible` family; servers that don't send it parse to an
  empty vec. Live http(s) URLs are never fetched (no unauthenticated
  egress) — the pixels must arrive inline.
- **OpenAI** — chat messages never carry images (assistant content is
  text/refusal only); the inline-in-response surface is the Responses
  API's `image_generation` tool, whose `image_generation_call` output
  items surface on `images()` too.

Families with no image-output surface (Claude, Copilot) always yield an
empty `images()` — `is_empty()` is the "no image this turn" check, the
same contract as `generate_image` safety blocks.

## The Bedrock provider (`bedrock`) — region-templated base URL

`bedrock` joins the OpenAI-compatible family through Amazon Bedrock's official
OpenAI-compatible endpoint on `bedrock-runtime` — no AWS SDK, no SigV4: auth
rides the shared `Authorization: Bearer` path with the Bedrock API key
(`AWS_BEARER_TOKEN_BEDROCK`, the endpoint's documented API-key mode). Because
the endpoint is region-scoped, the registry entry only pins a default region
(`us-east-1`); the live base URL is resolved at request time by:

```rust,ignore
pub fn resolved_provider_base_url(provider: &str) -> Cow<'static, str>
```

It re-points the registry URL to the ambient `AWS_REGION` (falling back to
`AWS_DEFAULT_REGION`). The region value is validated against the AWS region
grammar (`^[a-z]{2,4}(-[a-z]+)*-[0-9]+$`) before it is spliced into the
endpoint host — an unvalidated `AWS_REGION` would silently redirect the
bearer token to a foreign origin, so anything invalid (or blank) defers to
the next variable or falls back to the registry default. An explicit
`params.base_url` (connector `with_base_url`) always wins over the ambient
resolution.

Two endpoint surfaces are NOT part of Bedrock's OpenAI-compatible API, so
`embed` and `list_models` fail up front with an explanatory `NotImplemented`
error (unless an explicit base URL points the connector at the
`bedrock-mantle` compatibility endpoint, which serves the model catalog, or
at a proxy serving embeddings). Streaming, native tool calling, reasoning
effort, and image input work through the shared family path unchanged.

**Decision — no AWS SDK.** The full `aws-sdk-bedrockruntime` (and its
`Converse` API, SigV4 chain, Guardrails, embeddings) is deliberately NOT a
dependency: the OpenAI-compatible endpoint covers everything cosh needs and
the SDK would pull dozens of crates into the slim build. If a future
requirement genuinely needs the native surface, the plan is to add it as an
opt-in **feature gate** (`bedrock-sdk`) wrapping the SDK behind the existing
isolated-family pattern — not to replace this integration.

## API key resolution

```rust,ignore
pub const COSH_SERVICE: &str = "cosh"; // default keyring service (kept for compatibility)

pub fn get_api_key(provider: &str, service: Option<&str>) -> Option<String>
pub fn has_api_key(provider: &str) -> bool
pub fn detect_provider() -> Option<&'static str>

// Example: Check if a provider is configured
if has_api_key("openai") {
    println!("OpenAI is configured");
}

// Example: Get the API key (respects priority: keyring → env → explicit)
if let Some(key) = get_api_key("anthropic", None) {
    println!("Found Anthropic key");
}

// Example: Detect which provider the user is most likely using
if let Some(provider) = detect_provider() {
    println!("Default provider: {}", provider);
}
```

Resolution order:

1. **OS keyring** — the authoritative store for keys saved explicitly through
   the app (`keyring::Entry` under `cosh` by default). Preferring it over the
   environment means stale shell/`.env` exports can't shadow a key the user
   configured.

> **Why keyring takes priority:** Users who explicitly save keys through the app have made a deliberate configuration choice. Environment variables are often temporary (session exports, `.env` files for development) and can become stale. By preferring keyring, we ensure the user's intentional configuration isn't accidentally overridden by forgotten environment state.

2. **Environment variable** — per-provider fallback for providers never
   stored (`get_api_key("openai", None)` reads `OPENAI_API_KEY`).

Successful keyring lookups are cached in a process-lifetime map (keyed by
`(service, env_var)`) so repeated resolutions don't re-hit the OS credential
store; misses are never cached, so keys added externally appear immediately.
Cached values are `Zeroizing` — wiped from memory on invalidation.

```rust,ignore
pub fn invalidate_api_key(env_var: &str)   // forget one provider's cached key
pub fn clear_api_key_cache()               // forget everything
```

Call `invalidate_api_key` after **storing or updating** a key, or the cache
serves the stale value.

- `has_api_key(provider)` — "configured?" regardless of backend (keyring or
  env).
- `detect_provider()` — the first provider (in registration order) that has
  a key; the natural default for "which provider is this user using?"

---

Next: [discovery — context windows and reasoning](discovery.md).

---

## Summary

- The provider registry contains ~30 providers with `ProviderConfig` entries: name, family (OpenAICompatible/OpenAi/Gemini/Claude/Copilot), base URL, default model, extra-headers flag, and `local` flag.
- Providers include cloud services (openai, groq, mistral, together, openrouter, xai, deepseek, perplexity, bedrock, etc.) and local model servers (ollama, lmstudio, vllm, llamacpp, llamafile, koboldcpp, text-generation-webui, localai, jan, gpt4all, aphrodite, sglang, tabbyapi) pointing at localhost.
- Enumeration: `known_providers()` (iterator), `known_providers_with_env()` (pairs with env vars), and `get_provider_env_var()` (env var name for a provider). Local providers are excluded from `known_providers_with_env` — they have no key and must not be "detected" as configured cloud providers.
- `is_local_provider` / `known_local_providers` classify the registry; `get_provider` returns the full `ProviderConfig` (default port, family, model).
- `normalize_local_base_url(provider, url)` appends `/v1` for bare OpenAI-compatible URLs; it is applied at request time in the TUI layer, never persisted.
- API key resolution: OS keyring (authoritative store under `cosh` service) → environment variable (fallback) → explicit `with_api_key` (highest priority).
- Keyring lookups are cached per process lifetime (keyed by service + env var) and `Zeroizing`; misses are never cached so externally-added keys appear immediately.
- Cache invalidation: `invalidate_api_key(env_var)` after storing/updating a key, or `clear_api_key_cache()` to forget everything.
- `has_api_key(provider)` answers "configured?" regardless of backend; `detect_provider()` returns the first provider with a key (natural default).
