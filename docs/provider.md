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

~28 providers are registered: `openai`, `groq`, `mistral`, `together`,
`openrouter`, `xai`, `deepseek`, `perplexity`, `fireworks`, `cohere`,
`huggingface`, `sambanova`, `poe`, `cerebras`, `nvidia`, `anyscale`,
`vercel`, `cloudflare`, `azure`, `ollama`, `lmstudio`, `vllm`, `llamacpp`,
`gemini`, `claude`, `zai`, `charm`, `opencode`. The local model servers (ollama,
lmstudio, vllm, llamacpp, and the extra ones below) point at
`http://localhost:*`.

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
