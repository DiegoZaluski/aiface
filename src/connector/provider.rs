//! Provider registry — known LLM backends, their base URLs, default models,
//! and API key environment variable names.
//!
use keyring::Entry;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError, RwLock};
use zeroize::Zeroizing;

/// Default keyring service under which cosh stores its API keys.
///
/// `Connector`'s parameters already default `service_keyring` to this value,
/// so ordinary consumers don't need to reference it — it exists so the SDK
/// reads keys under the same service string that the TUI writes via
/// `save_provider_api_key`. Override only builders that pass a custom
/// service: [`Connector::with_service_keyring`](crate::connector::Connector::with_service_keyring).
pub const COSH_SERVICE: &str = "cosh";

/// Provider name of the DOCUMENTED OpenCode Zen gateway
/// (`https://opencode.ai/zen/v1`, `https://opencode.ai/docs/zen`).
///
/// Key-based access to the full Zen catalog: an account key from
/// `opencode.ai/auth` unlocks every model. A missing key fails with
/// [`ConnectorError::MissingApiKey`](crate::connector::error::ConnectorError::MissingApiKey).
pub const OPENCODE_ZEN_PROVIDER: &str = "zen";

/// Provider name of the OpenCode Go gateway
/// (`https://opencode.ai/zen/go/v1`, `https://opencode.ai/docs/go`).
///
/// A $10/month subscription serving popular open coding models (GLM, Kimi,
/// Qwen, DeepSeek, MiniMax, MiMo, …) behind the same OpenAI-compatible
/// surface as Zen. Subscriptions are managed through an OpenCode Zen
/// account, so the key copied from `opencode.ai/auth` works on both
/// gateways — `OPENCODE_GO_API_KEY` is tried first and
/// `OPENCODE_API_KEY` is accepted as a fallback. There is no anonymous
/// tier: a key is always required.
pub const OPENCODE_GO_PROVIDER: &str = "opencode-go";

/// Honest `User-Agent` for OpenCode gateway requests: identifies co-sh as
/// the calling client instead of imitating another client's identity.
pub const ZEN_USER_AGENT: &str = concat!("cosh/", env!("CARGO_PKG_VERSION"));

/// Whether the provider is hosted on the OpenCode gateway infrastructure
/// (`opencode.ai/zen/...`): the documented Zen gateway
/// ([`OPENCODE_ZEN_PROVIDER`]) and the Go subscription gateway
/// ([`OPENCODE_GO_PROVIDER`]). Both identify themselves with the
/// honest OpenCode Zen `User-Agent` identifier.
#[must_use]
pub fn is_opencode_gateway(name: &str) -> bool {
    matches!(name, OPENCODE_ZEN_PROVIDER | OPENCODE_GO_PROVIDER)
}

/// A missing credential is cached briefly so a UI render does not query the
/// OS keyring on every frame. Explicit saves/removals invalidate the entry.
const KEYRING_MISS_TTL: std::time::Duration = std::time::Duration::from_secs(30);

enum CachedKeyringValue {
    Found(Zeroizing<String>),
    Missing(std::time::Instant),
}

/// Process-lifetime cache mapping `(service, env_var)` to a key or recent miss.
type KeyringCache = HashMap<(String, String), CachedKeyringValue>;

/// Process-lifetime cache of resolved keyring keys, keyed by
/// `(service, env_var)`.
///
/// The first successful keyring lookup for a given key stays alive in process
/// memory, so repeated resolutions (model listing, chat, provider detection)
/// avoid re-hitting the OS credential store on every call. Successful
/// lookups are cached for the process lifetime. Misses expire after 30 seconds;
/// keys saved through cosh invalidate their entry immediately.
///
/// Cache omitting Zeroizing would hold keys in plain `String` for the whole
/// process too; `Zeroizing` makes sure the memory is wiped when an entry is
/// invalidated or dropped instead.
///
/// Use [`invalidate_api_key`] after storing or updating a key so the cache
/// never serves a stale value from a previous save.
static KEYRING_CACHE: LazyLock<Mutex<KeyringCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Read a key from the OS keyring, caching successful lookups so repeat
/// resolutions don't re-hit the credential store.
fn keyring_lookup(service: &str, env_var: &str) -> Option<String> {
    let key = (service.to_owned(), env_var.to_owned());
    {
        let cache = KEYRING_CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match cache.get(&key) {
            Some(CachedKeyringValue::Found(value)) => return Some((**value).clone()),
            Some(CachedKeyringValue::Missing(at)) if at.elapsed() < KEYRING_MISS_TTL => {
                return None;
            }
            _ => {}
        }
    }

    let found = Entry::new(service, env_var)
        .and_then(|e| e.get_password())
        .ok();
    KEYRING_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            key,
            match &found {
                Some(value) => CachedKeyringValue::Found(Zeroizing::new(value.clone())),
                None => CachedKeyringValue::Missing(std::time::Instant::now()),
            },
        );
    found
}

/// Forget cached keys for an env var, e.g. right after saving/updating that
/// provider's key so the freshest value is resolved next time.
pub fn invalidate_api_key(env_var: &str) {
    KEYRING_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .retain(|(_, var), _| var != env_var);
}

/// Forget every cached key. The process keeps resolving new lookups from the
/// OS keyring afterwards.
pub fn clear_api_key_cache() {
    KEYRING_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    OpenAICompatible,
    /// The OpenAI-native Responses API (`/v1/responses`). Only the `openai`
    /// provider uses this family; it is what exposes reasoning for OpenAI's
    /// reasoning models. `embed` and `list_models` still route through the
    /// shared `/v1/embeddings` and `/v1/models` endpoints.
    OpenAi,
    Gemini,
    Claude,
    /// GitHub's official Copilot agent runtime, driven through the official
    /// `github-copilot-sdk` crate (JSON-RPC over a spawned runtime process
    /// — no HTTP chat endpoint). An ISOLATED family: it shares none of the
    /// OpenAI-compatible request machinery, only the connector's output
    /// types. `embed` is unsupported; `list_models` does NOT use the
    /// runtime — it is a direct HTTP chain (GitHub token exchange, then
    /// the shared chat catalog), because the runtime's `models.list` RPC
    /// reads the agent-mode catalog, which stays empty for Free-tier
    /// accounts even though the account can chat fine (see
    /// `copilot::models`).
    Copilot,
    /// The System One DECISIONS API (`POST /v1/systemone`): typed
    /// questions in, calibrated probability distributions out — no
    /// generated prose. Served by decision models (TypeSafe Jev,
    /// Microsoft-Decision-1 on Azure Foundry) and by OpenRouter's
    /// System One route. An ISOLATED family like `Copilot`: it shares
    /// none of the chat request machinery; `chat`, `stream_chat` and
    /// `embed` are unsupported (the surface is
    /// [`Connector::decide`](crate::connector::Connector::decide)).
    SystemOne,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderConfig {
    pub name: &'static str,
    pub family: Family,
    pub base_url: &'static str,
    pub default_model: &'static str,
    pub needs_extra_headers: bool,
    /// Whether the endpoint runs on the user's machine (`localhost`). Local
    /// providers have no API key: they are configured with a base URL instead.
    pub local: bool,
    /// Path appended to `base_url` for the DECISIONS API
    /// ([`Connector::decide`](crate::connector::Connector::decide)), when the
    /// provider serves one. The System One contract travels on a provider-
    /// specific path — TypeSafe direct is `/v1/systemone`, Microsoft Foundry
    /// nests it at `/providers/microsoft/v1/systemone`, and OpenRouter's
    /// route lives under `/api/v1/systemone` (its `base_url` already carries
    /// `/api/v1`). `None` = the provider has no decisions surface.
    pub decisions_path: Option<&'static str>,
    /// Default model for the DECISIONS API when the provider's chat
    /// default is not a decision model (OpenRouter's `default_model` is a
    /// chat model, but its System One route serves Jev). `None` falls back
    /// to `default_model` — the case for the dedicated decision providers.
    pub decisions_default_model: Option<&'static str>,
}

impl ProviderConfig {
    const fn cloud(
        name: &'static str,
        family: Family,
        base_url: &'static str,
        default_model: &'static str,
    ) -> Self {
        Self {
            name,
            family,
            base_url,
            default_model,
            needs_extra_headers: false,
            local: false,
            decisions_path: None,
            decisions_default_model: None,
        }
    }

    /// Like [`cloud`](Self::cloud) but allows setting `needs_extra_headers`
    /// (currently only OpenRouter requires this).
    const fn cloud_extra(
        name: &'static str,
        family: Family,
        base_url: &'static str,
        default_model: &'static str,
        needs_extra_headers: bool,
    ) -> Self {
        Self {
            name,
            family,
            base_url,
            default_model,
            needs_extra_headers,
            local: false,
            decisions_path: None,
            decisions_default_model: None,
        }
    }

    /// A System One decision provider: the family's ONLY surface is the
    /// decisions API on `decisions_path` (chat/stream/embed are not part of
    /// this family's contract).
    const fn decisions(
        name: &'static str,
        base_url: &'static str,
        default_model: &'static str,
        decisions_path: &'static str,
    ) -> Self {
        Self {
            name,
            family: Family::SystemOne,
            base_url,
            default_model,
            needs_extra_headers: false,
            local: false,
            decisions_path: Some(decisions_path),
            decisions_default_model: None,
        }
    }

    const fn local(name: &'static str, base_url: &'static str) -> Self {
        Self {
            name,
            family: Family::OpenAICompatible,
            base_url,
            default_model: "llama3.3",
            needs_extra_headers: false,
            local: true,
            decisions_path: None,
            decisions_default_model: None,
        }
    }

    /// Attach the decisions-API path to a CHAT provider that also serves
    /// the System One route (OpenRouter: its `base_url` already carries
    /// `/api/v1`, so the System One endpoint is `/systemone` past it). The
    /// route's default model differs from the chat default (Jev, not the
    /// chat catalog's pick).
    const fn with_decisions_path(mut self, path: &'static str) -> Self {
        self.decisions_path = Some(path);
        self.decisions_default_model = Some("jev-latest");
        self
    }
}

const PROVIDERS: &[(&str, ProviderConfig)] = &[
    // ── Cloud providers ──────────────────────────────────────────────
    (
        "openai",
        ProviderConfig::cloud(
            "openai",
            Family::OpenAi,
            "https://api.openai.com/v1",
            "gpt-4o-mini",
        ),
    ),
    (
        "groq",
        ProviderConfig::cloud(
            "groq",
            Family::OpenAICompatible,
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
        ),
    ),
    (
        "mistral",
        ProviderConfig::cloud(
            "mistral",
            Family::OpenAICompatible,
            "https://api.mistral.ai/v1",
            "mistral-small-latest",
        ),
    ),
    (
        "together",
        ProviderConfig::cloud(
            "together",
            Family::OpenAICompatible,
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        ),
    ),
    (
        "openrouter",
        ProviderConfig::cloud_extra(
            "openrouter",
            Family::OpenAICompatible,
            "https://openrouter.ai/api/v1",
            "qwen/qwen-2.5-72b-instruct",
            true,
        )
        // OpenRouter also serves the System One route (documented at
        // openrouter.ai/docs/guides/community/typesafe-sdk): its base URL
        // already carries `/api/v1`, so the decisions path is `/systemone`
        // past it.
        .with_decisions_path("/systemone"),
    ),
    (
        "xai",
        ProviderConfig::cloud(
            "xai",
            Family::OpenAICompatible,
            "https://api.x.ai/v1",
            "grok-2-1212",
        ),
    ),
    (
        "deepseek",
        ProviderConfig::cloud(
            "deepseek",
            Family::OpenAICompatible,
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        ),
    ),
    (
        "perplexity",
        ProviderConfig::cloud(
            "perplexity",
            Family::OpenAICompatible,
            "https://api.perplexity.ai",
            "llama-3.1-sonar-small-128k-chat",
        ),
    ),
    (
        "fireworks",
        ProviderConfig::cloud(
            "fireworks",
            Family::OpenAICompatible,
            "https://api.fireworks.ai/inference/v1",
            "qwen2.5-72b-instruct",
        ),
    ),
    (
        "cohere",
        ProviderConfig::cloud(
            "cohere",
            Family::OpenAICompatible,
            "https://api.cohere.com/compatibility/v1",
            "command-r-plus",
        ),
    ),
    (
        "huggingface",
        ProviderConfig::cloud(
            "huggingface",
            Family::OpenAICompatible,
            "https://router.huggingface.co/v1",
            "meta-llama/Llama-3.3-70B-Instruct",
        ),
    ),
    (
        "sambanova",
        ProviderConfig::cloud(
            "sambanova",
            Family::OpenAICompatible,
            "https://api.sambanova.ai/v1",
            "Meta-Llama-3.1-8B-Instruct",
        ),
    ),
    (
        "poe",
        ProviderConfig::cloud(
            "poe",
            Family::OpenAICompatible,
            "https://api.poe.com/v1",
            "GPT-4o",
        ),
    ),
    (
        "cerebras",
        ProviderConfig::cloud(
            "cerebras",
            Family::OpenAICompatible,
            "https://api.cerebras.ai/v1",
            "llama-3.1-8b-chat-completion",
        ),
    ),
    (
        "nvidia",
        ProviderConfig::cloud(
            "nvidia",
            Family::OpenAICompatible,
            "https://integrate.api.nvidia.com/v1",
            "meta/llama-3.1-8b-instruct",
        ),
    ),
    (
        "anyscale",
        ProviderConfig::cloud(
            "anyscale",
            Family::OpenAICompatible,
            "https://api.endpoints.anyscale.com/v1",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
    ),
    (
        "vercel",
        ProviderConfig::cloud(
            "vercel",
            Family::OpenAICompatible,
            "https://ai-gateway.vercel.sh/v1",
            "meta-llama/Llama-3.1-8B-Instruct",
        ),
    ),
    (
        "cloudflare",
        ProviderConfig::cloud(
            "cloudflare",
            Family::OpenAICompatible,
            "https://gateway.ai.cloudflare.com/v1/_/cloudflare/workers-ai/openai",
            "@cf/meta/llama-3.1-8b-instruct",
        ),
    ),
    (
        "azure",
        ProviderConfig::cloud("azure", Family::OpenAICompatible, "", "gpt-4o"),
    ),
    // Amazon Bedrock through its official OpenAI-compatible endpoint. The
    // registry URL only pins the DEFAULT region — the live base URL is
    // resolved per request by [`bedrock_base_url`] from the standard AWS
    // environment variables (`AWS_REGION` / `AWS_DEFAULT_REGION`), and an
    // explicit `params.base_url` always wins. Auth rides the shared
    // `Authorization: Bearer` path with the Bedrock API key
    // (`AWS_BEARER_TOKEN_BEDROCK`) — the SigV4/IAM setup needs the
    // token-bearing env export or an explicit base URL.
    (
        "bedrock",
        ProviderConfig::cloud(
            "bedrock",
            Family::OpenAICompatible,
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            // Documented Chat Completions model on bedrock-runtime (the
            // endpoint's own docs use it as the example).
            "openai.gpt-oss-120b-1:0",
        ),
    ),
    // Non-OpenAI cloud providers
    // gemini-3.6-flash: stable v1 default — older models (1.5-flash, 2.5-flash)
    // return 404 on v1 and 2.0-flash is rate-limited on free tiers.
    (
        "gemini",
        ProviderConfig::cloud(
            "gemini",
            Family::Gemini,
            "https://generativelanguage.googleapis.com/v1",
            "gemini-3.6-flash",
        ),
    ),
    (
        "claude",
        ProviderConfig::cloud(
            "claude",
            Family::Claude,
            "https://api.anthropic.com/v1",
            "claude-sonnet-4-6",
        ),
    ),
    (
        "copilot",
        ProviderConfig::cloud(
            "copilot",
            Family::Copilot,
            // No HTTP endpoint: the family drives the official Copilot
            // runtime over JSON-RPC (see connector/copilot). The URL field
            // stays a stable identifier pointing at the runtime's home.
            "https://github.com/copilot",
            // Runtime default; overridable via `params.model`.
            "claude-sonnet-4.5",
        ),
    ),
    (
        "zai",
        ProviderConfig::cloud(
            "zai",
            Family::OpenAICompatible,
            "https://api.z.ai/api/paas/v4",
            "",
        ),
    ),
    (
        "charm",
        ProviderConfig::cloud(
            "charm",
            Family::OpenAICompatible,
            "https://hyper.charm.land/v1",
            "",
        ),
    ),
    // Documented OpenCode Zen gateway: strictly key-based — the full
    // catalog is served to the account key from `opencode.ai/auth`.
    (
        "zen",
        ProviderConfig::cloud(
            "zen",
            Family::OpenAICompatible,
            "https://opencode.ai/zen/v1",
            "gpt-5.6-luna",
        ),
    ),
    // OpenCode Go: $10/month subscription for open coding models, served
    // from the same account as Zen (hence the shared-key fallback).
    (
        "opencode-go",
        ProviderConfig::cloud(
            "opencode-go",
            Family::OpenAICompatible,
            "https://opencode.ai/zen/go/v1",
            "glm-5.3-flash",
        ),
    ),
    // Kimi (Moonshot AI): OpenAI-compatible endpoint documented at
    // `platform.kimi.ai/docs/api/overview` — base URL `https://api.moonshot.ai/v1`,
    // Bearer auth with the `MOONSHOT_API_KEY`. Current models: `kimi-k3`
    // (1M context), `kimi-k2.7-code[-highspeed]` and `kimi-k2.6` (256k); the
    // `moonshot-v1-*` and `kimi-k2` lines are discontinued. Key-only access:
    // no OAuth flow is documented for the platform.
    (
        "kimi",
        ProviderConfig::cloud(
            "kimi",
            Family::OpenAICompatible,
            "https://api.moonshot.ai/v1",
            // Current flagship: 1M context, always-thinking with top-level
            // `reasoning_effort` (docs: platform.kimi.ai/docs/models).
            "kimi-k3",
        ),
    ),
    // Venice.ai: privacy-first OpenAI-compatible gateway
    // (`https://api.venice.ai/api/v1`, docs.venice.ai) — a drop-in
    // replacement for the OpenAI Chat Completions endpoint with Bearer auth
    // (`VENICE_API_KEY`). The chat, embeddings and models surfaces are the
    // standard OpenAI shapes, so the shared family path serves them
    // unchanged; every request reports its REAL billed cost in a top-level
    // `cost: {usd, diem}` object on the response (the same shape Zen/Go
    // use), which the cost-reporting path reads. `GET /models` answers
    // without a key and carries `model_spec` with per-model pricing,
    // context length, capabilities and privacy tier. Default model is the
    // price/performance pick from the live catalog (2026-11): GLM 5.3
    // Flash — $0.15/$0.50 per 1M tokens, 1M context, code-optimized and
    // `private` (zero data retention).
    (
        "venice",
        ProviderConfig::cloud(
            "venice",
            Family::OpenAICompatible,
            "https://api.venice.ai/api/v1",
            "z-ai-glm-5-3-flash",
        ),
    ),
    // ── Local providers (configured by URL, no API key) ──────────────
    (
        "ollama",
        ProviderConfig::local("ollama", "http://localhost:11434/v1"),
    ),
    (
        "lmstudio",
        ProviderConfig::local("lmstudio", "http://localhost:1234/v1"),
    ),
    (
        "vllm",
        ProviderConfig::local("vllm", "http://localhost:8000/v1"),
    ),
    (
        "llamacpp",
        ProviderConfig::local("llamacpp", "http://localhost:8080/v1"),
    ),
    (
        "llamafile",
        ProviderConfig::local("llamafile", "http://localhost:8080/v1"),
    ),
    (
        "koboldcpp",
        ProviderConfig::local("koboldcpp", "http://localhost:5001/v1"),
    ),
    (
        "text-generation-webui",
        ProviderConfig::local("text-generation-webui", "http://localhost:5000/v1"),
    ),
    (
        "localai",
        ProviderConfig::local("localai", "http://localhost:8080/v1"),
    ),
    (
        "jan",
        ProviderConfig::local("jan", "http://localhost:1337/v1"),
    ),
    (
        "gpt4all",
        ProviderConfig::local("gpt4all", "http://localhost:4891/v1"),
    ),
    (
        "aphrodite",
        ProviderConfig::local("aphrodite", "http://localhost:2242/v1"),
    ),
    (
        "sglang",
        ProviderConfig::local("sglang", "http://localhost:30000/v1"),
    ),
    (
        "tabbyapi",
        ProviderConfig::local("tabbyapi", "http://localhost:5000/v1"),
    ),
    // ── System One decision providers ─────────────────────────────────
    // The decisions API (`POST /v1/systemone`): typed questions in,
    // calibrated probability distributions out. Chat/stream/embed are not
    // part of these families' contract — `Connector::decide` is the surface.
    (
        // TypeSafe direct — Jev is the reference System One model.
        "typesafe",
        ProviderConfig::decisions(
            "typesafe",
            "https://api.typesafe.ai",
            "jev-latest",
            "/v1/systemone",
        ),
    ),
    (
        // Microsoft-Decision-1 on Azure AI Foundry. The request's `model`
        // value is the DEPLOYMENT name (set with `with_model`), not the
        // underlying model. `with_base_url` points at the Foundry resource
        // endpoint; authentication is the resource API key via
        // `AZURE_FOUNDRY_KEY` (or `with_api_key`).
        "azure-decision",
        ProviderConfig::decisions(
            "azure-decision",
            "https://<your-foundry-resource>",
            "microsoft-decision-1",
            "/providers/microsoft/v1/systemone",
        ),
    ),
];

const API_KEY_ENVS: &[(&str, &str)] = &[
    ("openai", "OPENAI_API_KEY"),
    ("mistral", "MISTRAL_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("cerebras", "CEREBRAS_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
    ("huggingface", "HUGGINGFACE_API_KEY"),
    ("nvidia", "NVIDIA_API_KEY"),
    ("copilot", "GITHUB_TOKEN"),
    ("cloudflare", "CLOUDFLARE_API_KEY"),
    ("fireworks", "FIREWORKS_API_KEY"),
    ("xai", "XAI_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("perplexity", "PERPLEXITY_API_KEY"),
    ("cohere", "COHERE_API_KEY"),
    ("sambanova", "SAMBANOVA_API_KEY"),
    ("poe", "POE_API_KEY"),
    ("anyscale", "ANYSCALE_API_KEY"),
    ("vercel", "VERCEL_API_KEY"),
    ("azure", "AZURE_OPENAI_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("claude", "ANTHROPIC_API_KEY"),
    ("zai", "ZAI_API_KEY"),
    ("charm", "CHARM_API_KEY"),
    ("zen", "OPENCODE_API_KEY"),
    ("opencode-go", "OPENCODE_GO_API_KEY"),
    // AWS's Bedrock bearer token: the API-key auth mode of the OpenAI-
    // compatible endpoint. IAM/SigV4 setups must export this token (or point
    // the connector at a proxy via `with_base_url`) — no SigV4 signing here.
    ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
    // Kimi (Moonshot AI): the documented credential for the OpenAI-compatible
    // endpoint (platform.kimi.ai/docs/api/overview —
    // `Authorization: Bearer $MOONSHOT_API_KEY`).
    ("kimi", "MOONSHOT_API_KEY"),
    // Venice.ai: the API key created in the Venice account settings, sent as
    // `Authorization: Bearer $VENICE_API_KEY` (docs.venice.ai).
    ("venice", "VENICE_API_KEY"),
    // TypeSafe (Jev, System One decisions API): Bearer key issued by
    // console.typesafe.ai, documented as `TYPESAFE_API_KEY`.
    ("typesafe", "TYPESAFE_API_KEY"),
    // Microsoft-Decision-1 on Azure AI Foundry: the deployment resource's
    // API key (`api-key` header). Entra-ID bearer auth stays available via
    // `with_api_key` (the caller supplies the token itself).
    ("azure-decision", "AZURE_FOUNDRY_KEY"),
];

/// Secondary key sources for providers whose subscriptions share an account:
/// a Go subscription is managed through the same OpenCode Zen account, so a
/// key saved under the Zen variable also unlocks the Go gateway. Consulted
/// only after the provider's own env var (keyring then environment) misses.
const API_KEY_FALLBACK_ENVS: &[(&str, &str)] = &[("opencode-go", "OPENCODE_API_KEY")];

pub fn get_provider(name: &str) -> Option<&'static ProviderConfig> {
    PROVIDERS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, config)| config)
        .or_else(|| custom_provider_lookup(name))
}

/// User-registered provider endpoints (Settings → Integrations → Custom
/// providers). Registration leaks its [`ProviderConfig`] once per explicit
/// save (same contract as the ACP agent registry) and stores the
/// `&'static` pointer: registry replaces drop only the pointer Vec, never
/// a config another caller may still hold.
static CUSTOM_PROVIDERS: RwLock<Vec<&'static ProviderConfig>> = RwLock::new(Vec::new());

/// Look a name up in the user-registered table only. The built-in table
/// always wins (checked first in [`get_provider`]), so this never shadows.
fn custom_provider_lookup(name: &str) -> Option<&'static ProviderConfig> {
    CUSTOM_PROVIDERS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .copied()
        .find(|cfg| cfg.name == name)
}

/// Register the user-registered provider set (Settings save path).
/// `entries` maps base URL → family per provider name. Names must be
/// single words (they key the model picker and routing config), must not
/// collide with a built-in name even in different case, must not duplicate
/// each other, and the base URL must parse as `http(s)`. Any violation
/// rejects the whole call, so a typo can never shadow a built-in or hide a
/// sibling.
///
/// Keys resolve through the shared keyring path under the provider's own
/// name (`cosh` service, `<NAME>_API_KEY` env var) — the same storage the
/// API-key dialog writes.
///
/// # Errors
///
/// Returns an error when a name is empty or multi-word, duplicates a
/// built-in provider, duplicates another entry in the same call (even in
/// different case), or a base URL is not a valid http(s) URL. A rejected
/// call leaves the registry untouched; configs leaked before the
/// violation stay leaked (bounded by the call's size, unreachable from
/// any lookup — same tolerance as the ACP agent registry).
pub fn register_custom_providers(entries: Vec<(String, String, Family)>) -> Result<(), String> {
    let mut configs: Vec<&'static ProviderConfig> = Vec::with_capacity(entries.len());
    for (name, base_url, family) in entries {
        let name = name.trim();
        if name.is_empty() {
            return Err("Provider name is required.".into());
        }
        if name.split_whitespace().count() != 1 {
            return Err(format!(
                "'{name}' must be a single word (it is the provider id the model picker and routing config use)."
            ));
        }
        if PROVIDERS
            .iter()
            .any(|(key, _)| key.eq_ignore_ascii_case(name))
        {
            return Err(format!(
                "'{name}' is a built-in provider and cannot be registered again."
            ));
        }
        // Case-insensitive among customs too: `corp` and `CORP` would
        // otherwise both derive the same `<NAME>_API_KEY` credential and
        // silently fight over it.
        if configs
            .iter()
            .any(|cfg| cfg.name.eq_ignore_ascii_case(name))
        {
            return Err(format!("'{name}' is registered more than once."));
        }
        let url = reqwest::Url::parse(base_url.trim())
            .map_err(|_| format!("'{name}' base URL is not a valid URL."))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!(
                "'{name}' base URL must be http(s) (got '{}').",
                url.scheme()
            ));
        }
        // Family gating: the connector ships one wire implementation per
        // Family variant. Only OpenAI-compatible endpoints are accepted
        // today; a new family becomes registrable when its caller lands.
        if family != Family::OpenAICompatible {
            return Err(format!(
                "'{name}': family {family:?} has no connector support yet."
            ));
        }
        // Leak the whole config: the registry stores `&'static` pointers,
        // so every handed-out reference stays valid even after a later
        // replace swaps the Vec (one leak per explicit user save, the same
        // contract the ACP agent registry uses — never in a loop).
        configs.push(Box::leak(Box::new(ProviderConfig {
            name: Box::leak(name.to_string().into_boxed_str()),
            family,
            base_url: Box::leak(base_url.trim().to_string().into_boxed_str()),
            default_model: "",
            needs_extra_headers: false,
            local: false,
            decisions_path: None,
            decisions_default_model: None,
        })));
    }
    *CUSTOM_PROVIDERS
        .write()
        .unwrap_or_else(PoisonError::into_inner) = configs;
    Ok(())
}

/// Names of the user-registered custom providers (registration order), for
/// callers that enumerate providers beyond the built-in table — the model
/// picker offers a custom endpoint once its credential resolves.
#[must_use]
pub fn custom_provider_names() -> Vec<&'static str> {
    CUSTOM_PROVIDERS
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|cfg| cfg.name)
        .collect()
}

/// Delete a custom provider's keyring credential (Settings removal path):
/// the `<NAME>_API_KEY` entry under the cosh service, plus its cached
/// value. A missing entry is success — forgetting an endpoint that never
/// had a key saved is a no-op, not an error.
///
/// # Errors
///
/// Returns the keyring error when the credential exists but cannot be
/// deleted (locked store, platform failure).
pub fn forget_provider_key(provider: &str) -> Result<(), keyring::Error> {
    let env_var = custom_provider_env_var(provider);
    let result = match Entry::new(COSH_SERVICE, &env_var) {
        Ok(entry) => match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(err) => Err(err),
        },
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(err) => Err(err),
    };
    if result.is_ok() {
        invalidate_api_key(&env_var);
    }
    result
}

/// Keyring first, environment second — the shared resolution used for both
/// a provider's primary env var and its fallback (see
/// [`API_KEY_FALLBACK_ENVS`]). An empty (or blank) value is NOT a
/// credential: a shell profile exporting `FOO_API_KEY=` (placeholder,
/// template leak, cleanup leftover) must not make the provider look
/// configured — the empty string would ride the whole send path and fail
/// deep inside the HTTP call with an error shape no gate understands.
/// The UI save paths reject empty input and the OAuth adapters validate
/// their tokens, but the filter here is what actually holds the line:
/// it is symmetric over both backends and needs no caller cooperation.
fn resolve_env_var_key(service: &str, env_var: &str) -> Option<String> {
    keyring_lookup(service, env_var)
        .filter(|key| !key.trim().is_empty())
        .or_else(|| {
            std::env::var(env_var)
                .ok()
                .filter(|key| !key.trim().is_empty())
        })
}

/// The API-key variable derived for a custom provider: `<NAME>_API_KEY`,
/// uppercased with hyphens mapped to underscores so the identifier is a
/// valid shell/env name. The single derivation — `get_api_key` and
/// [`forget_provider_key`] must agree, or a saved key would resolve under
/// one spelling and be purged under another.
#[must_use]
pub fn custom_provider_env_var(provider: &str) -> String {
    format!(
        "{}_API_KEY",
        provider.to_ascii_uppercase().replace('-', "_")
    )
}

pub fn get_api_key(provider: &str, service: Option<&str>) -> Option<String> {
    // Custom providers resolve under a stable derived variable
    // (`<NAME>_API_KEY`, uppercased like every table entry) without
    // leaking a string per call — this runs on every request. The env var
    // stays a fallback: the keyring (written by the API-key dialog) is
    // consulted first under the same name.
    let custom_env = custom_provider_env_var(provider);
    let env_var: &str = API_KEY_ENVS
        .iter()
        .find(|(key, _)| *key == provider)
        .map_or_else(
            || {
                if custom_provider_lookup(provider).is_some() {
                    custom_env.as_str()
                } else {
                    "OPENAI_API_KEY"
                }
            },
            |(_, var)| *var,
        );

    // The OS keyring is the authoritative store for keys saved explicitly
    // through the app (ADD Provider). Prefer it over the process environment
    // so stale shell/.env exports don't shadow the key the user configured.
    // Environment variables remain a fallback for providers never stored.
    let service = service.unwrap_or(COSH_SERVICE);
    if let Some(key) = resolve_env_var_key(service, env_var) {
        return Some(key);
    }

    // Shared-account providers (opencode-go) accept the account key saved
    // under the parent gateway's variable when their own is unset.
    let fallback = API_KEY_FALLBACK_ENVS
        .iter()
        .find(|(name, _)| *name == provider)
        .map(|(_, var)| *var);
    if let Some(fallback_var) = fallback
        && let Some(key) = resolve_env_var_key(service, fallback_var)
    {
        return Some(key);
    }

    None
}

/// Return all known provider names: the built-in registry plus every
/// user-registered custom provider (registry order, then registration
/// order).
///
/// Useful for enumerating providers in the TUI front-end or for batch
/// operations (e.g. fetching model lists from every provider).
pub fn known_providers() -> impl Iterator<Item = &'static str> {
    let mut names: Vec<&'static str> = PROVIDERS.iter().map(|(name, _)| *name).collect();
    names.extend(
        CUSTOM_PROVIDERS
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|cfg| cfg.name),
    );
    names.into_iter()
}

/// Whether a provider runs a local model server (`localhost`) with no API
/// key. Local providers are configured with a base URL instead of a key.
#[must_use]
pub fn is_local_provider(name: &str) -> bool {
    get_provider(name).is_some_and(|cfg| cfg.local)
}

/// Return all known local provider names (ollama, llamacpp, lmstudio, …).
///
/// These are configured with a base URL (host + port) rather than an API key.
pub fn known_local_providers() -> impl Iterator<Item = &'static str> {
    PROVIDERS
        .iter()
        .filter(|(_, cfg)| cfg.local)
        .map(|(name, _)| *name)
}

/// Normalize a user-supplied local server URL for an OpenAI-compatible
/// provider: if the URL has no path, append `/v1` so it matches the endpoint
/// the server exposes (`/v1/chat/completions`). Non-OpenAI-compatible
/// providers and URLs that already carry a path are returned verbatim.
#[must_use]
pub fn normalize_local_base_url(provider: &str, url: &str) -> String {
    // Strip a trailing slash so bare URLs with `/` don't produce `//v1`.
    let url = url.trim().trim_end_matches('/');
    let Some(cfg) = get_provider(provider) else {
        return url.to_string();
    };
    if cfg.family != Family::OpenAICompatible {
        return url.to_string();
    }
    // Use a proper URL parser instead of fragile string splitting — handles
    // ports, query strings, and edge cases like `http://` or `https://host/`.
    match reqwest::Url::parse(url) {
        Ok(parsed) if parsed.path() == "/" || parsed.path().is_empty() => {
            format!("{url}/v1")
        }
        _ => url.to_string(),
    }
}

/// Return the environment variable name for a given provider's API key.
///
/// Returns `None` if the provider is unknown.
#[must_use]
pub fn get_provider_env_var(provider: &str) -> Option<&'static str> {
    API_KEY_ENVS
        .iter()
        .find(|(key, _)| *key == provider)
        .map(|(_, var)| *var)
}

/// Whether the provider is `bedrock` — the one registry entry whose base
/// URL is region-templated and whose endpoint family lacks the OpenAI-
/// compatible `GET /models` and embedding surfaces (see the gates in
/// `connector::openai_compatible::caller`).
pub(crate) fn is_bedrock(provider: &str) -> bool {
    provider == "bedrock"
}

/// Validate an AWS region name against the region grammar
/// (`^[a-z]{2,4}(-[a-z]+)*-[0-9]+$` — us-east-1, eu-west-2, us-gov-west-1,
/// cn-north-1, and the EU sovereign cloud's four-letter eusc-de-east-1, …).
///
/// The validation is a SECURITY gate, not cosmetic: the region is spliced
/// into the endpoint HOST, and the `Authorization: Bearer` header rides that
/// host. `AWS_REGION` is ambient state set by dotfiles, CI configs and repo
/// `.env` loaders — an unvalidated value like `evil.com/x#` would silently
/// redirect the bearer token to a foreign origin. Anything that does not
/// parse as a region falls back to the registry default (a connection error
/// in the user's real region beats silent token exfiltration).
fn is_valid_aws_region(region: &str) -> bool {
    static REGION_SHAPE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REGION_SHAPE
        .get_or_init(|| regex::Regex::new(r"^[a-z]{2,4}(-[a-z]+)*-[0-9]+$").expect("static regex"))
        .is_match(region)
}

/// Resolve the live base URL for the `bedrock` provider from the standard AWS
/// region environment variables. The registry entry only pins a default
/// region (`us-east-1`); real workloads run in the user's own region, which
/// AWS signals through `AWS_REGION` / `AWS_DEFAULT_REGION`.
///
/// Returns the provider's registry URL unchanged for every other provider —
/// and for bedrock when no VALID region variable is exported or when the
/// ambient region IS the registry default (an unregistered provider name
/// yields an empty URL — unreachable through [`Connector::new`](crate::connector::Connector::new), which
/// rejects unknown names). An explicit `params.base_url` (connector
/// `with_base_url`) always wins over this resolution; the helper exists so
/// the *default* path follows the ambient AWS configuration instead of
/// silently calling the wrong region.
#[must_use]
pub fn resolved_provider_base_url(provider: &str) -> std::borrow::Cow<'static, str> {
    let Some(config) = get_provider(provider) else {
        return std::borrow::Cow::Borrowed("");
    };
    if !is_bedrock(provider) {
        return std::borrow::Cow::Borrowed(config.base_url);
    }
    // First VALID region among the two variables. A PRESENT but blank or
    // invalid `AWS_REGION` must not shadow a valid `AWS_DEFAULT_REGION`
    // (the same placeholder rule the API-key resolution applies).
    let region = ["AWS_REGION", "AWS_DEFAULT_REGION"].iter().find_map(|var| {
        std::env::var(var)
            .ok()
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty() && is_valid_aws_region(r))
    });
    match region {
        // The registry URL is region-templated (`bedrock-runtime.{region}.
        // amazonaws.com` with `us-east-1` pinned); re-point it to the ambient
        // region by name substitution. `us-east-1` itself needs no rewrite.
        Some(region) if region != "us-east-1" && config.base_url.contains("us-east-1") => {
            std::borrow::Cow::Owned(config.base_url.replace("us-east-1", &region))
        }
        _ => std::borrow::Cow::Borrowed(config.base_url),
    }
}

/// Return all known providers with their API key environment variable names.
///
/// Useful for UI rendering that needs both the provider name and its key env var.
pub fn known_providers_with_env() -> impl Iterator<Item = (&'static str, &'static str)> {
    API_KEY_ENVS.iter().map(|(name, env)| (*name, *env))
}

/// Whether a provider has an API key available, either from the OS credential
/// store (keyring) under the cosh service or from its environment variable.
///
/// Useful for the UI to decide whether a provider is "configured" without
/// having to distinguish between the two storage backends. Resolution order
/// matches [`get_api_key`]: keyring first, then environment.
#[must_use]
pub fn has_api_key(provider: &str) -> bool {
    get_api_key(provider, Some(COSH_SERVICE)).is_some()
}

/// Return the first provider with an available API key.
///
/// Checks both the environment variables and the OS credential store. The
/// iteration order follows the registration order in the provider table.
#[must_use]
pub fn detect_provider() -> Option<&'static str> {
    PROVIDERS
        .iter()
        .find(|(name, _)| has_api_key(name))
        .map(|(name, _)| *name)
}
