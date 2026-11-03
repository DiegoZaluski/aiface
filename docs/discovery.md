# `connector::discovery` — context windows and reasoning

Knows things about models that the models' own APIs don't tell you: the
**context window** and the **reasoning capability**.

```rust,ignore
pub async fn discover_context_window(model_name: &str, cache_dir: Option<&str>) -> Option<usize>
pub fn model_reasoning(model_name: &str, cache_dir: Option<&str>) -> Option<ModelReasoning>
pub fn model_reasoning_from_catalog(model_name: &str, cache_dir: Option<&str>) -> Option<ModelReasoning>
pub fn resolve_reasoning_effort(model: &str, desired: &str, cache_dir: Option<&str>) -> Option<String>
pub fn effective_context_window(window: usize) -> usize
pub struct ModelReasoning { pub supported: bool, pub efforts: Option<Vec<String>> }
```

## Context windows

`discover_context_window` resolves the model's max input window, fastest
source first:

1. **Static table** — a hardcoded list of known, current models (`gpt-4o` →
   128k, `o1`/`o3` → 200k, `claude-sonnet-4-6`/`claude-opus-4-6` → 1M,
   `deepseek-v4-flash` → 131k, …). No network, no disk — the most common
   models can never silently fall back because of a slow network or a
   changed API. Matching is exact, case-insensitive, and tolerant of `~`
   aliases and vendor prefixes.
2. **models.dev catalog** — a cached JSON catalog (`models.dev.json` in
   `cache_dir`) for the long tail of models not worth hardcoding.
3. **OpenRouter catalog** — fallback (`openrouter.json`).
4. **Anthropic API** — for unknown `claude*` ids only (requires an
   `x-api-key`, so it only helps when a key is present).

`cache_dir: None` disables the disk cache entirely (network catalogs are
still consulted). The function returns `Option<usize>` — `None` when nothing
resolved.

`effective_context_window(window)` computes the *usable* budget from the
advertised maximum — the "sweet spot" the harness sizes its compaction
budget to, so compaction triggers *before* the model enters its degradation
zone. The effective fraction **decays logarithmically** with window size
(`2.612 − 0.443 · log10(window)`), clamped to `[0.20, 0.85]`:

| Advertised window | Effective fraction | Effective budget |
|---|---|---|
| 10k | ~84% | ~8.4k |
| 32k | ~62% | ~19.7k |
| 100k | ~40% | ~39.7k |
| 128k | ~35% | ~44.7k |
| 200k | ~26% | ~52.8k |
| ≥ ~278k | 20% (floor) | 20% of advertised |

The reasoning (per the cited research): small windows are genuinely usable
almost whole, while giant windows are strongly limited — "the bigger the
window, the bigger the illusion". The 20% floor keeps the function monotonic
(a larger advertised window never yields a smaller effective window). It is
only ever applied to a **real** discovered window — never to the harness's
default fallback budget.

## Reasoning capability

`model_reasoning` answers "does this model think?" — resolved from:

1. **Static reasoning table** (same matching rules as the window table), so
   the common models resolve instantly and offline;
2. **Cached models.dev catalog** — the catalog is the authoritative offline
   source because the providers' own `/models` endpoints (OpenAI, Gemini,
   Claude) don't expose reasoning capability;
3. `None` — caller falls back to a name heuristic.

```rust,ignore
pub struct ModelReasoning {
    pub supported: bool,                    // advertises thinking/reasoning
    pub efforts: Option<Vec<String>>,       // accepted effort levels, e.g. ["low","medium","high"]
}
```

`efforts` is `Some` only when the catalog advertises effort-style
configuration; `None` means it only advertises a toggle or token budget (call
the standard low/medium/high set). `model_reasoning_from_catalog` skips the
static tier and reads the catalog directly. Both are synchronous and offline,
so the TUI can consult them instantly when rendering the model dialog.

`resolve_reasoning_effort` maps a user-chosen effort onto a level the model
actually accepts — used when the choice must survive a model/family switch
(auto mode, provider fallback). It keeps the desired level verbatim when
supported, otherwise returns the **closest** supported level (ties broken
 toward more thinking), `None` for models known to have no reasoning knob,
and the desired level unchanged for models whose capability is unknown.

---

Back to the [module overview](connector.md).

---

## Summary

- `discover_context_window` resolves a model's max input window from multiple sources: static table (fastest, offline), models.dev catalog (cached JSON), OpenRouter catalog (fallback), or Anthropic API (for unknown `claude*` ids only).
- `effective_context_window` computes the usable budget from the advertised maximum using a logarithmic decay formula, clamped to [0.20, 0.85] — small windows are mostly usable, giant windows are strongly limited.
- The decay formula reflects research findings: larger advertised windows create an "illusion" of capacity, so the effective fraction decreases as window size increases.
- `model_reasoning` answers whether a model supports thinking/reasoning, resolved from static table (offline) or cached models.dev catalog.
- `resolve_reasoning_effort` maps a desired effort onto the closest level the model accepts (ties toward more thinking), drops it for known non-reasoning models, and passes it through for unknown ones — so a user-chosen level survives fallback switches without sending a knob the model rejects.
- `ModelReasoning` includes `supported` (boolean) and `efforts` (optional list of accepted effort levels like "low", "medium", "high").
- Both discovery functions are synchronous and offline for TUI instant rendering; network catalogs are only consulted when static resolution fails.
