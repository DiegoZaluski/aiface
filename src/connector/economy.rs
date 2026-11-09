//! Debug-only economy hints: warnings the connector emits when a call is
//! NOT using the token savings the called method could have used.
//!
//! Design rules (they are what makes these warnings useful instead of
//! noise):
//!
//! - **Debug-only.** Every emission is behind `log::debug!`, so release
//!   builds and quiet terminals never see them. Nothing here can affect a
//!   production user's logs.
//! - **About the method called, only.** A hint never suggests switching to
//!   another method or another provider — it points out what the CURRENT
//!   call left on the table, on the caller's own terms.
//! - **Never a method the call does not support.** A hint is emitted only
//!   when the connector's provider/family/model actually honors the
//!   missed knob (checked against the provider registry and the discovery
//!   tables), so the advice is always actionable.
//! - **Once per process.** [`Connector::chat`], `stream_chat*` and friends
//!   call [`maybe_note_for_chat`] before dispatching; each knob's advice
//!   is emitted once per process lifetime, the first time a call could
//!   have used it. An agent loop therefore sees a hint on its first turn
//!   and silence afterwards.
//!
//! The Rust-compiler-style framing ("help:" / "note:") is deliberate: the
//! audience is a developer who knows that format, and it keeps the hint
//! short, direct and didactic.

use std::sync::atomic::{AtomicBool, Ordering};

use super::provider::ProviderConfig;
use super::{Family, discovery};

/// A knob the called method supports but the request is not using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EconomyKnob {
    /// The model reasons, the request carries no reasoning effort, and the
    /// model's cheapest supported effort would bill fewer thinking tokens.
    ReasoningEffort,
    /// A multi-turn-style call (messages array or system prompt) on a
    /// provider that bills cached-prefix reads at a fraction of input
    /// price, with no cache-affinity routing set.
    SessionAffinity,
    /// The provider offers a prompt-cache surface for this call and the
    /// connector has it disabled (`claude` only; the default is ON).
    PromptCacheDisabled,
}

/// Number of variants in [`EconomyKnob`] — the emission-state array size.
/// A variant added without growing this constant fails to compile via the
/// match below (non-exhaustive), never at runtime via an out-of-bounds
/// index.
const KNOB_COUNT: usize = 3;

impl EconomyKnob {
    /// The builder method that enables the knob. Shown in the `help:` line.
    fn builder_method(self) -> &'static str {
        match self {
            Self::ReasoningEffort => ".with_reasoning_effort(\"low\")",
            Self::SessionAffinity => ".with_session_id(\"<stable-session-id>\")",
            Self::PromptCacheDisabled => ".with_prompt_cache(true)",
        }
    }

    /// What the knob buys, in one short clause. Shown in the `note:` line.
    fn note(self) -> &'static str {
        match self {
            Self::ReasoningEffort => "the cheapest supported effort bills fewer thinking tokens",
            Self::SessionAffinity => {
                "a stable id pins cached prompt prefixes to the same backend, so repeat turns read from cache"
            }
            Self::PromptCacheDisabled => {
                "cached prompt-prefix reads bill at 10% of the input price"
            }
        }
    }

    /// Stable index into the emission-state array (variant order).
    fn index(self) -> usize {
        match self {
            Self::ReasoningEffort => 0,
            Self::SessionAffinity => 1,
            Self::PromptCacheDisabled => 2,
        }
    }
}

/// One-shot emission state. DEBUG warnings are a developer tool: printing
/// the same hint on every turn of an agent loop would be spam, so each
/// knob's advice is emitted ONCE per process (the first call that could
/// have used it). The atomics keep the check lock-free on the hot path;
/// a `compare_exchange` claims the emission so concurrent first calls
/// cannot double-print.
static EMITTED: [AtomicBool; KNOB_COUNT] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

fn claim_emission(knob: EconomyKnob) -> bool {
    EMITTED[knob.index()]
        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

/// Reset the one-shot state (tests only — production code has no reason to
/// re-arm the hints).
#[cfg(test)]
pub(crate) fn reset_emitted_for_tests() {
    for flag in &EMITTED {
        flag.store(false, Ordering::Relaxed);
    }
}

/// Emit the debug hint for `knob` when the provider/model actually supports
/// it and this process has not emitted it yet.
fn maybe_note(provider: &ProviderConfig, model: &str, knob: EconomyKnob) {
    if !claim_emission(knob) {
        return;
    }
    if !knob_supported(provider, model, knob) {
        // Not supported HERE: give the knob back so a different provider or
        // model can still earn its hint later in the process.
        EMITTED[knob.index()].store(false, Ordering::Relaxed);
        return;
    }
    log::debug!(
        "warning: this call is not using the most economical request shape\n  \
         --> {}\n  \
         note: {}\n  \
         = help: hints are debug-only and appear once per knob",
        knob.builder_method(),
        knob.note()
    );
}

/// Whether the provider/model honors the knob — a hint must never recommend
/// something the called surface would silently drop.
///
/// - [`EconomyKnob::ReasoningEffort`]: the model must be reasoning-capable
///   AND the family must have a wire mapping for the knob. `Copilot` and
///   `SystemOne` have none. `cache_dir` is `None` on purpose: the check
///   consults ONLY the in-memory static table — no disk I/O or catalog
///   parse ever lands on the request path.
/// - [`EconomyKnob::SessionAffinity`]: cloud families with a cache layer
///   that discounts repeated prefixes. Local model servers have no
///   discounted cache-read billing, and the Copilot runtime has no HTTP
///   cache layer at all — advice there would be semantically empty.
/// - [`EconomyKnob::PromptCacheDisabled`]: the Anthropic surface only.
fn knob_supported(provider: &ProviderConfig, model: &str, knob: EconomyKnob) -> bool {
    match knob {
        EconomyKnob::ReasoningEffort => {
            matches!(
                provider.family,
                Family::OpenAICompatible | Family::OpenAi | Family::Gemini | Family::Claude
            ) && discovery::model_reasoning(model, None)
                .is_some_and(|meta| meta.supported)
        }
        EconomyKnob::SessionAffinity => !provider.local
            && matches!(
                provider.family,
                Family::OpenAICompatible | Family::OpenAi | Family::Claude | Family::Gemini
            ),
        EconomyKnob::PromptCacheDisabled => provider.family == Family::Claude,
    }
}

/// Where a debug hint is allowed. Called once by each chat/stream entry
/// point before dispatching to the family. The summarizer's
/// `stream_chat_with_system_no_tools` deliberately does NOT call this —
/// stripping tools is itself the economical shape for that call.
pub(crate) fn maybe_note_for_chat(
    provider: &ProviderConfig,
    params: &super::params::Parameters,
    model: &str,
    multi_turn: bool,
) {
    // Reasoning effort: the knob applies to THIS call whenever the model
    // can think and the request did not choose an effort.
    if params.reasoning_effort.is_none() {
        maybe_note(provider, model, EconomyKnob::ReasoningEffort);
    }
    // Cache-affinity routing only pays off across turns; a one-shot
    // single-prompt call has nothing to pin yet.
    if multi_turn && params.session_id.is_none() {
        maybe_note(provider, model, EconomyKnob::SessionAffinity);
    }
    // Anthropic prompt caching is ON by default; only an explicit
    // `.with_prompt_cache(false)` lands here.
    if provider.family == Family::Claude && !params.prompt_cache.enabled {
        maybe_note(provider, model, EconomyKnob::PromptCacheDisabled);
    }
}
