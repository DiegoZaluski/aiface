//! Tests for the debug-only economy hints (`connector::economy`).
//!
//! Emissions are process-global one-shots, so every test resets the state
//! and all hint-emitting tests serialize on [`ECON_LOCK`] (the log capture
//! sink is process-global too).

use super::super::economy;
use super::super::params::Parameters;
use super::super::{Connector, get_provider};
use std::sync::Mutex;

/// Process-global log capture sink. The `log` crate's logger is installed
/// ONCE per process — tests that need emissions serialize on [`ECON_LOCK`]
/// and read this sink between calls. Only ECONOMY hint records land here:
/// the capture logger keeps records carrying the hints' `-->` marker, so
/// unrelated `log` output from tests running in parallel cannot pollute
/// the assertions.
static SINK: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Serialize hint-emitting tests (one-shot state + global log sink).
static ECON_LOCK: Mutex<()> = Mutex::new(());

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let line = format!("{}: {}", record.level(), record.args());
            // Keep only hint-shaped records: every emission carries the
            // `-->` builder marker. Foreign log lines (other tests running
            // in parallel) are dropped, keeping the sink assertion-stable.
            if line.contains("-->") {
                SINK
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(line);
            }
        }
    }
    fn flush(&self) {}
}

/// Install the capture logger once and clear the sink. The `log` crate
/// rejects a second install; the first one already points at [`SINK`], so
/// the error is expected and harmless.
fn arm_capture() {
    let _ = log::set_boxed_logger(Box::new(CaptureLogger));
    log::set_max_level(log::LevelFilter::Debug);
    SINK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

fn captured() -> Vec<String> {
    SINK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

#[test]
fn reasoning_effort_hint_fires_once_for_a_reasoning_model() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    let connector = Connector::new("openai")
        .unwrap()
        .with_model("gpt-5.2")
        .with_api_key("sk-test");
    let provider = get_provider("openai").unwrap();
    // Same call twice: the second must be silent (one-shot per knob).
    for _ in 0..2 {
        economy::maybe_note_for_chat(provider, &connector.params, "gpt-5.2", false);
    }
    let lines = captured();
    let hits = lines
        .iter()
        .filter(|l| l.contains("with_reasoning_effort"))
        .count();
    assert_eq!(hits, 1, "reasoning hint must fire exactly once: {lines:?}");
    // The hint is about THIS method's shape, never another method.
    assert!(
        lines.iter().all(|l| !l.contains("stream_chat")),
        "hints must not recommend switching methods: {lines:?}"
    );
}

#[test]
fn reasoning_hint_is_silent_when_effort_is_set() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    let provider = get_provider("openai").unwrap();
    let params = Parameters {
        reasoning_effort: Some("low".to_string()),
        ..Parameters::default()
    };
    economy::maybe_note_for_chat(provider, &params, "gpt-5.2", false);
    assert!(
        captured().is_empty(),
        "an explicit effort must not trigger the reasoning hint"
    );
}

#[test]
fn reasoning_hint_is_silent_for_non_reasoning_models() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    let provider = get_provider("openai").unwrap();
    economy::maybe_note_for_chat(provider, &Parameters::default(), "gpt-4o", false);
    let lines = captured();
    assert!(
        lines.iter().all(|l| !l.contains("with_reasoning_effort")),
        "a non-reasoning model must never get the reasoning hint: {lines:?}"
    );
}

#[test]
fn session_affinity_hint_only_fires_on_multi_turn() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    let provider = get_provider("openai").unwrap();
    // One-shot single-prompt call: nothing to pin yet — silent.
    economy::maybe_note_for_chat(provider, &Parameters::default(), "gpt-4o", false);
    assert!(
        captured().is_empty(),
        "single-turn calls have no cache prefix to pin"
    );
    // Multi-turn call without a session id: the hint fires.
    economy::maybe_note_for_chat(provider, &Parameters::default(), "gpt-4o", true);
    let lines = captured();
    assert!(
        lines.iter().any(|l| l.contains("with_session_id")),
        "multi-turn calls should get the affinity hint: {lines:?}"
    );
}

#[test]
fn session_affinity_hint_never_fires_for_local_providers() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    // Local model servers bill no discounted cache reads — the affinity
    // advice would be semantically empty there.
    for name in ["ollama", "lmstudio", "vllm", "llamacpp"] {
        let provider = get_provider(name).unwrap();
        assert!(provider.local, "{name} must be registered as local");
        economy::maybe_note_for_chat(provider, &Parameters::default(), "gpt-4o", true);
    }
    let lines = captured();
    assert!(
        lines.iter().all(|l| !l.contains("with_session_id")),
        "local providers must never get the affinity hint: {lines:?}"
    );
}

#[test]
fn prompt_cache_hint_only_fires_for_disabled_claude_cache() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let provider = get_provider("claude").unwrap();

    // Default params: caching is ON — no cache hint (it is claude-only
    // anyway). Reset BEFORE each check so a prior test's one-shot state
    // cannot leak into this assertion.
    economy::reset_emitted_for_tests();
    arm_capture();
    economy::maybe_note_for_chat(provider, &Parameters::default(), "claude-sonnet-5", true);
    let lines = captured();
    assert!(
        lines.iter().all(|l| !l.contains("with_prompt_cache")),
        "default (enabled) cache must not warn: {lines:?}"
    );

    // Explicitly disabled cache on Claude: the hint fires.
    economy::reset_emitted_for_tests();
    arm_capture();
    let params = Parameters {
        prompt_cache: super::super::params::PromptCacheConfig {
            enabled: false,
            ..Default::default()
        },
        ..Parameters::default()
    };
    economy::maybe_note_for_chat(provider, &params, "claude-sonnet-5", true);
    let lines = captured();
    assert!(
        lines.iter().any(|l| l.contains("with_prompt_cache(true)")),
        "a disabled Claude cache should warn: {lines:?}"
    );
}

#[test]
fn hint_text_is_rust_style_and_direct() {
    let _lock = ECON_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    economy::reset_emitted_for_tests();
    arm_capture();

    let provider = get_provider("openai").unwrap();
    economy::maybe_note_for_chat(provider, &Parameters::default(), "gpt-5.2", false);
    let lines = captured().join("\n");
    // Compiler-diagnostic markers: `warning:`, `-->`, `note:`.
    assert!(lines.contains("warning:"), "style: {lines}");
    assert!(lines.contains("-->"), "style: {lines}");
    assert!(lines.contains("note:"), "style: {lines}");
    assert!(
        lines.contains("most economical request shape"),
        "style: {lines}"
    );
}
