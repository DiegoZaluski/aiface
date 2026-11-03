//! Bridge to the GitHub Copilot agent runtime (official
//! `github-copilot-sdk` crate).
//!
//! One runtime connection is created lazily per process and reused for
//! every turn. Each connector turn spawns a short EPHEMERAL session:
//! `create_session` → `send_and_wait` → `disconnect` → `delete_session`
//! (the transcript is REMOVED, not preserved). Nothing about a turn is
//! persisted by the wrapper: `ClientMode::Empty` + an explicit
//! `base_directory` (cosh's own cache dir, never the user's real
//! `~/.copilot`, and never a shared `/tmp`) keep the runtime's state
//! contained, and in Empty mode the runtime's built-in tools (shell, file
//! edits, …) stay DISABLED — the session only sees the tools the connector
//! declares, so nothing executes outside cosh's own permission flow.
//!
//! Events are funneled through [`RuntimeEvent`]:
//! - `assistant.message_delta` → [`RuntimeEvent::Token`]
//! - `assistant.reasoning_delta` → [`RuntimeEvent::Reasoning`]
//! - the complete `tool_requests` of the final `assistant.message` →
//!   [`RuntimeEvent::ToolCall`] (the authoritative, complete calls —
//!   `tool_call_delta` fragments are deliberately NOT surfaced, so a call
//!   can never be emitted twice)
//! - `assistant.usage` → [`RuntimeEvent::Usage`] (raw JSON, parsed by
//!   `tokens::extract_usage`; a multi-call turn's totals are SUMMED for the
//!   non-streaming path)
//! - `session.error` → [`RuntimeEvent::Failed`]
//!
//! The turn's terminal event comes from the SDK's idle-wait machinery
//! (`send_and_wait`); dropping the returned [`TurnHandle`] aborts the
//! session, so an abandoned stream never leaves a runtime session running.
//!
//! Known limitation: the shared runtime connection is cached for the
//! process lifetime; if the spawned runtime process dies mid-session every
//! subsequent turn fails with a transport error until process restart.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use github_copilot_sdk::{
    Client, ClientMode, ClientOptions, MessageOptions, SessionConfig, SessionEvent,
    session_events::{
        AssistantMessageData, AssistantMessageDeltaData, AssistantMessageToolRequest,
        AssistantReasoningDeltaData, SessionErrorData, SessionEventType,
    },
    subscription::RecvErrorKind,
    types::{SystemMessageConfig, Tool},
};
use tokio::sync::{mpsc, oneshot};

use super::super::error::ConnectorError;
use crate::connector::NativeToolCall;

/// Total time a single turn may run before the SDK's idle wait gives up.
///
/// Agentic turns execute tools and can legitimately take minutes; ten
/// minutes of a completely silent session is the dead-session signal.
/// Errors surfaced this way are NOT retried by the connector's retry
/// middleware — a half-executed agentic turn must not be blindly re-sent.
const TURN_WAIT_TIMEOUT: Duration = Duration::from_secs(600);

/// How a turn ended.
#[derive(Debug)]
pub(crate) enum TurnOutcome {
    /// The session went idle (terminal event consumed via the stream).
    Completed,
    /// The turn failed before reaching idle (transport error, wait timeout).
    Failed(String),
}

/// The final state of a completed turn.
#[derive(Debug, Default)]
pub(crate) struct TurnResult {
    /// Final assistant text: the last `assistant.message` content, or the
    /// concatenation of the streamed deltas when no final message arrived.
    pub message: String,
    /// Whether any `message_delta` arrived. When it did, the final
    /// `assistant.message` text is ALREADY visible to the consumer and must
    /// not be re-emitted as a token chunk (it would render twice).
    pub streamed: bool,
    /// SUMMED usage across every model call of the turn, serialized in the
    /// runtime's camelCase wire shape (what `tokens::extract_usage` parses).
    pub usage: Option<serde_json::Value>,
    /// Provider finish reason from the last usage event (`"stop"`,
    /// `"tool_calls"`, …).
    pub finish_reason: Option<String>,
    totals: UsageTotals,
}

/// Running sums across a turn's per-model-call `assistant.usage` events.
#[derive(Debug, Default)]
struct UsageTotals {
    input_tokens: i64,
    output_tokens: i64,
    cache_read_tokens: i64,
    cache_write_tokens: i64,
    reasoning_tokens: i64,
    cost: f64,
    /// The LAST raw `assistant.usage` event of the turn. Its non-counter
    /// metadata — cache TTL/expiry (`cacheExpiresAt`, `cacheTtlSeconds`),
    /// cache-frontier provenance (`frontierSource`), and the billing
    /// breakdown (`copilotUsage`) — is preserved in the serialized usage
    /// (the `raw` consumers parse); only the summed counters are
    /// overwritten.
    last_event: Option<serde_json::Value>,
}

impl UsageTotals {
    fn add(&mut self, usage: &serde_json::Value) {
        self.input_tokens += usage
            .get("inputTokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        self.output_tokens += usage
            .get("outputTokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        self.cache_read_tokens += usage
            .get("cacheReadTokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        self.cache_write_tokens += usage
            .get("cacheWriteTokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        self.reasoning_tokens += usage
            .get("reasoningTokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        // Cost is a per-call price, not a monotonic counter: the turn's
        // billed total is the SUM, but per-event max would lose money — sum
        // it like the token counters.
        self.cost += usage.get("cost").and_then(|v| v.as_f64()).unwrap_or(0.0);
        self.last_event = Some(usage.clone());
    }

    fn to_json(&self) -> serde_json::Value {
        // Start from the last raw event (preserving cache/billing metadata)
        // and overwrite the counters with the turn totals. The last call's
        // per-call attribution fields are dropped: the totals span several
        // calls, so keeping one call's id would misattribute the turn.
        let mut obj = match &self.last_event {
            Some(v) => v.clone(),
            None => serde_json::Value::Object(serde_json::Map::new()),
        };
        if let Some(map) = obj.as_object_mut() {
            map.remove("apiCallId");
            map.insert("inputTokens".into(), self.input_tokens.into());
            map.insert("outputTokens".into(), self.output_tokens.into());
            map.insert("cacheReadTokens".into(), self.cache_read_tokens.into());
            map.insert("cacheWriteTokens".into(), self.cache_write_tokens.into());
            map.insert("reasoningTokens".into(), self.reasoning_tokens.into());
            map.insert("cost".into(), serde_json::json!(self.cost));
        }
        obj
    }
}

/// A runtime failure with the upstream metadata needed for error mapping.
#[derive(Debug, Clone)]
pub(crate) struct SessionFailure {
    pub message: String,
    /// HTTP status from the upstream request, when the runtime surfaced one.
    pub status: Option<i32>,
    /// Runtime error category (`"rate_limit"`, `"quota"`, `"query"`, …).
    pub error_type: Option<String>,
}

/// Handle over one in-flight runtime turn.
///
/// Dropping the handle closes the completion channel, which makes the
/// driver task abort the underlying session — an abandoned stream (TUI
/// quit, superseded turn) never leaks a runtime session.
pub(crate) struct TurnHandle {
    rx: mpsc::Receiver<SessionEvent>,
    done: oneshot::Receiver<TurnOutcome>,
    /// sanitized → original tool-name mapping (see [`build_session_config`]).
    name_map: HashMap<String, String>,
}

impl TurnHandle {
    /// Drive the turn to completion, invoking `on_event` for every streamed
    /// runtime event.
    ///
    /// `on_event` failing (e.g. the consumer dropped the stream) aborts the
    /// turn. The returned [`TurnResult`] carries the final text and usage.
    pub async fn run(
        mut self,
        mut on_event: impl FnMut(RuntimeEvent) -> Result<(), ConnectorError>,
    ) -> Result<TurnResult, ConnectorError> {
        let name_map = std::mem::take(&mut self.name_map);
        let mut result = TurnResult::default();
        loop {
            tokio::select! {
                biased;
                outcome = &mut self.done => {
                    match outcome {
                        Ok(TurnOutcome::Completed) => break,
                        Ok(TurnOutcome::Failed(e)) => {
                            return Err(session_error(&SessionFailure {
                                message: e,
                                // Transport-level wait failures carry no
                                // upstream status or category: not transient
                                // (a half-executed agentic turn must never be
                                // blindly re-sent).
                                status: None,
                                error_type: None,
                            }))
                        }
                        Err(_) => break, // driver dropped: treat as clean end
                    }
                }
                event = self.rx.recv() => {
                    match event {
                        Some(ev) => apply_event(&ev, &mut result, &name_map, &mut on_event)?,
                        None => break, // event forwarder gone
                    }
                }
            }
        }
        // The terminal event can beat the last buffered stream events
        // through the broadcast channel — drain whatever is still queued so
        // the final usage / message deltas are not lost.
        while let Ok(ev) = self.rx.try_recv() {
            apply_event(&ev, &mut result, &name_map, &mut on_event)?;
        }
        Ok(result)
    }
}

/// One event fanned out from a runtime session to its waiter.
#[derive(Debug)]
pub(crate) enum RuntimeEvent {
    /// Streaming text delta (`assistant.message_delta`).
    Token(String),
    /// Streaming reasoning delta (`assistant.reasoning_delta`).
    Reasoning(String),
    /// A complete native tool call from the turn's final
    /// `assistant.message.tool_requests` (the authoritative envelope).
    ToolCall(NativeToolCall),
    /// The turn's final assistant message (`assistant.message`).
    Message(String),
    /// The raw `assistant.usage` event data (JSON) — parsed by `tokens.rs`.
    Usage(serde_json::Value),
    /// A fan-out carried by one session event: the final
    /// `assistant.message` emits its complete tool calls AND the text.
    Batch(Vec<RuntimeEvent>),
    /// The turn failed (`session.error`).
    Failed(SessionFailure),
}

/// Lazily-created, process-lifetime Copilot runtime connection.
static SHARED: tokio::sync::Mutex<Option<Arc<Runtime>>> = tokio::sync::Mutex::const_new(None);

/// A ready Copilot runtime connection.
pub(crate) struct Runtime {
    client: Client,
}

impl Runtime {
    async fn get(api_key: Option<&str>) -> Result<Arc<Self>, ConnectorError> {
        let mut guard = SHARED.lock().await;
        if let Some(rt) = guard.as_ref() {
            return Ok(Arc::clone(rt));
        }
        // `ClientMode::Empty` + explicit base directory: the runtime keeps
        // its state out of the user's real `~/.copilot`, and Empty mode
        // disables the CLI-style ambient behavior (including built-in
        // tools) — exactly the containment an embedded agent wants. The
        // cache dir comes from the `directories` crate (XDG-aware); there
        // is NO `/tmp` fallback — without a resolvable cache dir the turn
        // fails instead of scattering conversation state into a shared,
        // world-readable directory.
        //
        // A client-level GitHub token is set from the FIRST caller's
        // credential so client-level RPCs (`models.list`) authenticate too;
        // per-turn tokens are still applied at session scope and win for
        // model calls. CONSEQUENCE: whoever initializes the process's
        // runtime first determines the identity behind client-level RPCs
        // for the whole process — `list_models` in a multi-credential
        // process reflects that first credential, not the caller's.
        let cache = directories::ProjectDirs::from("", "", "cosh")
            .map(|dirs| dirs.cache_dir().to_path_buf().join("copilot"))
            .ok_or_else(|| {
                ConnectorError::Network(
                    "no resolvable cache directory for the Copilot runtime \
                     (set HOME or XDG_CACHE_HOME)"
                        .into(),
                )
            })?;
        let mut options = ClientOptions::default()
            .with_mode(ClientMode::Empty)
            .with_base_directory(cache);
        if let Some(token) = api_key {
            options = options.with_github_token(token.to_string());
        }
        let client = Client::start(options).await.map_err(|e| {
            ConnectorError::Network(format!("failed to start Copilot runtime: {e}"))
        })?;
        let rt = Arc::new(Self { client });
        *guard = Some(Arc::clone(&rt));
        Ok(rt)
    }
}

/// Everything needed to start one runtime turn.
pub(crate) struct TurnPlan {
    pub config: SessionConfig,
    /// sanitized → original tool-name mapping (see
    /// [`build_session_config`]); applied to returned tool calls so the
    /// harness validates against the ORIGINAL registry names.
    pub name_map: HashMap<String, String>,
    /// The source-qualified `available_tools` allowlist that was set on
    /// `config` (`custom:<sanitized-name>` entries) — exposed for tests and
    /// diagnostics; `ClientMode::Empty` requires the list to be explicit.
    #[cfg_attr(not(test), allow(dead_code))]
    pub available_tools: Vec<String>,
}

/// Spawn an ephemeral session and dispatch `prompt` on it.
///
/// The driver task owns the session: it runs the SDK's `send_and_wait`
/// idle wait, forwards the terminal event through `done`, and disconnects
/// AND deletes the session afterwards (no transcript persists on disk).
/// Dropping the `done` receiver (caller abandoned the turn) makes the
/// driver abort the session instead of waiting.
pub(crate) async fn start_turn(
    plan: TurnPlan,
    prompt: String,
    api_key: Option<String>,
) -> Result<TurnHandle, ConnectorError> {
    let runtime = Runtime::get(api_key.as_deref()).await?;
    // The GitHub token is a SESSION-scoped credential (`with_github_token`
    // on SessionConfig): per-session tokens avoid a process-wide identity
    // and let different connectors carry different tokens.
    let mut config = plan.config;
    if let Some(token) = api_key {
        config = config.with_github_token(token);
    }
    let session =
        runtime.client.create_session(config).await.map_err(|e| {
            ConnectorError::Network(format!("failed to create Copilot session: {e}"))
        })?;
    let name_map = plan.name_map;

    let (tx, rx) = mpsc::channel::<SessionEvent>(256);
    let (mut done_tx, done) = oneshot::channel();
    let mut sub = session.subscribe();
    let forwarder = tokio::spawn(async move {
        loop {
            match sub.recv().await {
                Ok(ev) => {
                    if tx.send(ev).await.is_err() {
                        break; // waiter gone
                    }
                }
                // `Lagged` means the subscription SKIPPED events but keeps
                // delivering from the next live one — continuing is the only
                // way to reach the terminal event; exiting would truncate a
                // still-running turn into a silent fake success.
                Err(e) if matches!(e.kind(), RecvErrorKind::Lagged(_)) => continue,
                Err(_) => break, // closed: the session shut down
            }
        }
    });

    let opts = MessageOptions::new(prompt).with_wait_timeout(TURN_WAIT_TIMEOUT);
    let client_for_delete = runtime.client.clone();
    tokio::spawn(async move {
        let wait = session.send_and_wait(opts);
        tokio::pin!(wait);
        tokio::select! {
            result = &mut wait => {
                forwarder.abort();
                let outcome = match result {
                    // `send_and_wait` resolves with the turn's last
                    // assistant message on success; the streamed events
                    // already carried the text, so the terminal event only
                    // marks completion.
                    Ok(_terminal) => TurnOutcome::Completed,
                    Err(e) => TurnOutcome::Failed(format!("Copilot runtime error: {e}")),
                };
                let _ = done_tx.send(outcome);
                // Disconnect AND delete: `disconnect` preserves the session
                // transcript on disk; `delete_session` removes it so cosh
                // never accumulates conversation state.
                let _ = session.disconnect().await;
                let _ = client_for_delete.delete_session(session.id()).await;
            }
            _ = done_tx.closed() => {
                // The consumer abandoned the turn — stop the session so it
                // does not keep running headless.
                forwarder.abort();
                let _ = session.abort().await;
                let _ = session.disconnect().await;
                let _ = client_for_delete.delete_session(session.id()).await;
            }
        }
    });
    Ok(TurnHandle { rx, done, name_map })
}

/// Lower the connector's parameters onto a runtime [`SessionConfig`].
///
/// Mapping:
/// - `model` → `SessionConfig::model` (the runtime validates the id);
/// - `reasoning_effort` → the runtime's native reasoning knob;
/// - system prompt → a system-message APPEND (`mode: "append"`): the
///   runtime always carries its own instructions, so the harness prompt
///   must be ADDED to them, never replace them;
/// - declared tools → handler-less runtime [`Tool`] declarations
///   (declaration-only: the runtime surfaces tool requests as events; the
///   harness executes them — nothing runs inside the runtime). Names are
///   sanitized to the runtime's `/^[a-zA-Z0-9_-]+$/` and the sanitized →
///   original mapping is returned so tool calls can be translated back to
///   the harness registry names;
/// - `temperature` and the other OpenAI-style sampling knobs have no
///   runtime surface for a session — they are ignored (documented on the
///   family, like providers that lack a knob).
///
/// In inline tool-call mode the declarations are withheld entirely: the
/// model then cannot produce structured tool requests and the harness
/// parses its inline-JSON text, keeping the two delivery paths from
/// crossing (same contract as every other family).
pub(crate) fn build_session_config(
    params: &super::super::params::Parameters,
    system: Option<&str>,
) -> TurnPlan {
    use super::super::params::ToolCallMode;

    let mut config = SessionConfig::default().with_streaming(true);
    if let Some(model) = params.model.as_deref().filter(|m| !m.is_empty()) {
        config = config.with_model(model.to_string());
    }
    if let Some(effort) = params.reasoning_effort.as_deref() {
        config = config.with_reasoning_effort(effort.to_string());
    }
    if let Some(sys) = system.filter(|s| !s.is_empty()) {
        config = config.with_system_message(
            SystemMessageConfig::new()
                .with_mode("append")
                .with_content(sys.to_string()),
        );
    }
    let mut name_map = HashMap::new();
    // `ClientMode::Empty` REQUIRES an explicit `available_tools` allowlist
    // (the SDK rejects the session otherwise) — and the filter is
    // source-qualified: `custom:<name>` matches tools declared via
    // `with_tools`. Native mode allowlists exactly the declared tools;
    // inline mode allows NONE (no declarations → the model cannot emit
    // structured tool calls, and Empty mode keeps built-ins out).
    let mut allow: Vec<String> = Vec::new();
    if params.tool_call_mode == ToolCallMode::Native
        && let Some(tools) = params.tools.as_ref().filter(|t| !t.is_empty())
    {
        let mut declarations: Vec<Tool> = Vec::with_capacity(tools.len());
        for tool in tools {
            let sanitized = sanitize_tool_name(&tool.function.name);
            if sanitized.is_empty() {
                continue;
            }
            // A sanitization COLLISION (two distinct names mapping to the
            // same runtime-safe name) must not silently misroute calls: the
            // mapping would translate calls for either original to the last
            // inserted one. The colliding tool is dropped from this turn's
            // declarations (fail loudly at the harness, never misroute).
            if name_map.contains_key(&sanitized) {
                continue;
            }
            // Map sanitized → original so returned calls are translated
            // back to the name the harness registry knows.
            name_map.insert(sanitized.clone(), tool.function.name.clone());
            // The sanitized name already satisfies the runtime's
            // `/^[a-zA-Z0-9_-]+$/`, so the `custom:` filter entry is valid.
            allow.push(format!("custom:{sanitized}"));
            // `overrides_built_in_tool` is REQUIRED for names that collide
            // with the runtime's own tools (cosh's `web_fetch` vs the
            // builtin `web_fetch`): without the flag the runtime's
            // `session.options.update` REJECTS the whole declaration list
            // ("External tool … conflicts with a built-in tool of the same
            // name") and every turn fails. Set UNCONDITIONALLY — the
            // runtime's builtin roster is not enumerable (no SDK API lists
            // it; the names drift across runtime versions) and a
            // collision-name list would rot silently; non-colliding names
            // ignore the flag. The flag makes cosh's handler the
            // authoritative implementation for that name — which is
            // exactly the containment contract: built-ins stay disabled,
            // calls route to the harness registry.
            let mut t = Tool::new(sanitized)
                .with_description(tool.function.description.clone().unwrap_or_default())
                .with_overrides_built_in_tool(true);
            if let Some(schema) = &tool.function.parameters {
                t = t.with_parameters(schema.clone());
            }
            declarations.push(t);
        }
        config = config.with_tools(declarations);
    }
    config = config.with_available_tools(allow.clone());
    TurnPlan {
        config,
        name_map,
        available_tools: allow,
    }
}

/// The runtime enforces `/^[a-zA-Z0-9_-]+$/` on tool names — map every
/// other character to `_` (and drop empty results) instead of failing the
/// whole turn over one oddly-named tool.
fn sanitize_tool_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.trim_matches('_').to_string()
}

/// Apply one runtime event to the turn result and the consumer callback.
fn apply_event(
    event: &SessionEvent,
    result: &mut TurnResult,
    name_map: &HashMap<String, String>,
    on_event: &mut impl FnMut(RuntimeEvent) -> Result<(), ConnectorError>,
) -> Result<(), ConnectorError> {
    match classify(event) {
        Some(RuntimeEvent::Token(t)) => {
            result.message.push_str(&t);
            result.streamed = true;
            on_event(RuntimeEvent::Token(t))
        }
        Some(RuntimeEvent::Reasoning(r)) => on_event(RuntimeEvent::Reasoning(r)),
        Some(RuntimeEvent::ToolCall(mut tc)) => {
            // Translate the sanitized declaration name back to the ORIGINAL
            // registry name the harness validates against.
            if let Some(original) = name_map.get(&tc.name) {
                tc.name = original.clone();
            }
            on_event(RuntimeEvent::ToolCall(tc))
        }
        Some(RuntimeEvent::Message(text)) => {
            // The final message supersedes the accumulated deltas for the
            // non-streaming result — but when deltas already streamed, the
            // consumer has SEEN the text; re-emitting the full message as a
            // token chunk would render the response twice.
            result.message = text;
            if !result.streamed {
                result.streamed = true;
                on_event(RuntimeEvent::Token(result.message.clone()))
            } else {
                Ok(())
            }
        }
        Some(RuntimeEvent::Batch(events)) => {
            for ev in events {
                apply_batch_member(ev, result, name_map, on_event)?;
            }
            Ok(())
        }
        Some(RuntimeEvent::Usage(u)) => {
            result.finish_reason = u
                .get("finishReason")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // Sum every model call's usage (agentic turns make several).
            // The stream carries the CUMULATIVE totals, not the raw
            // per-call event: `ChatStream::merge_stream` keeps per-field
            // maxima across frames, and max over cumulative sums IS the
            // turn total — whereas max over per-call values would
            // undercount (e.g. calls of 100 then 50 inputs report 100,
            // not 150).
            result.totals.add(&u);
            let totals = result.totals.to_json();
            result.usage = Some(totals.clone());
            on_event(RuntimeEvent::Usage(totals))
        }
        Some(RuntimeEvent::Failed(failure)) => Err(session_error(&failure)),
        None => Ok(()),
    }
}

/// Apply one member of a [`RuntimeEvent::Batch`] without re-classifying.
fn apply_batch_member(
    event: RuntimeEvent,
    result: &mut TurnResult,
    name_map: &HashMap<String, String>,
    on_event: &mut impl FnMut(RuntimeEvent) -> Result<(), ConnectorError>,
) -> Result<(), ConnectorError> {
    match event {
        RuntimeEvent::ToolCall(mut tc) => {
            if let Some(original) = name_map.get(&tc.name) {
                tc.name = original.clone();
            }
            on_event(RuntimeEvent::ToolCall(tc))
        }
        RuntimeEvent::Message(text) => {
            result.message = text.clone();
            if !result.streamed {
                result.streamed = true;
                on_event(RuntimeEvent::Token(text))
            } else {
                Ok(())
            }
        }
        other => on_event(other),
    }
}

/// Map one raw session event onto the wrapper's [`RuntimeEvent`] funnel.
///
/// Unrelated events (session lifecycle, tool execution progress, …)
/// classify to `None`. `assistant.tool_call_delta` fragments are
/// deliberately IGNORED: they stream argument FRAGMENTS, and emitting a
/// call as soon as one fragment parses would duplicate the authoritative
/// `tool_requests` of the final `assistant.message` (double execution).
pub(crate) fn classify(event: &SessionEvent) -> Option<RuntimeEvent> {
    match event.parsed_type() {
        SessionEventType::AssistantMessageDelta => {
            let d: AssistantMessageDeltaData = event.typed_data()?;
            (!d.delta_content.is_empty()).then(|| RuntimeEvent::Token(d.delta_content))
        }
        SessionEventType::AssistantReasoningDelta => {
            let d: AssistantReasoningDeltaData = event.typed_data()?;
            (!d.delta_content.is_empty()).then(|| RuntimeEvent::Reasoning(d.delta_content))
        }
        SessionEventType::AssistantMessage => {
            let d: AssistantMessageData = event.typed_data()?;
            // The final message's `tool_requests` carry the COMPLETE native
            // calls — the only authoritative source. Emit them first so the
            // extractor sees validated envelopes, then the message text.
            let mut events: Vec<RuntimeEvent> = d
                .tool_requests
                .unwrap_or_default()
                .iter()
                .filter_map(tool_request_to_call)
                .collect();
            events.push(RuntimeEvent::Message(d.content));
            Some(RuntimeEvent::Batch(events))
        }
        SessionEventType::AssistantUsage => Some(RuntimeEvent::Usage(event.data.clone())),
        SessionEventType::SessionError => {
            let d: SessionErrorData = event.typed_data()?;
            Some(RuntimeEvent::Failed(SessionFailure {
                message: d.message,
                status: d.status_code,
                error_type: Some(d.error_type),
            }))
        }
        _ => None,
    }
}

/// Convert a finalized `tool_requests` entry into a native call event.
fn tool_request_to_call(req: &AssistantMessageToolRequest) -> Option<RuntimeEvent> {
    let args = match &req.arguments {
        Some(v) if v.is_object() => v.to_string(),
        Some(_) | None => "{}".to_string(),
    };
    Some(RuntimeEvent::ToolCall(NativeToolCall {
        id: req.tool_call_id.clone(),
        name: req.name.clone(),
        arguments: args,
        // The Copilot runtime has no thought-signature replay contract.
        thought_signature: String::new(),
    }))
}

/// Map a runtime failure onto the connector's error taxonomy.
///
/// Rate-limit/quota failures are TRANSIENT (the retry middleware paces them
/// with backoff). Non-transient failures are mapped to status 400 even when
/// the upstream sent 5xx: `retry::is_retryable_error` treats any 5xx as
/// retryable, and a half-executed AGENTIC turn must never be blindly
/// re-sent — the real status stays visible in the error body.
pub(crate) fn session_error(failure: &SessionFailure) -> ConnectorError {
    let transient = matches!(
        failure.error_type.as_deref(),
        Some("rate_limit") | Some("quota")
    ) || matches!(failure.status, Some(429));
    let status = match failure.status {
        Some(s) if (100..=599).contains(&s) && transient => s as u16,
        _ if transient => 503,
        _ => 400,
    };
    ConnectorError::HttpError {
        status,
        body: failure.message.clone(),
        retry_after_ms: None,
        transient,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: &str, data: serde_json::Value) -> SessionEvent {
        serde_json::from_value(serde_json::json!({
            "id": "e1",
            "timestamp": "2026-01-01T00:00:00Z",
            "type": kind,
            "data": data,
        }))
        .expect("valid SessionEvent JSON")
    }

    fn collector(
        events: &mut Vec<RuntimeEvent>,
    ) -> impl FnMut(RuntimeEvent) -> Result<(), ConnectorError> + '_ {
        move |ev| {
            events.push(ev);
            Ok(())
        }
    }

    #[test]
    fn sanitize_tool_name_maps_invalid_chars() {
        assert_eq!(sanitize_tool_name("find_glob"), "find_glob");
        assert_eq!(sanitize_tool_name("mcp.server/tool"), "mcp_server_tool");
        assert_eq!(sanitize_tool_name("::"), "");
    }

    #[test]
    fn session_errors_classify_retryability() {
        let limited = session_error(&SessionFailure {
            message: "user rate limit exceeded".into(),
            status: Some(429),
            error_type: Some("rate_limit".into()),
        });
        assert!(limited.is_transient());
        let fatal = session_error(&SessionFailure {
            message: "invalid model id".into(),
            status: Some(500),
            error_type: Some("query".into()),
        });
        // Non-transient failures must NOT be retryable, even with a 5xx
        // upstream status (a half-executed agentic turn can't be re-sent).
        assert!(!fatal.is_transient());
    }

    #[test]
    fn build_session_config_maps_tool_names_back() {
        use super::super::super::params::{Parameters, ToolCallMode, ToolDefinition, ToolFunction};
        let params = Parameters {
            tool_call_mode: ToolCallMode::Native,
            tools: Some(vec![ToolDefinition::new(ToolFunction::new(
                "mcp.server/tool",
            ))]),
            ..Parameters::default()
        };
        let plan = build_session_config(&params, Some("sys"));
        assert_eq!(
            plan.name_map.get("mcp_server_tool").map(String::as_str),
            Some("mcp.server/tool")
        );
        // Inline mode withholds declarations entirely.
        let inline = Parameters {
            tool_call_mode: ToolCallMode::Inline,
            ..Parameters::default()
        };
        let plan = build_session_config(&inline, None);
        assert!(plan.name_map.is_empty());
    }

    #[test]
    fn session_config_always_carries_an_explicit_tool_allowlist() {
        use super::super::super::params::{Parameters, ToolCallMode, ToolDefinition, ToolFunction};
        // `ClientMode::Empty` REQUIRES `available_tools` (the SDK rejects
        // the session otherwise), so the allowlist must be explicit in BOTH
        // tool modes. Native mode allowlists the declared tools with the
        // runtime's source-qualified `custom:` form.
        let native = Parameters {
            tool_call_mode: ToolCallMode::Native,
            tools: Some(vec![ToolDefinition::new(ToolFunction::new("find_glob"))]),
            ..Parameters::default()
        };
        let plan = build_session_config(&native, None);
        assert_eq!(plan.available_tools, vec!["custom:find_glob".to_string()]);

        // Inline mode: no declarations, and the allowlist is EMPTY (not
        // absent — an empty Vec still satisfies the Empty-mode invariant).
        let inline = Parameters {
            tool_call_mode: ToolCallMode::Inline,
            tools: None,
            ..Parameters::default()
        };
        let plan = build_session_config(&inline, None);
        assert!(plan.available_tools.is_empty());
    }

    #[test]
    fn tool_name_collision_drops_the_second_tool() {
        use super::super::super::params::{Parameters, ToolCallMode, ToolDefinition, ToolFunction};
        // Both names sanitize to "mcp_a_b" — a collision must NOT overwrite
        // the mapping (that would silently misroute calls to the wrong
        // tool); the second tool is dropped from the declarations instead.
        let params = Parameters {
            tool_call_mode: ToolCallMode::Native,
            tools: Some(vec![
                ToolDefinition::new(ToolFunction::new("mcp.a/b")),
                ToolDefinition::new(ToolFunction::new("mcp_a_b")),
            ]),
            ..Parameters::default()
        };
        let plan = build_session_config(&params, None);
        assert_eq!(
            plan.name_map.get("mcp_a_b").map(String::as_str),
            Some("mcp.a/b"),
            "first insertion wins; the colliding tool is dropped"
        );
    }

    #[test]
    fn declared_tools_override_built_ins_of_the_same_name() {
        use super::super::super::params::{Parameters, ToolCallMode, ToolDefinition, ToolFunction};
        // cosh declares tools whose names collide with the runtime's
        // built-ins (e.g. `web_fetch`). Without `overrides_built_in_tool`
        // the runtime's `session.options.update` REJECTS the whole
        // declaration list and every turn fails — the flag must be set on
        // EVERY declaration, unconditionally.
        let params = Parameters {
            tool_call_mode: ToolCallMode::Native,
            tools: Some(vec![
                ToolDefinition::new(ToolFunction::new("web_fetch")),
                ToolDefinition::new(ToolFunction::new("find_glob")),
            ]),
            ..Parameters::default()
        };
        let plan = build_session_config(&params, None);
        let declarations = plan.config.tools.expect("native mode declares tools");
        assert_eq!(declarations.len(), 2);
        for tool in declarations {
            assert!(
                tool.overrides_built_in_tool,
                "{} must declare the override flag or the runtime rejects the turn",
                tool.name
            );
        }
        // The override flag must not change the allowlist shape: an
        // overridden builtin is still classified `custom:` at registration
        // time (source-qualified filter, mode.rs), so the colliding name
        // keeps its `custom:` entry.
        assert!(
            plan.available_tools
                .contains(&"custom:web_fetch".to_string()),
            "the overridden builtin keeps its custom: allowlist entry"
        );
    }

    #[test]
    fn final_message_not_re_emitted_after_deltas() {
        let mut events = Vec::new();
        let mut result = TurnResult::default();
        let delta = event(
            "assistant.message_delta",
            serde_json::json!({ "deltaContent": "Hi ", "messageId": "m1" }),
        );
        apply_event(
            &delta,
            &mut result,
            &HashMap::new(),
            &mut collector(&mut events),
        )
        .expect("delta applies");
        let final_msg = event(
            "assistant.message",
            serde_json::json!({ "content": "Hi there", "messageId": "m1" }),
        );
        apply_event(
            &final_msg,
            &mut result,
            &HashMap::new(),
            &mut collector(&mut events),
        )
        .expect("final message applies");
        // Exactly ONE token chunk (the delta); the full message is NOT
        // re-emitted, or every consumer would render the response twice.
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], RuntimeEvent::Token(ref t) if t == "Hi "));
        assert_eq!(result.message, "Hi there");
    }

    #[test]
    fn final_message_emitted_when_nothing_streamed() {
        let mut events = Vec::new();
        let mut result = TurnResult::default();
        let final_msg = event(
            "assistant.message",
            serde_json::json!({ "content": "full answer", "messageId": "m1" }),
        );
        apply_event(
            &final_msg,
            &mut result,
            &HashMap::new(),
            &mut collector(&mut events),
        )
        .expect("final message applies");
        // Non-streaming fallback: the message IS surfaced as one token.
        assert!(matches!(events[0], RuntimeEvent::Token(ref t) if t == "full answer"));
        assert_eq!(result.message, "full answer");
    }

    #[test]
    fn tool_calls_come_only_from_the_final_message() {
        let mut events = Vec::new();
        let mut result = TurnResult::default();
        // A mid-stream fragment that happens to parse as a complete object
        // must NOT produce a tool call (the final tool_requests are the only
        // authoritative source — otherwise short-arg calls would run twice).
        let delta = event(
            "assistant.tool_call_delta",
            serde_json::json!({
                "inputDelta": r#"{"path":"a"}"#,
                "toolCallId": "t1",
                "toolName": "find_glob",
            }),
        );
        apply_event(
            &delta,
            &mut result,
            &HashMap::new(),
            &mut collector(&mut events),
        )
        .expect("delta applies");
        let final_msg = event(
            "assistant.message",
            serde_json::json!({
                "content": "",
                "messageId": "m2",
                "toolRequests": [
                    { "toolCallId": "t1", "name": "find_glob", "arguments": { "path": "a" } }
                ],
            }),
        );
        apply_event(
            &final_msg,
            &mut result,
            &HashMap::new(),
            &mut collector(&mut events),
        )
        .expect("final message applies");
        let calls: Vec<_> = events
            .iter()
            .filter(|ev| matches!(ev, RuntimeEvent::ToolCall(_)))
            .collect();
        assert_eq!(calls.len(), 1, "exactly one ToolCall event");
    }

    #[test]
    fn tool_call_names_translate_back_to_original() {
        let mut events = Vec::new();
        let mut result = TurnResult::default();
        let mut name_map = HashMap::new();
        name_map.insert("mcp_server_tool".to_string(), "mcp.server/tool".to_string());
        let final_msg = event(
            "assistant.message",
            serde_json::json!({
                "content": "",
                "messageId": "m3",
                "toolRequests": [
                    { "toolCallId": "t9", "name": "mcp_server_tool", "arguments": {} }
                ],
            }),
        );
        apply_event(
            &final_msg,
            &mut result,
            &name_map,
            &mut collector(&mut events),
        )
        .expect("applies");
        match &events[0] {
            RuntimeEvent::ToolCall(tc) => assert_eq!(tc.name, "mcp.server/tool"),
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[test]
    fn usage_sums_across_model_calls() {
        let mut events = Vec::new();
        let mut result = TurnResult::default();
        for (input, output, cost) in [(100i64, 20i64, 0.001f64), (50, 10, 0.002)] {
            let ev = event(
                "assistant.usage",
                serde_json::json!({
                    "inputTokens": input,
                    "outputTokens": output,
                    "cost": cost,
                    "finishReason": "stop",
                }),
            );
            apply_event(
                &ev,
                &mut result,
                &HashMap::new(),
                &mut collector(&mut events),
            )
            .expect("usage applies");
        }
        let usage = result.usage.expect("usage accumulated");
        assert_eq!(usage["inputTokens"], 150);
        assert_eq!(usage["outputTokens"], 30);
        assert_eq!(usage["cost"], 0.003);
    }

    #[test]
    fn unrelated_events_classify_to_none() {
        let idle = event("session.idle", serde_json::json!({ "title": "t" }));
        assert!(classify(&idle).is_none());
        let exec = event(
            "tool.execution_start",
            serde_json::json!({ "toolCallId": "t1" }),
        );
        assert!(classify(&exec).is_none());
    }
}
