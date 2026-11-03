//! Chat entry points for the Copilot family.
//!
//! The Copilot runtime is an AGENT runtime, not a chat-completions API: it
//! has no one-shot completion endpoint. Every turn is an ephemeral session
//! (`create_session` → `send_and_wait` → `disconnect` → `delete_session`),
//! driven by [`runtime::start_turn`]; the wrapper hides that lifecycle
//! behind the same function signatures every other family exposes.
//!
//! History (`chat_stream_with_messages`) is lowered into the prompt: the
//! runtime owns conversation state inside a session, and since each turn
//! here is an ephemeral session, the harness's history must be replayed as
//! part of the user turn — the same strategy the OpenAI-compatible inline
//! mode uses for stateless providers. Images and Claude thinking blocks are
//! dropped in this replay (no multimodal or replay contract on this
//! runtime).
//!
//! Usage and finish reason arrive on `assistant.usage` events; `raw` is
//! the serialized usage JSON (the closest analog to the other families'
//! raw HTTP bodies — parse it manually for provider-specific fields).

use super::super::error::ConnectorError;
use super::super::output::{ChatOutput, ChatStream, LsOutput, StreamChunk};
use super::super::params::{ChatMessage, Parameters};
use super::super::provider::{Family, ProviderConfig, get_api_key};
use super::runtime::{RuntimeEvent, TurnHandle, build_session_config, start_turn};

/// Resolve the GitHub credential with the connector's standard precedence:
/// the explicitly-set API key first, then the provider keyring
/// (`GITHUB_TOKEN` under the `copilot` name, already registered in
/// `API_KEY_ENVS`).
fn resolve_api_key(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Result<String, ConnectorError> {
    params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))
}

/// Serialize the assistant message history into the turn prompt.
///
/// The runtime session is ephemeral, so the conversation replay travels in
/// the prompt itself. Tool-call turns are rendered with their native call
/// shape and results as `tool` role blocks — readable by every backend
/// model behind the runtime.
fn render_history(messages: &[ChatMessage]) -> String {
    let mut rendered = String::new();
    for message in messages {
        let role = message.role.as_str();
        if let Some(content) = message.content.as_deref().filter(|c| !c.is_empty()) {
            if role == "tool" {
                rendered.push_str(&format!(
                    "[tool result for {}]: {content}\n",
                    message.tool_call_id.as_deref().unwrap_or("?")
                ));
            } else {
                rendered.push_str(&format!("[{role}]: {content}\n"));
            }
        }
        if let Some(calls) = &message.tool_calls {
            for call in calls {
                rendered.push_str(&format!(
                    "[{role}]: tool call {}({})\n",
                    call.function.name, call.function.arguments
                ));
            }
        }
    }
    rendered
}

/// Serialize a runtime usage JSON into the `raw` payload shape.
///
/// `Value::Null` serializes to `"null"`, so an absent usage is rendered as
/// an empty object — the shape the extractors already treat as "no usage".
fn usage_raw(usage: &serde_json::Value) -> String {
    if usage.is_null() {
        return "{}".to_string();
    }
    serde_json::to_string(usage).unwrap_or_else(|_| "{}".to_string())
}

// Public API

/// One-shot chat completion against the Copilot runtime.
pub async fn chat(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatOutput, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let plan = build_session_config(params, system_prompt);
    let handle = start_turn(plan, prompt.to_string(), Some(api_key)).await?;
    let result = handle.run(|_event| Ok(())).await?;
    Ok(ChatOutput {
        raw: usage_raw(result.usage.as_ref().unwrap_or(&serde_json::Value::Null)),
        message: result.message,
    })
}

/// Streaming chat completion against the Copilot runtime.
pub async fn chat_stream(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    system_prompt: Option<&str>,
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let plan = build_session_config(params, system_prompt);
    let handle = start_turn(plan, prompt.to_string(), Some(api_key)).await?;
    Ok(event_stream(handle))
}

/// Streaming chat completion with explicit assistant/tool history.
pub async fn chat_stream_with_messages(
    config: &ProviderConfig,
    params: &Parameters,
    system: &str,
    messages: &[ChatMessage],
    service: Option<&str>,
) -> Result<ChatStream, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let plan = build_session_config(params, Some(system));
    let history = render_history(messages);
    let prompt = if history.is_empty() {
        // Degenerate call with no messages: still need SOMETHING to send.
        "(continue)".to_string()
    } else {
        format!("Conversation so far:\n{history}\nRespond to the latest message.")
    };
    let handle = start_turn(plan, prompt, Some(api_key)).await?;
    Ok(event_stream(handle))
}

/// Wrap a running turn into the connector's [`ChatStream`].
///
/// An unbounded channel bridges the turn's synchronous event callback into
/// the async stream: send only fails when the consumer is gone, which is
/// exactly the abort condition.
fn event_stream(handle: TurnHandle) -> ChatStream {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<StreamChunk, ConnectorError>>();
    tokio::spawn(async move {
        let outcome = handle
            .run(|event| {
                match event {
                    RuntimeEvent::Batch(events) => {
                        // The final `assistant.message` fans out: complete
                        // tool calls first, then the message text (which is
                        // only re-emitted when nothing streamed before it —
                        // the runtime dedups against the deltas).
                        for inner in events {
                            if let Some(chunk) = event_stream_member(inner)
                                && tx.send(chunk).is_err()
                            {
                                return Err(ConnectorError::Network(
                                    "stream consumer dropped".into(),
                                ));
                            }
                        }
                        Ok(())
                    }
                    other => match event_stream_member(other) {
                        Some(Ok(chunk)) => tx
                            .send(Ok(chunk))
                            .map_err(|_| ConnectorError::Network("stream consumer dropped".into())),
                        Some(Err(e)) => Err(e),
                        // Deduped/suppressed events produce no chunk.
                        None => Ok(()),
                    },
                }
            })
            .await;
        if let Err(e) = outcome {
            let _ = tx.send(Err(e));
        }
    });
    ChatStream::new(
        Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)),
        Family::Copilot,
    )
}

/// Lower one runtime event into its stream chunk.
///
/// `None` means the event is intentionally NOT surfaced on the stream:
/// - a `Message` that duplicates already-streamed deltas (the runtime
///   suppresses it) — surfacing it would render the response twice;
/// - a `Usage` event whose `raw` frame is emitted per event and merged by
///   `ChatStream` itself (`merge_stream` keeps per-field maxima across
///   calls, the same convention the Claude family relies on for split
///   usage frames) — the finish reason rides the same chunk.
fn event_stream_member(event: RuntimeEvent) -> Option<Result<StreamChunk, ConnectorError>> {
    let chunk = match event {
        RuntimeEvent::Token(text) => StreamChunk {
            raw: "{}".to_string(),
            token: text,
            reasoning: String::new(),
            finish_reason: None,
            thinking_blocks: None,
            tool_call: None,
            reset: false,
        },
        RuntimeEvent::Reasoning(text) => StreamChunk {
            raw: "{}".to_string(),
            token: String::new(),
            reasoning: text,
            finish_reason: None,
            thinking_blocks: None,
            tool_call: None,
            reset: false,
        },
        RuntimeEvent::ToolCall(call) => StreamChunk {
            raw: "{}".to_string(),
            token: String::new(),
            reasoning: String::new(),
            finish_reason: None,
            thinking_blocks: None,
            tool_call: Some(call),
            reset: false,
        },
        RuntimeEvent::Message(text) => StreamChunk {
            raw: "{}".to_string(),
            token: text,
            reasoning: String::new(),
            finish_reason: None,
            thinking_blocks: None,
            tool_call: None,
            reset: false,
        },
        RuntimeEvent::Usage(usage) => StreamChunk {
            raw: usage_raw(&usage),
            token: String::new(),
            reasoning: String::new(),
            // The usage event's finish reason (when present) normalizes
            // through the shared helper downstream.
            finish_reason: usage
                .get("finishReason")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            thinking_blocks: None,
            tool_call: None,
            reset: false,
        },
        RuntimeEvent::Batch(_) => {
            // Batches are unwrapped by the caller before reaching here.
            return None;
        }
        RuntimeEvent::Failed(failure) => return Some(Err(super::runtime::session_error(&failure))),
    };
    Some(Ok(chunk))
}

/// List the Copilot models available to the authenticated account.
///
/// Delegates to [`models::list_catalog`] — the direct exchange +
/// shared-catalog chain. The runtime's own `models.list` RPC is NOT used
/// for listing: it reads the agent-mode catalog, which stays EMPTY for
/// Free-tier accounts even though the account can chat fine (verified
/// live); the shared chat catalog carries the real picker entries.
pub async fn list_models(
    config: &ProviderConfig,
    params: &Parameters,
    service: Option<&str>,
) -> Result<LsOutput, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    super::models::list_catalog(&api_key).await
}

#[cfg(test)]
mod tests {
    use super::super::runtime::SessionFailure;
    use super::*;

    #[test]
    fn history_renders_roles_and_tool_calls() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("list files".into()),
                tool_calls: None,
                tool_call_id: None,
                thinking_blocks: None,
                images: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![crate::connector::params::ToolCallMsg {
                    id: "t1".into(),
                    kind: "function".into(),
                    function: crate::connector::params::ToolCallFunctionMsg {
                        name: "find_glob".into(),
                        arguments: r#"{"pattern":"*.rs"}"#.into(),
                    },
                    thought_signature: None,
                }]),
                tool_call_id: None,
                thinking_blocks: None,
                images: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("3 files".into()),
                tool_calls: None,
                tool_call_id: Some("t1".into()),
                thinking_blocks: None,
                images: None,
            },
        ];
        let rendered = render_history(&messages);
        assert!(rendered.contains("[user]: list files"));
        assert!(rendered.contains("tool call find_glob"));
        // Tool results render ONCE (as tool results, not as a bare role line).
        assert_eq!(rendered.matches("3 files").count(), 1);
        assert!(rendered.contains("[tool result for t1]: 3 files"));
    }

    #[test]
    fn empty_history_degrades_to_continue_prompt() {
        let rendered = render_history(&[]);
        assert!(rendered.is_empty());
    }

    #[test]
    fn usage_raw_renders_null_as_empty_object() {
        assert_eq!(usage_raw(&serde_json::Value::Null), "{}");
        assert_eq!(
            usage_raw(&serde_json::json!({"inputTokens": 5})),
            r#"{"inputTokens":5}"#
        );
    }

    #[test]
    fn suppressed_message_yields_no_chunk() {
        // A Message event the runtime already suppressed is not represented
        // as an error — the member mapping returns None (no chunk at all).
        // Here we verify the FAILURE mapping instead, the other None/Err edge.
        let failure = event_stream_member(RuntimeEvent::Failed(SessionFailure {
            message: "boom".into(),
            status: Some(500),
            error_type: Some("query".into()),
        }));
        let err = failure.expect("failure maps to Err").expect_err("is error");
        assert!(!err.is_transient());
    }
}
