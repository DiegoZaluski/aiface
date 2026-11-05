//! The System One LLM wrapper — decisions from a GENERATIVE model under the
//! same contract the native decision APIs answer.
//!
//! This is the path TypeSafe's own evals use ("System One LLM wrapper"):
//! the declared questions are compiled into a schema prompt, the LLM's JSON
//! self-report is parsed, and the result flows through the SAME validation
//! funnel as a native decision response. The output is
//! [`DecisionOutput`] with [`ProbabilitySource::DerivedFromLlm`] — the
//! caller can always tell the two backend classes apart.
//!
//! Reuses `Connector::chat_with_system` verbatim (retry, auth, response
//! format, error classification are all inherited) — no new wire code.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{
    DecisionAnswer, DecisionOutput, DecisionQuestion, Probabilities, ProbabilitySource,
    validate_answers,
};
use crate::connector::client::Connector;
use crate::connector::error::ConnectorError;
use crate::connector::params::ResponseFormat;
use crate::connector::usage::TokenUsage;

/// The prompt contract's preamble: the model must answer as a decisions
/// engine, never as a chat partner.
const DECISIONS_PREAMBLE: &str = "\
You are a decision engine. You answer ONLY with a JSON object mapping the \
requested question keys to typed answers — never prose, never explanations. \
Rules:
- Every requested key MUST appear in the answer object, with the exact type asked.
- Probabilities are finite numbers in [0, 1]; a distribution's values must \
sum to (approximately) 1.0 and cover EVERY allowed option.
- For a `choice` answer, the winner MUST be one of the declared option keys — \
never invent options, never merge or rename them.
- For a `noul` answer, emit the single probability that the statement is TRUE.
- For a `score` answer, emit a probability for EVERY level index (`\"0\"` \
through the highest declared index), not prose labels.
- Be epistemically honest: when the state does not determine an answer, put \
the probability mass near the middle instead of a confident guess.";

/// Compile the declared questions into the schema section of the system
/// prompt — the exact JSON shape the model must emit, derived from the
/// caller's declaration so prompt and validator can never drift apart.
#[must_use]
pub(crate) fn schema_prompt(questions: &BTreeMap<String, DecisionQuestion>) -> String {
    let mut schema = String::from(DECISIONS_PREAMBLE);
    schema.push_str("\n\nAnswer exactly this JSON shape:\n{\n  \"answers\": {\n");
    for (name, question) in questions {
        match question {
            DecisionQuestion::Noul { .. } => {
                schema.push_str(&format!(
                    "    \"{name}\": {{ \"type\": \"noul\", \"probability\": <0..1> }},\n"
                ));
            }
            DecisionQuestion::Choice { criteria, .. } => {
                let options: Vec<String> =
                    criteria.iter().map(|(k, _)| format!("\"{k}\"")).collect();
                schema.push_str(&format!(
                    "    \"{name}\": {{ \"type\": \"choice\", \"choice\": <one of {}>, \
                     \"probabilities\": {{ {} }}, \"confidence\": <0..1> }},\n",
                    options.join(", "),
                    options
                        .iter()
                        .map(|o| format!("{o}: <0..1>"))
                        .collect::<Vec<_>>()
                        .join(", "),
                ));
            }
            DecisionQuestion::Score { levels, .. } => {
                let max = levels.len().saturating_sub(1);
                schema.push_str(&format!(
                    "    \"{name}\": {{ \"type\": \"score\", \"probabilities\": \
                     {{ \"0\": <0..1>, ..., \"{max}\": <0..1> }}, \"confidence\": <0..1> }},\n"
                ));
            }
        }
    }
    schema.push_str("  }\n}\n");
    schema
}

/// Build the schema system prompt for a decision call.
#[must_use]
pub(crate) fn system_prompt(questions: &BTreeMap<String, DecisionQuestion>) -> String {
    format!("{}\n\nDecide about the `state` provided in the user message.", schema_prompt(questions))
}

/// Serialize the state into the user prompt — JSON objects travel as their
/// canonical JSON rendering, everything else as its string form.
#[must_use]
pub(crate) fn state_prompt(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

/// Run one decision request through a GENERATIVE backend.
///
/// The questions are compiled into a schema prompt, the request rides the
/// standard chat path (`chat_with_system` with JSON-object response
/// format), the reply is parsed and validated against the declared
/// questions by the same funnel the native System One caller uses.
///
/// The returned [`DecisionOutput`] carries
/// [`ProbabilitySource::DerivedFromLlm`]: the probabilities are the model's
/// SELF-REPORTED numbers, not calibrated outputs.
///
/// # Errors
///
/// Everything `chat_with_system` can return, plus the decision-contract
/// errors from [`validate_answers`] when the model's JSON breaks the
/// declared schema (hallucinated option, missing question, ...).
pub(crate) async fn decide(
    connector: &Connector,
    state: Value,
    questions: &BTreeMap<String, DecisionQuestion>,
) -> Result<DecisionOutput, ConnectorError> {
    for (name, question) in questions {
        question.validate().map_err(|e| match e {
            ConnectorError::InvalidQuestion(detail) => {
                ConnectorError::InvalidQuestion(format!("{name}: {detail}"))
            }
            other => other,
        })?;
    }

    let system = system_prompt(questions);
    let prompt = state_prompt(&state);

    // The clone is the whole request configuration: strip tool plumbing
    // (a connector in an agent loop must not send tool schemas on a
    // decision call — the model might answer with a tool call instead of
    // the JSON contract) and force JSON-object output.
    let mut request_connector = connector.clone();
    request_connector.params.tools = None;
    request_connector.params.tool_choice = None;
    let output = request_connector
        .with_response_format(ResponseFormat::json_object())
        .chat_with_system(&prompt, &system)
        .await?;

    let parsed = extract_answers_json(output.message())?;
    let answers = validate_answers(questions, parsed)?;
    Ok(DecisionOutput::from_parts(
        answers,
        output.token_usage(),
        output.raw().to_string(),
        ProbabilitySource::DerivedFromLlm,
    ))
}

/// Pull the `answers` map out of the model's reply.
///
/// Tolerates the model wrapping the object in markdown fences or emitting
/// the map bare (without the `"answers"` key) — but the CONTENT must be the
/// declared JSON, never prose.
fn extract_answers_json(
    message: &str,
) -> Result<BTreeMap<String, DecisionAnswer>, ConnectorError> {
    let text = strip_fences(message);
    let value: Value = serde_json::from_str(text.trim()).map_err(|e| {
        ConnectorError::Deserialization(format!(
            "the model's decision reply is not JSON ({e}); raw: {message}"
        ))
    })?;
    let map = value
        .get("answers")
        .unwrap_or(&value)
        .as_object()
        .ok_or_else(|| {
            ConnectorError::Deserialization(format!(
                "the model's decision reply has no `answers` object; raw: {message}"
            ))
        })?;
    let mut parsed = BTreeMap::new();
    for (name, v) in map {
        parsed.insert(name.clone(), parse_llm_answer(name, v)?);
    }
    Ok(parsed)
}

/// Strip markdown code fences a model may wrap its JSON in.
fn strip_fences(message: &str) -> &str {
    let trimmed = message.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Skip the (optional) language tag line.
    let rest = rest.split_once('\n').map_or(rest, |(_, after)| after);
    rest.trim()
        .strip_suffix("```")
        .unwrap_or(rest)
        .trim()
}

/// Parse one answer object from the LLM's JSON.
///
/// Accepts BOTH spellings the native contract uses — `probability` /
/// `noul` for the binary answer, `choice` / `value` for the winner —
/// because the wrapper is the LLM's first exposure to the schema and
/// strictness here would reject models that are structurally correct.
fn parse_llm_answer(name: &str, value: &Value) -> Result<DecisionAnswer, ConnectorError> {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "noul" => {
            let p = value
                .get("probability")
                .or_else(|| value.get("noul"))
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    ConnectorError::Deserialization(format!(
                        "noul answer for `{name}` has no numeric probability; raw: {value}"
                    ))
                })?;
            Ok(DecisionAnswer::Noul(p))
        }
        "choice" | "" => {
            // An answer without an explicit `type` is treated as `choice`
            // when it carries a winner + distribution — the shape the
            // schema asks for.
            let top = value
                .get("choice")
                .or_else(|| value.get("value"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ConnectorError::Deserialization(format!(
                        "choice answer for `{name}` has no `choice` winner; raw: {value}"
                    ))
                })?
                .to_string();
            let per_label = probabilities(name, value)?;
            let confidence = value
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or_else(|| per_label.get(&top).copied().unwrap_or(0.0));
            Ok(DecisionAnswer::Choice(Probabilities {
                top,
                per_label,
                confidence,
            }))
        }
        "score" => {
            let per_label = probabilities(name, value)?;
            let top = per_label
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone())
                .ok_or_else(|| {
                    ConnectorError::Deserialization(format!(
                        "score answer for `{name}` carries an empty probability map"
                    ))
                })?;
            let confidence = value
                .get("confidence")
                .and_then(Value::as_f64)
                .unwrap_or_else(|| per_label.get(&top).copied().unwrap_or(0.0));
            Ok(DecisionAnswer::Score(Probabilities {
                top,
                per_label,
                confidence,
            }))
        }
        other => Err(ConnectorError::Deserialization(format!(
            "answer for `{name}` has unknown type `{other}`"
        ))),
    }
}

/// The `probabilities` map of a wrapper answer, tolerating numeric strings.
fn probabilities(
    name: &str,
    value: &Value,
) -> Result<BTreeMap<String, f64>, ConnectorError> {
    let map = value
        .get("probabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ConnectorError::Deserialization(format!(
                "answer for `{name}` has no `probabilities` object; raw: {value}"
            ))
        })?;
    let mut out = BTreeMap::new();
    for (key, v) in map {
        let p = v
            .as_f64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
            .ok_or_else(|| {
                ConnectorError::Deserialization(format!(
                    "probability for `{name}.{key}` is not a number: {v}"
                ))
            })?;
        out.insert(key.clone(), p);
    }
    Ok(out)
}

/// The token usage of a chat response, extracted through the connector's
/// family dispatcher.
trait ChatUsage {
    fn token_usage(&self) -> Option<TokenUsage>;
}

impl ChatUsage for crate::connector::output::ChatOutput {
    fn token_usage(&self) -> Option<TokenUsage> {
        // `Connector::token_usage` needs a connector; the raw response
        // carries the usage object in every family's native shape. Reuse
        // the shared extractor by parsing through the connector that made
        // the call — but `ChatOutput` does not carry the connector, so
        // parse the common shapes here.
        let value: Value = serde_json::from_str(self.raw()).ok()?;
        let usage = value.get("usage")?;
        let input = usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Some(TokenUsage {
            input_tokens: u32::try_from(input).unwrap_or(u32::MAX),
            output_tokens: u32::try_from(output).unwrap_or(u32::MAX),
            reported_cost: crate::connector::output::extract_reported_cost(self.raw()),
            ..TokenUsage::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn questions() -> BTreeMap<String, DecisionQuestion> {
        let mut q = BTreeMap::new();
        q.insert(
            "urgent".to_string(),
            DecisionQuestion::Noul {
                instructions: "is urgent".to_string(),
                criteria: None,
            },
        );
        q.insert(
            "team".to_string(),
            DecisionQuestion::Choice {
                instructions: "route".to_string(),
                criteria: vec![
                    ("billing".to_string(), "charges".to_string()),
                    ("technical".to_string(), "bugs".to_string()),
                ],
            },
        );
        q
    }

    #[test]
    fn schema_prompt_names_every_question_and_option() {
        let prompt = schema_prompt(&questions());
        assert!(prompt.contains("\"urgent\""));
        assert!(prompt.contains("\"team\""));
        assert!(prompt.contains("\"billing\""));
        assert!(prompt.contains("\"technical\""));
        assert!(prompt.contains("noul"));
        assert!(prompt.contains("choice"));
    }

    #[test]
    fn state_prompt_renders_objects_as_json() {
        assert_eq!(state_prompt(&Value::String("hello".into())), "hello");
        let obj = serde_json::json!({"a": 1});
        assert!(state_prompt(&obj).contains("\"a\""));
    }

    #[test]
    fn extract_answers_json_parses_fenced_and_bare_shapes() {
        let mut parsed = extract_answers_json(
            "```json\n{\"answers\": {\"urgent\": {\"type\": \"noul\", \"probability\": 0.9}}}\n```",
        )
        .unwrap();
        assert_eq!(parsed["urgent"].as_noul().unwrap(), 0.9);

        // Bare map without the `answers` wrapper.
        parsed = extract_answers_json(
            "{\"urgent\": {\"type\": \"noul\", \"probability\": 0.5}}",
        )
        .unwrap();
        assert_eq!(parsed["urgent"].as_noul().unwrap(), 0.5);

        // Prose instead of JSON is a clear error, never a silent fallback.
        assert!(extract_answers_json("I think this is urgent!").is_err());
    }

    #[test]
    fn parse_llm_answer_accepts_both_spellings() {
        let a = parse_llm_answer(
            "team",
            &serde_json::json!({
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.8, "technical": 0.2},
                "confidence": 0.8
            }),
        )
        .unwrap();
        assert_eq!(a.as_choice().unwrap().top, "billing");

        // `value` as the winner spelling, no explicit type.
        let b = parse_llm_answer(
            "team",
            &serde_json::json!({
                "value": "technical",
                "probabilities": {"billing": "0.3", "technical": "0.7"}
            }),
        )
        .unwrap();
        assert_eq!(b.as_choice().unwrap().top, "technical");
        assert_eq!(b.as_choice().unwrap().per_label["billing"], 0.3);

        // Stringly-typed probabilities parse.
        let c = parse_llm_answer(
            "sev",
            &serde_json::json!({
                "type": "score",
                "probabilities": {"0": 0.1, "1": 0.9}
            }),
        )
        .unwrap();
        assert_eq!(c.as_score().unwrap().top, "1");
    }

    #[test]
    fn wrapper_answers_hit_the_same_validation_funnel() {
        // A hallucinated option in the LLM's JSON is rejected by the same
        // validator the native path uses.
        let mut q = BTreeMap::new();
        q.insert(
            "team".to_string(),
            DecisionQuestion::Choice {
                instructions: "route".to_string(),
                criteria: vec![
                    ("billing".to_string(), "charges".to_string()),
                    ("technical".to_string(), "bugs".to_string()),
                ],
            },
        );
        let parsed = extract_answers_json(
            "{\"answers\": {\"team\": {\"type\": \"choice\", \"choice\": \"sales\", \
             \"probabilities\": {\"billing\": 0.5, \"technical\": 0.5, \"sales\": 0.0}}}}",
        )
        .unwrap();
        assert!(validate_answers(&q, parsed).is_err());
    }
}
