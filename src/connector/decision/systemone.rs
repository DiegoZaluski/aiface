//! Native System One decisions caller — the wire contract shared by
//! TypeSafe's `/v1/systemone` (Jev), Microsoft Foundry's
//! `/providers/microsoft/v1/systemone` (Microsoft-Decision-1) and
//! OpenRouter's System One route.
//!
//! One HTTP POST: `state` plus typed `questions` in, an `answers` map of
//! probability distributions out. The answers are parsed here and validated
//! against the caller's declared questions by the shared funnel
//! ([`super::validate_answers`]) — the same contract the LLM wrapper is
//! held to.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{
    DecisionAnswer, DecisionOutput, DecisionQuestion, Probabilities, ProbabilitySource,
    validate_answers,
};
use crate::connector::error::ConnectorError;
use crate::connector::output::extract_reported_cost;
use crate::connector::provider::ProviderConfig;
use crate::connector::provider::get_api_key;
use crate::connector::usage::TokenUsage;

/// The request body exactly as the System One wire contract defines it.
#[derive(Debug, serde::Serialize)]
struct SystemOneRequest {
    model: String,
    state: Value,
    questions: BTreeMap<String, SystemOneQuestion>,
}

/// One typed question on the wire. `criteria` is the type-dependent field:
/// an option map for `choice`, an ordered level array for `score`, and the
/// optional true/false anchors for `noul`.
#[derive(Debug, serde::Serialize)]
struct SystemOneQuestion {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    criteria: Option<Value>,
}

/// Lower a declared question onto the wire shape.
fn wire_question(question: &DecisionQuestion) -> SystemOneQuestion {
    match question {
        DecisionQuestion::Noul {
            instructions,
            criteria,
        } => SystemOneQuestion {
            kind: "noul",
            instructions: instructions.clone(),
            criteria: criteria.as_ref().map(|(t, f)| {
                json!({
                    "true": t,
                    "false": f,
                })
            }),
        },
        DecisionQuestion::Choice {
            instructions,
            criteria,
        } => SystemOneQuestion {
            kind: "choice",
            instructions: instructions.clone(),
            criteria: Some(json!(criteria
                .iter()
                .map(|(k, d)| (k.clone(), Value::from(d.clone())))
                .collect::<BTreeMap<_, _>>())),
        },
        DecisionQuestion::Score { instructions, levels } => SystemOneQuestion {
            kind: "score",
            instructions: instructions.clone(),
            criteria: Some(json!(levels)),
        },
    }
}

/// Parse one answer object from the `answers` map into a
/// [`DecisionAnswer`], before validation.
///
/// The wire shapes (per the System One contract):
/// - `noul`: `{ "type": "noul", "noul": <p> }` — some deployments spell the
///   probability field `probability`; both are accepted.
/// - `choice`: `{ "type": "choice", "choice": <winner>, "probabilities":
///   {…}, "confidence": <c> }`.
/// - `score`: `{ "type": "score", "score": <weighted>, "probabilities":
///   {"0": …}, "confidence": <c> }` — the winner is derived as the ARGMAX
///   level index (the weighted `score` field may fall between levels and is
///   not a label).
fn parse_answer(name: &str, value: &Value) -> Result<DecisionAnswer, ConnectorError> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| ConnectorError::Deserialization(format!(
            "answer for `{name}` has no `type` field; raw: {value}"
        )))?;
    match kind {
        "noul" => {
            let p = value
                .get("noul")
                .or_else(|| value.get("probability"))
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    ConnectorError::Deserialization(format!(
                        "noul answer for `{name}` has no numeric probability; raw: {value}"
                    ))
                })?;
            Ok(DecisionAnswer::Noul(p))
        }
        "choice" => {
            let top = value
                .get("choice")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ConnectorError::Deserialization(format!(
                        "choice answer for `{name}` has no `choice` winner; raw: {value}"
                    ))
                })?
                .to_string();
            let per_label = probability_map(name, value)?;
            let confidence = value
                .get("confidence")
                .and_then(Value::as_f64)
                // The contract always sends a confidence for choice answers;
                // fall back to the winner's own probability when a gateway
                // omits it (the two coincide by definition).
                .unwrap_or_else(|| per_label.get(&top).copied().unwrap_or(0.0));
            Ok(DecisionAnswer::Choice(Probabilities {
                top,
                per_label,
                confidence,
            }))
        }
        "score" => {
            let per_label = probability_map(name, value)?;
            // The winner is the ARGMAX level index — the contract's `score`
            // field is the weighted average and may fall BETWEEN levels.
            let top = per_label
                .iter()
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
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

/// The `probabilities` map of a choice/score answer, keys as sent.
fn probability_map(
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
            // Some gateways type the values as numeric strings.
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

/// Send one System One decisions request and return the validated,
/// normalized output.
///
/// `state` may be a JSON string, object or array (the contract accepts all
/// three); `questions` are the caller's declarations; `service` is the
/// keyring service for API-key resolution (see
/// [`Connector::with_service_keyring`](crate::connector::Connector::with_service_keyring)).
///
/// # Errors
///
/// [`ConnectorError::MissingApiKey`], `HttpError` (with the retry
/// middleware's classification), `Deserialization` on a malformed body,
/// and the decision-contract errors ([`ConnectorError::InvalidQuestion`],
/// [`ConnectorError::DecisionSchemaViolation`],
/// [`ConnectorError::MissingDecisionAnswer`]) when the answers break the
/// declared schema.
pub async fn decide(
    config: &'static ProviderConfig,
    params: &crate::connector::params::Parameters,
    state: Value,
    questions: &BTreeMap<String, DecisionQuestion>,
    service: Option<&str>,
) -> Result<DecisionOutput, ConnectorError> {
    // Local contract check BEFORE the request: a question outside the
    // System One bounds never reaches the wire.
    for (name, question) in questions {
        question.validate().map_err(|e| match e {
            ConnectorError::InvalidQuestion(detail) => ConnectorError::InvalidQuestion(format!(
                "{name}: {detail}"
            )),
            other => other,
        })?;
    }

    let model = params
        .model
        .clone()
        .unwrap_or_else(|| {
            config
                .decisions_default_model
                .unwrap_or(config.default_model)
                .to_string()
        });
    let request = SystemOneRequest {
        model,
        state,
        questions: questions
            .iter()
            .map(|(k, q)| (k.clone(), wire_question(q)))
            .collect(),
    };

    let api_key = resolve_api_key(config, params, service)?;
    // A caller-provided base_url often carries a trailing slash (copied
    // from a browser) — strip it so `{base}{path}` never doubles up.
    let base_url = params
        .base_url
        .as_deref()
        .unwrap_or(config.base_url)
        .trim_end_matches('/');
    let path = config
        .decisions_path
        .ok_or(ConnectorError::NotImplemented("decisions"))?;
    let url = format!("{base_url}{path}");

    // Auth dialects: TypeSafe and OpenRouter are `Authorization: Bearer`;
    // Microsoft Foundry authenticates the resource with the `api-key`
    // header (Entra-ID callers pass the bearer token via `with_api_key`).
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(key) = &api_key {
        if config.name == "azure-decision" {
            headers.push(("api-key", key.clone()));
        } else {
            headers.push(("Authorization", format!("Bearer {key}")));
        }
    }
    // OpenRouter's identification extras ride via `apply_provider_headers`
    // (shared with the chat path) — not duplicated here.

    let raw =
        crate::connector::common::send_request_retrying(
            config,
            &url,
            &request,
            &headers,
            params.retry_enabled,
            params.max_retries,
            params.retry_delay_override,
        )
        .await?;

    let body: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            return Err(ConnectorError::Deserialization(format!(
                "{e}. Raw response: {raw}"
            )));
        }
    };

    let answers_json = body
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ConnectorError::Deserialization(format!(
                "System One response has no `answers` object; raw: {raw}"
            ))
        })?;

    let mut parsed = BTreeMap::new();
    for (name, value) in answers_json {
        parsed.insert(name.clone(), parse_answer(name, value)?);
    }

    let answers = validate_answers(questions, parsed)?;

    let usage = extract_usage(&body, &raw);
    Ok(DecisionOutput::from_parts(
        answers,
        usage,
        raw,
        ProbabilitySource::Native,
    ))
}

/// Normalize the System One `usage` object onto [`TokenUsage`].
///
/// The contract's fields are `input_tokens` / `output_tokens` (OpenAI-style
/// names); OpenRouter additionally reports the REAL billed amount as
/// `usage.cost`, which the shared cost extractor already recognizes.
fn extract_usage(body: &Value, raw: &str) -> Option<TokenUsage> {
    let usage = body.get("usage")?;
    let input = usage
        .get("input_tokens")
        .or_else(|| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("output_tokens")
        .or_else(|| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Some(TokenUsage {
        input_tokens: u32::try_from(input).unwrap_or(u32::MAX),
        output_tokens: u32::try_from(output).unwrap_or(u32::MAX),
        reported_cost: extract_reported_cost(raw),
        ..TokenUsage::default()
    })
}

/// Resolve the API key for a decisions request (explicit key, then
/// keyring/environment) — same precedence as the chat callers.
fn resolve_api_key(
    config: &ProviderConfig,
    params: &crate::connector::params::Parameters,
    service: Option<&str>,
) -> Result<Option<String>, ConnectorError> {
    let key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service));
    if key.is_none() {
        return Err(ConnectorError::MissingApiKey(config.name.to_string()));
    }
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_question_lowering_matches_the_contract() {
        let req = wire_question(&DecisionQuestion::Noul {
            instructions: "is urgent?".to_string(),
            criteria: None,
        });
        assert_eq!(req.kind, "noul");
        assert!(req.criteria.is_none());

        let choice = wire_question(&DecisionQuestion::Choice {
            instructions: "route".to_string(),
            criteria: vec![
                ("billing".to_string(), "charges".to_string()),
                ("technical".to_string(), "bugs".to_string()),
            ],
        });
        let criteria = choice.criteria.unwrap();
        assert_eq!(criteria["billing"], "charges");
        assert_eq!(criteria["technical"], "bugs");

        let score = wire_question(&DecisionQuestion::Score {
            instructions: "severity".to_string(),
            levels: vec!["low".to_string(), "high".to_string()],
        });
        assert_eq!(score.kind, "score");
        assert_eq!(score.criteria.unwrap(), json!(["low", "high"]));

        let noul_anchored = wire_question(&DecisionQuestion::Noul {
            instructions: "refund".to_string(),
            criteria: Some(("commits money".to_string(), "no commitment".to_string())),
        });
        let criteria = noul_anchored.criteria.unwrap();
        assert_eq!(criteria["true"], "commits money");
        assert_eq!(criteria["false"], "no commitment");
    }

    #[test]
    fn parses_every_wire_answer_shape() {
        // The TypeSafe direct shape.
        let noul = parse_answer(
            "urgent",
            &json!({"type": "noul", "noul": 0.942}),
        )
        .unwrap();
        assert_eq!(noul.as_noul().unwrap(), 0.942);

        // The alternative `probability` field spelling.
        let noul_alt = parse_answer(
            "urgent",
            &json!({"type": "noul", "probability": 0.5}),
        )
        .unwrap();
        assert_eq!(noul_alt.as_noul().unwrap(), 0.5);

        let choice = parse_answer(
            "team",
            &json!({
                "type": "choice",
                "choice": "retention",
                "probabilities": {"billing": 0.048, "retention": 0.942, "support": 0.010},
                "confidence": 0.942
            }),
        )
        .unwrap();
        let p = choice.as_choice().unwrap();
        assert_eq!(p.top, "retention");
        assert_eq!(p.confidence, 0.942);

        let score = parse_answer(
            "severity",
            &json!({
                "type": "score",
                "score": 1.3,
                "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
                "confidence": 0.81
            }),
        )
        .unwrap();
        let p = score.as_score().unwrap();
        // The winner is the ARGMAX level index, not the weighted score.
        assert_eq!(p.top, "1");
        assert!((p.score_value() - 1.1).abs() < 1e-12);
    }

    #[test]
    fn malformed_answers_deserialize_with_clear_errors() {
        let err = parse_answer("x", &json!({"no_type": 1})).unwrap_err();
        assert!(err.to_string().contains("no `type`"));

        let err = parse_answer("x", &json!({"type": "noul"})).unwrap_err();
        assert!(err.to_string().contains("no numeric probability"));

        let err = parse_answer(
            "x",
            &json!({"type": "choice", "choice": "a", "probabilities": {"a": "high"}}),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not a number"));

        let err = parse_answer("x", &json!({"type": "box"})).unwrap_err();
        assert!(err.to_string().contains("unknown type"));
    }

    #[test]
    fn extracts_usage_in_both_field_dialects() {
        let body = json!({"usage": {"input_tokens": 82, "output_tokens": 18}});
        let usage = extract_usage(&body, body.to_string().as_str()).unwrap();
        assert_eq!(usage.input_tokens, 82);
        assert_eq!(usage.output_tokens, 18);

        let openrouter = json!({"usage": {"prompt_tokens": 275, "completion_tokens": 20, "cost": 0.00003}});
        let raw = openrouter.to_string();
        let usage = extract_usage(&openrouter, &raw).unwrap();
        assert_eq!(usage.input_tokens, 275);
        assert_eq!(usage.reported_cost, Some(0.00003));

        assert!(extract_usage(&json!({}), "{}").is_none());
    }

    #[test]
    fn decide_validates_questions_before_the_request() {
        // A question outside the contract bounds fails locally — no network.
        let mut questions = BTreeMap::new();
        questions.insert(
            "bad".to_string(),
            DecisionQuestion::Choice {
                instructions: "pick".to_string(),
                criteria: vec![("only".to_string(), "one".to_string())],
            },
        );
        let config = crate::connector::provider::get_provider("typesafe").unwrap();
        let params = crate::connector::params::Parameters::default();
        let err = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(decide(config, &params, json!("state"), &questions, None))
            .unwrap_err();
        assert!(matches!(err, ConnectorError::InvalidQuestion(_)));
    }
}
