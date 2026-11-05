//! Decision (System One) tests: the native decisions API path against a
//! mock server, the LLM wrapper path against an OpenAI-compatible mock,
//! and the shared schema-validation contract.

use super::common::mock_server;
use super::super::Connector;
use crate::connector::decision::{DecisionOutput, ProbabilitySource};
use std::sync::{Arc, Mutex};

/// A complete System One response in the TypeSafe wire shape.
const NATIVE_RESPONSE: &str = r#"{
  "model": "jev-latest",
  "answers": {
    "urgent": { "type": "noul", "noul": 0.94 },
    "team": {
      "type": "choice",
      "choice": "technical",
      "probabilities": { "billing": 0.1, "technical": 0.9 },
      "confidence": 0.9
    },
    "severity": {
      "type": "score",
      "score": 1.3,
      "probabilities": { "0": 0.2, "1": 0.5, "2": 0.3 },
      "confidence": 0.81
    }
  },
  "usage": { "input_tokens": 82, "output_tokens": 18 }
}"#;

/// A chat-completion wrapper around an LLM decision self-report — what the
/// LLM wrapper path reads out of `choices[0].message.content`.
const LLM_CHAT_RESPONSE: &str = r#"{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"answers\": {\"urgent\": {\"type\": \"noul\", \"probability\": 0.91}, \"team\": {\"type\": \"choice\", \"choice\": \"billing\", \"probabilities\": {\"billing\": 0.8, \"technical\": 0.2}, \"confidence\": 0.8}}}"
    }
  }],
  "usage": { "prompt_tokens": 300, "completion_tokens": 60 }
}"#;

fn typesafe_connector(port: u16) -> Connector {
    Connector::new("typesafe")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("ts-test")
        .with_retry(false)
}

fn wait_for_capture(captured: &Arc<Mutex<Option<String>>>) -> String {
    for _ in 0..100 {
        if let Some(body) = captured.lock().unwrap().as_ref() {
            return body.clone();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the mock server never captured a request body");
}

#[tokio::test]
async fn native_decide_round_trips_every_question_type() {
    let (port, captured_body, _, _handle) = mock_server(NATIVE_RESPONSE, 200);
    let output = typesafe_connector(port)
        .decide(serde_json::json!({"ticket": "The API returns 500 on every call."}))
        .noul("urgent", "Is this urgent?")
        .choice("team", "Which team?", [
            ("billing", "Charges and refunds"),
            ("technical", "Bugs and outages"),
        ])
        .score("severity", "How severe?", ["low", "medium", "high"])
        .send()
        .await
        .unwrap();

    // The request must carry the wire contract's three fields.
    let request: serde_json::Value =
        serde_json::from_str(&wait_for_capture(&captured_body)).unwrap();
    assert_eq!(request["model"], "jev-latest");
    assert!(request["state"]["ticket"]
        .as_str()
        .unwrap()
        .contains("500"));
    let questions = request["questions"].as_object().unwrap();
    assert_eq!(questions["urgent"]["type"], "noul");
    assert_eq!(questions["team"]["type"], "choice");
    assert_eq!(questions["team"]["criteria"]["billing"], "Charges and refunds");
    assert_eq!(questions["severity"]["type"], "score");
    assert_eq!(questions["severity"]["criteria"], serde_json::json!(["low", "medium", "high"]));

    // Native source, typed accessors.
    assert_eq!(output.source, ProbabilitySource::Native);
    assert!((output.noul("urgent").unwrap() - 0.94).abs() < 1e-12);
    let team = output.choice("team").unwrap();
    assert_eq!(team.top, "technical");
    assert!((team.per_label["billing"] - 0.1).abs() < 1e-12);
    let severity = output.score("severity").unwrap();
    // The score accessor's winner is the argmax level index.
    assert_eq!(severity.top, "1");
    assert!((severity.score_value() - 1.1).abs() < 1e-12);

    // Usage lands normalized.
    let usage = output.usage.unwrap();
    assert_eq!(usage.input_tokens, 82);
    assert_eq!(usage.output_tokens, 18);
}

#[tokio::test]
async fn native_decide_rejects_a_schema_violating_response() {
    // `team` answered as `noul` while declared as `choice` — a contract
    // break, even from a native backend.
    let body = r#"{
        "model": "jev-latest",
        "answers": {
            "team": { "type": "noul", "noul": 0.5 }
        }
    }"#;
    let (port, _, _, _handle) = mock_server(body, 200);
    let err = typesafe_connector(port)
        .decide("state")
        .choice("team", "Which team?", [
            ("billing", "Charges"),
            ("technical", "Bugs"),
        ])
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::connector::ConnectorError::DecisionSchemaViolation { .. }
        ),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn native_decide_rejects_a_hallucinated_option() {
    let body = r#"{
        "model": "jev-latest",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "sales",
                "probabilities": { "billing": 0.5, "technical": 0.5, "sales": 0.0 }
            }
        }
    }"#;
    let (port, _, _, _handle) = mock_server(body, 200);
    let err = typesafe_connector(port)
        .decide("state")
        .choice("team", "Which team?", [
            ("billing", "Charges"),
            ("technical", "Bugs"),
        ])
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::connector::ConnectorError::DecisionSchemaViolation { .. }
        ),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn azure_decision_uses_the_api_key_header_and_foundry_path() {
    let (port, captured_body, captured_raw, _handle) = mock_server(
        r#"{"model":"microsoft-decision-1","answers":{"in_scope":{"type":"noul","noul":0.97}}}"#,
        200,
    );
    let output = Connector::new("azure-decision")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("azure-key")
        .with_retry(false)
        .decide("The API returns 500 on every call.")
        .noul("in_scope", "Is this in scope?")
        .send()
        .await
        .unwrap();

    // Foundry authenticates the resource with the `api-key` header, and the
    // decision endpoint nests under /providers/microsoft/v1/systemone.
    let raw = captured_raw.lock().unwrap().clone().unwrap();
    assert!(
        raw.to_lowercase().contains("api-key: azure-key"),
        "expected the api-key header in: {raw}"
    );
    assert!(
        raw.contains("/providers/microsoft/v1/systemone"),
        "expected the Foundry decision path in: {raw}"
    );
    // The request `model` is the DEPLOYMENT name.
    let request: serde_json::Value =
        serde_json::from_str(&wait_for_capture(&captured_body)).unwrap();
    assert_eq!(request["model"], "microsoft-decision-1");
    assert!((output.noul("in_scope").unwrap() - 0.97).abs() < 1e-12);
}

#[tokio::test]
async fn llm_wrapper_produces_the_same_output_contract() {
    let (port, captured_body, _, _handle) = mock_server(LLM_CHAT_RESPONSE, 200);
    let output = Connector::new("groq")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .decide(serde_json::json!({"ticket": "I was charged twice."}))
        .noul("urgent", "Is this urgent?")
        .choice("team", "Which team?", [
            ("billing", "Charges and refunds"),
            ("technical", "Bugs and outages"),
        ])
        .send()
        .await
        .unwrap();

    // Same output contract, different source flag.
    assert_eq!(output.source, ProbabilitySource::DerivedFromLlm);
    assert!((output.noul("urgent").unwrap() - 0.91).abs() < 1e-12);
    let team = output.choice("team").unwrap();
    assert_eq!(team.top, "billing");
    assert!((team.per_label["technical"] - 0.2).abs() < 1e-12);

    // The request rode the standard chat path with the JSON response format
    // and the decisions schema in the system prompt. (The captured body is
    // JSON, so quotes inside the prompt text are escaped — assert on the
    // bare tokens.)
    let raw_request = wait_for_capture(&captured_body);
    assert!(raw_request.contains("response_format"));
    assert!(raw_request.contains("decision engine"));
    assert!(raw_request.contains("urgent"));
    assert!(raw_request.contains("billing"));
}

#[tokio::test]
async fn llm_wrapper_rejects_prose_instead_of_json() {
    let body = r#"{
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "I think this is urgent! The team should be billing."
            }
        }],
        "usage": {}
    }"#;
    let (port, _, _, _handle) = mock_server(body, 200);
    let err = Connector::new("groq")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .decide("state")
        .noul("urgent", "Is this urgent?")
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::connector::ConnectorError::Deserialization(_)),
        "got: {err:?}"
    );
}

#[tokio::test]
async fn decide_without_questions_fails_before_the_network() {
    let (port, _, _, _handle) = mock_server("{}", 200);
    let err = typesafe_connector(port)
        .decide("state")
        .send()
        .await
        .unwrap_err();
    assert!(matches!(err, crate::connector::ConnectorError::InvalidQuestion(_)));
}

#[tokio::test]
async fn decide_bounds_are_checked_before_the_network() {
    let (port, _, _, _handle) = mock_server("{}", 200);
    // One option is below the choice contract minimum.
    let err = typesafe_connector(port)
        .decide("state")
        .choice("team", "Which team?", [("billing", "Charges")])
        .send()
        .await
        .unwrap_err();
    assert!(matches!(err, crate::connector::ConnectorError::InvalidQuestion(_)));
}

#[tokio::test]
async fn chat_on_a_decision_provider_is_rejected() {
    let (port, _, _, _handle) = mock_server("{}", 200);
    let err = typesafe_connector(port).chat("hello").await.unwrap_err();
    assert!(matches!(err, crate::connector::ConnectorError::NotImplemented(_)));
}

#[tokio::test]
async fn into_model_materializes_the_declared_struct() {
    let (port, _, _, _handle) = mock_server(NATIVE_RESPONSE, 200);
    // Every question the fixture answers must be declared — the validator
    // rejects answers for questions the caller never declared.
    let output = typesafe_connector(port)
        .decide("The API returns 500 on every call.")
        .noul("urgent", "Is this urgent?")
        .choice("team", "Which team?", [
            ("billing", "Charges and refunds"),
            ("technical", "Bugs and outages"),
        ])
        .score("severity", "How severe?", ["low", "medium", "high"])
        .send()
        .await
        .unwrap();

    #[derive(serde::Deserialize)]
    struct Routing {
        urgent: f64,
        team: String,
        severity: f64,
    }
    let routing: Routing = output.into_model().unwrap();
    assert!((routing.urgent - 0.94).abs() < 1e-12);
    assert_eq!(routing.team, "technical");
    // The score materializes as the probability-weighted average (1.1 here).
    assert!((routing.severity - 1.1).abs() < 1e-12);
}

/// A missing usage object must not fail the call — usage stays `None`.
#[tokio::test]
async fn native_decide_without_usage_is_still_valid() {
    let (port, _, _, _handle) = mock_server(
        r#"{"model":"jev-latest","answers":{"x":{"type":"noul","noul":0.5}}}"#,
        200,
    );
    let output: DecisionOutput = typesafe_connector(port)
        .decide("state")
        .noul("x", "is true?")
        .send()
        .await
        .unwrap();
    assert!(output.usage.is_none());
    assert!((output.noul("x").unwrap() - 0.5).abs() < 1e-12);
}

/// Dispatch is capability-based: openrouter declares `decisions_path`, so
/// its `decide` rides the NATIVE System One route (chat keeps its own path)
/// — the docs' contract, verified on the wire here.
#[tokio::test]
async fn openrouter_decide_rides_the_native_systemone_route() {
    let (port, captured_body, captured_raw, _handle) = mock_server(
        r#"{"model":"jev-latest","answers":{"x":{"type":"noul","noul":0.8}}}"#,
        200,
    );
    let output = Connector::new("openrouter")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-or-test")
        .with_retry(false)
        .decide("state")
        .noul("x", "is true?")
        .send()
        .await
        .unwrap();

    // The native route: /api/v1/systemone, model = the System One model
    // (NOT a chat model), Bearer auth.
    let raw = captured_raw.lock().unwrap().clone().unwrap();
    assert!(
        raw.contains("/api/v1/systemone"),
        "expected the System One route in: {raw}"
    );
    let request: serde_json::Value =
        serde_json::from_str(&wait_for_capture(&captured_body)).unwrap();
    assert_eq!(request["model"], "jev-latest");
    assert!(
        raw.to_lowercase().contains("authorization: bearer"),
        "expected Bearer auth in: {raw}"
    );
    assert_eq!(output.source, ProbabilitySource::Native);
}

/// Declaring the same question name twice is a caller bug — rejected
/// locally instead of silently overwriting the first declaration.
#[tokio::test]
async fn duplicate_question_names_are_rejected_locally() {
    let (port, _, _, _handle) = mock_server("{}", 200);
    let err = typesafe_connector(port)
        .decide("state")
        .noul("x", "first declaration")
        .noul("x", "overwrites the first one")
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::connector::ConnectorError::InvalidQuestion(ref d) if d.contains("more than once")),
        "got: {err:?}"
    );
}

/// Duplicate option keys inside one `choice` declaration would make the
/// declared label set and the wire's criteria map disagree — rejected
/// locally.
#[tokio::test]
async fn duplicate_option_keys_are_rejected_locally() {
    let (port, _, _, _handle) = mock_server("{}", 200);
    let err = typesafe_connector(port)
        .decide("state")
        .choice("team", "Which team?", [
            ("billing", "Charges"),
            ("billing", "Charges again"),
        ])
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::connector::ConnectorError::InvalidQuestion(ref d) if d.contains("twice")),
        "got: {err:?}"
    );
}
