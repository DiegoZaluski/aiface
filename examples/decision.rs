//! Demonstrate the `decision` module: the System One contract
//! (`Connector::decide`) under BOTH backend classes — a native decision
//! model (TypeSafe Jev) and the LLM wrapper — plus the typed
//! materialization into user structs and the local contract checks.
//!
//! Run with:
//!
//! ```bash
//! cargo run --example decision
//! ```
//!
//! Live calls run only when a key resolves (`TYPESAFE_API_KEY`, or any
//! chat provider's key for the wrapper path); everything else is offline.

use aiface::connector::{
    Connector, ConnectorError, DecisionOutput, DecisionRequest, ProbabilitySource,
    known_providers_with_env,
};
use std::collections::BTreeMap;

/// The routing decision the example declares — the SAME questions for both
/// backends, so the outputs are directly comparable.
fn declare(connector: &Connector, state: serde_json::Value) -> DecisionRequest<'_> {
    connector
        .decide(state)
        .noul("urgent", "Is this request urgent?")
        .choice("team", "Which team should handle this request?", [
            ("billing", "Charges, invoices and refunds"),
            ("technical", "Bugs, errors and outages"),
            ("support", "How-to and account questions"),
        ])
        .score("severity", "How severe is this issue?", [
            "low", "medium", "high",
        ])
}

/// The typed shape the answers materialize into — field names are the
/// question names, field types follow the declared question kinds.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)] // the fields are the demonstration: they line up with the declared questions
struct Routing {
    urgent: f64,
    team: String,
    severity: f64,
}

fn show(output: &DecisionOutput) {
    println!("  source: {:?}", output.source);
    println!("  urgent  (noul):   {:.3}", output.noul("urgent").unwrap());
    let team = output.choice("team").unwrap();
    println!("  team    (choice): {} — {team:#?}", team.top);
    let severity = output.score("severity").unwrap();
    println!(
        "  severity (score): {:.2} (winner level {}) — confidence {:.2}",
        severity.score_value(),
        severity.top,
        severity.confidence
    );
    if let Some(usage) = &output.usage {
        println!(
            "  usage: {} in / {} out tokens{}",
            usage.input_tokens,
            usage.output_tokens,
            usage
                .reported_cost
                .map(|c| format!(", ${c:.6}"))
                .unwrap_or_default()
        );
    }
    let routing: Routing = output.into_model().unwrap();
    println!("  into_model: {routing:?}");
}

#[tokio::main]
async fn main() -> Result<(), ConnectorError> {
    // ── 1. The contract, offline ─────────────────────────────────────────────
    println!("== 1. the System One contract ==");
    let state = serde_json::json!({
        "ticket": "Checkout is down for all EU customers since 09:00. This is my third request.",
        "customer_contact_count": 3,
    });

    // Local contract checks fire BEFORE any network call: bounds are part
    // of the declaration, not the backend's business.
    let out_of_bounds = Connector::new("typesafe")?
        .decide("state")
        .choice("bad", "Fewer than two options is outside the contract", [
            ("only", "one"),
        ])
        .send()
        .await;
    match out_of_bounds {
        Err(err) => println!("  local check: {err}"),
        Ok(_) => println!("  unexpected success"),
    }
    let no_questions = Connector::new("typesafe")?
        .decide("state")
        .send()
        .await;
    match no_questions {
        Err(err) => println!("  local check: {err}"),
        Ok(_) => println!("  unexpected success"),
    }
    println!();

    // ── 2. Native decision model (needs TYPESAFE_API_KEY) ────────────────────
    println!("== 2. native decision model (typesafe/Jev) ==");
    match Connector::new("typesafe") {
        Ok(connector) => match declare(&connector, state.clone()).send().await {
            Ok(output) => {
                assert_eq!(output.source, ProbabilitySource::Native);
                show(&output);
            }
            Err(err) => println!("  decide skipped: {err}"),
        },
        Err(err) => println!("  Connector::new failed: {err}"),
    }
    println!();

    // ── 3. The LLM wrapper (needs any chat provider key) ─────────────────────
    // Same declaration, same output contract — the source flag says the
    // probabilities are the model's SELF-REPORTED numbers, not calibrated
    // scoring output.
    println!("== 3. LLM wrapper (openai) ==");
    match Connector::new("openai") {
        Ok(connector) => match declare(&connector, state).send().await {
            Ok(output) => {
                assert_eq!(output.source, ProbabilitySource::DerivedFromLlm);
                show(&output);
            }
            Err(err) => println!("  decide skipped: {err}"),
        },
        Err(err) => println!("  Connector::new failed: {err}"),
    }
    println!();

    // ── 4. Which decision providers are registered ───────────────────────────
    println!("== 4. decision providers ==");
    let known: BTreeMap<_, _> = known_providers_with_env()
        .filter(|(name, _)| {
            matches!(
                *name,
                "typesafe" | "azure-decision" | "openrouter"
            )
        })
        .collect();
    for (name, env) in &known {
        println!("  {name:<16} key env: {env}");
    }
    Ok(())
}
