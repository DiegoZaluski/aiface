# Decisions — the System One contract

Typed decisions from two backend classes under ONE interface: native
decision models (TypeSafe Jev, Microsoft-Decision-1, OpenRouter's System
One route) and generative LLMs wrapped in the same schema. Swap the name
passed to `Connector::new` to change the backend — the output type never
changes.

## The contract

A decision request carries a `state` (text or JSON) and one or more typed
questions. Three question kinds:

| Kind | Answers with | Bounds |
|---|---|---|
| `noul` | one probability in [0, 1] that the statement is TRUE | — |
| `choice` | the selected option + a probability per option + `confidence` | 2–255 options |
| `score` | the probability-weighted level average + a probability per level + `confidence` | 2–10 ordered levels |

Every answer is validated against the questions the caller declared
BEFORE the call (option sets, probability ranges, answer types, no
undeclared answers), then delivered through [`DecisionOutput`].

## Native decision models

```rust
use aiface::connector::Connector;

let output = Connector::new("typesafe")?   // or "azure-decision", "openrouter"
    .with_api_key(&std::env::var("TYPESAFE_API_KEY").unwrap_or_default())
    .decide(serde_json::json!({ "ticket": "The API returns 500 on every call." }))
    .noul("urgent", "Is this request urgent?")
    .choice("team", "Which team should handle this?", [
        ("billing", "Charges and refunds"),
        ("technical", "Bugs and outages"),
    ])
    .score("severity", "How severe is this?", ["low", "medium", "high"])
    .send()
    .await?;

assert_eq!(output.noul("urgent")?, 0.94);
let team = output.choice("team")?;
assert_eq!(team.top, "technical");       // the winning option key
let severity = output.score("severity")?;
assert!((severity.score_value() - 1.1).abs() < 1e-12); // weighted average
```

Decision providers registered in the crate:

| Provider | Endpoint | Default model | API key env |
|---|---|---|---|
| `typesafe` | `https://api.typesafe.ai/v1/systemone` | `jev-latest` | `TYPESAFE_API_KEY` |
| `azure-decision` | `<resource>/providers/microsoft/v1/systemone` | `microsoft-decision-1` | `AZURE_FOUNDRY_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1/systemone` | `jev-latest` | `OPENROUTER_API_KEY` |

Notes per provider:

- **`typesafe`** — Jev reads the `state` once and scores every question in
  parallel; batching questions into one request is the intended posture
  (one round trip, one state).
- **`azure-decision`** — the request `model` value is the DEPLOYMENT name
  (`with_model("pi-decision-1")`), not the underlying model. Point
  `with_base_url` at your Foundry resource endpoint. Authentication is the
  resource API key (`api-key` header); Entra-ID callers pass the bearer
  token through `with_api_key` instead.
- **`openrouter`** — chat stays on the normal path and `decide` rides the
  System One route on the same account key.

## Decisions through an LLM

Any chat provider works too. The declared questions are compiled into a
schema prompt, the model is constrained to JSON-object output, and its
self-report flows through the SAME validation funnel:

```rust
# use aiface::connector::Connector;
let output = Connector::new("openai")?    // any chat backend
    .decide("Customer says their production service is down.")
    .noul("urgent", "Is this request urgent?")
    .choice("team", "Which team?", [
        ("billing", "Charges and refunds"),
        ("technical", "Bugs and outages"),
    ])
    .send()
    .await?;

// The source flag separates calibrated from self-reported probabilities.
assert_eq!(output.source, aiface::connector::ProbabilitySource::DerivedFromLlm);
```

The two backend classes are distinguishable in the output via
[`ProbabilitySource`] (`Native` vs `DerivedFromLlm`). An LLM's numbers are
SELF-REPORTED, not calibrated — thresholds tuned against a native model
must be re-validated on this source.

## Typed output materialization

Answers materialize into a caller-defined struct whose fields match the
declared questions: `f64` for `noul`, the option key (string or enum) for
`choice`, the weighted score (`f64`) for `score`:

```rust
# use aiface::connector::Connector;
# async fn example() -> Result<(), aiface::connector::ConnectorError> {
#[derive(serde::Deserialize)]
struct Routing {
    urgent: f64,
    team: String,
    severity: f64,
}

let routing: Routing = Connector::new("typesafe")?
    .with_api_key("ts-...")
    .decide("The API returns 500 on every call.")
    .noul("urgent", "Is this urgent?")
    .choice("team", "Which team?", [
        ("billing", "Charges and refunds"),
        ("technical", "Bugs and outages"),
    ])
    .score("severity", "How severe?", ["low", "medium", "high"])
    .send()
    .await?
    .into_model()?;
# Ok(())
# }
```

## Error surface

| Error | Meaning |
|---|---|
| `InvalidQuestion` | the declaration is outside the contract (checked locally, before any network call) — fewer than 2 options, more than 255 choices, more than 10 score levels, no question declared |
| `DecisionSchemaViolation` | an answer broke the declared schema: wrong answer type, a hallucinated option, a probability outside [0, 1], an answer for an undeclared question |
| `MissingDecisionAnswer` | a declared question has no answer in the response |
| `DecisionTypeMismatch` | the accessor's kind does not match the answer's (`output.choice` on a `noul` answer) |

`chat`, `stream_chat` and `embed` return `NotImplemented` on a decision
provider — the family's only surface is `decide`.
