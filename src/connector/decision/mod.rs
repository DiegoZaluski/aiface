//! System One decisions: a unified, typed decision contract for both
//! native decision models (TypeSafe Jev, Microsoft-Decision-1) and
//! decision-shaped LLM calls.
//!
//! A decision request carries a `state` (text or JSON) and one or more
//! typed questions — `noul` (binary probability), `choice` (pick one of a
//! fixed option set) or `score` (a position on an ordered scale). Every
//! answer comes back with probabilities, and the shape is validated
//! against the questions the caller declared BEFORE the call: the answer
//! for a `choice` question must name an option from that question's own
//! criteria, the probabilities must be finite values in [0, 1], and so on.
//!
//! The same contract holds whether the backend is a native decision API
//! (`Family::SystemOne`) or a generative LLM wrapped in a schema prompt
//! ([`ProbabilitySource::DerivedFromLlm`]): callers switch providers via
//! [`Connector::new`](crate::connector::Connector::new) and the output
//! type never changes.

use std::collections::BTreeMap;

pub(crate) mod llm;
pub(crate) mod systemone;

use crate::connector::error::ConnectorError;
use crate::connector::usage::TokenUsage;

/// How many options a single `choice` question may declare (the System One
/// wire contract's upper bound; higher cardinalities are layered manually
/// in code).
pub(crate) const MAX_CHOICE_OPTIONS: usize = 255;

/// How many levels a single `score` question may declare (2–10 on the
/// System One wire contract).
pub(crate) const MAX_SCORE_LEVELS: usize = 10;

/// The probability data behind a `choice` or `score` decision answer.
///
/// Native decision models report calibrated distributions from their
/// scoring pass; an LLM wrapper derives them from a structured self-report
/// (see [`ProbabilitySource`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Probabilities {
    /// The winning label — the option key for a `choice` answer, the level
    /// index string (`"0"`, `"1"`, …) for a `score` answer.
    pub top: String,
    /// Probability (0.0–1.0) for each declared label. Normalized to sum to
    /// 1.0 when the backend's raw numbers do not (LLM wrappers often emit
    /// unnormalized or partial mass).
    pub per_label: BTreeMap<String, f64>,
    /// Concentration of probability mass on the winning label — the
    /// "how sure" signal callers threshold on. Always present for `choice`
    /// and `score` answers; `noul` answers carry the whole signal in the
    /// single probability itself (see [`DecisionAnswer::Noul`]).
    pub confidence: f64,
}

impl Probabilities {
    /// The probability-weighted average of the level indexes — the
    /// System One contract's `score` value: `Σ i·pᵢ` over the distribution
    /// keyed by index strings. A 4-level scale spans 0.0–3.0 and the
    /// result can fall between levels.
    ///
    /// Labels that do not parse as level indexes (never happens for a
    /// validated `score` answer) contribute nothing.
    #[must_use]
    pub fn score_value(&self) -> f64 {
        self.per_label
            .iter()
            .filter_map(|(k, p)| k.parse::<usize>().ok().map(|i| (i, p)))
            .map(|(i, p)| i as f64 * p)
            .sum()
    }
}

/// A validated answer to one declared question.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionAnswer {
    /// Binary probability in [0, 1] — the probability the condition is
    /// TRUE. There is no separate `confidence`: this single number is the
    /// whole signal (System One contract — `noul` answers carry no
    /// confidence field).
    Noul(f64),
    /// One option selected from the question's declared set, with a
    /// probability for every declared option.
    Choice(Probabilities),
    /// A position on the question's ordered scale — the probability-weighted
    /// average of the level indexes (so a 4-level scale spans 0.0–3.0 and a
    /// result can fall between levels) — with the per-level distribution.
    Score(Probabilities),
}

impl DecisionAnswer {
    /// The probability the condition holds, for a `noul` answer.
    ///
    /// # Errors
    ///
    /// [`ConnectorError::DecisionTypeMismatch`] when this answer is not a
    /// `noul` answer (e.g. the question was declared as `choice` but the
    /// accessor reads it as a probability).
    pub fn as_noul(&self) -> Result<f64, ConnectorError> {
        match self {
            Self::Noul(p) => Ok(*p),
            other => Err(ConnectorError::DecisionTypeMismatch {
                expected: "noul",
                got: other.kind(),
            }),
        }
    }

    /// The selected option and its full probability distribution, for a
    /// `choice` answer.
    ///
    /// # Errors
    ///
    /// [`ConnectorError::DecisionTypeMismatch`] when this answer is not a
    /// `choice` answer.
    pub fn as_choice(&self) -> Result<&Probabilities, ConnectorError> {
        match self {
            Self::Choice(p) => Ok(p),
            other => Err(ConnectorError::DecisionTypeMismatch {
                expected: "choice",
                got: other.kind(),
            }),
        }
    }

    /// The weighted score and its per-level distribution, for a `score`
    /// answer.
    ///
    /// # Errors
    ///
    /// [`ConnectorError::DecisionTypeMismatch`] when this answer is not a
    /// `score` answer.
    pub fn as_score(&self) -> Result<&Probabilities, ConnectorError> {
        match self {
            Self::Score(p) => Ok(p),
            other => Err(ConnectorError::DecisionTypeMismatch {
                expected: "score",
                got: other.kind(),
            }),
        }
    }

    /// The answer's kind, as the wire-contract type string.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Noul(_) => "noul",
            Self::Choice(_) => "choice",
            Self::Score(_) => "score",
        }
    }

    /// The answer's PLAIN value — the shape [`DecisionOutput::into_model`]
    /// materializes into user structs: the probability itself for `noul`,
    /// the winning option key for `choice`, the probability-weighted score
    /// for `score`.
    #[must_use]
    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Self::Noul(p) => serde_json::json!(p),
            Self::Choice(p) => serde_json::json!(p.top),
            Self::Score(p) => serde_json::json!(p.score_value()),
        }
    }
}

/// Where a decision's probabilities came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbabilitySource {
    /// The backend is a native decision model (TypeSafe Jev,
    /// Microsoft-Decision-1): the probabilities are the model's calibrated
    /// scoring output.
    Native,
    /// The backend is a generative LLM wrapped in the decision schema: the
    /// numbers are the model's SELF-REPORTED probabilities, not calibrated
    /// outputs. Thresholds tuned against a native model must be re-validated
    /// on this source (the System One wrapper trade-off: flexible backend,
    /// weaker calibration guarantees).
    DerivedFromLlm,
}

/// The validated result of a decision request.
///
/// Answers are keyed by the question name the caller declared — the same
/// keys travel through every backend, so downstream code is
/// backend-agnostic.
#[derive(Clone, Debug)]
pub struct DecisionOutput {
    answers: BTreeMap<String, DecisionAnswer>,
    /// Normalized token usage (input/output/reported cost) when the
    /// backend reported one.
    pub usage: Option<TokenUsage>,
    /// Raw JSON response as received (unmodified), mirroring
    /// [`ChatOutput::raw`](crate::connector::ChatOutput::raw).
    pub raw: String,
    /// Which backend class produced these probabilities.
    pub source: ProbabilitySource,
}

impl DecisionOutput {
    /// Build an output from parts (also usable from doctests and
    /// downstream code that needs to assemble one by hand).
    pub fn from_parts(
        answers: BTreeMap<String, DecisionAnswer>,
        usage: Option<TokenUsage>,
        raw: String,
        source: ProbabilitySource,
    ) -> Self {
        Self {
            answers,
            usage,
            raw,
            source,
        }
    }

    /// The validated answers, keyed by question name.
    #[must_use]
    pub fn answers(&self) -> &BTreeMap<String, DecisionAnswer> {
        &self.answers
    }

    /// The answer for one question by name.
    ///
    /// # Errors
    ///
    /// [`ConnectorError::MissingDecisionAnswer`] when no answer arrived for
    /// that question name.
    pub fn answer(&self, question: &str) -> Result<&DecisionAnswer, ConnectorError> {
        self.answers
            .get(question)
            .ok_or_else(|| ConnectorError::MissingDecisionAnswer(question.to_string()))
    }

    /// The `noul` probability for one question.
    ///
    /// # Errors
    ///
    /// Missing answer or a type mismatch (see
    /// [`DecisionAnswer::as_noul`]).
    pub fn noul(&self, question: &str) -> Result<f64, ConnectorError> {
        self.answer(question)?.as_noul()
    }

    /// The `choice` distribution for one question.
    ///
    /// # Errors
    ///
    /// Missing answer or a type mismatch (see
    /// [`DecisionAnswer::as_choice`]).
    pub fn choice(&self, question: &str) -> Result<&Probabilities, ConnectorError> {
        self.answer(question)?.as_choice()
    }

    /// The `score` distribution for one question.
    ///
    /// # Errors
    ///
    /// Missing answer or a type mismatch (see
    /// [`DecisionAnswer::as_score`]).
    pub fn score(&self, question: &str) -> Result<&Probabilities, ConnectorError> {
        self.answer(question)?.as_score()
    }

    /// Materialize the answers into a caller-defined struct via serde.
    ///
    /// This is the "typed decision" payoff of the contract: declare a Rust
    /// struct whose field names match the question names and whose field
    /// types match the declared question kinds (`f64` for `noul`, a string
    /// or enum for the `choice` winner, `f64` for a `score`), then
    /// deserialize straight into it.
    ///
    /// ```
    /// # use aiface::connector::DecisionOutput;
    /// # use std::collections::BTreeMap;
    /// # fn build() -> DecisionOutput {
    /// #     let mut answers = BTreeMap::new();
    /// #     answers.insert("team".to_string(), aiface::connector::DecisionAnswer::Noul(0.9));
    /// #     DecisionOutput::from_parts(answers, None, "{}".to_string(), aiface::connector::ProbabilitySource::Native)
    /// # }
    /// #[derive(serde::Deserialize)]
    /// struct Routing {
    ///     team: f64, // the `noul` probability
    /// }
    /// let routing: Routing = build().into_model().unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// [`ConnectorError::Deserialization`] when the answers do not fit the
    /// target type (missing question, wrong shape, unmatchable enum
    /// variant, ...).
    pub fn into_model<T: serde::de::DeserializeOwned>(&self) -> Result<T, ConnectorError> {
        // Each answer materializes to its PLAIN contract value (probability
        // number / winning option / weighted score) — NOT the internal
        // enum shape — so the user struct fields line up with the question
        // kinds they declared.
        let plain: BTreeMap<&str, serde_json::Value> = self
            .answers
            .iter()
            .map(|(k, a)| (k.as_str(), a.to_value()))
            .collect();
        serde_json::from_value(serde_json::to_value(&plain).expect("plain answers serialize"))
            .map_err(|e| ConnectorError::Deserialization(e.to_string()))
    }
}

/// A declared question: what the backend should judge, and the fixed label
/// set it may answer with.
///
/// The Rust-side declaration doubles as the validation schema — every
/// backend must answer within the declared set (see
/// [`validate_answers`]).
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionQuestion {
    /// Binary probability gate ("is this true?"). `criteria` anchors the
    /// TRUE and FALSE conditions for borderline states.
    Noul {
        instructions: String,
        criteria: Option<(String, String)>,
    },
    /// Pick exactly one option from a fixed set of 2–255 named,
    /// described options (routing, classification, model/tool selection).
    Choice {
        instructions: String,
        criteria: Vec<(String, String)>,
    },
    /// Position on an ordered scale of 2–10 levels, lowest index first
    /// (severity, priority, rubric grading). The score is the
    /// probability-weighted average of the level indexes.
    Score {
        instructions: String,
        levels: Vec<String>,
    },
}

impl DecisionQuestion {
    /// The question's wire-contract type string (`noul` / `choice` /
    /// `score`).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }

    /// The declared label set: for `choice` the option keys, for `score`
    /// the level index strings (`"0"`, `"1"`, …), `None` for `noul`.
    #[must_use]
    pub fn labels(&self) -> Option<Vec<String>> {
        match self {
            Self::Noul { .. } => None,
            Self::Choice { criteria, .. } => {
                Some(criteria.iter().map(|(k, _)| k.clone()).collect())
            }
            Self::Score { levels, .. } => Some(
                (0..levels.len())
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    /// Validate the question's declaration against the System One
    /// contract's bounds (2–255 choices, 2–10 ordered levels).
    ///
    /// # Errors
    ///
    /// [`ConnectorError::InvalidQuestion`] when the declaration is outside
    /// the contract.
    pub fn validate(&self) -> Result<(), ConnectorError> {
        match self {
            Self::Noul { .. } => Ok(()),
            Self::Choice { criteria, .. } => {
                if criteria.len() < 2 || criteria.len() > MAX_CHOICE_OPTIONS {
                    return Err(ConnectorError::InvalidQuestion(format!(
                        "choice question declares {} options; the System One contract allows 2–{MAX_CHOICE_OPTIONS}",
                        criteria.len()
                    )));
                }
                // Duplicate option keys would make the declared label set
                // and the wire's criteria map disagree — reject the
                // declaration instead of quietly answering fewer options.
                let mut seen = std::collections::HashSet::new();
                if let Some(dup) = criteria
                    .iter()
                    .map(|(k, _)| k)
                    .find(|k| !seen.insert(k.as_str()))
                {
                    return Err(ConnectorError::InvalidQuestion(format!(
                        "choice question declares option `{dup}` twice"
                    )));
                }
                Ok(())
            }
            Self::Score { levels, .. } => {
                if levels.len() < 2 || levels.len() > MAX_SCORE_LEVELS {
                    return Err(ConnectorError::InvalidQuestion(format!(
                        "score question declares {} levels; the System One contract allows 2–{MAX_SCORE_LEVELS}",
                        levels.len()
                    )));
                }
                Ok(())
            }
        }
    }
}

/// Validate a set of parsed answers against the declared questions.
///
/// Every declared question must have an answer, every answer must have the
/// question's type, every label the answer references must belong to the
/// question's own declared set, and every probability must be a finite
/// value in [0, 1]. This is the single funnel BOTH backends share — a
/// native decision API response and an LLM's JSON self-report are held to
/// exactly the same contract.
///
/// Returns the validated answers with normalized distributions (each
/// distribution's mass is scaled to sum to 1.0 when the backend's raw
/// numbers drift; the winner and the order of the distribution are
/// preserved).
///
/// # Errors
///
/// [`ConnectorError::DecisionSchemaViolation`] when an answer breaks the
/// declared schema; [`ConnectorError::MissingDecisionAnswer`] when a
/// declared question has no answer.
pub(crate) fn validate_answers(
    questions: &BTreeMap<String, DecisionQuestion>,
    parsed: BTreeMap<String, DecisionAnswer>,
) -> Result<BTreeMap<String, DecisionAnswer>, ConnectorError> {
    let mut out = BTreeMap::new();
    for (name, question) in questions {
        let Some(answer) = parsed.get(name) else {
            return Err(ConnectorError::MissingDecisionAnswer(name.clone()));
        };
        if answer.kind() != question.kind() {
            return Err(ConnectorError::DecisionSchemaViolation {
                question: name.clone(),
                detail: format!(
                    "declared as `{}` but answered as `{}`",
                    question.kind(),
                    answer.kind()
                ),
            });
        }
        let validated = match answer {
            DecisionAnswer::Noul(p) => {
                let p = *p;
                if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                    return Err(ConnectorError::DecisionSchemaViolation {
                        question: name.clone(),
                        detail: format!("noul probability {p} is outside [0, 1]"),
                    });
                }
                DecisionAnswer::Noul(p)
            }
            DecisionAnswer::Choice(_) | DecisionAnswer::Score(_) => {
                let labels = question.labels().unwrap_or_default();
                let normalized = normalize_distribution(name, answer, &labels)?;
                match question.kind() {
                    "choice" => DecisionAnswer::Choice(normalized),
                    _ => DecisionAnswer::Score(normalized),
                }
            }
        };
        out.insert(name.clone(), validated);
    }
    // An answer for a question the caller never declared is a schema
    // violation too — the contract is exactly the declared set, nothing
    // more.
    for name in parsed.keys() {
        if !questions.contains_key(name) {
            return Err(ConnectorError::DecisionSchemaViolation {
                question: name.clone(),
                detail: "answer for an undeclared question".to_string(),
            });
        }
    }
    Ok(out)
}

/// Extract the probability payload from a `choice`/`score` answer (the
/// validation result keeps the answer's declared kind).
fn answer_probs(answer: &DecisionAnswer) -> &Probabilities {
    match answer {
        DecisionAnswer::Choice(p) | DecisionAnswer::Score(p) => p,
        DecisionAnswer::Noul(_) => unreachable!("guarded by the caller's kind check"),
    }
}

/// Normalize one answer's distribution against its declared label set.
///
/// Enforces: every probability finite and in [0, 1]; exactly the declared
/// label set (no missing label, no invented label); positive total mass.
/// The distribution is then rescaled to sum to 1.0, preserving the winner.
fn normalize_distribution(
    question: &str,
    answer: &DecisionAnswer,
    labels: &[String],
) -> Result<Probabilities, ConnectorError> {
    let violation = |detail: String| ConnectorError::DecisionSchemaViolation {
        question: question.to_string(),
        detail,
    };
    let probs = answer_probs(answer);
    for (label, p) in &probs.per_label {
        if !p.is_finite() || !(0.0..=1.0).contains(p) {
            return Err(violation(format!(
                "probability for `{label}` is {p}, outside [0, 1]"
            )));
        }
    }
    for label in labels {
        if !probs.per_label.contains_key(label) {
            return Err(violation(format!(
                "declared option `{label}` is missing from the answer's probabilities"
            )));
        }
    }
    for label in probs.per_label.keys() {
        if !labels.contains(label) {
            return Err(violation(format!(
                "answer carries probability for undeclared option `{label}`"
            )));
        }
    }
    // The winner must be one of the declared labels — a hallucinated
    // option key is the classic LLM wrapper failure and a contract break
    // even from a native backend.
    if !labels.contains(&probs.top) {
        return Err(violation(format!(
            "selected `{}`, which is not one of the declared options",
            probs.top
        )));
    }
    let total: f64 = probs.per_label.values().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(violation(format!(
            "probability mass totals {total}; it must be positive"
        )));
    }
    let per_label = probs
        .per_label
        .iter()
        .map(|(k, v)| (k.clone(), v / total))
        .collect();
    // NaN survives clamp, so a missing/garbled confidence falls back to the
    // winner's own share of the distribution — the strongest available
    // "how sure" signal.
    let confidence = if probs.confidence.is_finite() {
        probs.confidence.clamp(0.0, 1.0)
    } else {
        probs.per_label[&probs.top] / total
    };
    Ok(Probabilities {
        top: probs.top.clone(),
        per_label,
        confidence,
    })
}

/// Builder for a decision request: declare the state and the typed
/// questions, then `send()`.
///
/// Obtained from [`Connector::decide`](crate::connector::Connector::decide);
/// the backend (native decision model or LLM wrapper) is chosen by the
/// connector's provider — the output type is the same either way.
///
/// ```
/// # async fn example() -> Result<(), aiface::connector::ConnectorError> {
/// # use aiface::connector::Connector;
/// let output = Connector::new("typesafe")?
///     .with_api_key("ts-...")
///     .decide("Customer says their production service is down.")
///     .noul("urgent", "Is this request urgent?")
///     .choice("team", "Which team should handle this?", [
///         ("billing", "Charges and refunds"),
///         ("technical", "Bugs and outages"),
///     ])
///     .score("severity", "How severe?", ["low", "medium", "high"])
///     .send()
///     .await?;
/// assert!(output.noul("urgent").is_ok());
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct DecisionRequest<'a> {
    connector: &'a crate::connector::client::Connector,
    state: serde_json::Value,
    questions: BTreeMap<String, DecisionQuestion>,
    /// Question names declared more than once — the BTreeMap silently
    /// overwrites, so the builder records the collisions for `send()`
    /// to reject (a repeat declaration is almost always a caller bug).
    duplicate_names: Vec<String>,
}

impl<'a> DecisionRequest<'a> {
    pub(crate) fn new(
        connector: &'a crate::connector::client::Connector,
        state: serde_json::Value,
    ) -> DecisionRequest<'a> {
        Self {
            connector,
            state,
            questions: BTreeMap::new(),
            duplicate_names: Vec::new(),
        }
    }

    /// Record one question declaration, noting a repeated name instead of
    /// silently overwriting it (`send()` rejects the request).
    fn declare(mut self, name: String, question: DecisionQuestion) -> Self {
        if self.questions.insert(name.clone(), question).is_some() {
            self.duplicate_names.push(name);
        }
        self
    }

    /// Declare a `noul` (binary probability) question: "is this true?"
    pub fn noul(self, name: impl Into<String>, instructions: impl Into<String>) -> Self {
        self.declare(
            name.into(),
            DecisionQuestion::Noul {
                instructions: instructions.into(),
                criteria: None,
            },
        )
    }

    /// Declare a `noul` question with TRUE/FALSE criteria anchors — sharper
    /// boundaries for borderline states.
    pub fn noul_with_criteria(
        self,
        name: impl Into<String>,
        instructions: impl Into<String>,
        true_when: impl Into<String>,
        false_when: impl Into<String>,
    ) -> Self {
        self.declare(
            name.into(),
            DecisionQuestion::Noul {
                instructions: instructions.into(),
                criteria: Some((true_when.into(), false_when.into())),
            },
        )
    }

    /// Declare a `choice` question: pick exactly one option from a fixed
    /// set of `(option, description)` pairs (2–255 options).
    pub fn choice(
        self,
        name: impl Into<String>,
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.declare(
            name.into(),
            DecisionQuestion::Choice {
                instructions: instructions.into(),
                criteria: criteria
                    .into_iter()
                    .map(|(k, d)| (k.into(), d.into()))
                    .collect(),
            },
        )
    }

    /// Declare a `score` question: a position on an ordered scale of 2–10
    /// levels, lowest index first.
    pub fn score(
        self,
        name: impl Into<String>,
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.declare(
            name.into(),
            DecisionQuestion::Score {
                instructions: instructions.into(),
                levels: levels.into_iter().map(Into::into).collect(),
            },
        )
    }

    /// Send the decision request to the connector's provider.
    ///
    /// Dispatch is CAPABILITY-based: any provider that declares a decisions
    /// endpoint (`decisions_path` — TypeSafe, Azure Foundry decisions,
    /// OpenRouter's System One route) calls the native decisions API; every
    /// other provider routes through the LLM wrapper (same contract,
    /// self-reported probabilities).
    ///
    /// # Errors
    ///
    /// [`ConnectorError::InvalidQuestion`] when a declaration is outside
    /// the System One bounds (checked locally, before any network call),
    /// a question name was declared twice, or no question was declared;
    /// the provider's transport errors; and the decision-contract errors
    /// when the answers break the declared schema.
    pub async fn send(self) -> Result<DecisionOutput, ConnectorError> {
        if self.questions.is_empty() {
            return Err(ConnectorError::InvalidQuestion(
                "a decision request needs at least one question".to_string(),
            ));
        }
        if let Some(name) = self.duplicate_names.first() {
            return Err(ConnectorError::InvalidQuestion(format!(
                "question `{name}` was declared more than once"
            )));
        }
        let provider = self.connector.provider()?;
        if provider.decisions_path.is_some() {
            systemone::decide(
                provider,
                &self.connector.params,
                self.state,
                &self.questions,
                self.connector.params.service_keyring.as_deref(),
            )
            .await
        } else {
            llm::decide(self.connector, self.state, &self.questions).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice_q(options: &[&str]) -> DecisionQuestion {
        DecisionQuestion::Choice {
            instructions: "pick one".to_string(),
            criteria: options
                .iter()
                .map(|o| (o.to_string(), "desc".to_string()))
                .collect(),
        }
    }

    fn score_q(levels: &[&str]) -> DecisionQuestion {
        DecisionQuestion::Score {
            instructions: "rate it".to_string(),
            levels: levels.iter().map(|l| l.to_string()).collect(),
        }
    }

    fn probs(top: &str, per: &[(&str, f64)]) -> Probabilities {
        Probabilities {
            top: top.to_string(),
            per_label: per.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            confidence: 0.9,
        }
    }

    #[test]
    fn noul_probability_passes_validation() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "urgent".to_string(),
            DecisionQuestion::Noul {
                instructions: "is urgent".to_string(),
                criteria: None,
            },
        );
        let mut parsed = BTreeMap::new();
        parsed.insert("urgent".to_string(), DecisionAnswer::Noul(0.94));
        let out = validate_answers(&questions, parsed).unwrap();
        assert_eq!(out["urgent"].as_noul().unwrap(), 0.94);
    }

    #[test]
    fn noul_probability_outside_unit_interval_is_rejected() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "urgent".to_string(),
            DecisionQuestion::Noul {
                instructions: "is urgent".to_string(),
                criteria: None,
            },
        );
        let mut parsed = BTreeMap::new();
        parsed.insert("urgent".to_string(), DecisionAnswer::Noul(1.5));
        assert!(validate_answers(&questions, parsed).is_err());
    }

    #[test]
    fn unnormalized_choice_distribution_is_normalized() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        let mut parsed = BTreeMap::new();
        // Raw sub-unit mass (an LLM wrapper's common drift): 0.4 + 0.1 = 0.5
        // → normalized to 0.8 / 0.2.
        parsed.insert(
            "team".to_string(),
            DecisionAnswer::Choice(probs("billing", &[("billing", 0.4), ("technical", 0.1)])),
        );
        let out = validate_answers(&questions, parsed).unwrap();
        let p = out["team"].as_choice().unwrap();
        assert_eq!(p.top, "billing");
        assert!((p.per_label["billing"] - 0.8).abs() < 1e-12);
        assert!((p.per_label["technical"] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn hallucinated_option_is_a_schema_violation() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        let mut parsed = BTreeMap::new();
        parsed.insert(
            "team".to_string(),
            DecisionAnswer::Choice(probs(
                "sales",
                &[("billing", 0.5), ("technical", 0.5), ("sales", 0.0)],
            )),
        );
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn missing_declared_option_is_a_schema_violation() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        let mut parsed = BTreeMap::new();
        // The LLM answered with only ONE option's probability.
        parsed.insert(
            "team".to_string(),
            DecisionAnswer::Choice(probs("billing", &[("billing", 1.0)])),
        );
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn answer_type_must_match_declaration() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        let mut parsed = BTreeMap::new();
        parsed.insert("team".to_string(), DecisionAnswer::Noul(0.7));
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn missing_answer_for_declared_question_is_rejected() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        questions.insert(
            "urgent".to_string(),
            DecisionQuestion::Noul {
                instructions: "is urgent".to_string(),
                criteria: None,
            },
        );
        let mut parsed = BTreeMap::new();
        parsed.insert(
            "team".to_string(),
            DecisionAnswer::Choice(probs("billing", &[("billing", 0.5), ("technical", 0.5)])),
        );
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(err, ConnectorError::MissingDecisionAnswer(_)));
    }

    #[test]
    fn undeclared_answer_is_a_schema_violation() {
        let mut questions = BTreeMap::new();
        questions.insert(
            "urgent".to_string(),
            DecisionQuestion::Noul {
                instructions: "is urgent".to_string(),
                criteria: None,
            },
        );
        let mut parsed = BTreeMap::new();
        parsed.insert("urgent".to_string(), DecisionAnswer::Noul(0.5));
        parsed.insert("extra".to_string(), DecisionAnswer::Noul(0.5));
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn score_levels_validate_bounds_and_indexes() {
        // 2 levels is the minimum.
        assert!(score_q(&["low", "high"]).validate().is_ok());
        // 1 level is below the contract minimum.
        assert!(score_q(&["low"]).validate().is_err());
        // 11 levels is above the contract maximum.
        assert!(score_q(&["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])
            .validate()
            .is_err());
    }

    #[test]
    fn choice_bounds_match_the_contract() {
        assert!(choice_q(&["a", "b"]).validate().is_ok());
        assert!(choice_q(&["a"]).validate().is_err());
        // 255 options is the maximum — build it procedurally.
        let max: Vec<String> = (0..MAX_CHOICE_OPTIONS).map(|i| format!("o{i}")).collect();
        let max_q = DecisionQuestion::Choice {
            instructions: "pick".to_string(),
            criteria: max.iter().map(|o| (o.clone(), "d".to_string())).collect(),
        };
        assert!(max_q.validate().is_ok());
        let over: Vec<String> = (0..=MAX_CHOICE_OPTIONS).map(|i| format!("o{i}")).collect();
        let over_q = DecisionQuestion::Choice {
            instructions: "pick".to_string(),
            criteria: over.iter().map(|o| (o.clone(), "d".to_string())).collect(),
        };
        assert!(over_q.validate().is_err());
    }

    #[test]
    fn zero_total_mass_is_rejected() {
        let mut questions = BTreeMap::new();
        questions.insert("team".to_string(), choice_q(&["billing", "technical"]));
        let mut parsed = BTreeMap::new();
        parsed.insert(
            "team".to_string(),
            DecisionAnswer::Choice(probs("billing", &[("billing", 0.0), ("technical", 0.0)])),
        );
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn score_indexes_validate_against_declared_levels() {
        let mut questions = BTreeMap::new();
        questions.insert("severity".to_string(), score_q(&["low", "med", "high"]));
        let mut parsed = BTreeMap::new();
        // A level index the question never declared ("3" on a 3-level scale).
        parsed.insert(
            "severity".to_string(),
            DecisionAnswer::Score(probs(
                "1",
                &[("0", 0.2), ("1", 0.5), ("2", 0.3), ("3", 0.0)],
            )),
        );
        let err = validate_answers(&questions, parsed).unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionSchemaViolation { .. }
        ));
    }

    #[test]
    fn accessor_type_mismatch_is_typed() {
        let mut answers = BTreeMap::new();
        answers.insert("x".to_string(), DecisionAnswer::Noul(0.5));
        let out =
            DecisionOutput::from_parts(answers, None, "{}".to_string(), ProbabilitySource::Native);
        assert_eq!(out.noul("x").unwrap(), 0.5);
        let err = out.choice("x").unwrap_err();
        assert!(matches!(
            err,
            ConnectorError::DecisionTypeMismatch {
                expected: "choice",
                ..
            }
        ));
        let err = out.answer("missing").unwrap_err();
        assert!(matches!(err, ConnectorError::MissingDecisionAnswer(_)));
    }

    #[test]
    fn into_model_materializes_a_user_struct() {
        let mut answers = BTreeMap::new();
        answers.insert("team".to_string(), DecisionAnswer::Noul(0.94));
        let out =
            DecisionOutput::from_parts(answers, None, "{}".to_string(), ProbabilitySource::Native);
        #[derive(serde::Deserialize)]
        struct Routing {
            team: f64,
        }
        let routing: Routing = out.into_model().unwrap();
        assert!((routing.team - 0.94).abs() < 1e-12);
    }
}
