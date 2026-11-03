//! Request retry middleware, mirroring the crush/fantasy semantics.
//!
//! Retryable failures — rate limits (429), HTTP 5xx, and every
//! network/transport-level error (connection refused, timeouts, reset
//! streams) — are retried with exponential backoff (5s → 10s → 20s),
//! honoring the server's `retry-after` / `retry-after-ms` headers when they
//! carry a sane hint. Deterministic failures (auth, context-window
//! overflow, malformed responses) are never retried.
//!
//! When a retry fires AFTER partial content was already streamed, a
//! [`StreamChunk::reset`](crate::connector::output::StreamChunk::reset) marker is emitted first so the consumer discards
//! the failed attempt's buffered text — the retried response restarts from
//! the beginning and must not concatenate with the partial content.

use super::error::ConnectorError;
use std::time::Duration;

/// Maximum number of RETRIES for a single request (crush/fantasy:
/// `MaxRetries = 3`) — i.e. up to 4 total attempts.
pub const RETRY_MAX_RETRIES: usize = 3;

/// Initial backoff before the first retry (crush/fantasy: 5s).
pub const RETRY_INITIAL_DELAY: Duration = Duration::from_secs(5);

/// Backoff multiplier per retry (crush/fantasy: 2.0).
pub const RETRY_BACKOFF_FACTOR: f64 = 2.0;

/// Upper bound for a server-provided retry hint (crush/fantasy: 60s).
const RETRY_HINT_MAX: Duration = Duration::from_secs(60);

/// Whether a connector error should be retried: rate limits (429), HTTP 5xx,
/// and every network/transport-level failure (connection refused, timeouts,
/// reset streams). Deterministic failures — auth (401/403), context-window
/// overflow, malformed responses — are NOT retried: re-running them would
/// only waste time and tokens.
#[must_use]
pub fn is_retryable_error(e: &ConnectorError) -> bool {
    match e {
        ConnectorError::Network(_) | ConnectorError::StreamTerminated => true,
        ConnectorError::HttpError {
            status, transient, ..
        } => *transient || *status == 429 || (500..=599).contains(status),
        _ => false,
    }
}

/// Backoff delay before retry `attempt` (1-based): exponential by default
/// (`initial * factor^(attempt-1)`), overridden by the server's retry hint
/// when one was attached to the error and it is smaller and within the sane
/// bounds (mirrors fantasy's `getRetryDelayInMs`).
#[must_use]
pub fn retry_delay(e: &ConnectorError, attempt: usize) -> Duration {
    let base =
        RETRY_INITIAL_DELAY.mul_f64(RETRY_BACKOFF_FACTOR.powi(attempt.saturating_sub(1) as i32));
    if let ConnectorError::HttpError {
        retry_after_ms: Some(ms),
        ..
    } = e
    {
        let hinted = Duration::from_millis(*ms);
        if hinted < RETRY_HINT_MAX && hinted < base {
            return hinted;
        }
    }
    base
}

/// Apply "equal jitter" to a computed backoff: the result is uniformly in
/// `[0.5 * delay, delay)`.
///
/// Without jitter, every concurrent session that hits the same rate limit
/// re-sends at exactly the same instant (thundering herd), repeating the
/// spike that caused the limit. The entropy source is wall-clock nanoseconds
/// mixed with the delay itself — not cryptographic, plenty for spacing out
/// retries.
#[must_use]
pub fn jittered(delay: Duration) -> Duration {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        // A per-process counter keeps concurrent calls distinct even when
        // the clock tick did not move between them.
        .wrapping_add(CALL_COUNTER.fetch_add(1, Ordering::Relaxed) << 20);
    // SplitMix64 finalizer: cheap avalanche so consecutive calls differ even
    // when the clock tick did not.
    let mut z = nanos.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let half = delay / 2;
    let extra = delay
        .saturating_sub(half)
        .mul_f64((z >> 11) as f64 / (1u64 << 53) as f64);
    half + extra
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_stays_within_half_to_full_delay() {
        for delay in [
            Duration::from_millis(1),
            RETRY_INITIAL_DELAY,
            Duration::from_secs(60),
        ] {
            for _ in 0..50 {
                let j = jittered(delay);
                assert!(j >= delay / 2, "{j:?} below half of {delay:?}");
                assert!(j < delay, "{j:?} above {delay:?}");
            }
        }
    }

    #[test]
    fn jitter_varies_across_calls() {
        let delay = Duration::from_secs(5);
        let seen: std::collections::HashSet<_> = (0..20).map(|_| jittered(delay)).collect();
        assert!(seen.len() > 3, "jitter produced too few distinct delays");
    }

    #[test]
    fn zero_delay_stays_zero() {
        assert_eq!(jittered(Duration::ZERO), Duration::ZERO);
    }
}
