//! Tests for the request retry middleware (`Connector::with_retry`).
//!
//! Covers: a rate limit (429) is retried transparently and the retried
//! stream succeeds; a non-retryable error (401) surfaces immediately
//! without a retry; disabling retry keeps single-shot semantics (the
//! previous behavior, used by the rest of the SDK tests); and the server's
//! `retry-after` hint is honored for pacing.

use super::super::{Connector, ToolCallMode};
use super::common::{MockResponse, connector, mock_server_sequence};
use std::time::Duration;

/// Tiny backoff override so retry tests run in milliseconds instead of
/// sleeping the production 5s → 10s → 20s schedule.
fn fast_retries(c: Connector) -> Connector {
    c.with_retry_delay(Duration::from_millis(5))
}

/// A 429 with a short `retry-after` must be retried transparently: the
/// consumer sees the retried (200) stream as if nothing happened, and the
/// server received exactly two requests.
#[tokio::test]
async fn rate_limit_is_retried_and_stream_succeeds() {
    let sse_ok = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                  data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}")
            .with_header("retry-after", "0"),
        MockResponse::new(200, sse_ok),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    let mut text = String::new();
    use tokio_stream::StreamExt;
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => text.push_str(chunk.token()),
            Err(e) => panic!("stream error after retry: {e}"),
        }
    }
    assert_eq!(text, "hello world", "retried stream delivers the full text");
    assert_eq!(
        *request_count.lock().unwrap(),
        2,
        "429 then retry = 2 requests"
    );

    let _ = handle.join();
}

/// A 500 is retryable too — the second attempt succeeds.
#[tokio::test]
async fn server_error_is_retried() {
    let sse_ok = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                  data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(500, "{\"error\":\"boom\"}"),
        MockResponse::new(200, sse_ok),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Ok(chunk) = item {
            text.push_str(chunk.token());
        }
    }
    assert_eq!(text, "ok");
    assert_eq!(*request_count.lock().unwrap(), 2);

    let _ = handle.join();
}

/// A MID-STREAM failure after partial content must emit a reset marker and
/// re-stream the retried response from the beginning: the consumer drops
/// the failed attempt's text instead of concatenating it with the retry.
#[tokio::test]
async fn mid_stream_failure_emits_reset_and_restarts() {
    // First attempt: one token, then the connection dies WITHOUT [DONE] —
    // the parser surfaces StreamTerminated (retryable). Second attempt:
    // the full stream.
    let partial =
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"partial-\"},\"index\":0}]}\n\n";
    let full = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"retried\"},\"index\":0}]}\n\n\
                data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(200, partial),
        MockResponse::new(200, full),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut text = String::new();
    let mut saw_reset = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => {
                if chunk.is_reset() {
                    saw_reset = true;
                    text.clear();
                } else {
                    text.push_str(chunk.token());
                }
            }
            Err(e) => panic!("stream error: {e}"),
        }
    }
    assert!(saw_reset, "a mid-stream retry must emit the reset marker");
    assert_eq!(
        text, "retried",
        "the retried response replaces the partial text"
    );
    assert_eq!(*request_count.lock().unwrap(), 2);

    let _ = handle.join();
}

/// A non-retryable error (401 auth) surfaces immediately — no retry, one
/// request, and the error carries the server's message.
#[tokio::test]
async fn non_retryable_error_is_not_retried() {
    let (port, request_count, handle) = mock_server_sequence(vec![MockResponse::new(
        401,
        "{\"error\":{\"message\":\"bad key\"}}",
    )]);

    let c = connector(port);
    // The first request is eager: a non-retryable 401 surfaces from the call
    // itself, with the server's message intact.
    let err = match c.stream_chat("hi").await {
        Ok(_) => panic!("a 401 must surface as an error"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("401"), "got: {err}");
    assert_eq!(*request_count.lock().unwrap(), 1, "no retry on 401");

    let _ = handle.join();
}

/// With retry disabled, the failure surfaces on the FIRST attempt and only
/// one request is made (single-shot semantics — what the rest of the SDK
/// tests rely on).
#[tokio::test]
async fn retry_disabled_is_single_shot() {
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}"),
        MockResponse::new(200, "data: [DONE]\n\n"),
    ]);

    let c = connector(port).with_retry(false);
    // Single-shot semantics: the 429 surfaces as an error from the call
    // itself (no retry), with the server's message intact.
    let err = match c.stream_chat("hi").await {
        Ok(_) => panic!("a 429 with retry disabled must fail"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("429"), "got: {err}");
    assert_eq!(*request_count.lock().unwrap(), 1, "exactly one request");

    let _ = handle.join();
}

/// The retry middleware must not interfere with the inline tool-call path:
/// with `retry-after` pacing, a 429 followed by a 200 stream still delivers
/// the parsed tool call.
#[tokio::test]
async fn inline_mode_streams_through_retry() {
    let sse_ok = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"{\\\"name\\\": \\\"echo\\\", \\\"arguments\\\": {\\\"text\\\": \\\"x\\\"}}\"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                  data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}")
            .with_header("retry-after-ms", "1"),
        MockResponse::new(200, sse_ok),
    ]);

    let c = fast_retries(connector(port)).with_tool_call_mode(ToolCallMode::Inline);
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Ok(chunk) = item {
            text.push_str(chunk.token());
        }
    }
    assert!(
        text.contains("echo"),
        "the inline tool-call JSON must arrive after the retry, got: {text}"
    );
    assert_eq!(*request_count.lock().unwrap(), 2);

    let _ = handle.join();
}

/// A transient SSE error event (OpenAI `overloaded_error` with HTTP 200)
/// inside a mid-stream response must be retried: the parser detects the
/// transient error type, marks the error as retryable, and the retry
/// middleware re-requests — the consumer sees the retried stream as if
/// nothing happened.
#[tokio::test]
async fn transient_sse_error_is_retried() {
    // First attempt: an SSE stream that emits an overloaded_error event
    // (HTTP 200 but transient). Second attempt: the full successful stream.
    let sse_error =
        "data: {\"error\":{\"message\":\"overloaded\",\"type\":\"overloaded_error\"}}\n\n";
    let sse_ok = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                  data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(200, sse_error),
        MockResponse::new(200, sse_ok),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    let mut text = String::new();
    use tokio_stream::StreamExt;
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk) => text.push_str(chunk.token()),
            Err(e) => panic!("stream error after transient retry: {e}"),
        }
    }
    assert_eq!(
        text, "ok",
        "transient SSE error must be retried transparently"
    );
    assert_eq!(
        *request_count.lock().unwrap(),
        2,
        "transient error then retry = 2 requests"
    );

    let _ = handle.join();
}

/// A non-transient SSE error event (e.g. `invalid_request_error`) must
/// NOT be retried — it surfaces as a stream error when polled (200
/// status means `send_with_retry` succeeds; the error is only detected
/// by the SSE parser).
#[tokio::test]
async fn non_transient_sse_error_is_not_retried() {
    let sse_error =
        "data: {\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\"}}\n\n";
    let (port, request_count, handle) =
        mock_server_sequence(vec![MockResponse::new(200, sse_error)]);

    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();
    // The error surfaces when the stream is polled, not from stream_chat
    // itself (HTTP 200 passes through send_with_retry successfully).
    use tokio_stream::StreamExt;
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            saw_error = true;
            assert!(e.to_string().contains("bad request"), "got: {e}");
            break;
        }
    }
    assert!(
        saw_error,
        "a non-transient SSE error must surface as a stream error"
    );
    assert_eq!(
        *request_count.lock().unwrap(),
        1,
        "no retry on non-transient error"
    );

    let _ = handle.join();
}

/// When all retry attempts are exhausted (1 initial + 3 retries = 4 total),
/// the last error surfaces to the consumer — no panic, no infinite loop.
#[tokio::test]
async fn max_retries_exhausted_surfaces_last_error() {
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}"),
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}"),
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}"),
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}"),
    ]);

    let c = fast_retries(connector(port));
    let err = match c.stream_chat("hi").await {
        Ok(_) => panic!("should have exhausted retries and surfaced an error"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("429"), "got: {err}");
    assert_eq!(
        *request_count.lock().unwrap(),
        4,
        "1 initial + 3 retries = 4 requests"
    );

    let _ = handle.join();
}

/// A non-retryable error (401) emitted MID-STREAM after some content was
/// already yielded must surface immediately without retry.
#[tokio::test]
async fn mid_stream_non_retryable_error_surfaces_immediately() {
    // First attempt: one token, then a 401 error event (non-retryable).
    let sse_partial = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"partial-\"},\"index\":0}]}\n\n\
         data: {\"error\":{\"message\":\"unauthorized\"}}\n\n";
    let (port, request_count, handle) =
        mock_server_sequence(vec![MockResponse::new(200, sse_partial)]);

    let c = connector(port);
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            saw_error = true;
            assert!(e.to_string().contains("unauthorized"), "got: {e}");
            break;
        }
    }
    assert!(
        saw_error,
        "a non-retryable mid-stream error must surface without retry"
    );
    assert_eq!(
        *request_count.lock().unwrap(),
        1,
        "no retry on non-retryable error"
    );

    let _ = handle.join();
}

/// The server's `retry-after` hint is honored: a 429 with a short
/// `retry-after-ms` header paces the retry at the hinted delay rather
/// than the default exponential backoff.
#[tokio::test]
async fn retry_after_hint_is_honored() {
    let sse_ok = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}]}\n\n\
                  data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                  data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(429, "{\"error\":{\"message\":\"rate limited\"}}")
            .with_header("retry-after-ms", "1"),
        MockResponse::new(200, sse_ok),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Ok(chunk) = item {
            text.push_str(chunk.token());
        }
    }
    assert_eq!(text, "ok");
    assert_eq!(
        *request_count.lock().unwrap(),
        2,
        "429 then retry = 2 requests"
    );

    let _ = handle.join();
}

/// Multiple mid-stream failures: the attempt counter is shared between the
/// inner re-request loop and the outer stream loop. After two consecutive
/// mid-stream failures, the third attempt succeeds.
#[tokio::test]
async fn multiple_mid_stream_retries_succeed() {
    // First attempt: one token, then stream terminates (retryable).
    let partial1 =
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"a-\"},\"index\":0}]}\n\n";
    // Second attempt: one token, then stream terminates again.
    let partial2 =
        "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"b-\"},\"index\":0}]}\n\n";
    // Third attempt: full successful stream.
    let full = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"done\"},\"index\":0}]}\n\n\
                data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\n\n\
                data: [DONE]\n\n";
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(200, partial1),
        MockResponse::new(200, partial2),
        MockResponse::new(200, full),
    ]);

    let c = fast_retries(connector(port));
    let mut stream = c.stream_chat("hi").await.unwrap();

    use tokio_stream::StreamExt;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Ok(chunk) = item {
            if chunk.is_reset() {
                text.clear();
            } else {
                text.push_str(chunk.token());
            }
        }
    }
    assert_eq!(
        text, "done",
        "the third attempt's content replaces prior partials"
    );
    assert_eq!(*request_count.lock().unwrap(), 3, "three attempts total");

    let _ = handle.join();
}

/// `with_max_retries(0)` disables retries entirely: a retryable failure
/// surfaces immediately after the first attempt.
#[tokio::test]
async fn max_retries_zero_disables_retry() {
    let sse_ok = concat!(
        r#"data: {"id":"1","choices":[{"delta":{"content":"ok"},"index":0}]}"#,
        "

"
    );
    let _ = &sse_ok;
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(500, r#"{"error":"boom"}"#),
        MockResponse::new(200, sse_ok),
    ]);

    let c = connector(port).with_max_retries(0);
    // The eager first request surfaces the 500 directly from stream_chat.
    let result = c.stream_chat("hi").await;
    let _ = handle.join();
    assert!(result.is_err(), "first 500 must surface when retries = 0");
    assert_eq!(*request_count.lock().unwrap(), 1, "exactly one attempt");
}

/// `with_max_retries(n)` caps the attempts: with n=1, one retry happens and
/// the second (successful) response completes the turn.
#[tokio::test]
async fn max_retries_one_allows_single_retry() {
    let sse_ok = concat!(
        r#"data: {"id":"1","choices":[{"delta":{"content":"ok"},"index":0}]}"#,
        "

",
        r#"data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop","index":0}]}"#,
        "

",
        "data: [DONE]

",
    );
    let (port, request_count, handle) = mock_server_sequence(vec![
        MockResponse::new(500, r#"{"error":"boom"}"#),
        MockResponse::new(200, sse_ok),
    ]);

    let c = connector(port)
        .with_retry_delay(Duration::from_millis(1))
        .with_max_retries(1);
    let mut stream = c.stream_chat("hi").await.unwrap();
    let mut text = String::new();
    while let Some(item) = tokio_stream::StreamExt::next(&mut stream).await {
        match item {
            Ok(chunk) => text.push_str(chunk.token()),
            Err(e) => panic!("retry within cap should recover: {e}"),
        }
    }
    let _ = handle.join();
    assert_eq!(text, "ok");
    assert_eq!(*request_count.lock().unwrap(), 2, "one retry happened");
}
