use super::error::ConnectorError;
use super::output::StreamChunk;
use super::provider::{ProviderConfig, ZEN_USER_AGENT, is_opencode_gateway};
use async_stream::stream;
use std::pin::Pin;
use std::sync::OnceLock;
use std::time::Duration;
use tokio_stream::Stream;

/// Per-chunk read timeout for SSE streams. Generous (120s) so a slow local
/// model or a reasoning pause between tokens does not abort the agent loop
/// mid-generation; only a truly dead connection trips it.
pub(crate) const SSE_CHUNK_TIMEOUT: Duration = Duration::from_secs(120);

/// Shared [`reqwest::Client`] reused across every provider request.
///
/// Creating a fresh client per request discards the connection pool, so every
/// LLM call pays a full DNS + TCP + TLS handshake — the dominant source of the
/// slow intervals between agent-loop calls. One long-lived client keeps
/// keep-alive connections alive between calls (and between sessions).
pub(crate) fn shared_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        // Only a CONNECT timeout lives on the client: an SSE stream's body is
        // read incrementally by the caller, so a total request timeout here
        // would kill long generations. Per-chunk SSE timeouts govern the body.
        reqwest::Client::builder()
            .pool_max_idle_per_host(4)
            .connect_timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

pub struct SseBuffer {
    buf: Vec<u8>,
}

impl SseBuffer {
    pub(crate) const fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(crate) fn push_and_drain(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut frames = Vec::new();
        // An SSE event ends with a blank line. Both newline styles occur in
        // the wild: LF LF ("\n\n" — OpenAI, Claude) and CRLF CRLF
        // ("\r\n\r\n" — which the Gemini API emits even with `?alt=sse`).
        // Split on whichever separator comes FIRST: a parser that only
        // accepts "\n\n" never splits Gemini's CRLF stream, the whole body
        // stays buffered, and the caller silently receives zero frames (no
        // tokens, no error) — the "provider fails silently" symptom.
        while let Some(sep_end) = next_sse_separator(&self.buf) {
            let raw: Vec<u8> = self.buf.drain(..sep_end).collect();
            let text = String::from_utf8_lossy(&raw);
            if let Some(data) = text.lines().find_map(|l| strip_data_prefix(l)) {
                frames.push(data.to_owned());
            }
        }
        frames
    }

    /// Flush any remaining buffered data as frames at end-of-stream.
    ///
    /// The SSE spec dispatches pending data when the stream ends, so a
    /// server that closes the connection right after the last `data:` line
    /// (no trailing blank line) must not lose its final frame silently —
    /// the same class of failure as the CRLF separator bug. A no-op when
    /// the stream ended cleanly (every event was blank-line terminated).
    ///
    /// Note: a TRUNCATED stream (connection cut mid-frame) leaves a partial
    /// line here, which is emitted as-is and will fail JSON parsing at the
    /// consumer — surfacing the truncation as an error is intentional and
    /// preferable to silently dropping the tail.
    pub(crate) fn flush(&mut self) -> Vec<String> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let raw: Vec<u8> = std::mem::take(&mut self.buf);
        let text = String::from_utf8_lossy(&raw);
        text.lines()
            .filter_map(|l| strip_data_prefix(l).map(String::from))
            .collect()
    }
}

/// Strip the SSE `data:` field prefix from a line, returning the field value
/// (trimmed of surrounding whitespace) or `None` for non-data lines.
///
/// Per the WHATWG SSE spec, the field value starts after `data:` followed by
/// an optional single U+0020. Both `data: payload` (with space) and
/// `data:payload` (without space) are valid — the space is stripped when
/// present. This mirrors Goose's `strip_data_prefix` implementation.
fn strip_data_prefix(line: &str) -> Option<&str> {
    line.strip_prefix("data: ")
        .or_else(|| line.strip_prefix("data:"))
        .map(|s| s.trim())
}

/// Byte offset just past the first SSE event separator in `buf` — the blank
/// line between events, written as either LF LF ("\n\n") or CRLF CRLF
/// ("\r\n\r\n"). The two patterns never overlap (CRLF contains no "\n\n"
/// and vice versa), so taking the earliest match is unambiguous.
fn next_sse_separator(buf: &[u8]) -> Option<usize> {
    let lf = buf.windows(2).position(|w| w == b"\n\n");
    let crlf = buf.windows(4).position(|w| w == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(a), Some(b)) => Some((a + 2).min(b + 4)),
        (Some(a), None) => Some(a + 2),
        (None, Some(b)) => Some(b + 4),
        (None, None) => None,
    }
}

pub async fn send_request(
    config: &ProviderConfig,
    url: &str,
    body: &(impl serde::Serialize + Sync),
    headers: &[(&str, String)],
) -> Result<String, ConnectorError> {
    let response = send_request_stream(config, url, body, headers).await?;
    // Non-streaming call: the whole body must arrive within the request
    // timeout (the stream path governs the body via per-chunk timeouts).
    let text = tokio::time::timeout(Duration::from_mins(1), response.text())
        .await
        .map_err(|_| ConnectorError::Network("request timed out after 60s".to_string()))??;
    Ok(text)
}

pub async fn send_get_request(
    config: &ProviderConfig,
    url: &str,
    headers: &[(&str, String)],
) -> Result<String, ConnectorError> {
    let mut request_builder = apply_provider_headers(
        shared_client().get(url).timeout(Duration::from_mins(1)),
        config,
    );

    for (key, value) in headers {
        request_builder = request_builder.header(*key, value.as_str());
    }

    let response = request_builder.send().await?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error response".to_string());
        let status = status.as_u16();
        let error_msg = format!("HTTP {status} - {error_text}");
        log::error!("HTTP Error captured: {error_msg}");
        return Err(ConnectorError::classify_http(status, error_text));
    }

    Ok(response.text().await?)
}

pub(crate) async fn send_request_stream(
    config: &ProviderConfig,
    url: &str,
    body: &(impl serde::Serialize + Sync),
    headers: &[(&str, String)],
) -> Result<reqwest::Response, ConnectorError> {
    let mut request_builder = apply_provider_headers(
        shared_client()
            .post(url)
            .header("Content-Type", "application/json"),
        config,
    );

    for (key, value) in headers {
        request_builder = request_builder.header(*key, value.as_str());
    }

    let json_body = serde_json::to_string(body)?;
    send_builder(request_builder, json_body).await
}

/// Attach cache-affinity session headers (`x-session-id` /
/// `x-session-affinity`, same opaque value) to a request builder. No-op when
/// no session id is configured.
pub(crate) fn apply_session_headers(
    request_builder: reqwest::RequestBuilder,
    session: Option<&str>,
) -> reqwest::RequestBuilder {
    match session {
        Some(id) => request_builder
            .header("x-session-id", id)
            .header("x-session-affinity", id),
        None => request_builder,
    }
}

/// Build the provider-extra headers for the given provider — shared by the
/// plain send paths and the retry wrapper. Two concerns live here:
/// - OpenRouter's required `HTTP-Referer` / `X-Title` identification;
/// - the OpenCode gateways' honest `User-Agent` (documented Zen, Go):
///   co-sh identifies itself as the calling client instead of imitating
///   another client's identity.
pub(crate) fn apply_provider_headers(
    request_builder: reqwest::RequestBuilder,
    config: &ProviderConfig,
) -> reqwest::RequestBuilder {
    let request_builder = if is_opencode_gateway(config.name) {
        request_builder.header("User-Agent", ZEN_USER_AGENT)
    } else {
        request_builder
    };
    if config.needs_extra_headers {
        request_builder
            .header("HTTP-Referer", "https://localhost")
            .header("X-Title", "provider")
    } else {
        request_builder
    }
}

/// Execute a fully-configured [`reqwest::RequestBuilder`] (URL, headers,
/// provider extras, serialized JSON body attached) once, mapping non-2xx
/// responses to a classified [`ConnectorError`] with the server's retry hint
/// attached when one was sent. The builder is cloned per attempt so the
/// original stays reusable across retries.
async fn send_builder(
    request_builder: reqwest::RequestBuilder,
    json_body: String,
) -> Result<reqwest::Response, ConnectorError> {
    let response = tokio::time::timeout(
        Duration::from_mins(1),
        request_builder.body(json_body).send(),
    )
    .await
    .map_err(|_| ConnectorError::Network("request timed out after 60s".to_string()))?
    .map_err(ConnectorError::from)?;

    let status = response.status();
    if !status.is_success() {
        // Capture the retry hint BEFORE consuming the body (the retry
        // middleware paces rate-limit retries with the server's own
        // `retry-after` / `retry-after-ms` when present).
        let retry_after_ms = parse_retry_after_ms(response.headers());
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unable to read error response".to_string());
        let status = status.as_u16();
        let error_msg = format!("HTTP {status} - {error_text}");
        log::error!("HTTP Error captured: {error_msg}");
        let mut err = ConnectorError::classify_http(status, error_text);
        if let ConnectorError::HttpError {
            retry_after_ms: slot,
            ..
        } = &mut err
        {
            *slot = retry_after_ms;
        }
        return Err(err);
    }

    Ok(response)
}

/// Retry wrapper around a single chat-stream request, mirroring the
/// crush/fantasy semantics (see [`crate::connector::retry`]): a retryable
/// failure — rate limit (429), HTTP 5xx, network/transport error — is
/// re-sent with exponential backoff (honoring the server's `retry-after`
/// hint) up to [`RETRY_MAX_RETRIES`]. When a mid-stream failure happens
/// AFTER partial content was already yielded, a [`StreamChunk::reset`]
/// marker is emitted first so the consumer discards the failed attempt's
/// buffered text (the retried response restarts from the beginning).
///
/// `request_builder` must already carry the URL, headers and provider
/// extras; the (already serialized) `json_body` is re-attached per attempt.
/// `retry_delay_override` (usually `None`) replaces the production
/// 5s → 10s → 20s backoff — tests pass a tiny delay so retries run in
/// milliseconds.
pub async fn send_with_retry(
    request_builder: &reqwest::RequestBuilder,
    json_body: String,
    retry_delay_override: Option<Duration>,
    max_retries: Option<usize>,
) -> Result<reqwest::Response, ConnectorError> {
    use super::retry::{RETRY_MAX_RETRIES, is_retryable_error, jittered, retry_delay};

    let mut attempts = 0usize;
    loop {
        attempts += 1;
        let Some(builder) = request_builder.try_clone() else {
            return Err(ConnectorError::Network(
                "request builder not clonable".to_string(),
            ));
        };
        match send_builder(builder, json_body.clone()).await {
            Err(e)
                if attempts <= max_retries.unwrap_or(RETRY_MAX_RETRIES)
                    && is_retryable_error(&e) =>
            {
                log::warn!("retry #{attempts} after request error: {e}");
                let delay =
                    retry_delay_override.unwrap_or_else(|| jittered(retry_delay(&e, attempts)));
                tokio::time::sleep(delay).await;
            }
            other => return other,
        }
    }
}

/// Wrap the SSE stream of an ALREADY-SUCCESSFUL response in mid-stream
/// retry handling: if the stream fails with a retryable error (connection
/// cut, 5xx chunk), the request is re-sent (with backoff, honoring
/// `retry_delay_override`) and the response re-parsed. When partial content
/// was already yielded, a [`StreamChunk::reset`] marker is emitted first so
/// the consumer discards the failed attempt's text — the retried response
/// restarts from the beginning.
///
/// The FIRST request stays eager (performed by [`send_with_retry`] before
/// this wrapper runs) so connect-phase failures surface from the call, not
/// the first poll.
pub fn retry_mid_stream(
    initial_response: reqwest::Response,
    request_builder: reqwest::RequestBuilder,
    json_body: String,
    retry_delay_override: Option<Duration>,
    max_retries: Option<usize>,
    parse: impl Fn(
        reqwest::Response,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>>
    + Send
    + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, ConnectorError>> + Send>> {
    use super::retry::{RETRY_MAX_RETRIES, is_retryable_error, jittered, retry_delay};
    use tokio_stream::StreamExt;

    Box::pin(stream! {
        let mut response = initial_response;
        let mut attempts = 1usize;
        let mut emitted_any = false;
        'stream: loop {
            let mut inner = parse(response);
            while let Some(item) = inner.next().await {
                match item {
                    Ok(chunk) => {
                        emitted_any = true;
                        yield Ok(chunk);
                    }
                    Err(e)
                    if attempts
                        <= max_retries.unwrap_or(RETRY_MAX_RETRIES)
                        && is_retryable_error(&e) =>
                {
                        log::warn!("retry #{attempts} after mid-stream error: {e}");
                        if emitted_any {
                            // The consumer must drop the partial content of
                            // the failed attempt; the retried response
                            // restarts from the beginning.
                            yield Ok(StreamChunk::reset());
                        }
                        // Re-request (with its own request-phase retries).
                        loop {
                            attempts += 1;
                            let delay = retry_delay_override
                                .unwrap_or_else(|| {
                                    jittered(retry_delay(&e, attempts))
                                });
                            tokio::time::sleep(delay).await;
                            let Some(builder) = request_builder.try_clone() else {
                                yield Err(ConnectorError::Network(
                                    "request builder not clonable".to_string(),
                                ));
                                return;
                            };
                            match send_builder(builder, json_body.clone()).await {
                                Ok(r) => {
                                    response = r;
                                    continue 'stream;
                                }
                                Err(e2)
                                    if attempts
                                        <= max_retries.unwrap_or(RETRY_MAX_RETRIES)
                                        && is_retryable_error(&e2) =>
                                {
                                    log::warn!("retry #{attempts} after re-request error: {e2}");
                                }
                                Err(e2) => {
                                    yield Err(e2);
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            // The parse stream ended cleanly (finish reason / [DONE] / EOF
            // without a retryable error): the response completed.
            return;
        }
    })
}

/// Parse the server's retry hint from response headers: `retry-after-ms`
/// (milliseconds, used e.g. by OpenAI) or `retry-after` (seconds or an HTTP
/// date). Mirrors the crush/fantasy sanity bounds: only a delay in
/// (0, 60s] is honored — a hint outside that range is ignored in favor of
/// the exponential backoff.
fn parse_retry_after_ms(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers
        .get("retry-after-ms")
        .or_else(|| headers.get("retry-after"))?
        .to_str()
        .ok()?
        .trim()
        .to_ascii_lowercase();
    let ms = raw.parse::<u64>().ok().map(|secs| {
        if headers.contains_key("retry-after-ms") {
            secs
        } else {
            secs.saturating_mul(1000)
        }
    });
    // `retry-after-ms` may itself be fractional ("1500.5") — fall back to a
    // whole-second parse for the plain `retry-after` case.
    let ms = ms.or_else(|| {
        let secs = raw.parse::<f64>().ok()?;
        Some((secs * 1000.0) as u64)
    });
    let ms = ms?;
    if ms > 0 && ms <= 60_000 {
        Some(ms)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{SseBuffer, next_sse_separator};

    #[test]
    fn splits_lf_frames() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    /// The Gemini API emits CRLF CRLF between SSE events — the exact wire
    /// format captured from a live `streamGenerateContent?alt=sse` call.
    /// A parser that only splits on LF LF would buffer the whole body and
    /// emit zero frames (the silent-failure bug).
    #[test]
    fn splits_crlf_frames() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\r\n\r\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    #[test]
    fn buffers_partial_frame_until_separator() {
        let mut buf = SseBuffer::new();
        // First chunk ends mid-frame (no blank line yet).
        assert!(buf.push_and_drain(b"data: {\"a\":1}\r\n").is_empty());
        // The rest completes the frame and carries the next one.
        let frames = buf.push_and_drain(b"\r\ndata: {\"b\":2}\r\n\r\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    #[test]
    fn mixed_separator_styles_are_both_split() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data: {\"a\":1}\n\ndata: {\"b\":2}\r\n\r\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    #[test]
    fn separator_offsets_are_correct() {
        assert_eq!(
            next_sse_separator(b"data: x\n\n"),
            Some(b"data: x\n\n".len())
        );
        assert_eq!(
            next_sse_separator(b"data: x\r\n\r\n"),
            Some(b"data: x\r\n\r\n".len())
        );
        assert_eq!(next_sse_separator(b"data: x"), None);
    }

    /// A stream that ends without a trailing blank line must still deliver
    /// its final frame at EOF — the "silently lost tail frame" class of
    /// failure (same family as the CRLF separator bug).
    #[test]
    fn flush_emits_final_frame_without_separator() {
        let mut buf = SseBuffer::new();
        // First frame ends cleanly; the second never gets its blank line,
        // so it must be recovered by flush() at EOF.
        let frames = buf.push_and_drain(b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}");
        assert_eq!(frames, vec![r#"{"a":1}"#]);
        assert_eq!(buf.flush(), vec![r#"{"b":2}"#]);
        assert!(buf.flush().is_empty());
    }

    #[test]
    fn flush_is_noop_when_stream_ended_cleanly() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data: {\"a\":1}\n\n");
        assert_eq!(frames, vec![r#"{"a":1}"#]);
        assert!(buf.flush().is_empty());
    }

    #[test]
    fn flush_recovers_multiple_unseparated_data_lines() {
        // Server that never emits blank lines (NDJSON-ish): every `data:`
        // line still becomes a frame at EOF.
        let mut buf = SseBuffer::new();
        assert!(
            buf.push_and_drain(b"data: {\"a\":1}\ndata: {\"b\":2}")
                .is_empty()
        );
        assert_eq!(buf.flush(), vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    // ── strip_data_prefix tests (SSE spec compliance) ─────────────────

    #[test]
    fn strip_data_prefix_with_space() {
        assert_eq!(
            super::strip_data_prefix("data: {\"a\":1}"),
            Some(r#"{"a":1}"#)
        );
    }

    #[test]
    fn strip_data_prefix_without_space() {
        // SSE spec: space after colon is optional.
        assert_eq!(
            super::strip_data_prefix("data:{\"a\":1}"),
            Some(r#"{"a":1}"#)
        );
    }

    #[test]
    fn strip_data_prefix_non_data_line() {
        assert_eq!(super::strip_data_prefix("event: message"), None);
        assert_eq!(super::strip_data_prefix("id: 123"), None);
        assert_eq!(super::strip_data_prefix(""), None);
    }

    #[test]
    fn strip_data_prefix_trims_whitespace() {
        assert_eq!(
            super::strip_data_prefix("data:  {\"a\":1}  "),
            Some(r#"{"a":1}"#)
        );
    }

    #[test]
    fn strip_data_prefix_empty_value() {
        // SSE spec: empty data field is valid (event dispatch without data).
        assert_eq!(super::strip_data_prefix("data:"), Some(""));
        assert_eq!(super::strip_data_prefix("data: "), Some(""));
    }

    // ── SseBuffer with no-space data prefix ────────────────────────────

    #[test]
    fn pushes_no_space_data_prefix() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data:{\"a\":1}\n\ndata:{\"b\":2}\n\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }

    #[test]
    fn mixed_space_and_no_space_prefix() {
        let mut buf = SseBuffer::new();
        let frames = buf.push_and_drain(b"data: {\"a\":1}\n\ndata:{\"b\":2}\n\n");
        assert_eq!(frames, vec![r#"{"a":1}"#, r#"{"b":2}"#]);
    }
}
