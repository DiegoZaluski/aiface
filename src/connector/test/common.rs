//! Shared test infrastructure: mock HTTP server, env guard, connector builders.

use super::super::Connector;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// Saves and restores an env var within its scope.
pub struct EnvGuard {
    key: String,
    old: Option<String>,
}

impl EnvGuard {
    pub fn remove(key: &str) -> Self {
        let old = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self {
            key: key.to_owned(),
            old,
        }
    }
    pub fn set(key: &str, val: &str) -> Self {
        let old = std::env::var(key).ok();
        unsafe { std::env::set_var(key, val) };
        Self {
            key: key.to_owned(),
            old,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.old {
            Some(v) => unsafe { std::env::set_var(&self.key, v) },
            None => unsafe { std::env::remove_var(&self.key) },
        }
    }
}

/// Serializes tests that mutate process-wide environment variables.
/// Uses an async-aware mutex because the guard is held across awaits.
pub static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Handle to a running mock server.
///
/// `(port, captured_request_body, captured_raw_http, thread_handle)`.
pub type MockServerHandle = (
    u16,
    Arc<Mutex<Option<String>>>,
    Arc<Mutex<Option<String>>>,
    std::thread::JoinHandle<()>,
);

/// Starts a mock HTTP server that returns the given response.
pub fn mock_server(response: &str, status_code: u16) -> MockServerHandle {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = response.to_owned();
    let captured_body = Arc::new(Mutex::new(None::<String>));
    let captured_body_clone = Arc::clone(&captured_body);
    let captured_raw = Arc::new(Mutex::new(None::<String>));
    let captured_raw_clone = Arc::clone(&captured_raw);

    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();

        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];

        loop {
            let n = stream.read(&mut tmp).unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);

            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&buf[..pos]);
                let content_len: usize = headers
                    .lines()
                    .filter_map(|l| {
                        l.trim()
                            .to_lowercase()
                            .strip_prefix("content-length:")?
                            .trim()
                            .parse()
                            .ok()
                    })
                    .next()
                    .unwrap_or(0);
                let body_start = pos + 4;
                if buf.len() >= body_start + content_len {
                    let req_body =
                        String::from_utf8_lossy(&buf[body_start..body_start + content_len])
                            .to_string();
                    *captured_body_clone.lock().unwrap() = Some(req_body);
                    let raw = String::from_utf8_lossy(&buf[..body_start + content_len]).to_string();
                    *captured_raw_clone.lock().unwrap() = Some(raw);
                    break;
                }
            }
        }

        let status_line = if status_code == 200 {
            format!("HTTP/1.1 {status_code} OK")
        } else {
            format!("HTTP/1.1 {status_code} Error")
        };
        let resp = format!(
            "{status_line}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(resp.as_bytes());
    });

    (port, captured_body, captured_raw, handle)
}

/// A canned response for [`mock_server_sequence`]: a status code, an optional
/// body, and optional extra headers (e.g. `retry-after` to pace the retry).
pub struct MockResponse {
    pub status: u16,
    pub body: String,
    pub headers: Vec<(String, String)>,
}

impl MockResponse {
    pub fn new(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_owned(),
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.push((key.to_owned(), value.to_owned()));
        self
    }
}

/// How long the sequence server waits for the next request before exiting.
const IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1500);

/// Starts a mock HTTP server that serves the given responses in order, one
/// per connection — the retry tests need the server to survive a failed
/// first attempt and answer the retried request.
///
/// Returns `(port, request_count, thread_handle)`.
pub fn mock_server_sequence(
    responses: Vec<MockResponse>,
) -> (u16, Arc<Mutex<usize>>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let responses = Arc::new(Mutex::new(responses.into_iter()));
    let request_count = Arc::new(Mutex::new(0usize));
    let count_clone = Arc::clone(&request_count);

    let handle = std::thread::spawn(move || {
        // Poll accept with a bounded total wait so the thread exits even when
        // the client makes FEWER requests than the sequence (e.g. a
        // non-retried failure): after IDLE_TIMEOUT with no connection the
        // loop ends.
        let _ = listener.set_nonblocking(true);
        for response in responses.lock().unwrap().by_ref() {
            let deadline = std::time::Instant::now() + IDLE_TIMEOUT;
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(accepted) => break accepted,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if std::time::Instant::now() >= deadline {
                            return; // no next request in time — done serving
                        }
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(e) => panic!("mock accept error: {e}"),
                }
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));

            // Drain the request (headers + body) before answering.
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            loop {
                match stream.read(&mut tmp) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        buf.extend_from_slice(&tmp[..n]);
                        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let headers = String::from_utf8_lossy(&buf[..pos]);
                            let content_len: usize = headers
                                .lines()
                                .filter_map(|l| {
                                    l.trim()
                                        .to_lowercase()
                                        .strip_prefix("content-length:")?
                                        .trim()
                                        .parse()
                                        .ok()
                                })
                                .next()
                                .unwrap_or(0);
                            if buf.len() >= pos + 4 + content_len {
                                break;
                            }
                        }
                    }
                }
            }
            *count_clone.lock().unwrap() += 1;

            let status_line = if response.status == 200 {
                format!("HTTP/1.1 {} OK", response.status)
            } else {
                format!("HTTP/1.1 {} Error", response.status)
            };
            let extra: String = response
                .headers
                .iter()
                .map(|(k, v)| format!("{k}: {v}\r\n"))
                .collect();
            let resp = format!(
                "{status_line}\r\nContent-Length: {}\r\nContent-Type: application/json\r\n{extra}Connection: close\r\n\r\n{}",
                response.body.len(),
                response.body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });

    (port, request_count, handle)
}

/// Build a Connector pointed at a mock server on the given port.
///
/// Uses `groq` — a generic OpenAI-compatible cloud provider — so these tests
/// exercise the shared `/v1/chat/completions` path. The `openai` provider now
/// routes to the Responses API and is covered separately via
/// [`openai_connector`].
pub fn connector(port: u16) -> Connector {
    Connector::new("groq")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
}

/// Build an OpenAI (Responses API) Connector pointed at a mock server on the
/// given port. Used by the `test::openai` module to exercise `/v1/responses`.
pub fn openai_connector(port: u16) -> Connector {
    Connector::new("openai")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
}

/// Build a Claude Connector pointed at a mock server on the given port.
pub fn claude_connector(port: u16) -> Connector {
    Connector::new("claude")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("sk-ant-test")
}

/// Like `connector()` but without an API key — for testing env var fallback.
pub fn connector_no_key(port: u16) -> Connector {
    Connector::new("groq")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
}

/// Like `claude_connector()` but without an API key — for testing env var fallback.
pub fn claude_connector_no_key(port: u16) -> Connector {
    Connector::new("claude")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}"))
}

/// Build a DeepSeek Connector pointed at a mock server on the given port.
pub fn deepseek_connector(port: u16) -> Connector {
    Connector::new("deepseek")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-deepseek-test")
}

/// Build a Kimi (Moonshot) Connector pointed at a mock server on the given
/// port — exercises the same OpenAI-compatible caller as the real
/// `api.moonshot.ai/v1` endpoint.
pub fn kimi_connector(port: u16) -> Connector {
    Connector::new("kimi")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-kimi-test")
}

/// Build a Gemini Connector pointed at a mock server on the given port.
pub fn gemini_connector(port: u16) -> Connector {
    Connector::new("gemini")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("gk-test")
}
