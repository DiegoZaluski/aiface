//! Integration tests for `Connector::generate_speech()` — Venice-only
//! text-to-speech over the OpenAI-shaped `/audio/speech` surface.
//!
//! The wire contract under test: Bearer-keyed POST to `{base}/audio/speech`
//! with `{model, input}` (no unset knobs), a RAW BINARY 200 body typed by
//! `Content-Type`, JSON error bodies on non-2xx, and the local gates
//! (empty/over-length input, non-Venice providers) that must fail BEFORE
//! any request leaves.

use super::super::Connector;
use super::common::{ENV_LOCK, EnvGuard, mock_server};

/// A fake WAV payload: the RIFF container magic plus the WAVE format
/// discriminator — valid UTF-8 so the text-mode mock server can serve it
/// verbatim, and exactly the bytes a real `audio/wav` response starts with.
const WAV_BYTES: &[u8] = b"RIFF\x24\x08\x00\x00WAVEfmt ";

/// Happy path: the request hits `/audio/speech` with the model knob and
/// NOTHING else (voice/response_format/speed stay off the wire), the binary
/// body comes back verbatim, and the MIME falls back to `audio/mpeg` (the
/// text mock always answers `Content-Type: application/json`, which does
/// not name audio — the documented fallback applies).
#[tokio::test]
async fn venice_synthesizes_speech() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let (port, body, raw, handle) = mock_server("RIFF\x24\x08\x00\x00WAVEfmt ", 200);
    let out = Connector::new("venice")
        .unwrap()
        .with_model("tts-kokoro")
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_speech("Hello from aiface.")
        .await
        .unwrap();
    handle.join().unwrap();

    let req = raw.lock().unwrap().take().unwrap().to_lowercase();
    assert!(
        req.contains("authorization: bearer sk-venice-test"),
        "Bearer key missing from speech request"
    );
    assert!(
        req.contains("post /api/v1/audio/speech"),
        "speech request did not hit /audio/speech: {req}"
    );

    let sent: serde_json::Value =
        serde_json::from_str(&body.lock().unwrap().clone().unwrap()).unwrap();
    assert_eq!(sent["model"], "tts-kokoro");
    assert_eq!(sent["input"], "Hello from aiface.");
    // No knobs yet: everything optional must stay OFF the wire.
    assert!(sent.get("voice").is_none());
    assert!(sent.get("response_format").is_none());
    assert!(sent.get("speed").is_none());
    assert!(sent.get("streaming").is_none());

    let clip = out.audio();
    assert_eq!(clip.bytes(), WAV_BYTES);
    assert_eq!(clip.mime_type(), "audio/mpeg");
    assert_eq!(clip.extension(), "mp3");
}

/// A 400 with a JSON error body classifies through the same HTTP-error
/// funnel as every other family call (`classify_http`).
#[tokio::test]
async fn http_400_classifies_as_http_error() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let (port, _body, _raw, handle) =
        mock_server(r#"{"error":{"message":"unsupported response_format","type":"invalid_request_error"}}"#, 400);
    let err = Connector::new("venice")
        .unwrap()
        .with_model("tts-kokoro")
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_speech("hello")
        .await
        .unwrap_err();
    handle.join().unwrap();
    assert!(
        err.to_string().contains("unsupported response_format"),
        "provider error body lost in classification: {err}"
    );
}

/// The Venice-only gate: every non-Venice provider returns `NotImplemented`
/// BEFORE any request — the error must be the gate itself, never an HTTP
/// round trip (pointed at a live mock no request may reach it; the handle
/// is dropped, not joined, exactly like the other no-request tests —
/// joining would hang the mock's accept loop forever).
#[tokio::test]
async fn non_venice_provider_is_not_implemented() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let (port, _body, _raw, _handle) = mock_server("binary", 200);
    let err = Connector::new("openai")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .generate_speech("hello")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("text-to-speech is supported only on venice"),
        "got: {err}"
    );
}

/// Empty input fails locally — the connector pointed at an unroutable port
/// proves no request was attempted (a Network error would mean the gate
/// leaked).
#[tokio::test]
async fn empty_input_fails_before_any_request() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let err = Connector::new("venice")
        .unwrap()
        .with_base_url("http://127.0.0.1:1/api/v1")
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_speech("   ")
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("nothing to synthesize"),
        "got: {err}"
    );
}

/// Over-length input (Venice caps TTS at 4096 chars) fails locally too —
/// same unroutable-endpoint proof.
#[tokio::test]
async fn oversize_input_fails_before_any_request() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("VENICE_API_KEY");
    let text = "a".repeat(4097);
    let err = Connector::new("venice")
        .unwrap()
        .with_base_url("http://127.0.0.1:1/api/v1")
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_speech(&text)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("4096"),
        "over-length error must name the cap: {err}"
    );
    // Exactly at the cap is fine locally (the request would fail on the
    // unroutable port with a NETWORK error, not InvalidInput).
    let at_cap = "a".repeat(4096);
    let err = Connector::new("venice")
        .unwrap()
        .with_base_url("http://127.0.0.1:1/api/v1")
        .with_api_key("sk-venice-test")
        .with_retry(false)
        .generate_speech(&at_cap)
        .await
        .unwrap_err();
    assert!(
        !err.to_string().contains("Invalid input"),
        "at-cap input must pass local validation, got: {err}"
    );
}
