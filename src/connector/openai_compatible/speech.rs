//! Venice `/audio/speech` text-to-speech — the connector's ONLY speech
//! surface, and deliberately Venice-only.
//!
//! Wire contract (docs.venice.ai — OpenAI-compatible endpoint):
//! `POST {base}/audio/speech`, JSON body `{model, input, voice?,
//! response_format?, speed?}`; a SUCCESS response is a RAW BINARY audio
//! body (never JSON) whose `Content-Type` names the format (`audio/mpeg`,
//! `audio/wav`, `audio/opus`, `audio/aac`, `audio/flac`, `audio/pcm`); an
//! ERROR response (4xx/5xx) is a JSON error body, classified like every
//! other non-2xx in the family.
//!
//! Three wire quirks shape the code:
//!
//! - `response_format` support is MODEL-specific (Venice's model catalog
//!   lists each TTS model's `supported_formats`; an unsupported format is
//!   an HTTP 400). The connector sends no format knob, so the model's
//!   documented default (`tts-kokoro`: mp3) is what comes back.
//! - Every supported model caps `input` at 4096 characters. The check runs
//!   LOCALLY, before any request: an oversized input is a caller bug, not
//!   a server round trip.
//! - Streaming (`streaming: true`) is not implemented — every request is
//!   single-shot. And per the crate's no-egress policy nothing is ever
//!   fetched: the audio bytes arrive inline in this response.

use super::super::common::send_request_retrying_bytes;
use super::super::error::ConnectorError;
use super::super::output::{AudioOutput, GeneratedAudio};
use super::super::params::Parameters;
use super::super::provider::ProviderConfig;
use super::caller::{resolve_api_key, resolve_base_url};

/// Venice caps TTS input at 4096 characters (docs.venice.ai). Enforced
/// locally so a rejected-before-send error is cheap and deterministic.
const MAX_INPUT_CHARS: usize = 4096;

/// Default TTS model when none is configured — Venice's documented
/// default, the cheapest broadly-supported voice model on the catalog.
const DEFAULT_TTS_MODEL: &str = "tts-kokoro";

/// Request body for `POST /audio/speech`. `skip_serializing_if`
/// throughout — the API rejects model-inappropriate params, so an unset
/// knob must stay OFF the wire entirely (same reason as
/// `ImageGenerationRequest`). `voice`/`response_format`/`speed` have no
/// connector knobs yet: they are modeled because the wire contract
/// carries them, and they stay unset until a knob exists.
#[derive(serde::Serialize)]
struct SpeechRequest {
    model: String,
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
}

/// Synthesize speech through Venice's OpenAI-shaped `/audio/speech`
/// route.
///
/// The model follows `params.model` (`tts-kokoro` when unset); voice,
/// response format and speed are not knobbed yet and stay off the wire —
/// the model's documented defaults apply. Returns the audio bytes with
/// their MIME type resolved from the response `Content-Type`.
pub(crate) async fn generate_speech(
    config: &ProviderConfig,
    params: &Parameters,
    text: &str,
    service: Option<&str>,
) -> Result<AudioOutput, ConnectorError> {
    // Local validation BEFORE any request: an empty prompt has nothing to
    // synthesize, and Venice hard-rejects over-length input with a 400 —
    // both are caller bugs that must not spend a round trip.
    if text.trim().is_empty() {
        return Err(ConnectorError::InvalidInput(
            "speech input is empty (nothing to synthesize)".to_string(),
        ));
    }
    let chars = text.chars().count();
    if chars > MAX_INPUT_CHARS {
        return Err(ConnectorError::InvalidInput(format!(
            "speech input is {chars} characters (Venice caps TTS input at {MAX_INPUT_CHARS})"
        )));
    }

    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| DEFAULT_TTS_MODEL.to_string());
    let request = SpeechRequest {
        model,
        input: text.to_string(),
        voice: None,
        response_format: None,
        speed: None,
    };
    let url = format!(
        "{}/audio/speech",
        resolve_base_url(params, config).trim_end_matches('/')
    );

    let auth = api_key.as_ref().map(|key| format!("Bearer {key}"));
    let mut headers: Vec<(&str, String)> = Vec::new();
    if let Some(sid) = params.session_id.as_deref() {
        headers.push(("x-session-id", sid.to_string()));
        headers.push(("x-session-affinity", sid.to_string()));
    }
    if let Some(a) = &auth {
        headers.push(("Authorization", a.to_string()));
    }

    let (bytes, content_type) = send_request_retrying_bytes(
        config,
        &url,
        &request,
        &headers,
        params.retry_enabled,
        params.max_retries,
        params.retry_delay_override,
    )
    .await?;
    if bytes.is_empty() {
        // A 200 with a zero-byte body carries no audio at all — surface it
        // instead of handing back a silently empty clip.
        return Err(ConnectorError::Deserialization(
            "speech response contained no audio bytes".to_string(),
        ));
    }
    let mime = audio_mime(content_type.as_deref(), request.response_format.as_deref());
    Ok(AudioOutput::new(GeneratedAudio::new(bytes, mime)))
}

/// Resolve the audio MIME type for the response bytes.
///
/// The response `Content-Type` is authoritative WHEN it actually names an
/// audio type (`audio/mpeg`, `audio/wav`, ...); anything else — a gateway
/// answering `application/json` or `application/octet-stream` on a
/// success — falls back to the requested format, then to `audio/mpeg`
/// (Venice `tts-kokoro`'s documented default).
fn audio_mime(content_type: Option<&str>, response_format: Option<&str>) -> String {
    content_type
        .map(|ct| ct.split(';').next().unwrap_or(ct).trim().to_ascii_lowercase())
        .filter(|ct| ct.starts_with("audio/"))
        .or_else(|| {
            response_format.map(|fmt| match fmt {
                "wav" => "audio/wav".to_string(),
                "pcm" => "audio/pcm".to_string(),
                "opus" => "audio/opus".to_string(),
                "aac" => "audio/aac".to_string(),
                "flac" => "audio/flac".to_string(),
                // "mp3" and any unknown spelling fall back to the family's
                // dominant type rather than a guess.
                _ => "audio/mpeg".to_string(),
            })
        })
        .unwrap_or_else(|| "audio/mpeg".to_string())
}

#[cfg(test)]
mod speech_tests {
    use super::*;

    #[test]
    fn request_omits_unset_fields() {
        // Venice rejects model-inappropriate params; the request body must
        // carry only what was configured (here: model + input only).
        let req = SpeechRequest {
            model: "tts-kokoro".into(),
            input: "hello".into(),
            voice: None,
            response_format: None,
            speed: None,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"model": "tts-kokoro", "input": "hello"})
        );
    }

    #[test]
    fn header_content_type_wins_when_it_names_audio() {
        // A charset suffix is stripped; the value is normalized.
        assert_eq!(
            audio_mime(Some("Audio/Wav; charset=binary"), None),
            "audio/wav"
        );
        assert_eq!(audio_mime(Some("audio/opus"), Some("wav")), "audio/opus");
    }

    #[test]
    fn non_audio_content_type_falls_back_to_requested_format() {
        // Gateways answering application/json (the test mocks do) or
        // octet-stream must not poison the MIME.
        assert_eq!(audio_mime(Some("application/json"), Some("wav")), "audio/wav");
        assert_eq!(
            audio_mime(Some("application/octet-stream"), None),
            "audio/mpeg"
        );
        assert_eq!(audio_mime(None, None), "audio/mpeg");
        // Unknown format spellings fall back to the default, not a guess.
        assert_eq!(audio_mime(None, Some("whatever")), "audio/mpeg");
    }

    #[test]
    fn audio_extension_mapping() {
        // (mime, extension) — every Venice response format plus the fallback.
        for (mime, ext) in [
            ("audio/mpeg", "mp3"),
            ("audio/wav", "wav"),
            ("audio/pcm", "pcm"),
            ("audio/opus", "opus"),
            ("audio/aac", "aac"),
            ("audio/flac", "flac"),
            ("something/else", "mp3"),
        ] {
            let clip = GeneratedAudio::new(Vec::new(), mime.to_string());
            assert_eq!(clip.extension(), ext, "mime {mime}");
        }
    }

    #[test]
    fn input_limits_are_enforced_locally() {
        // The validation lives in `generate_speech` (async, network-bound);
        // here only the constants' relationship is pinned so a limit change
        // is a conscious one.
        assert_eq!(MAX_INPUT_CHARS, 4096);
    }
}
