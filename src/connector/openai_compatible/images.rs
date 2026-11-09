//! OpenAI `/v1/images/generations` image generation — the OpenAI-native
//! surface shared by the `openai` provider and every `OpenAICompatible`
//! gateway that ships the same route.
//!
//! Wire contract (platform.openai.com/docs/api-reference/images):
//! `POST {base}/images/generations`, JSON body `{model, prompt, n, size,
//! quality, response_format, ...}`; response `{created, data:
//! [{b64_json|url, revised_prompt}], usage?}`. Model classes differ:
//!
//! - DALL·E 2/3 honor `response_format` (default `url`, expiring in 60
//!   minutes) and return `revised_prompt`.
//! - GPT Image models ALWAYS return `b64_json`; sending `response_format`
//!   is a 400. They carry the `usage` token object.
//!
//! The codec therefore only ever requests `b64_json` (legacy models) and
//! parses URLs as data-URLs when a gateway inlines one — live http(s)
//! URLs are left in place and surface as a parse error, because fetching
//! arbitrary hosts from the connector's request path would turn a
//! generation call into an unauthenticated egress the caller never asked
//! for.

use super::super::common::send_request_retrying;
use super::super::error::ConnectorError;
use super::super::output::{GeneratedImage, ImageOutput};
use super::super::params::Parameters;
use super::super::provider::ProviderConfig;
use base64::Engine as _;
use super::caller::{resolve_api_key, resolve_base_url};

/// Request body for `POST /images/generations`. `skip_serializing_if`
/// throughout — the API rejects model-inappropriate params, so an unset
/// knob must stay OFF the wire entirely.
#[derive(serde::Serialize)]
struct ImageGenerationRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    /// Only sent when explicitly configured: GPT Image models reject the
    /// field, and their b64-only default is what this client wants anyway.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
}

/// One entry of the response `data` array: exactly one of `b64_json`/`url`
/// is populated depending on the model class.
#[derive(serde::Deserialize)]
struct ApiImage {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

#[derive(serde::Deserialize)]
struct ApiImagesResponse {
    #[allow(dead_code)]
    #[serde(default)]
    created: Option<serde_json::Value>,
    data: Vec<ApiImage>,
}

/// Generate an image through the OpenAI-shaped `/images/generations`
/// route.
///
/// Applies to every provider documented with this contract — OpenAI
/// itself, xAI, Together, Nebius, Venice's OpenAI-compat surface — and to
/// any `with_base_url` proxy that mirrors it. Model-specific parameters
/// (GPT Image quality tiers, DALL·E style, xAI aspect ratio) ride the
/// generic `image_size`/`image_quality` knobs; everything else stays off
/// the wire.
pub(crate) async fn generate_image(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    service: Option<&str>,
) -> Result<ImageOutput, ConnectorError> {
    let api_key = resolve_api_key(config, params, service)?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| "gpt-image-1".to_string());

    let request = ImageGenerationRequest {
        model,
        prompt: prompt.to_string(),
        n: params.image_count,
        size: params.image_size.clone(),
        quality: params.image_quality.clone(),
        // Only for the legacy DALL·E class, which defaults to a URL that
        // expires in 60 minutes — request base64 instead. Never set when
        // the model is a GPT Image model: the API rejects the field.
        response_format: params
            .image_response_format
            .clone()
            .or_else(|| legacy_b64_default(config, &params.model)),
    };
    let url = format!(
        "{}/images/generations",
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
    let response_text = send_request_retrying(
        config,
        &url,
        &request,
        &headers,
        params.retry_enabled,
        params.max_retries,
        params.retry_delay_override,
    )
    .await?;
    images_from_openai_json(&response_text)
        .map_err(|e| ConnectorError::Deserialization(format!("images response: {e}")))
}

/// DALL·E models default to `url` responses (links expire in 60 minutes)
/// — request base64 instead. GPT Image models reject the field, so they
/// never see it. Explicit `image_response_format` always wins.
fn legacy_b64_default(config: &ProviderConfig, model: &Option<String>) -> Option<String> {
    let effective = model
        .as_deref()
        .unwrap_or(config.default_model);
    (effective.starts_with("dall-e")).then(|| "b64_json".to_string())
}

/// Parse an OpenAI-shaped images response into decoded images.
///
/// Tolerant to the family's variations: nullable `created` (xAI omits
/// it), either `b64_json` or `url` per entry, and data-URI vs bare base64
/// payloads (xAI has shipped both spellings). A live http(s) `url` entry
/// is NOT fetched — see the module doc for why — and surfaces as a
/// deserialization error telling the caller to request `b64_json`.
pub(crate) fn images_from_openai_json(response_text: &str) -> Result<ImageOutput, String> {
    let parsed: ApiImagesResponse = serde_json::from_str(response_text)
        .map_err(|e| format!("invalid images response JSON: {e}"))?;
    if parsed.data.is_empty() {
        return Err("images response contained no data entries".to_string());
    }
    let mut images = Vec::with_capacity(parsed.data.len());
    for entry in &parsed.data {
        let (bytes, mime) = match (&entry.b64_json, &entry.url) {
            // b64_json wins when a gateway populates BOTH fields (the
            // documented precedence for the OpenAI family).
            (Some(b64), _) => (decode_b64(b64)?, sniff_mime(b64)),
            (None, Some(url)) if url.starts_with("data:") => {
                (decode_b64(url)?, sniff_mime(url))
            }
            (None, Some(url)) => {
                return Err(format!(
                    "images response returned a URL instead of image bytes ({url}); \
                     request response_format \"b64_json\" to get decodable bytes"
                ));
            }
            (None, None) => {
                return Err("images response entry had neither b64_json nor url".to_string());
            }
        };
        images.push(GeneratedImage::new(
            bytes,
            mime,
            entry.revised_prompt.clone(),
            response_text.to_string(),
        ));
    }
    Ok(ImageOutput::new(images))
}

/// Decode a base64 payload, tolerating the data-URI prefix
/// (`data:image/png;base64,…`) some gateways send in `url` fields.
fn decode_b64(payload: &str) -> Result<Vec<u8>, String> {
    let encoded = payload
        .split_once("base64,")
        .map_or(payload, |(_, rest)| rest)
        .trim();
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("image payload is not valid base64: {e}"))
}

/// Sniff the image MIME type from a base64 payload's magic bytes
/// (`data:` URIs carry it explicitly; bare payloads are decoded by their
/// leading bytes). Falls back to `image/png`, the family's dominant
/// output format.
fn sniff_mime(payload: &str) -> String {
    let encoded = payload
        .split_once("base64,")
        .map_or(payload, |(_, rest)| rest)
        .trim();
    // 16 base64 chars → 12 bytes, exactly the deepest magic check
    // (RIFF container family: 4-byte "RIFF" + 4-byte size + the format
    // discriminator at byte 8..12).
    let Some(prefix) = encoded.get(..16) else {
        return "image/png".to_string();
    };
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(prefix) else {
        return "image/png".to_string();
    };
    match bytes.as_slice() {
        [0x89, b'P', b'N', b'G', ..] => "image/png".to_string(),
        [0xFF, 0xD8, ..] => "image/jpeg".to_string(),
        // RIFF is a container family (WAV, AVI, WebP) — the WebP
        // discriminator lives at bytes 8..12, not in the first four.
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P'] => "image/webp".to_string(),
        [b'G', b'I', b'F', b'8', ..] => "image/gif".to_string(),
        _ => "image/png".to_string(),
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use crate::connector::Family;

    fn provider() -> ProviderConfig {
        ProviderConfig {
            name: "test-provider",
            family: Family::OpenAi,
            base_url: "https://example.invalid/v1",
            default_model: "gpt-image-1",
            needs_extra_headers: false,
            local: false,
            decisions_path: None,
            decisions_default_model: None,
        }
    }

    #[test]
    fn parses_b64_response() {
        // "hello" base64-encoded.
        let json = r#"{"created":1713833628,"data":[{"b64_json":"aGVsbG8="}]}"#;
        let out = images_from_openai_json(json).unwrap();
        assert_eq!(out.images().len(), 1);
        assert_eq!(out.first().unwrap().bytes(), b"hello");
        assert_eq!(out.first().unwrap().mime_type(), "image/png");
        assert_eq!(out.first().unwrap().extension(), "png");
    }

    #[test]
    fn parses_data_url_entry() {
        // A webp payload: "RIFF" magic sniffs through the data-URI prefix.
        let webp_b64 = base64::engine::general_purpose::STANDARD.encode(b"RIFFxxxxWEBPx");
        let json = format!(
            r#"{{"data":[{{"url":"data:image/webp;base64,{webp_b64}"}}]}}"#
        );
        let out = images_from_openai_json(&json).unwrap();
        assert_eq!(out.first().unwrap().mime_type(), "image/webp");
        assert_eq!(out.first().unwrap().extension(), "webp");
    }

    #[test]
    fn live_url_entry_is_an_error_not_a_fetch() {
        let json = r#"{"data":[{"url":"https://evil.example/img.png"}]}"#;
        let err = images_from_openai_json(json).unwrap_err();
        assert!(err.contains("b64_json"), "got: {err}");
    }

    #[test]
    fn jpeg_magic_sniffs_as_jpeg() {
        // 12+ bytes so the sniff window (16 base64 chars) is satisfied.
        let jpeg_b64 = base64::engine::general_purpose::STANDARD
            .encode([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01]);
        let json = format!(r#"{{"data":[{{"b64_json":"{jpeg_b64}"}}]}}"#);
        let out = images_from_openai_json(&json).unwrap();
        assert_eq!(out.first().unwrap().mime_type(), "image/jpeg");
        assert_eq!(out.first().unwrap().extension(), "jpg");
    }

    #[test]
    fn short_payloads_fall_back_to_png() {
        // 4 bytes (8 base64 chars) cannot satisfy the sniff window.
        let tiny_b64 = base64::engine::general_purpose::STANDARD.encode([0xFF, 0xD8, 0xFF, 0xE0]);
        let json = format!(r#"{{"data":[{{"b64_json":"{tiny_b64}"}}]}}"#);
        let out = images_from_openai_json(&json).unwrap();
        assert_eq!(out.first().unwrap().mime_type(), "image/png");
    }

    #[test]
    fn entry_without_payload_is_an_error() {
        let json = r#"{"data":[{"revised_prompt":"x"}]}"#;
        assert!(images_from_openai_json(json).is_err());
    }

    #[test]
    fn empty_data_is_an_error() {
        assert!(images_from_openai_json(r#"{"data":[]}"#).is_err());
    }

    #[test]
    fn revised_prompt_is_carried() {
        let json = r#"{"data":[{"b64_json":"aGVsbG8=","revised_prompt":"a better prompt"}]}"#;
        let out = images_from_openai_json(json).unwrap();
        assert_eq!(out.first().unwrap().revised_prompt(), Some("a better prompt"));
    }

    #[test]
    fn legacy_b64_default_only_for_dalle() {
        // Provider default (fixture: gpt-image-1) is NOT legacy → the
        // field stays off the wire.
        assert_eq!(legacy_b64_default(&provider(), &None), None);
        // An explicit DALL·E model flips the default to b64.
        assert_eq!(
            legacy_b64_default(&provider(), &Some("dall-e-3".into())),
            Some("b64_json".into())
        );
        // An explicit non-legacy model overrides a legacy provider default.
        let dalle_provider = ProviderConfig {
            default_model: "dall-e-3",
            ..provider()
        };
        assert_eq!(
            legacy_b64_default(&dalle_provider, &Some("gpt-image-1".into())),
            None
        );
    }

    #[test]
    fn request_omits_unset_fields() {
        // GPT Image models 400 on unknown/unset-model params; the request
        // body must carry only what was configured.
        let req = ImageGenerationRequest {
            model: "gpt-image-1".into(),
            prompt: "a cat".into(),
            n: None,
            size: None,
            quality: None,
            response_format: None,
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"model": "gpt-image-1", "prompt": "a cat"})
        );
    }
}
