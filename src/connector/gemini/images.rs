//! Gemini image generation via `models/{model}:generateContent` —
//! image-capable Gemini models (`gemini-2.5-flash-image` et al.) return
//! images as base64 `inlineData` parts inside the normal chat response.
//!
//! Wire contract (ai.google.dev/gemini-api/docs/image-generation):
//! `POST {base}/models/{model}:generateContent` with
//! `generationConfig.responseModalities: ["TEXT","IMAGE"]` (the 2.0 Flash
//! preview REQUIRES the pair; image-only `["IMAGE"]` errors) and optional
//! `generationConfig.imageConfig` (`aspectRatio`, `imageSize`).
//!
//! Imagen models (`imagen-3.0-generate-002`, `imagen-4.0-*`) travel a
//! DIFFERENT route (`:predict`, `instances`/`parameters`, paid tier only)
//! and are deliberately not wired: the shapes share nothing and the
//! caller can select an image-capable Gemini model explicitly.
//!
//! Two provider-specific safety states are surfaced as `Ok` with NO
//! image rather than an error — a withheld generation is a normal
//! answer, not a transport failure: `promptFeedback.blockReason` (no
//! candidates at all) and a `finishReason != "STOP"` candidate.

use super::super::common::send_request_retrying;
use super::super::error::ConnectorError;
use super::super::output::{GeneratedImage, ImageOutput};
use super::super::params::Parameters;
use super::super::provider::ProviderConfig;
use super::caller::{gemini_headers, resolve_base_url};
use super::super::provider::get_api_key;

/// `generationConfig` for an image-generation request. Serialized
/// untagged-style via serde with `skip_serializing_if` — fields the model
/// does not support must stay off the wire.
#[derive(serde::Serialize)]
struct ImageGenerationConfig {
    #[serde(rename = "responseModalities")]
    response_modalities: [&'static str; 2],
    #[serde(rename = "imageConfig", skip_serializing_if = "Option::is_none")]
    image_config: Option<ImageConfig>,
}

/// `generationConfig.imageConfig` — aspect ratio and output size. The
/// builder knobs map to the knobs this surface actually has:
///
/// - `image_size` → `aspectRatio` (`"16:9"`, `"1:1"`, …) — this surface
///   has no pixel-size field.
/// - `image_quality` → `imageSize` (`"1K"`, `"2K"`, `"4K"`) — Gemini's
///   output-resolution tier.
/// - `image_count` has NO equivalent (no `n` on this surface) and stays
///   off the wire.
#[derive(serde::Serialize)]
struct ImageConfig {
    #[serde(rename = "aspectRatio", skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
    #[serde(rename = "imageSize", skip_serializing_if = "Option::is_none")]
    image_size: Option<String>,
}

#[derive(serde::Serialize)]
struct ImageGenerateRequest {
    contents: [ImageContent; 1],
    #[serde(rename = "generationConfig")]
    generation_config: ImageGenerationConfig,
}

#[derive(serde::Serialize)]
struct ImageContent {
    parts: [ImagePart; 1],
}

#[derive(serde::Serialize)]
struct ImagePart {
    text: String,
}

/// A response `Part` — only `inlineData` is read; text parts are
/// intentionally ignored (the images are the payload this call asked
/// for) and serde skips unknown fields by default.
#[derive(serde::Deserialize)]
struct ResponsePart {
    #[serde(rename = "inlineData", default)]
    inline_data: Option<InlineResponseData>,
}

#[derive(serde::Deserialize)]
struct InlineResponseData {
    #[serde(rename = "mimeType", default)]
    mime_type: Option<String>,
    #[serde(default)]
    data: String,
}

#[derive(serde::Deserialize)]
struct ImageCandidate {
    #[serde(default)]
    content: Option<CandidateContent>,
    #[serde(default, rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<ResponsePart>,
}

#[derive(serde::Deserialize)]
struct ImageGenerateResponse {
    #[serde(default)]
    candidates: Vec<ImageCandidate>,
    #[serde(default, rename = "promptFeedback")]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(serde::Deserialize)]
struct PromptFeedback {
    #[serde(default, rename = "blockReason")]
    block_reason: Option<String>,
}

/// Generate images through the Gemini `generateContent` route.
///
/// The model must be image-capable (`gemini-2.5-flash-image`, the
/// `gemini-2.0-flash-preview-image-generation` preview) — requesting
/// image modalities from a text model is an API error, and the connector
/// does not second-guess the caller's model choice.
pub(crate) async fn generate_image(
    config: &ProviderConfig,
    params: &Parameters,
    prompt: &str,
    service: Option<&str>,
) -> Result<ImageOutput, ConnectorError> {
    let api_key = params
        .api_key
        .clone()
        .or_else(|| get_api_key(config.name, service))
        .ok_or_else(|| ConnectorError::MissingApiKey(config.name.to_string()))?;
    let model = params
        .model
        .clone()
        .unwrap_or_else(|| "gemini-2.5-flash-image".to_string());

    let request = ImageGenerateRequest {
        contents: [ImageContent {
            parts: [ImagePart {
                text: prompt.to_string(),
            }],
        }],
        generation_config: ImageGenerationConfig {
            // The pair is mandatory: image-only modality errors on the
            // 2.0 Flash preview, and the 2.5 Flash Image model answers
            // text-when-appropriate with the same setting.
            response_modalities: ["TEXT", "IMAGE"],
            image_config: if params.image_size.is_some() || params.image_quality.is_some() {
                Some(ImageConfig {
                    // `image_size` carries the provider's own vocabulary; the
                    // Gemini knob here is the aspect ratio ("16:9", …).
                    aspect_ratio: params.image_size.clone(),
                    // `image_quality` maps to Gemini's output-resolution
                    // tier ("1K"/"2K"/"4K").
                    image_size: params.image_quality.clone(),
                })
            } else {
                None
            },
        },
    };

    let base_url = resolve_base_url(config, params);
    let url = format!("{base_url}/models/{model}:generateContent");
    let response_text = send_request_retrying(
        config,
        &url,
        &request,
        &gemini_headers(api_key.as_str(), params.session_id.as_deref()),
        params.retry_enabled,
        params.max_retries,
        params.retry_delay_override,
    )
    .await?;
    images_from_gemini_json(&response_text)
        .map_err(|e| ConnectorError::Deserialization(format!("images response: {e}")))
}

/// Parse a `generateContent` response into decoded images.
///
/// Collects every `inlineData` part across all candidates (a multi-image
/// answer arrives as several parts) and base64-decodes each. Text parts
/// are ignored — the images are the payload this call asked for.
///
/// SAFETY states that yield no image are normal answers, not errors:
/// `promptFeedback.blockReason` (request blocked before any candidate)
/// and `finishReason != "STOP"` (e.g. `IMAGE_SAFETY`, `PROHIBITED_CONTENT`
/// — part or all of the generation withheld) return an EMPTY
/// [`ImageOutput`]. Callers distinguish via `images().is_empty()`.
pub(crate) fn images_from_gemini_json(response_text: &str) -> Result<ImageOutput, String> {
    let parsed: ImageGenerateResponse = serde_json::from_str(response_text)
        .map_err(|e| format!("invalid images response JSON: {e}"))?;

    // Request-level block: no candidates exist at all.
    if parsed.candidates.is_empty() {
        if parsed
            .prompt_feedback
            .and_then(|f| f.block_reason)
            .is_some()
        {
            return Ok(ImageOutput::new(Vec::new()));
        }
        return Err("images response contained no candidates".to_string());
    }

    let mut images = Vec::new();
    for candidate in &parsed.candidates {
        // A non-STOP finish reason means the image (or part of it) was
        // withheld — skip the candidate's parts rather than fail the
        // whole call.
        if candidate.finish_reason.as_deref().is_some_and(|r| r != "STOP") {
            continue;
        }
        let Some(content) = &candidate.content else {
            continue;
        };
        for part in &content.parts {
            if let Some(inline) = &part.inline_data {
                let bytes = decode_b64(&inline.data)?;
                let mime = inline
                    .mime_type
                    .clone()
                    .unwrap_or_else(|| "image/png".to_string());
                images.push(GeneratedImage::new(
                    bytes,
                    mime,
                    None,
                    response_text.to_string(),
                ));
            }
        }
    }
    Ok(ImageOutput::new(images))
}

/// Standard-base64 decode (Gemini payloads carry no data-URI prefix).
fn decode_b64(payload: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| format!("image payload is not valid base64: {e}"))
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn parses_inline_data_part() {
        // "hello" in standard base64.
        let json = r#"{"candidates":[{"content":{"parts":[
            {"text":"here is your image"},
            {"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
        ]}}]}"#;
        let out = images_from_gemini_json(json).unwrap();
        assert_eq!(out.images().len(), 1);
        assert_eq!(out.first().unwrap().bytes(), b"hello");
        assert_eq!(out.first().unwrap().mime_type(), "image/png");
    }

    #[test]
    fn multiple_images_arrive_as_multiple_parts() {
        let json = r#"{"candidates":[{"content":{"parts":[
            {"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}},
            {"inlineData":{"mimeType":"image/png","data":"d29ybGQ="}}
        ]}}]}"#;
        let out = images_from_gemini_json(json).unwrap();
        assert_eq!(out.images().len(), 2);
    }

    #[test]
    fn blocked_prompt_is_empty_output_not_error() {
        let json = r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#;
        let out = images_from_gemini_json(json).unwrap();
        assert!(out.images().is_empty());
    }

    #[test]
    fn withheld_finish_reason_skips_candidate() {
        let json = r#"{"candidates":[
            {"finishReason":"IMAGE_SAFETY","content":{"parts":[
                {"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]}}
        ]}"#;
        let out = images_from_gemini_json(json).unwrap();
        assert!(out.images().is_empty());
    }

    #[test]
    fn stop_candidate_with_image_parses() {
        let json = r#"{"candidates":[{"finishReason":"STOP","content":{"parts":[
            {"inlineData":{"mimeType":"image/jpeg","data":"aGVsbG8="}}]}}]}"#;
        let out = images_from_gemini_json(json).unwrap();
        assert_eq!(out.images().len(), 1);
        assert_eq!(out.first().unwrap().mime_type(), "image/jpeg");
    }

    #[test]
    fn no_candidates_no_block_is_an_error() {
        assert!(images_from_gemini_json(r#"{"candidates":[]}"#).is_err());
    }

    #[test]
    fn invalid_base64_is_an_error() {
        let json = r#"{"candidates":[{"content":{"parts":[
            {"inlineData":{"mimeType":"image/png","data":"!!!not base64!!!"}}]}}]}"#;
        assert!(images_from_gemini_json(json).is_err());
    }

    #[test]
    fn request_serializes_modalities_and_optional_config() {
        let req = ImageGenerateRequest {
            contents: [ImageContent {
                parts: [ImagePart { text: "a cat".into() }],
            }],
            generation_config: ImageGenerationConfig {
                response_modalities: ["TEXT", "IMAGE"],
                image_config: Some(ImageConfig {
                    aspect_ratio: Some("16:9".into()),
                    image_size: Some("2K".into()),
                }),
            },
        };
        let json: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT", "IMAGE"])
        );
        assert_eq!(
            json["generationConfig"]["imageConfig"]["aspectRatio"],
            serde_json::json!("16:9")
        );
        // No `n`/`size`/`quality` knobs exist on this surface.
        assert!(json.get("n").is_none());
    }
}
