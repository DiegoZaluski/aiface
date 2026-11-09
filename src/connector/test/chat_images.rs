//! Integration tests for images the model embeds in a CHAT ANSWER
//! (multi-modal output) — the `ChatOutput::images` / `ChatStream::images`
//! channel, round-tripped through mock servers per family wire shape.

use super::super::Connector;
use super::common::{ENV_LOCK, EnvGuard, mock_server};

/// OpenRouter-style gateway (the `OpenAICompatible` normalization):
/// generated images ride `choices[].message.images[]` as data URIs.
#[tokio::test]
async fn openrouter_gateway_images_arrive_decoded() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENROUTER_API_KEY");
    let (port, _body, _raw, handle) = mock_server(
        r#"{"choices":[{"message":{"content":"here is your image","images":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}
        ]}}]}"#,
        200,
    );
    let out = Connector::new("openrouter")
        .unwrap()
        .with_model("google/gemini-2.5-flash-image")
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .chat("draw a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    assert_eq!(out.message(), "here is your image");
    assert_eq!(out.images().len(), 1);
    assert_eq!(out.images()[0].bytes(), b"hello");
    assert_eq!(out.images()[0].mime_type(), "image/png");
}

/// A text-only reply yields NO images — the common path pays nothing.
#[tokio::test]
async fn text_only_chat_answer_has_no_images() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENROUTER_API_KEY");
    let (port, _body, _raw, handle) =
        mock_server(r#"{"choices":[{"message":{"content":"just words"}}]}"#, 200);
    let out = Connector::new("openrouter")
        .unwrap()
        .with_base_url(format!("http://127.0.0.1:{port}/api/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .chat("hi")
        .await
        .unwrap();
    handle.join().unwrap();
    assert!(out.images().is_empty());
}

/// Gemini: an image-capable model answers with interleaved text/image
/// parts in the normal `generateContent` response.
#[tokio::test]
async fn gemini_chat_answer_images_arrive_decoded() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("GEMINI_API_KEY");
    let (port, _body, _raw, handle) = mock_server(
        r#"{"candidates":[{"finishReason":"STOP","content":{"parts":[
            {"text":"here it is"},
            {"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
        ]}}]}"#,
        200,
    );
    let out = Connector::new("gemini")
        .unwrap()
        .with_model("gemini-2.5-flash-image")
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("test-key")
        .with_retry(false)
        .chat("draw a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    assert_eq!(out.message(), "here it is");
    assert_eq!(out.images().len(), 1);
    assert_eq!(out.images()[0].bytes(), b"hello");
    assert_eq!(out.images()[0].mime_type(), "image/png");
}

/// OpenAI Responses API: the `image_generation` tool delivers its output
/// as `image_generation_call` items inside the same response.
#[tokio::test]
async fn openai_responses_tool_images_arrive_decoded() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("OPENAI_API_KEY");
    let (port, _body, _raw, handle) = mock_server(
        r#"{"output":[
            {"type":"message","content":[{"type":"output_text","text":"done"}]},
            {"type":"image_generation_call","result":"aGVsbG8=","output_format":"png"}
        ]}"#,
        200,
    );
    let out = Connector::new("openai")
        .unwrap()
        .with_model("gpt-5.2")
        .with_base_url(format!("http://127.0.0.1:{port}/v1"))
        .with_api_key("sk-test")
        .with_retry(false)
        .chat("draw a red panda")
        .await
        .unwrap();
    handle.join().unwrap();

    assert_eq!(out.message(), "done");
    assert_eq!(out.images().len(), 1);
    assert_eq!(out.images()[0].bytes(), b"hello");
    assert_eq!(out.images()[0].mime_type(), "image/png");
}

/// Streaming: images accumulate across frames in wire order and dedupe
/// (Gemini's final frame re-sends the parts earlier frames carried).
#[tokio::test]
async fn gemini_stream_accumulates_images_with_dedupe() {
    let _lock = ENV_LOCK.lock().await;
    let _guard = EnvGuard::remove("GEMINI_API_KEY");
    // Single-line JSON per frame: SSE frames cannot span lines (each line
    // after `data:` needs its own prefix, and this mock emulates the REAL
    // Gemini CRLF framing captured in test/gemini.rs).
    let part = r#"{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]}}]}"#;
    let final_frame = r#"{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"done"},{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}]}}]}"#;
    let sse = format!("data: {part}\r\n\r\ndata: {final_frame}\r\n\r\n");
    let (port, _body, _raw, handle) = mock_server(&sse, 200);
    let mut stream = Connector::new("gemini")
        .unwrap()
        .with_model("gemini-2.5-flash-image")
        .with_base_url(format!("http://127.0.0.1:{port}"))
        .with_api_key("test-key")
        .with_retry(false)
        .stream_chat("draw a red panda")
        .await
        .unwrap();
    let images = stream.images().await.to_vec();
    handle.join().unwrap();

    // The part arrives twice (stream frame + final frame) but decodes to
    // the same bytes — one image, not two.
    assert_eq!(images.len(), 1);
    assert_eq!(images[0].bytes(), b"hello");
    assert_eq!(images[0].mime_type(), "image/png");
}
