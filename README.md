# aiface

[![Crates.io](https://img.shields.io/crates/v/aiface.svg)](https://crates.io/crates/aiface)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A unified multi-provider LLM client for Rust — one [`Connector`], three
provider families, one interface.

`aiface` is the standalone connector crate of
[cosh](https://github.com/DiegoZaluski/co-sh)'s SDK: a single client that
speaks Claude, OpenAI, Gemini, GitHub Copilot and every OpenAI-compatible
endpoint (DeepSeek, Groq, xAI, Ollama, LM Studio, vLLM, …), with streaming
chat, tool calling, embeddings, token usage and cost tracking.

## Highlights

- **One `Connector`, 28 providers** — native adapters for Claude, OpenAI and
  Gemini, plus OpenAI-compatible endpoints. Swap the name passed to
  `Connector::new` to switch backends.
- **Streaming with mid-stream retry** — automatic retry on 429/5xx/network
  errors with backoff and `retry-after` support; a reset marker lets you
  discard partial text from a failed attempt.
- **Context-window intelligence** — model catalog with context-window
  discovery and `ContextWindowExceeded` classification, even from ambiguous
  SSE error bodies.
- **OAuth built in** — Authorization Code + PKCE, loopback redirect and
  Device Grant flows (RFC 7636 / 8252 / 8628), plus a GitHub Copilot
  device-flow adapter, with tokens stored in the OS keyring.
- **Multimodal** — image blocks travel inline (base64) on the message
  channel.

## Usage

```rust
use aiface::connector::Connector;

let reply = Connector::new("openai")?
    .with_model("gpt-4o")
    .with_temperature(0.7)
    .with_api_key("sk-...")
    .chat("Explain Rust ownership")
    .await?;
```

## Documentation

Guides for every module live in [`docs/`](docs/README.md):

`connector` (overview) · `client` (builder API) · `params` (request
vocabulary) · `output` (output types) · `discovery` (context windows) ·
`provider` (registry & keys) · `error` (classification & retry).

A runnable walkthrough lives in [`examples/client.rs`](examples/client.rs):

```bash
cargo run --example client
```

## Requirements

- Rust 2024 edition (≥ 1.85)

## License

Apache-2.0 — see [LICENSE](LICENSE).
