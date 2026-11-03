# Aiface documentation

Module guides for the `aiface` crate — a unified multi-provider LLM client.

## Map

| Document | Scope |
|---|---|
| [`connector.md`](connector.md) | Overview: one client, three provider families |
| [`client.md`](client.md) | The `Connector` builder API: chat, streaming, embeddings |
| [`params.md`](params.md) | Request vocabulary: messages, tools, response formats |
| [`output.md`](output.md) | Output types: `ChatOutput`, streams, model listing |
| [`discovery.md`](discovery.md) | Context-window discovery and reasoning capability detection |
| [`provider.md`](provider.md) | Provider registry, API-key resolution, keyring, gateways |
| [`error.md`](error.md) | `ConnectorError` classification and retry semantics |

A runnable walkthrough lives in [`examples/client.rs`](../examples/client.rs).
