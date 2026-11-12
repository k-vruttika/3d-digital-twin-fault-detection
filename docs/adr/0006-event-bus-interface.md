# ADR 0006: In-process event bus behind a swappable interface

- **Status:** Accepted
- **Date:** 2026-11-12
- **Phase:** 2 (adapters and ingestion)

## Context

The ingest pipeline produces readings and state changes. Several consumers need them: the event store (Phase 3), the twin engine, the inference service and the WebSocket gateway (Phase 4). The whole backend runs in one process today. Later it may be split (for example, running inference as a separate worker), and then the bus would need to cross processes, e.g. through Redis.

## Decision

- **Interface** (`twin_backend.bus.base`):
  - `EventBus` Protocol with `publish(topic, message)`, `subscribe(topic) -> Subscription` and `close()`.
  - `Subscription` provides `get()`, `get_nowait()`, async iteration, `close()` and a `dropped` counter.
  - Topics are the `Topic` enum: `readings`, `states`, `predictions`, `commands`, `alerts`.
  - Messages are the Pydantic contract models from `twin_common` (`Reading`, `StateChange`, `Prediction`, `Command`, `CommandAck`). They serialize to JSON, so a networked implementation can carry them unchanged.
  - `publish` is `async`, even though the in-memory implementation never awaits anything, so a networked implementation can be dropped in without changing callers.
- **Implementation** (`InMemoryEventBus`):
  - Fan-out to every subscriber of a topic, with one bounded queue per subscriber (`bus.queue_size` in `sources.yaml`).
  - **Publishers never block.** A subscriber that falls behind loses its oldest messages, and `Subscription.dropped` counts them. A slow WebSocket client must not stall ingestion.
  - No persistence and no replay. Durability and replay belong to the event store (Phase 3), not the bus.
- **Usage:** the bus is created at application start and handed to the components (`SourceManager`, later the engine and gateway). Nothing imports a global bus.

## Consequences

- **Swapping to Redis** later means adding a `RedisEventBus`, e.g. with Redis Streams per topic. It would use JSON via `model_dump_json`, a consumer group per subscriber kind, and `MAXLEN` for the same bounded drop-oldest behaviour. Callers stay the same. Message ordering per topic is preserved by both implementations.
- **Losing messages on overflow is deliberate:**
  - Live consumers get the most recent data.
  - Consumers that must see everything have to read from the event store.
  - The `dropped` counter makes losses visible, and the API will expose it as a metric in Phase 4.
- **Message timing:** in-process delivery is synchronous fan-out into queues; consumers run when the event loop schedules them. Tests drain a subscription with `get_nowait()` and do not need to sleep.
