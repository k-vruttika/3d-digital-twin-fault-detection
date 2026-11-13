# ADR 0017: Frontend API types generated from the backend, including WebSocket frames

- **Status:** Accepted
- **Date:** 2026-11-14
- **Phase:** 5 (3D twin engine, frontend)

## Context

The frontend consumes the REST API and the `/stream` WebSocket. Hand-written TypeScript types would drift from the Pydantic models. The WebSocket protocol was documented by hand only: the hub built its control frames (auth, subscribed, snapshot, error, dropped, pong) as ad-hoc dicts.

## Decision

- **WebSocket frames are Pydantic models** (`backend/twin_backend/api/ws_models.py`):
  - client frames `auth`, `subscribe` and `ping`;
  - server frames `auth`, `subscribed`, `snapshot`, `reading`, `state`, `line`, `dataQuality`, `alert`, `command`, `ack`, `prediction`, `dropped`, `error` and `pong`;
  - they form discriminated unions on `type`: `WsClientMessage` and `WsServerMessage`.
- **The hub uses the models:** it builds its control frames through them. Streamed frames keep their single serialisation per message (the contract model inside the `{type, data}` envelope).
- **OpenAPI carries them:** a custom `app.openapi()` merges both unions into `components.schemas`. A model also used by a REST route must give the same schema in both places (FastAPI drops `"default": null`, which is normalised), otherwise schema generation fails.
- **Committed generated files:**
  - `uv run poe api-docs` writes `docs/api.md`, `frontend/src/api/openapi.json` (sorted keys, LF) and the `GET /line` fixtures for both shipped line configs (`frontend/src/engine/__fixtures__/`);
  - it then runs `openapi-typescript` (devDependency, pinned 7.13.0), which writes `frontend/src/api/schema.gen.ts`;
  - `frontend/src/api/types.ts` only aliases the generated types.
- **Staleness tests:**
  - pytest checks `openapi.json` and the fixtures against the running code;
  - Vitest regenerates `schema.gen.ts` from `openapi.json` in memory and compares;
  - the REST client's endpoint list is checked against `openapi.json`;
  - the existing WebSocket tests validate every frame they receive against `WsServerMessage`.

## Consequences

- A backend contract change breaks a test until `uv run poe api-docs` is run, and then usually breaks the TypeScript build where the frontend has to adapt.
- `openapi-typescript` 7.13 declares a peer dependency on TypeScript 5. An npm `overrides` entry lets it use the project's TypeScript 6. Generation works with it (checked in the staleness test). If a future TypeScript breaks it, the generator can be pinned to its own TypeScript instead.
- Fields with defaults are generated as required (openapi-typescript's `defaultNonNullable`), matching what the backend always sends.
- `schema.gen.ts` is excluded from ESLint (generated); `tsc` still type-checks it.

## Alternatives considered

- **Hand-written types with a key-set check:** possible, but every change would mean two edits.
- **A Python TypeScript generator:** no npm dependency, but more code to maintain than a standard tool.
- **A separate JSON Schema file for the WebSocket:** the OpenAPI components already hold the shared models (Reading, Alert, ...), so one schema avoids duplicate definitions.
