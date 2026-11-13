# ADR 0014: Command path: idempotency, safety rules (STOP fail-safe), audit trail

- **Status:** Accepted
- **Date:** 2026-11-13
- **Phase:** 4 (twin engine, API, commands)

## Context

Operators send commands (START, STOP, SET_SPEED, SET_LOAD, SCHEDULE_MAINTENANCE, RESET) back to the machines through whichever adapter is active. A command must be validated, must not run twice when a client retries, must be refused when it is unsafe, must give the operator a clear outcome, and must leave a complete audit trail.

## Decision

`POST /commands` (operator) takes `{machineId, type, params, idempotencyKey}`. The server sets `commandId`, `issuedBy` (the key's name) and `ts`. `CommandService.submit` then works in this order:

1. **Contract validation:** the `Command` Pydantic contract, for example SET_SPEED needs a numeric `speed` (422). An unknown machine is 404.
2. **Idempotency:** the same `idempotencyKey` with the same request returns the original final ack and sends nothing, with the header `Idempotent-Replay: true`. A duplicate that arrives while the first request is still in flight waits for the same result. The same key with a different request is 409. Keys are kept for `idempotency_ttl_s`, at most `idempotency_max_entries`, and are reloaded from the event store on startup.
3. **Safety rules** (`config/commands.yaml`): forbidden or required states per command type, plus speed and load limits per machine kind. Without any state for the machine yet, only STOP passes.
4. **Capability:** a source without `Capability.COMMANDS` (the file replay) gives REJECTED, with the reason "the active source (file adapter) does not accept commands; switch to a live source ...".
5. **Routing** to the active adapter, waiting at most `command_timeout_s`. No answer in time gives TIMEOUT.

Every outcome of the command itself is a 200 with the final `CommandAck`: `EXECUTED`, `REJECTED` with a reason, or `TIMEOUT`. HTTP errors are reserved for the request itself: 401, 403, 404, 409 and 422.

### Fail-safe principle: STOP is always allowed

STOP passes the safety rules **in every state**: OFFLINE, STALE, FAULT, MAINTENANCE, and an unknown state. Stopping must never be blocked by the twin's view of the machine, because that view may be stale or wrong exactly when a stop is needed. The machine decides whether the stop is still meaningful; for example, the simulator answers "machine is already stopped". The evaluator exempts STOP in code, and the config schema rejects any rule that lists STOP, so a configuration change cannot disable it. Tests cover STOP in every state, and over a live source while FAULT, STALE and OFFLINE.

### Shipped rules

| Commands | Rule |
|---|---|
| START, SET_SPEED, SET_LOAD, SCHEDULE_MAINTENANCE, RESET | not while OFFLINE or STALE (only STOP until the link recovers) |
| SET_SPEED, SET_LOAD | not while FAULT or MAINTENANCE |
| START | not while FAULT (RESET first) |
| SCHEDULE_MAINTENANCE | only when stopped (IDLE or FAULT) |
| SET_SPEED, SET_LOAD | within the per-kind limits (the simulator's maximum speed and load) |

### Audit trail

- **What is published:** the Command; then an `ACCEPTED` ack once the safety rules and the capability check have passed; then the final ack. All go on `Topic.COMMANDS`, so the store keeps every command, rejected ones included. A safety rejection has no ACCEPTED ack, which tells it apart from a machine's refusal.
- **Timestamps:** audit `ts` values are on the event-time axis (ADR 0010): the target machine's watermark, the line's if the machine has none, and the wall clock only before any data has arrived. The wall-clock time is kept as each row's `ingested_at`.
- **Queries:** `GET /commands/{id}` and `GET /commands?from=&to=&machines=` combine the store with this process's in-memory records, because the store lags by up to one writer flush.

`SourceManager.route_command` does the routing without publishing. `send_command` (route plus publish) stays for direct use.

## Consequences

- Retries are safe. A client that lost the response resends with the same key and gets the original ack.
- The command reaches the machine only through the adapter. The command bus does not change the twin's state; the state changes when the machine reports it.
- A command whose ack never arrived before a backend restart is not remembered, so a retry sends it again. The machine-side idempotency (the simulator keeps an idempotency table) catches that case.
