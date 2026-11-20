# ADR 0031: What-if projections, drift detection and the demo scenarios

- **Status:** Accepted (what-if, Group 2 of Phase 11 Part C). The drift and demo sections are added with their groups.
- **Date:** 2026-11-20
- **Phase:** 11 Part C

## Context

The dashboard can show predictions (ADR 0030). Operators also want to ask "what if I stop this machine, or change its speed?" before doing it. CLAUDE.md §3 puts this on the architecture: dashboard → `POST /whatif` → simulator snapshot → fast-forward copy → model → projected trajectory. Constraints:

- **Simulation only.** Real equipment cannot be forked, so the capability belongs to the source adapter like scenario injection (ADR 0015); other sources answer 409.
- **Minimal mode:** it must keep working with no ML. Without inference a what-if shows the simulator only.
- **The live line must not be disturbed:** no command is sent, nothing is stored, and the live inference (buffers, alarms, alerts) is untouched.
- **The evaluated alarm rule** must be the one applied to projections (ADR 0022 / 0030), not a new one.
- **Safety:** a change must pass the same rules as a command.

## Decision

### What-if

- **One change per request**, applied at the fork point (now): a command type, its parameters, a machine and a projection length (`config/whatif.yaml`: default 120 s, maximum 300 s). That is exactly what "apply as command" can send. Scheduling changes later in the run is not offered.
- **Validation in the backend first:** the command contract (a missing speed is a 422), the safety rules and limits of `commands.yaml` against the machine's current state (`SafetyRules`; STOP always passes), and the horizon limit. The simulator's own verdict is checked as well.
- **The fork lives in the simulator** (`twin_sim/whatif.py`, mock server `POST /whatif`, contracts in `twin_common/whatif.py`):
  - the state is copied and the recent history read in one tick-boundary call, so the history ends at the fork point;
  - an unchanged and a changed copy are built from the snapshot and fast-forwarded in a thread, so the live simulation keeps its pace;
  - the random-number streams are restored, so the unchanged copy equals what the live simulation does next (tested).
- **Scoring in the backend with its own worker pool** (`whatif/scoring.py`): the same bundles, gate and alarm rule (`MachineAlarms`), no attributions. The pool starts on the first request (a few seconds) and is reused; the live pool is not shared, because each projection needs empty buffers and a what-if must not delay or alter live predictions. Each variant is replayed from empty buffers over the recent readings plus its projection.
  - **Cost:** a second set of worker processes (memory) once a what-if has been asked.
  - **Failure:** if the workers cannot start or answer, the projection is still returned, with a note that it shows the simulator only.
- **Response:** per machine and variant, the projected readings per second, the projected simulator states at +10 s / +30 s / the end ("ghost states"), the first simulator WARNING / FAULT and, with a model, calibrated P(WARNING) / P(FAULT) at +30 s, top class, the evaluated alarm, time to fault, and the first alarm.
- **Status codes:** 404 unknown machine, 409 source without the capability (or no `whatif.yaml`), 422 refused (contract, safety, limits, simulator), 502 the simulator could not run it. Viewer role is enough: a what-if changes nothing.
- **Dashboard:** ghost states, dashed projection lines, outcome text, and "apply as command" for operators in live mode after a confirmation, through `POST /commands`.

## Consequences

- **A projection is the simulator's future, not a promise.** The model's numbers on it are as trustworthy as on live data from the same simulator, and no more. The dashboard says it is a simulation.
- **The what-if shares the simulator's world model:** with the mock source it is the same simulator that produces the data, so projected states are what the simulator would do (including a planned fault later in the run). With real equipment this feature would need a different mechanism and does not exist.
- **Worker memory:** a second pool after the first what-if with inference on.
- **Not covered:** several changes at once, delayed changes, and projections from replay time (the fork is always the running simulation).
