# Simulator (`twin_sim`)

A deterministic, physically motivated simulator of the line CNV-01 → B1 → CNC-01 → B2 → INSP-01 → B3 → CNV-02. It generates 1 Hz sensor data, gradual and sudden machine faults, sensor faults, and machine-to-machine disruption (STARVED/BLOCKED). The parameters live in [`config/simulator.yaml`](../config/simulator.yaml) (physics, faults, thresholds) and [`config/scenarios/*.yaml`](../config/scenarios) (operating conditions, fault schedules, seeds). The code does not hardcode any of these values.

The common runs are poe tasks, and the commands work the same in any shell on Windows, Linux and macOS:

```text
uv run poe sim                # 200 runs x 3600 s, seed 42 -> data/sim/
uv run poe sim-quick          # 5 fault-dense runs + CSV/JSONL -> data/sim_quick/
uv run poe sim-quick-single   # the same on the single-station line (section 5)
uv run poe sim-plots          # example plots -> experiments/results/sim_examples/
```

The CLI behind them, for custom runs:

```text
uv run python -m twin_sim generate --runs 200 --duration 3600 --seed 42 --out data/sim/
uv run python -m twin_sim generate --quick --export csv,jsonl --out data/sim_quick/
uv run python -m twin_sim run --speed 10 --duration 120     # real time x10, JSON-lines Readings on stdout
uv run python -m twin_sim run --line config/line_single.yaml --scenario config/scenarios/quick_single.yaml --speed 10
uv run python -m twin_sim generate --help
```

`generate` and `run` take `--line` to choose the line layout. The default is `config/line.yaml`. `plot-examples` always uses the full line.

## 1. Machine model

Notation:
- `L`: effective load (%), with `ℓ = L/100`
- `ω`: speed, with `ω_set` the setpoint and `ω_nom` the configured nominal speed
- `f`: friction index
- `c`: cooling loss (0..1)
- `w`: progress of the active gradual fault (0 = healthy, **uncapped**)

| quantity | equation |
|---|---|
| load demand | `μ + y`, clipped to the run's load band. `y` is an Ornstein–Uhlenbeck process (τ = 300 s, stationary σ = 15 % of band width). `μ ~ U(band)` per machine; SET_LOAD moves `μ` and the band. |
| effective load | first-order lag (τ = 3 s) towards the demand when processing, `idle_load_pct` (10 %) when STARVED/BLOCKED, and 0 when stopped |
| speed | `ω = ω_set·(1 − s·ℓ − k_ov·max(0, ℓ−1))·fault_speed_factor·(1 + σ_ω·jitter·ε)` |
| temperature | `C·dT/dt = q0 + qL·ℓ² + qf·f·(ω/ω_nom) + Q_fault − k·(1 − c)·(T − T_amb)`, integrated exactly per substep: `T ← T_ss + (T − T_ss)·exp(−dt·k(1−c)/C)` |
| vibration (true) | `v = (v0·(1 + βL·ℓ) + V_bearing)·(ω/ω_nom) + V_imbalance·(ω/ω_nom)²` |
| current (true) | `I = I0 + IL·ℓ + I_fault` |

The ℓ² term is copper loss: I²R with I ∝ load. A stopped machine has no heat input, current or vibration, and its temperature relaxes to ambient. `T_amb ~ U(18, 28) °C` per run.

**Spindle speed changes between jobs.** Real CNCs change spindle speed from job to job. The `default` scenario, and `quick`, `sensor_faults` and `e5_lubrication_loss`, which mirror its operating conditions, contain a `speed_changes` entry for CNC-01:

```yaml
speed_changes:
  - { machine: CNC-01, interval_s: [300, 900], speed_rpm: [7500, 10000] }
```

- **How it is issued:** every U(300, 900) s, a SET_SPEED to a setpoint drawn from U(7500, 10000) RPM, rounded to 1 RPM. The values come from a dedicated seeded stream (`speed_plan`), so adding or removing speed changes leaves the fault plan of a run unchanged. The commands are planned into `RunPlan.commands` and executed through the same command path as the auto-recovery operator.
- **Why it matters:** spindle_imbalance vibration scales with speed², while bearing-wear vibration and the healthy baseline scale with speed. Dividing vibration by the speed ratio separates the two faults: bearing wear becomes smooth and imbalance keeps a step at each speed change (`imbalance_vs_bearing_speed_steps.png`). It also adds operating-condition variety.
- **Guards:** a scenario's speeds must lie inside the kind's validated `speed_operating_rpm`. They must also keep the cycle time below the arrival interval: at 7500 RPM the CNC cycle is 6 s × 8000/7500 = 6.4 s, against 6.5 s arrivals. `check_scenario_compat` refuses scenarios that violate either rule.

**Measurement layer.** It uses a separate RNG stream, so measurement noise never changes the physics.

| sensor | measured value |
|---|---|
| temp | `T + 0.2·ε` |
| vib | `max(v·(1 + 0.03ε) + 0.03ε', 0) + stick-slip burst` |
| current | `max(I·(1 + 0.01ε) + 0.05ε', 0)` |
| load | `L + 0.5ε` |

### Default parameters

| param | CONVEYOR | CNC (spindle) | INSPECTION (indexer) |
|---|---|---|---|
| ω_nom / ω_max (RPM) | 1450 / 1750 | 8000 / 12000 | 600 / 900 |
| validated operating setpoints `speed_operating_rpm` (RPM) | 1450 only | 7500–10000 | 600 only |
| slip s at full load, speed noise σ_ω | 0.03, 0.3 % | 0.015, 0.2 % | 0.02, 0.3 % |
| C (J/K), k (W/K), τ = C/k | 6000, 25, 240 s | 15000, 50, 300 s | 4000, 20, 200 s |
| q0, qL, qf (W) | 100, 750, 600 | 250, 1300, 1500 | 60, 500, 400 |
| I0, IL, If (A); I_rated = I0 + IL | 1.5, 6.0, 2.5; 7.5 | 4.0, 18.0, 6.0; 22.0 | 1.0, 3.5, 1.5; 4.5 |
| v0 (mm/s), βL | 0.45, 0.3 | 1.2, 0.3 | 0.40, 0.3 |
| cycle time (s) | 4 | 6 (bottleneck) | 5 |

## 2. Faults

Each machine has at most one active machine fault at a time. An onset that arrives while another fault is active, or during maintenance, is skipped and logged in the manifest (`realised_faults[].skipped_reason`).

**Gradual progress.** `dw/dt = g(L)/D · exp(0.1·ε − 0.005)`. The noise is lognormal and mean-corrected, and progress advances only while the motor runs.
- `D ~ U(720, 1500)` s is the time for w to go from 0 to 1 at g = 1.
- `g(L) = clip((L/62.5)^m, 0.5, 1.5)` is load-accelerated degradation. It is motivated by the L10 bearing-life law (life ∝ load⁻³), softened to m = 1.5.
- The **progression time** is `D/g`.

**w is uncapped.** It keeps growing until the machine trips, then freezes until maintenance resets it. Physical fractions are clipped, for example `c ≤ 0.6`.

| fault | machines | effects | m | leading signal |
|---|---|---|---|---|
| `bearing_wear` | all | `V_bearing = 1.6·vib_fault·(e^{3w} − 1)/(e³ − 1)`, `f = 0.2w`, `I_fault = If·f` | 1.5 | vibration mean: smooth, exponential |
| `overheating` | all | `c = min(0.6w, 0.6)`, `Q_fault = ΔT_oh·k·w` with ΔT_oh = 100 K (CNC 5.0 kW, conveyor 2.5 kW, indexer 2.0 kW at w = 1) | 0 | temperature, with thermal lag |
| `spindle_imbalance` | CNC only | `V_imbalance = 1.5·vib_fault·w²` (×(ω/ω_nom)², i.e. force ∝ ω²) | 0 | vibration, speed-dependent |
| `belt_jam` (sudden) | conveyors, indexer | at onset: FAULT; ω → 0 with τ = 0.5 s; I = 3·I_rated until the protection trips after 2 s, then 0 | — | none (no precursor) |
| `lubrication_loss` (**held out**) | all | `f_l = w³`, `I_fault = 1.5·I_rated·f_l`, `qf·f_l` heat, speed `×(1 − 0.04w²)`, speed noise `×(1 + 3w)`, stick-slip vibration bursts | 0 | current |

**Overheating.** A failing coolant or fan both removes cooling and adds heat from the struggling component. Cooling loss alone, capped at 0.75, does not reach FAULT at low load: the CNC at 30 % load and 18 °C settles near 47 °C, against a FAULT range of 76.5–93.5 °C. The extra heat term fixes this. With `c ≤ 0.6`, the thermal time constant grows by at most 2.5×.

**How `lubrication_loss` differs (experiment E5).** It is designed so that no training fault shares its signature:

| aspect | lubrication_loss |
|---|---|
| leading channel | current, through friction torque that accelerates as w³, together with speed sag and speed jitter. In training data, current only moves in a jam, and only as an instant 2 s spike. |
| vibration | spikiness and variance rise instead of the mean: stick-slip bursts with probability 0.3·w per second and amplitude U(0.5, 1.5)·0.3·vib_fault. Bursts are added to the *observed* vibration only. bearing_wear and spindle_imbalance instead show a smooth rise in the vibration mean with flat current. |
| temperature | lags and rises only moderately. overheating instead raises temperature alone, with vibration, current and speed flat. |
| trip channel | **overcurrent**, which no training fault reaches through gradual drift |

The fault is marked `held_out: true` in `simulator.yaml`. Random sampling skips it unless a scenario sets `include_held_out: true`, which is what [`e5_lubrication_loss.yaml`](../config/scenarios/e5_lubrication_loss.yaml) does. An explicit `scheduled` entry also injects it. The manifest records `eligible_fault_types` and `held_out_fault_types`.

### Feasibility guarantee (validated whenever `simulator.yaml` is loaded)

`twin_sim.feasibility` simulates the noise-free single-machine model at constant load and ambient, for every gradual fault, applicable kind, progression bound and **both ends of the kind's operating speed range**. Every effect is monotone in speed, so the ends are the extremes. It runs three checks and refuses configs that fail any of them:

- **Reachability, best case:** L = 30 %, T_amb = 18 °C, every threshold +10 %. FAULT must be crossed before w = 0.95, i.e. within the progression time.
- **Minimum lead time, worst case:** L = 95 %, T_amb = 28 °C, every threshold −10 %, D = 720 s. FAULT must come at least `min_lead_s` = 300 s after onset.
- **Healthy margin:** a healthy machine at the worst case stays below every WARNING threshold.

The tables below show the validator's numbers for the default config. They come from `simulate_noise_free`, the function the validator runs.

Conveyors and indexer, at their single setpoint:

| fault | kind | best case: w at FAULT (D = 1500 s / 720 s) | worst case: lead time |
|---|---|---|---|
| overheating | CNV / INSP | 0.73 / 0.68, and 0.90 / 0.83 | 327 / 316 s |
| bearing_wear | CNV / INSP | 0.85 / 0.86 | 377 / 378 s |
| lubrication_loss | CNV / INSP | 0.87 / 0.87 | 374 / 374 s |

CNC-01 at both ends of its 7500–10000 RPM setpoint range. Reachability is hardest at low speed; lead time is shortest at high speed.

| fault | best case: w at FAULT, 7500 / 10000 RPM | worst case: lead time, 7500 / 10000 RPM |
|---|---|---|
| overheating | 0.70 / 0.70 (D = 1500 s), 0.89 / 0.89 (D = 720 s) | 309 / 309 s |
| bearing_wear | 0.85 / 0.75 | 369 / **315** s |
| spindle_imbalance | 0.84 / 0.61 | 532 / 380 s |
| lubrication_loss | 0.87 / 0.87 | 375 / 375 s |

No parameter change was needed to cover the speed range. The tightest case is bearing wear at 10000 RPM, with a worst-case lead time of 315 s against the 300 s minimum. A wider range would fail: at 12000 RPM, bearing wear reaches FAULT too soon after onset, and the validator refuses the config (tested).

A step up in speed can push a degraded spindle over the FAULT threshold at the moment of the step, because imbalance vibration jumps by (10000/7500)² ≈ 1.8×. This is realistic, and the lead time from onset still meets the minimum.

At low load, bearing wear can take up to `0.86·1500/0.5 ≈ 43 min` from onset. A late onset in a 1 h run is therefore often censored, which is realistic. Tests check reachability and lead time empirically, with noise, over several seeds.

### Sensor faults (observation layer)

| type | effect |
|---|---|
| `stuck` | holds the last observed value |
| `drift` | adds `rate·(t − onset)`, with rate = 10 % of the sensor's catalog range per hour |
| `dropout` | the whole reading is lost: NaN in Parquet, and no `Reading` from the real-time runner or in JSONL |

Sensor faults never change states or labels; the twin should still infer the truth. They are labelled in `sensor_fault` (`none`, `stuck:temp`, `drift:vib`, `dropout`). They are off in `default.yaml` and on in `sensor_faults.yaml`.

## 3. Thresholds and states

Base values are below. Every threshold of every machine is multiplied per run by an independent `U(0.9, 1.1)` factor (`threshold_perturbation_pct`), so a model cannot learn one fixed limit. The perturbed values are written to the manifest.

| | CONVEYOR | CNC | INSPECTION |
|---|---|---|---|
| vib WARNING / FAULT (mm/s RMS) | 1.8 / 4.5 | 4.5 / 7.1 | 1.8 / 4.5 |
| temp WARNING / FAULT (°C, housing) | 80 / 95 | 70 / 85 | 75 / 90 |
| current WARNING / FAULT (× I_rated) | 1.15 / 1.30 | 1.15 / 1.30 | 1.15 / 1.30 |

Sources. These are "inspired by" values, not compliance limits:

- **Vibration:** ISO 10816 severity zones. WARNING = zone B/C boundary ("unsatisfactory for long-term operation") and FAULT = zone C/D boundary ("damage may occur").
  - The conveyors and the indexer use ISO 10816-1 class I (small machines; boundaries 0.71 / 1.8 / 4.5 mm/s).
  - The spindle uses ISO 10816-3 group 2 with flexible mounting (boundaries 2.3 / 4.5 / 7.1 mm/s).
  - ISO 10816 has been superseded by ISO 20816; the zone concept is unchanged.
- **Temperature:** motor insulation classes in IEC 60034-1 (class B 130 °C, class F 155 °C winding limits). The simulated sensor sits on the housing, assumed to run some tens of kelvin below the winding hotspot. CNC spindle limits are set lower to protect the spindle bearings.
- **Current:** motor overload protection, motivated by the overload-relay trip classes in IEC 60947-4-1. A SET_LOAD above about 105 % can therefore legitimately cause an overload WARNING or FAULT, labelled `fault_type = overload`.

**State logic.** States are computed from the deterministic signals, as a controller-filtered value would be, not from measurement noise.
- **Hysteresis:** a machine leaves WARNING once all signals are 5 % below their thresholds.
- **Priority:** `MAINTENANCE > FAULT (latched) > IDLE > WARNING > STARVED/BLOCKED > RUNNING`. WARNING ranks above the flow states, so a degrading machine keeps its health label while starved.
- **Trips:** FAULT latches and stops the motor.

## 4. Line coupling

Parts are discrete and buffer levels are integers (capacities from `line.yaml`; every buffer starts with 2 parts). A source feeds CNV-01 every 6.5 s ± 10 %. Each machine takes a part, processes it for `cycle_time·ω_nom/ω_set`, and pushes it downstream.

- **STARVED:** no input part for ≥ 15 s.
- **BLOCKED:** a finished part could not leave for ≥ 15 s.
- The grace period stops the normal gaps between parts from flickering the state.
- **`root_cause_machine`:**
  - FAULT, WARNING, IDLE and MAINTENANCE: the machine itself.
  - STARVED or BLOCKED: follows the chain upstream or downstream to the first machine that is not STARVED/BLOCKED (`SOURCE` if the source is starving CNV-01).
  - RUNNING: empty.

Example ([`cnc_stop.yaml`](../config/scenarios/cnc_stop.yaml)): STOP CNC-01 at 600 s. B1 fills in about 65 s and CNV-01 goes BLOCKED; INSP-01 and CNV-02 go STARVED. All of them report root cause CNC-01.

## 5. Single-station mode

The original project synopsis asks for "a 3D conveyor or CNC station on desktop". [`config/line_single.yaml`](../config/line_single.yaml) provides exactly that: one infeed conveyor, CNV-01, with no buffers. The simulator itself needs no special case.

- **Flow:** the source feeds CNV-01 directly, and finished parts leave to an unbounded sink. The 4 s conveyor cycle is faster than the 6.5 s arrivals, so a healthy station only shows RUNNING, never STARVED or BLOCKED. Throughput is the source rate.
- **States:** STARVED appears only if the source stops delivering, and its root cause is then `SOURCE`. BLOCKED cannot occur, because the sink is unbounded. Every FAULT, WARNING, IDLE or MAINTENANCE state has root cause CNV-01.
- **Faults:** only faults applicable to a conveyor occur: bearing_wear, overheating, belt_jam and the held-out lubrication_loss. The physics, thresholds, feasibility guarantee, labels, commands and snapshots are identical to the full line. In the dataset, `buffer_in` and `buffer_out` are null.
- **Scenarios:** a scenario must only reference machines that exist on the chosen line. The full-line scenarios (`default`, `quick`, `sensor_faults`, `e5_lubrication_loss`, `cnc_stop`) reference CNC-01, through spindle speed changes or commands. They are therefore refused on the single line with `unknown machine 'CNC-01' (not on line 'line1_single')`. Use [`quick_single.yaml`](../config/scenarios/quick_single.yaml), which is `quick` without the CNC-only parts, or write a scenario in the same style.

## 6. Commands and recovery

`LineSimulator.apply_command(Command) -> CommandAck` returns EXECUTED or REJECTED with a reason. Commands are idempotent per `idempotencyKey`.

| type | behaviour | rejected when |
|---|---|---|
| START | the machine runs | in FAULT (needs RESET) or MAINTENANCE |
| STOP | IDLE | already stopped |
| SET_SPEED | new setpoint; cycle time scales | outside (0, ω_max] |
| SET_LOAD | new load operating point | outside [0, 120] % |
| SCHEDULE_MAINTENANCE | MAINTENANCE for `durationS` (default 600 s); the fault, wear and cooling loss are cleared; then RUNNING | already in maintenance |
| RESET | clears the FAULT latch, then IDLE. Wear is kept, so a worn machine trips again | not in FAULT |

The scenario's auto-recovery operator issues these same commands, so one run contains several fault cycles:
- jam: RESET + START after U(30, 120) s
- gradual fault or overload: SCHEDULE_MAINTENANCE after U(120, 600) s

## 7. API, determinism, snapshots

- **Stepping:** `LineSimulator(line, sim, plan)` with `plan = make_run_plan(scenario, sim, line, seed, run_index)`. `step(dt)` splits `dt` into substeps of at most 1 s and returns a `StepFrame`; `readings(frame)` turns it into wire-format `Reading`s.
- **Snapshots:** `snapshot()` returns a `SimSnapshot`, a Pydantic model that includes the RNG bit-generator states. `restore(snap)` and `LineSimulator.from_snapshot(...)` load one. A snapshot round-tripped through JSON continues **bit-identically**.
- **Fast-forward:** `fast_forward(seconds)` is identical to repeated `step(1)` but needs no sleeping; one simulated hour takes about 0.25 s.
- **Seeding:** run `i` uses `SeedSequence(seed, spawn_key=(i,))`, so output does not depend on the number of runs or workers. Each run spawns five independent named streams: `plan`, `sensor_plan`, `process`, `measure` and `ops`.
- **Real time:** `RealtimeRunner(sim, on_reading, speed)` ticks on absolute deadlines at `1/speed` s per simulated second. The callback may be sync or async. `submit(cmd)` applies a command at the next tick.

## 8. Dataset format and labels

`generate` writes `run_NNNN.parquet` (zstd) per run, an optional `.csv` (same columns) and an optional `.jsonl` (wire-format `Reading`s, time-ordered, for minimal-mode replay). It also writes `manifest.json`. The manifest contains:
- the CLI args, base seed, the resolved scenario and sim config with SHA-256 hashes, the package version and the git commit;
- per run: the seed spawn key, load band, ambient, load operating points, perturbed thresholds, planned and realised faults, sensor faults, and parts produced.

Parquet rows: one per machine per second, grouped by machine and then by time. The columns are:

| column | meaning |
|---|---|
| `run_id, machine_id, seq, ts, t_s` | identity and time. `seq` increments every second, even during dropout. |
| `state, speed, temp, vib, load, current, run_time, quality` | the Reading fields. `quality` is always `OK`; the ingest layer assigns other values later. |
| `fault_type` | the fault mode active in the machine from degradation start through FAULT. `overload` for a trip without an injected fault, `none` otherwise. A jam has no precursor rows. |
| `time_to_fault_s` | time to the next transition into FAULT, capped at 1800 s. It is 0 while in FAULT. |
| `ttf_censored` | true when no FAULT onset lies ahead in the run. `time_to_fault_s` is then the observed lower bound `min(t_end − t, cap)`. |
| `root_cause_machine` | see §4 |
| `load_range` | the run's load band (`low`, `medium` or `high`) |
| `sensor_fault` | see §2 |
| `buffer_in, buffer_out` | level of the upstream and downstream buffer (null at the ends of the line) |

## 9. Performance

Measured on the development laptop by running `uv run poe sim`: 200 runs × 3600 s, 2.88 M rows, 132 MB of Parquet, generated in **9.8 s** with 17 worker processes. A rerun after the spindle speed changes were added took 16.2 s, within normal run-to-run variation. With one worker, 10 runs took 2.7 s, which puts the full dataset at about 1 minute on a single core. The target was under 15 minutes. The time for each generation is recorded in `manifest.json` (`generation_time_s`).

## 10. Example plots

`uv run poe sim-plots` writes the following to `experiments/results/sim_examples/`:
- one figure per fault type (sensor panels with WARNING/FAULT threshold lines and a state band)
- `sensor_faults.png`
- `line_coupling_cnc_stop.png` (machine states and buffer levels)
- `imbalance_vs_bearing_speed_steps.png`: CNC-01 stepping between 7500 and 10000 RPM every 200 s, with spindle_imbalance in one run and bearing_wear in the other (onset 300 s). The bottom panel shows vibration divided by the speed ratio: it is smooth for bearing wear and keeps the speed steps for imbalance.
