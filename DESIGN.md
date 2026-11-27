# LineTwin dashboard: design rules

These rules apply to every change in `frontend/src/dashboard/`. If a change needs to break one, raise it first. Do not quietly deviate.

Reference images: [`docs/design-refs/ref-1.png`](docs/design-refs/ref-1.png) (warehouse robots) and [`docs/design-refs/ref-2.png`](docs/design-refs/ref-2.png) (city map). The rendered status palette is in [`docs/design-refs/status-palette.png`](docs/design-refs/status-palette.png).

## 1. Direction: "Night Shift": a high-performance HMI

The dashboard follows **ISA-101 high-performance HMI** practice, in a dark version for a control room at night. ISA-101 is the standard real plant control systems use. Its central rule:

> **The world is grey. Colour means something is abnormal.**

Machines, panels and text are graphite. Only machine states, alarms and model reliability carry chromatic colour. A healthy line therefore looks calm and nearly monochrome, and a fault is the brightest thing on the screen.

### What the references confirm

- **ref-1 (warehouse) is the closer match.** It has a neutral graphite world with one hot colour: the faulted robot, its floor glow, its row in the queue and its `● Fault` pill are all orange, and nothing else is. That is exactly the rule above.
- **Dense floating panels over the 3D scene.** ref-1 has a queue on the left and an inspector on the right; ref-2 has panels in all four corners.
- **Thin hairlines.** In ref-1 the `BODY | LIFT | TOP SPEED` stats are separated by 1 px vertical rules, not boxes.
- **Uppercase tracked micro labels:** `ORDERS QUEUE`, `BATTERY · LAST 2 H`.
- **Large light-weight numerals with small muted units:** `341`, `81 %`, `0.0 m/s` in ref-1; `320 vehicles` in ref-2.
- **Monospace IDs in lists:** `#A-20419`, `HR-047 → P-02`.

### What the references changed in this document

1. **Rounded panels.** Both references use about 8 px panel radii, so panels move from 0 to `--radius-panel: 8px` and controls and badges from 2 to 4 px. (§4)
2. **Abnormal badges are tinted, not solid.** ref-1's `● Fault` pill is a tint with a coloured border, dot and text. A solid fill is now reserved for alarms that still need acknowledging. (§3.4)
3. **The incident banner goes top centre.** ref-1 puts `HR-047 stopped · drive motor fault 00:04`, with lifecycle steps, at the top centre of the scene. The annunciator moves there and shows the real ISA-18.2 lifecycle: `Raised → Acknowledged → Cleared`. The line strip moves to the bottom edge of the stage. (§5)
4. **Stat rows instead of cards.** Readings use ref-1's hairline-divided stat row: big value, small muted unit, a micro label above. (§5, §9)
5. **Sparkline conventions.** From ref-1's battery chart: start and end time labels (`12:30 … NOW`), a dot on the last value with its value label, and a short micro-label annotation at event markers (`CHARGED AT C-02` → `FAULT 00:12:57`). (§8)
6. **Mono for IDs.** Machine IDs, alert kinds and timestamps in lists and tables use Plex Mono. (§2)
7. **A ground ring under the selected or faulted object.** Both references draw one. This needs engine changes, so it is listed as an *additional* proposal in §10. It is not in the approved stage 7 list.

### Where we deliberately differ from the references

- **ref-2's grid of equal KPI tiles** stays forbidden (§9). We use hairline stat rows.
- **ref-2's teal accent** is not adopted. Teal and cyan sit next to our "explained" colour, and the accent must never look like a status. Bone (§3.2) stays the accent.
- **ref-2's blue-slate surfaces.** We keep ref-1's neutral graphite: a tinted world competes with status hues.

## 2. Type

All three fonts are OFL-licensed and self-hosted as `.woff2` files in `frontend/src/dashboard/fonts/`, wired up in `fonts.css`. There is no CDN and no npm font package. The app works offline and falls back gracefully if a file is missing.

| Role | Family | Weights | Used for |
|---|---|---|---|
| Display | **Barlow Condensed** | 200, 800 | 800: uppercase micro labels, section heads, the inspector's machine ID. 200: **only at ≥ 40 px** (the selected machine's state word) |
| Body | **IBM Plex Sans** | 400, 500, 600 | Running text, table text, controls |
| Mono | **IBM Plex Mono** | 200, 400, 500 | 200: **only at ≥ 40 px** (large readings). 400: table numerics, IDs, timestamps, alert kinds, units. 500: emphasised values |

**Weight 200 is never used below 40 px**, in any family.

Expected font files (in `frontend/src/dashboard/fonts/`):

| File | Family / weight |
|---|---|
| `BarlowCondensed-ExtraLight.woff2` | Barlow Condensed 200 |
| `BarlowCondensed-ExtraBold.woff2` | Barlow Condensed 800 |
| `IBMPlexSans-Regular.woff2` | IBM Plex Sans 400 |
| `IBMPlexSans-Medium.woff2` | IBM Plex Sans 500 |
| `IBMPlexSans-SemiBold.woff2` | IBM Plex Sans 600 |
| `IBMPlexMono-ExtraLight.woff2` | IBM Plex Mono 200 |
| `IBMPlexMono-Regular.woff2` | IBM Plex Mono 400 |
| `IBMPlexMono-Medium.woff2` | IBM Plex Mono 500 |
| `OFL-BarlowCondensed.txt`, `OFL-IBMPlex.txt` | Licences |

- **Never** use Inter, Roboto, Open Sans, Arial, `system-ui`, `-apple-system`, `Segoe UI` or a bare `sans-serif` stack. The stacks are in `tokens.css` (`--font-display`, `--font-body`, `--font-mono`). Their fallbacks are metric-close families; a generic family appears only at the very end.
- **Every number** (readings, probabilities, timestamps, counts) is in Plex Mono, or carries `font-variant-numeric: tabular-nums` inside body text. Number columns are right-aligned.

### Type scale

| Token | Size / line height | Family and weight | Use |
|---|---|---|---|
| `--type-micro` | 11 / 16 px, +0.08 em tracking, uppercase | Barlow Condensed 800 | Labels: `READINGS`, `TIME TO FAULT`, axis titles, column heads |
| `--type-body` | 13 / 20 px | Plex Sans 400 (500/600 for emphasis) | Everything readable |
| `--type-title` | 16 / 24 px | Barlow Condensed 800 | Machine ID in the inspector, dialog titles |
| `--type-display` | 40 / 48 px | Barlow Condensed 200 or Plex Mono 200 | The selected machine's state word; its key readings (`7941` RPM) |

40 px display text against 11 px labels (about 3.6×), and weight 200 against 800, carry the hierarchy. **Hierarchy comes from type, space and hairlines, never from boxes inside boxes.**

**Micro labels on the HUD use `--text-2`**, never `--text-3`. On docked surfaces, micro labels may use `--text-3`.

## 3. Colour

All colours are CSS custom properties in `frontend/src/dashboard/tokens.css`: `:root` holds dark (the default, "night shift") and `:root[data-theme="light"]` holds light ("day shift", kept because the theme toggle is an existing feature). Hex values appear **only** there and in the 3D engine config in `engineFactory.ts`, which mirrors the same values.

### 3.1 Surfaces: one neutral family (graphite)

| Token | Dark | Light | Use |
|---|---|---|---|
| `--surface-0` | `#0D0F12` | `#F2F1ED` | App background, behind the stage |
| `--surface-1` | `#14171B` | `#FBFAF8` | Docked panels (inspector, console) |
| `--surface-2` | `#1B1F24` | `#F2F0EB` | Inputs, table header row, tooltips |
| `--surface-3` | `#242930` | `#EAE7E0` | Hover and selected rows |
| `--hud` | `rgb(20 23 27 / 0.88)` + `backdrop-filter: blur(12px)` | `rgb(251 250 248 / 0.9)` | **Only** the floating HUD layer over the 3D stage |
| `--hairline` | `#2A3037` | `#DAD6CC` | 1 px dividers |
| `--hairline-strong` | `#3A424B` | `#C4BFB3` | Input borders, the active tab's edge |
| `--text-1` | `#ECEEF0` | `#15181C` | Primary text, values |
| `--text-2` | `#A7AFB8` | `#4A525B` | Secondary text, units, HUD micro labels |
| `--text-3` | `#8E97A1` | `#5C646D` | Micro labels and captions on docked surfaces only |

### 3.2 Brand accent: one colour, "bone"

| Token | Dark | Light |
|---|---|---|
| `--accent` / `--on-accent` | `#E8DFC8` / `#0D0F12` | `#2B2620` / `#FBFAF8` |

When every chromatic pixel means a machine state, the interaction accent has to be **light, not hue**. In the light theme it becomes near-black ink. It marks the focus ring, the selected machine, the active tab, primary buttons, and the what-if "with change" series. It is never used for status. There are no purple or indigo gradients and no decorative gradients at all.

### 3.3 Status palette (semantic, exempt from the one-accent rule)

The first draft had 11 status colours. Rendered side by side (`status-palette.png`) and measured in OKLab, several pairs could not be told apart at badge size: idle/offline 0.056, running/drift 0.057, maintenance/explained 0.076, waiting/warning 0.095; in the light theme idle/offline was 0.011 and waiting/degraded 0.033. **Merged result: 8 status colours.**

| Token | Dark | Light fill / ink | Meaning | Merged from |
|---|---|---|---|---|
| `--st-running` | `#6FC28E` | `#2F6E1F` / same | RUNNING. Normal, so **subdued**: dot plus text, never a filled badge | |
| `--st-inactive` | `#A0A8B1` | `#5C646D` / same | IDLE, STARVED, BLOCKED, OFFLINE, STALE, NO DATA, link lost. None of these is a fault, so ISA-101 keeps them grey. The word (`STARVED by INSP-01`) and the dot shape tell them apart (§3.4) | idle, waiting, offline |
| `--st-warning` | `#F4A42C` | `#AD620C` / `#8A4D08` | WARNING state, WARNING alerts | |
| `--st-fault` | `#FF5B4F` (ink `#FF6E63`) | `#C6413C` / `#A3302C` | FAULT state, CRITICAL alerts. **The one hot colour.** | |
| `--st-maintenance` | `#6E9BFF` (ink `#7DA6FF`) | `#3C6EC0` / `#30599F` | MAINTENANCE, INFO alerts | |
| `--st-unusual` | `#C58AFF` | `#794DA5` / same | **"Unusual behaviour, cause unknown" only.** Purple is used for nothing else. | |
| `--st-degraded` | `#E6DC5A` | `#786914` / `#6A5D10` | Do not fully trust this: reliability DEGRADED, data quality DEGRADED (`DQ`), **model drift** (banner, `model drift` badge) | degraded, drift |
| `--st-explained` | `#3FE0E8` | `#217F96` / `#1A6676` | Parts the model's explanation points at (3D glow, attribution chips) | |

- **Fill vs ink.** Each status has a **fill** (`--st-x`: dots, badge fills and borders, chart markers, the 3D scene) and an **ink** (`--st-x-ink`: the colour used as text). In the dark theme they are nearly equal. In the light theme, colours that pass AA as text get so dark that they become indistinguishable, so the light fills stay saturated and the inks are darker.
- **On a solid fill:** text is `--on-status` (`#0D0F12` dark, `#FFFFFF` light).
- **Tints:** `color-mix(in srgb, var(--st-x) var(--tint), var(--surface-1))`, where `--tint` is 14 % dark and 12 % light.
- **Changes from today, on purpose:**
  - The 3D data-quality colour `#B39DDB` (lavender) becomes `--st-degraded`, because purple is reserved.
  - Reliability DEGRADED stops borrowing warning amber.
  - The DQ flag stops borrowing maintenance blue.
  - Predicted alerts stop using blue and take the predicted class's own hue, with a dashed border (§3.4).

### 3.4 Status grammar: the shape says how sure we are

| Style | Means | Example |
|---|---|---|
| **Dot + text**, no fill | Observed, normal (running, idle) | `● RUNNING` |
| **Ring + text** (hollow dot) | Observed, no data or not producing (offline, stale, starved, blocked) | `○ STARVED by INSP-01` |
| **Tint + 1 px border + dot + ink text** | Observed, abnormal (as ref-1's `● Fault`) | `● FAULT`, `● WARNING`, `● unusual behaviour · cause unknown` |
| **Solid fill + `--on-status`** | Alarm active **and unacknowledged**: needs a person (ISA-18.2) | Annunciator, unacknowledged count, the CRITICAL row marker |
| **Hollow** (border + ink, no tint) | A statement about trust | `DEGRADED`, `DQ`, `model drift` |
| **Dashed border**, class hue | Predicted or projected, not observed | `predicted WARNING`, the +10 s / +30 s cells, what-if ghosts |

Colour is never the only cue. Every badge carries its word.

### 3.5 Contrast record (asserted by `frontend/scripts/check-contrast.mjs`)

`npm --prefix frontend test` runs the script first. It reads `tokens.css`, recomputes tints and the HUD glass exactly as the browser composites them, and fails the build if any of these checks fails:

| Check | Threshold |
|---|---|
| `--text-1/2/3`, `--accent`, every `--st-*-ink` on `--surface-0` … `--surface-3` | ≥ 4.5 |
| Every `--st-*-ink` on its own tint | ≥ 4.5 |
| `--on-status` on every `--st-*` fill; `--on-accent` on `--accent` | ≥ 4.5 |
| Every `--st-*` fill against `--surface-1` and `--surface-3` (dots, borders: WCAG 1.4.11) | ≥ 3.0 |
| `--text-1`, `--text-2`, `--accent`, every `--st-*-ink` on the HUD glass over `--surface-0` **and** over the lightest machine body `--scene-body` `#B8C0C8` | ≥ 4.5 |
| Minimum OKLab distance between the status fills and the accent | ≥ 0.11 dark, ≥ 0.09 light |

Current result: **208 checks pass.**

- The tightest dark pair is `--text-3` on `--surface-3`, at 4.94. The closest status pair is running/inactive, 0.116 apart.
- The tightest light pair is `--on-status` on `--st-explained`, at 4.62. The closest status pair is maintenance/explained, 0.094 apart.

**Known limit (light theme):** with 8 statuses plus the accent and AA on a light surface, about 0.094 is the ceiling. A search over each hue's lightness and chroma did not beat it without turning amber brown. In the light theme the shape grammar (§3.4) and the words carry the difference alongside the colour.

## 4. Space, edges, depth

- **8 px rhythm.** Tokens: `--s-1: 4px` (only inside a control), `--s-2: 8px`, `--s-3: 16px`, `--s-4: 24px`, `--s-5: 32px`, `--s-6: 48px`. Every margin, padding and gap is a multiple of 4, and layout-level ones are multiples of 8.
- **Radii.** `--radius-panel: 8px` on panels (docked and floating); `--radius-control: 4px` on controls and badges.
- **Depth.** There are three levels: surface-0 (the stage), surface-1 docked panels, and the floating `--hud`. Panels sit 8 px apart over surface-0 and are separated by space, not shadows. The only shadow is the modal dialog's (`0 24px 64px rgb(0 0 0 / 0.5)`).
- **No boxes inside boxes.** Inside a panel, sections are separated by a hairline and a 16 px gap, headed by a micro label. Tables use row hairlines only. Stat rows may use vertical hairlines between cells (ref-1).

## 5. Layout

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ LINETWIN  line1   ● Connected   00:13:09 UTC        [Replay]   key ▸ operator  Labels Theme │ top bar 48 px
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ (system bands, only when active: link lost · model drift · REPLAY scrubber)               │
├──────────────────────────────────────────────────────────────┬───────────────────────────┤
│ STAGE (3D, full-bleed)                                       │ INSPECTOR  (surface-1)    │
│        ┌ HUD annunciator ───────────────────────────┐        │ CNC-01                    │
│        │ ■ INSP-01 is FAULT        CRITICAL 00:12:57 │        │ CNC machining station     │
│        │ Raised ━━━━ Acknowledged ──── Cleared       │        │ FAULT            (40 px)  │
│        └────────────────────────────────────────────┘        │ ───────────────────────── │
│                                                              │ SPEED │ TEMP │ VIB         │
│        conveyor ── CNC ── inspection ── conveyor             │ 7941  │ 36.4 │ 4.21        │
│                                                              │ RPM   │ °C   │ mm/s        │
│ ┌ HUD line strip (flow order) ────────────────────────────┐  │ ───────────────────────── │
│ │ CNV-01 ● RUNNING → ▮▮ → CNC-01 ● FAULT → ▮ → INSP-01 …  │  │ PREDICTION · +30 s        │
│ └─────────────────────────────────────────────────────────┘  │ ACTIVE ALERTS             │
├──────────────────────────────────────────────────────────────┤ ───────────────────────── │
│ CONSOLE (surface-1, collapsible)                             │ COMMANDS                  │
│ Live · History · Alerts 1 · Prediction · What-if · Scenario  │ [STOP] Start Reset …      │
└──────────────────────────────────────────────────────────────┴───────────────────────────┘
```

- **The stage** is the 3D view. It fills its area edge to edge, with no border and no card. The engine fits the line to the container width, so HUD elements sit **above and below** the line, in the scene's empty sky and foreground. They never cover the sides, where they would hide the end conveyors. (Stage 7's view insets lift this restriction.)
- **The annunciator** (HUD, top centre, after ref-1) shows the most severe active alert:
  - Content: its message (`INSP-01 is FAULT`), its severity and time, and its ISA-18.2 lifecycle as three steps (`Raised → Acknowledged → Cleared`), coloured as far as the alert has got.
  - Behaviour: it pulses only while the alert is active and unacknowledged; clicking it opens the Alerts tab. With no active alerts it shows one quiet line: `No active alerts`.
- **The line strip** (HUD, bottom edge of the stage) replaces today's left-hand machine list:
  - Content: the machines in topology order, joined by flow arrows and buffer fill marks, so the HUD mirrors the physical line beneath it. Each cell shows the mono ID, the state in §3.4 grammar, the cause line (`BLOCKED by CNC-01`), and DQ and alert flags.
  - Behaviour: it keeps the listbox semantics and every `data-testid` (`machine-list`, `machine-item-<id>`): ArrowLeft/Right **and** ArrowUp/Down, Home, End, Escape. With more machines than fit, it scrolls horizontally.
- **The inspector** replaces today's details panel and commands panel. It is a docked surface (not glass, because it holds dense text) with sections separated by hairlines, in this order: identity, state, readings stat row, prediction, active alerts, commands. STOP stays the only red button.
- **The console** replaces the tabbed bottom panel. It is a docked surface with a collapse toggle (expanded, or tab strip only). Dense tables (alerts, predictions, what-if ghosts) live here, never on glass.
- **System bands** stay full width under the top bar: connection, drift and replay. The replay band keeps the amber frame on the stage.
- **Difference from today.** Today there are three boxed columns (list | 3D card plus tab box | details box) and `min-width: 1200px`. The new layout drops the left column (the stage gains about 250 px), moves machines onto the stage as a topology strip, and removes the nested cards.
- **Breakpoints.** At ≥ 1600 px the inspector is 440 px. From 1280 to 1599 px it is 360 px. From 1024 to 1279 px the inspector becomes a drawer over the stage that opens on selection. Below 1024 px everything stacks: stage at 50 vh, strip, inspector, console. Body text is never smaller than 13 px.

## 6. Components: reuse, don't hand-roll

There is **no shadcn/ui, no Tailwind and no component library**. The styling is plain CSS: `tokens.css` + `fonts.css` + `styles.css` in `frontend/src/dashboard/`. The redesign keeps that: no Tailwind, no CSS-in-JS. These are the existing pieces to reuse:

| Need | Use | Path |
|---|---|---|
| Tabs (ARIA tablist) | `Tabs`, `TabSpec` | `frontend/src/dashboard/components/Tabs.tsx` |
| Confirm modal | `ConfirmDialog` | `frontend/src/dashboard/components/ConfirmDialog.tsx` |
| State badge | `StateBadge` (+ `stateTone` from `format.ts`) | `frontend/src/dashboard/components/StateBadge.tsx` |
| Alert-kind badge (predicted / drift / unusual) | `AlertKindBadge` | `frontend/src/dashboard/components/AlertsPanel.tsx` |
| Alert times | `AlertTimesText` | `frontend/src/dashboard/components/AlertTimesText.tsx` |
| Connection | `ConnectionPill`, `ConnectionBanner` | `frontend/src/dashboard/components/ConnectionBanner.tsx` |
| Drift banner | `DriftBanner` | `frontend/src/dashboard/components/DriftBanner.tsx` |
| Replay | `ReplayButton`, `ReplayBar` | `frontend/src/dashboard/components/ReplayBar.tsx` |
| API key and role | `ApiKeyField` | `frontend/src/dashboard/components/ApiKeyField.tsx` |
| Prediction pieces | `PredictionSummary`, `UnusualIndicator`, `AboutPredictions` | `frontend/src/dashboard/components/PredictionPanel.tsx` |
| Confidence words | `confidenceText`, `ttfText`, `gateView`, `classTone` | `frontend/src/dashboard/state/predictions.ts` |
| Alert grouping | `groupAlerts`, `alertKindView` | `frontend/src/dashboard/state/alerts.ts` |
| Charts | `SensorChart`, `ProjectionChart`, `lineChartConfig`, `CHART_COLORS`, `chartFactory` | `frontend/src/dashboard/charts/` |
| Formatting | `formatValue`, `formatTime`, `formatAge`, `formatPct`, `axisTitle`, `displayUnit` | `frontend/src/dashboard/format.ts` |
| 3D host | `TwinView`, `twinEngineFactory`, `ENGINE_THEME` | `components/TwinView.tsx`, `engineFactory.ts`, `theme.ts` |

Data comes only from the existing model and services (`useModel`, `useServices`, `useStore`). Never call `fetch` from a component. **Keep every `data-testid`, ARIA role, keyboard behaviour and user-visible sentence** that a test or e2e spec relies on.

## 7. Motion

- **One curve:** `--ease: cubic-bezier(0.2, 0, 0, 1)`. **Durations:** `--dur-1: 120ms` (hover, press), `--dur-2: 200ms` (panel open, tab change, value tint).
- **Only state changes animate:**
  - A **fault pulse** plays only while an alarm is active and unacknowledged. The solid annunciator or badge pulses opacity 1 → 0.45 on a 1 s cycle; once acknowledged it holds steady (ISA-18.2).
  - On a **value change**, a reading whose machine state changes tints in the new status ink for 200 ms. Ordinary ticks do not animate.
  - The inspector drawer and console collapse use `--dur-2`.
- Nothing decorative: no entrance animations, no shimmer skeletons, no parallax.
- **`prefers-reduced-motion: reduce`:** the duration tokens become 0. The pulse becomes a static 2 px double border.

## 8. Data visualisation

- **Trend line:** `--text-1` at 85 % opacity, 1.25 px, no points, `tension: 0`. Charts show measurements, not status, so they stay neutral. A history min–max band uses the line colour at 10 %.
- **Grid:** horizontal gridlines only, in `--hairline`, at most 4. No vertical grid, no chart border.
- **Axes:** tick labels in Plex Mono 400 at 11 px. Axis titles are micro labels (`TEMP · °C`, `TIME · UTC`).
- **Sparkline details (after ref-1):** a 4 px dot on the last value with its value in mono, and start and end time labels.
- **Gaps stay gaps.** `spanGaps: false` on every dataset is unchanged, and a null is never drawn as zero. Runs of nulls also get a hatched `--hairline` band labelled `NO DATA` when it is wide enough (≥ 48 px). This is an inline Chart.js plugin; no dependency. (Stage 6)
- **Fault markers:** a vertical 1 px line at each alert's raise time for that machine, in its severity fill, with a 6 px cap and a micro-label annotation (`FAULT 00:12:57`). Predicted kinds are dashed; `unusual_behaviour` uses `--st-unusual`. This is an inline plugin, reading alerts already in the model. (Stage 6)
- **Threshold lines: none.** The backend provides no alarm limits (`docs/frontend.md`: the simulator's limits are perturbed per run, and the `/line` ranges are ingest plausibility bounds). Drawing a threshold would mean inventing one.
- **Tooltip:** `--surface-2`, 1 px `--hairline-strong`, 4 px radius, no colour swatches. The title is the UTC time in mono; the value is in mono with its unit. A gap shows `no data`.
- **What-if projection:** "no change" uses `--text-2`; "with change" uses `--accent`. The projected part is dashed (§3.4).

## 9. Forbidden

- Three (or N) equal rounded cards in a row; KPI tile grids (ref-2 notwithstanding). Use hairline stat rows.
- Centred layouts and centred body text. Text aligns left, numbers align right. The only centred element is the annunciator.
- Low-opacity drop shadows on everything; any shadow except the modal's.
- Glassmorphism anywhere except the floating HUD layer over the stage (annunciator, line strip).
- Emoji or decorative icons. Glyphs allowed: `●` `○` status dots, `→` flow, `▸` disclosure, `×` close.
- Placeholder or Lorem ipsum copy. Use real content: `CNC-01 · CNC machining station`, `vib 4.21 mm/s`, `INSP-01 is FAULT`, `Most likely WARNING (0.88) · no other state in the set`.
- Purple or indigo outside "unusual behaviour, cause unknown". Gradients as decoration.
- Status colours used for decoration, or the accent used for status.
- Weight 200 below 40 px. `--text-3` on the HUD.
- Removing or rewording anything on the mandatory list (§11).

## 10. 3D realism

### Achievable now (dashboard code only, through the public API)

| Aspect | How | Through |
|---|---|---|
| Palette and materials | Pass a `VisualConfig` to `new TwinEngine(container, { config })` in `engineFactory.ts`: graphite machine body (`body #9EA6AE`, `frame #59616A`, `belt #2B3036`), cardboard items (`item #B89A6A`), and state colours from §3.3 (`dataQuality` → degraded, `unusual` → purple only, `highlight` → explained, `selection` → bone, `waiting` and `offline` → inactive, `label` → text-1 on surface-0) | `TwinEngineOptions.config` |
| Background and floor | Scene `#111418`, floor `#171B20`, grid `#222830` | `setTheme(EngineTheme)` |
| State glow | `flashHz` 2 → 1.25 (a calmer, ISA-like blink); outline thickness | `config.flashHz`, `config.outlineThickness` |
| Labels | Size and fade distances, so the strip carries names and the scene stays quiet | `config.labels` |
| Sharpness and perspective | `render.maxPixelRatio`; `render.fov` 45 → 38 | `config.render` |
| Edge treatment | CSS radial vignette on the stage (`::after`, `pointer-events: none`) and the replay frame | CSS on `.twin` |
| Camera | `focus(machineId)` on double-click in the strip | `engine.focus` |

### Stage 7: TwinEngine changes (approved list; additive and opt-in through `config`; separate commits; diff summary before starting)

1. **Light rig.** Today: a hemisphere light plus one white sun. New: warm key, cool fill and rim lights from `config.lighting`.
2. **Soft shadows.** `PCFSoftShadowMap`, and a shadow camera fitted tightly to the line.
3. **Environment reflections.** `RoomEnvironment` + `PMREMGenerator` (both ship with three.js; no new dependency).
4. **Fog,** so the floor fades out instead of ending in a hard-edged grid.
5. **View insets.** `setViewInsets({ top, right, bottom, left })`, so the auto-fit frames the line inside the part of the stage the HUD leaves free.

**Additional proposals, not approved yet:**

6. A ground ring or glow under the selected and faulted machine (both references).
7. Label and tooltip fonts from config. `labels.ts` and `tooltipView.ts` hard-code `system-ui`, which §2 forbids.
8. Tone-mapping exposure in config.

## 11. Mandatory elements: restyle, never remove

| Element | Today |
|---|---|
| "About these predictions" disclosure | `AboutPredictions`, Prediction tab |
| "Most likely … · cannot rule out …" wording | `confidenceText` (predictions table, `PredictionSummary`) |
| Purple "unusual behaviour · cause unknown" alarms | `AlertKindBadge`, `UnusualIndicator`, tinted alert row, 3D whole-machine outline |
| Reliability `OK` / `DEGRADED` badges | Predictions table, `PredictionSummary` |
| Drift banner | `DriftBanner` (+ `model drift` alert badge) |
| "SIMULATION ONLY" labels | `ScenarioPanel` (`role="note"`); What-if copy: "a simulation, not a promise" |
| Gaps (not zeros) for missing values | `charts/series.ts` null points, `spanGaps: false` |
