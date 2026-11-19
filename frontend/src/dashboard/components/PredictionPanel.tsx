/**
 * Live predictions (Phase 11, ADR 0030): per machine the current state next to the predicted
 * state at +10 s / +30 s with calibrated confidence, the conformal set in words, time to fault
 * with its interval, the reliability flag and the alarm; for the selected machine the
 * attributions (supervised model only) and the autoencoder score; the model versions and an
 * honest note about where the models come from.
 */
import type { Prediction } from "../../api/types.ts";
import { useModel } from "../context.ts";
import { formatTime } from "../format.ts";
import { classTone, gateView, predictionAt, setText, topClass, ttfText } from "../state/predictions.ts";
import { useStore } from "../state/store.ts";
import { StateBadge } from "./StateBadge.tsx";

const HORIZONS = [10, 30];

function ClassCell({ p, testId }: { p: Prediction | null; testId: string }) {
  if (p === null) return <td data-testid={testId}>–</td>;
  const { cls, prob } = topClass(p);
  return (
    <td data-testid={testId}>
      <span className={`badge tone-${classTone(cls)}`}>{cls}</span>{" "}
      <span className="small">{prob.toFixed(2)}</span>
    </td>
  );
}

function useInferenceInfo() {
  const model = useModel();
  const inference = useStore(model.inference);
  return { inference, info: inference?.inference ?? null };
}

/** Model versions and inference state, one line. */
function InferenceLine() {
  const { info } = useInferenceInfo();
  const mode = useStore(useModel().modeStore);
  if (info === null) {
    return <p className="small muted">Inference state not known yet (GET /predict has not answered).</p>;
  }
  if (!info.enabled) {
    return (
      <p className="small" data-testid="prediction-off">
        Live inference is off. {info.reason ?? ""}
      </p>
    );
  }
  return (
    <p className="small" data-testid="prediction-model">
      <span className={`badge ${info.state === "ready" ? "tone-ok" : "tone-warning"}`}>{info.state}</span>{" "}
      Model: <strong>{info.modelVersion ?? info.model ?? "–"}</strong>
      {mode === "replay" && <span className="muted"> · replay shows the stored predictions as they were</span>}
      {info.reason !== null && info.reason !== undefined && info.state !== "ready" && (
        <span className="error"> · {info.reason}</span>
      )}
    </p>
  );
}

export function AboutPredictions() {
  const { info } = useInferenceInfo();
  const card = info?.supervised?.card ?? null;
  const alpha = info?.supervised?.alpha;
  const coverage = alpha === null || alpha === undefined ? "90 %" : `${String(Math.round((1 - alpha) * 100))} %`;
  return (
    <details className="about small" data-testid="prediction-about">
      <summary>About these predictions</summary>
      <ul>
        <li>
          <strong>Simulator data only.</strong> The models were trained, validated and calibrated on simulated data
          from the LineTwin simulator.
        </li>
        <li>
          <strong>Not shown to work on real machines.</strong> Phase 10 tested the same methods on two public datasets
          and did not demonstrate early warning on real machines.
        </li>
        <li>
          Confidence is calibrated, and the sets hold the true state about {coverage} of the time on clean simulated
          data. Neither is guaranteed when reliability is DEGRADED (noisy or missing data, drift).
        </li>
        <li>
          Each state has its own threshold for the set (Mondrian), so a set can leave out the most likely state or
          include an unlikely one: a healthy machine at NORMAL 0.996 (shown as 1.00) can get the set “FAULT”. The set
          is a coverage statement, not a second prediction.
        </li>
        <li>
          An alarm is raised by the evaluated rule (the +30 s prediction is WARNING or FAULT for 3 s in a row), not by
          the confidence.
        </li>
        <li>
          “Unusual behaviour” comes from an autoencoder: the inputs look unlike normal operation. It names no cause and
          no part, and it raises false alarms now and then.
        </li>
        <li>
          Highlighted parts come from the supervised model’s explanation of its own alarm, shown only while it predicts
          WARNING or FAULT.
        </li>
        <li>No prediction is made while a machine is stopped, waiting, or within its first 30 s of running.</li>
      </ul>
      {card !== null && (
        <div data-testid="prediction-card">
          <p>{card.trainingData}</p>
          <p>{card.realData}</p>
          <ul>
            {card.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </details>
  );
}

function SelectedDetails({ id }: { id: string }) {
  const model = useModel();
  const preds = useStore(model.predictions)[id];
  const state = useStore(model.machines)[id]?.state ?? null;
  const asOf = useStore(model.asOf);
  const mode = useStore(model.modeStore);
  const { inference, info } = useInferenceInfo();
  const p = predictionAt(preds, model.primaryHorizonS());
  if (p === null) return null;
  // The same gate as the table row: no stale alarm / score once the machine stops or waits.
  const gate = gateView(inference, mode, id, state, p, asOf, model.predictionStaleMs());
  if (gate.status !== "predict") {
    return (
      <div className="prediction-details" data-testid="prediction-details">
        <h3>{id}</h3>
        <p className="small muted" data-testid="prediction-details-gate">
          {gate.text}
        </p>
      </div>
    );
  }
  const att = p.attributions ?? [];
  return (
    <div className="prediction-details" data-testid="prediction-details">
      <h3>
        {id} at +{String(p.horizonS)} s <span className="muted small">· predicted at {formatTime(p.ts)}</span>
      </h3>
      <p className="small">
        Alarm (evaluated rule): <strong data-testid="prediction-alarm">{p.alarm === true ? "on" : "off"}</strong>
      </p>
      <p className="small" data-testid="prediction-attributions">
        Signals behind the prediction:{" "}
        {att.length === 0
          ? "none (shown only while the supervised model predicts WARNING or FAULT)"
          : att
              .slice(0, 3)
              .map((a) => `${a.sensor} → ${a.part} (${a.weight >= 0 ? "+" : ""}${a.weight.toFixed(2)})`)
              .join(", ")}
        {att.length > 0 && p.attributionsTs !== null && p.attributionsTs !== undefined && (
          <span className="muted"> · explains the window ending {formatTime(p.attributionsTs)}</span>
        )}
      </p>
      <p className="small" data-testid="prediction-anomaly">
        Unusual-behaviour detector (autoencoder): score {p.anomalyScore.toFixed(4)}
        {p.anomalyThreshold !== null && p.anomalyThreshold !== undefined && (
          <> / threshold {p.anomalyThreshold.toFixed(4)}</>
        )}{" "}
        {p.anomalyFlag === true ? (
          <span className="badge kind-unusual">above threshold · cause unknown</span>
        ) : (
          <span className="muted">normal</span>
        )}
      </p>
      {info?.anomaly && <p className="small muted">Detector: {info.anomaly.label}</p>}
    </div>
  );
}

export function PredictionPanel() {
  const model = useModel();
  const line = useStore(model.line);
  const machines = useStore(model.machines);
  const predictions = useStore(model.predictions);
  const selection = useStore(model.selection);
  const asOf = useStore(model.asOf);
  const mode = useStore(model.modeStore);
  const { inference, info } = useInferenceInfo();
  const staleMs = model.predictionStaleMs();
  const primary = model.primaryHorizonS();
  const cap = info?.supervised?.ttfCapS ?? null;
  return (
    <div data-testid="prediction-panel">
      <InferenceLine />
      <table className="predictions">
        <thead>
          <tr>
            <th scope="col">Machine</th>
            <th scope="col">Now</th>
            {HORIZONS.map((h) => (
              <th scope="col" key={h}>
                +{h} s
              </th>
            ))}
            <th scope="col">Set (+{primary} s)</th>
            <th scope="col">Time to fault</th>
            <th scope="col">Reliability</th>
          </tr>
        </thead>
        <tbody>
          {(line?.machines ?? []).map((m) => {
            const state = machines[m.id]?.state ?? null;
            const byH = predictions[m.id];
            const pPrimary = predictionAt(byH, primary);
            const gate = gateView(inference, mode, m.id, state, pPrimary, asOf, staleMs);
            const show = gate.status === "predict" && pPrimary !== null;
            return (
              <tr
                key={m.id}
                data-testid={`prediction-row-${m.id}`}
                className={selection === m.id ? "selected" : undefined}
                onClick={() => {
                  model.select(m.id);
                }}
              >
                <th scope="row">{m.id}</th>
                <td>
                  <StateBadge state={state} />
                </td>
                {show ? (
                  <>
                    {HORIZONS.map((h) => (
                      <ClassCell key={h} p={predictionAt(byH, h)} testId={`prediction-cell-${m.id}-${String(h)}`} />
                    ))}
                    <td data-testid={`prediction-set-${m.id}`}>{setText(pPrimary.conformalSet)}</td>
                    <td data-testid={`prediction-ttf-${m.id}`}>{ttfText(pPrimary, cap)}</td>
                    <td data-testid={`prediction-reliability-${m.id}`}>
                      <span className={`badge ${pPrimary.reliability === "OK" ? "tone-ok" : "tone-warning"}`}>
                        {pPrimary.reliability}
                      </span>
                      {pPrimary.alarm === true && <span className="badge tone-warning">alarm</span>}
                    </td>
                  </>
                ) : (
                  <td colSpan={HORIZONS.length + 3} className="muted" data-testid={`prediction-gate-${m.id}`}>
                    {gate.text}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {selection !== null && <SelectedDetails id={selection} />}
      <AboutPredictions />
    </div>
  );
}

/** One line for the details panel: the +30 s prediction, or why there is none. */
export function PredictionSummary({ id }: { id: string }) {
  const model = useModel();
  const predictions = useStore(model.predictions);
  const machines = useStore(model.machines);
  const asOf = useStore(model.asOf);
  const mode = useStore(model.modeStore);
  const { inference, info } = useInferenceInfo();
  if (info !== null && !info.enabled) return <p className="small muted">Live inference is off.</p>;
  const primary = model.primaryHorizonS();
  const p = predictionAt(predictions[id], primary);
  const gate = gateView(inference, mode, id, machines[id]?.state ?? null, p, asOf, model.predictionStaleMs());
  if (gate.status !== "predict" || p === null) {
    return (
      <p className="small muted" data-testid="details-prediction">
        {gate.text}
      </p>
    );
  }
  const { cls, prob } = topClass(p);
  return (
    <p className="small" data-testid="details-prediction">
      +{String(primary)} s: <span className={`badge tone-${classTone(cls)}`}>{cls}</span> {prob.toFixed(2)} · set{" "}
      {setText(p.conformalSet)} · {p.reliability}
      {p.alarm === true && <span className="badge tone-warning">alarm</span>}
      {p.anomalyFlag === true && <span className="badge kind-unusual">unusual</span>}
    </p>
  );
}
