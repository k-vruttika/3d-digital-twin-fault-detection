/**
 * Prediction and what-if placeholders. The backend answers 501 until Phase 11; the panels say
 * so instead of failing.
 */
import { useEffect, useState } from "react";

import { ApiError } from "../../api/rest.ts";
import { useRest } from "../context.ts";

type Outcome = { kind: "idle" } | { kind: "loading" } | { kind: "notYet"; detail: string } | { kind: "error"; detail: string } | { kind: "ok" };

function describe(e: unknown): Outcome {
  if (e instanceof ApiError && e.status === 501) {
    return { kind: "notYet", detail: typeof e.detail === "string" ? e.detail : "not implemented" };
  }
  return { kind: "error", detail: e instanceof ApiError ? `HTTP ${String(e.status)}` : String(e) };
}

function OutcomeView({ outcome, what }: { outcome: Outcome; what: string }) {
  switch (outcome.kind) {
    case "idle":
      return null;
    case "loading":
      return <p className="muted small">Asking the backend…</p>;
    case "notYet":
      return (
        <div className="result" data-testid="placeholder-501">
          <strong>{what}: available in Phase 11.</strong>
          <div className="small muted">Backend: {outcome.detail}</div>
        </div>
      );
    case "error":
      return (
        <p className="error small" role="alert">
          {what} could not be loaded ({outcome.detail}).
        </p>
      );
    case "ok":
      return <p className="small">The backend answered; this panel is filled in Phase 11.</p>;
  }
}

export function PredictionPanel() {
  const rest = useRest();
  const [outcome, setOutcome] = useState<Outcome>({ kind: "loading" });
  useEffect(() => {
    let cancelled = false;
    rest
      .getPredict()
      .then(() => {
        if (!cancelled) setOutcome({ kind: "ok" });
      })
      .catch((e: unknown) => {
        if (!cancelled) setOutcome(describe(e));
      });
    return () => {
      cancelled = true;
    };
  }, [rest]);
  return (
    <div data-testid="prediction-panel">
      <p className="small muted">
        Fault predictions (state at +10 s / +30 s, time to fault) with calibrated confidence and explanations on the 3D parts.
      </p>
      <OutcomeView outcome={outcome} what="Predictions" />
    </div>
  );
}

export function WhatIfPanel() {
  const rest = useRest();
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const run = (): void => {
    setOutcome({ kind: "loading" });
    rest
      .postWhatif({})
      .then(() => {
        setOutcome({ kind: "ok" });
      })
      .catch((e: unknown) => {
        setOutcome(describe(e));
      });
  };
  return (
    <div data-testid="whatif-panel">
      <p className="small muted">What-if: fast-forward a copy of the simulator from now with changed settings and see the projected trajectory.</p>
      <button type="button" onClick={run} data-testid="whatif-run">
        Run what-if
      </button>
      <OutcomeView outcome={outcome} what="What-if" />
    </div>
  );
}
