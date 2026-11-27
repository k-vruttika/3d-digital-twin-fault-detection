/**
 * The dashboard layout (DESIGN.md §5): top bar, system bands, then the stage (3D view with the
 * HUD layer) over the console (tabs) on the left, and the inspector (details + commands) right.
 */
import { useEffect, useState } from "react";

import { HistoryPanel } from "./charts/HistoryPanel.tsx";
import { LiveCharts } from "./charts/LiveCharts.tsx";
import { AlertsPanel } from "./components/AlertsPanel.tsx";
import { CommandPanel } from "./components/CommandPanel.tsx";
import { WhatIfPanel } from "./components/WhatIfPanel.tsx";
import { PredictionPanel } from "./components/PredictionPanel.tsx";
import { ReplayBar, ReplayButton } from "./components/ReplayBar.tsx";
import { ScenarioPanel } from "./components/ScenarioPanel.tsx";
import { SourcesPanel } from "./components/SourcesPanel.tsx";
import { ConnectionBanner } from "./components/ConnectionBanner.tsx";
import { DriftBanner } from "./components/DriftBanner.tsx";
import { DetailsPanel } from "./components/DetailsPanel.tsx";
import { Header } from "./components/Header.tsx";
import { MachineList } from "./components/MachineList.tsx";
import { Tabs, type TabSpec } from "./components/Tabs.tsx";
import { TwinView } from "./components/TwinView.tsx";
import { useModel, useServices } from "./context.ts";
import { groupAlerts } from "./state/alerts.ts";
import { useStore } from "./state/store.ts";
import { useTheme } from "./theme.ts";
import { useAlertBackfill } from "./useAlertBackfill.ts";
import { usePredictPolling } from "./usePredictPolling.ts";
import { useSourcesPolling } from "./useSourcesPolling.ts";

function AlertsTabLabel() {
  const alerts = useStore(useModel().alerts);
  const n = groupAlerts(alerts.values()).activeUnacked.length;
  return (
    <>
      Alerts{n > 0 && <span className="count" aria-label={`${String(n)} active unacknowledged`}>{n}</span>}
    </>
  );
}

export function App() {
  const { model, replay } = useServices();
  const theme = useTheme();
  const sources = useStore(model.sources);
  const [consoleCollapsed, setConsoleCollapsed] = useState(false);
  useSourcesPolling();
  usePredictPolling();
  useAlertBackfill();
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(
    () => () => {
      replay.dispose();
    },
    [replay],
  );
  const scenario = (sources?.capabilities ?? []).includes("scenario");
  const tabs: TabSpec[] = [
    { id: "live", label: "Live charts", content: <LiveCharts /> },
    { id: "history", label: "History", content: <HistoryPanel /> },
    { id: "alerts", label: <AlertsTabLabel />, content: <AlertsPanel /> },
    { id: "sources", label: "Sources", content: <SourcesPanel /> },
    ...(scenario ? [{ id: "scenario", label: "Scenario (sim)", content: <ScenarioPanel /> }] : []),
    { id: "prediction", label: "Prediction", content: <PredictionPanel /> },
    { id: "whatif", label: "What-if", content: <WhatIfPanel /> },
  ];
  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to the 3D view
      </a>
      <Header>
        <ReplayButton replay={replay} />
      </Header>
      <ConnectionBanner />
      <DriftBanner />
      <ReplayBar replay={replay} />
      <div className="layout">
        <main id="main" className="stage-col" tabIndex={-1}>
          <div className="stage">
            <TwinView />
            <div className="hud hud-machines">
              <MachineList />
            </div>
          </div>
          <section
            className="console"
            id="console"
            aria-label="Charts, history and alerts"
            data-collapsed={consoleCollapsed}
            data-testid="console"
          >
            <button
              type="button"
              className="console-toggle"
              aria-expanded={!consoleCollapsed}
              aria-controls="console"
              onClick={() => {
                setConsoleCollapsed((c) => !c);
              }}
              data-testid="console-toggle"
            >
              {consoleCollapsed ? "Expand panels ▸" : "Collapse panels"}
            </button>
            <Tabs label="Panels" tabs={tabs} />
          </section>
        </main>
        <aside className="inspector" aria-label="Selected machine">
          <DetailsPanel />
          <CommandPanel />
        </aside>
      </div>
    </div>
  );
}
