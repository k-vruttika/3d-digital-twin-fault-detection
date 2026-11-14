/** The dashboard layout: header, banners, machine list | 3D view + tabs | details + commands. */
import { useEffect } from "react";

import { HistoryPanel } from "./charts/HistoryPanel.tsx";
import { LiveCharts } from "./charts/LiveCharts.tsx";
import { AlertsPanel } from "./components/AlertsPanel.tsx";
import { CommandPanel } from "./components/CommandPanel.tsx";
import { PredictionPanel, WhatIfPanel } from "./components/PlaceholderPanels.tsx";
import { ReplayBar, ReplayButton } from "./components/ReplayBar.tsx";
import { ScenarioPanel } from "./components/ScenarioPanel.tsx";
import { SourcesPanel } from "./components/SourcesPanel.tsx";
import { ConnectionBanner } from "./components/ConnectionBanner.tsx";
import { DetailsPanel } from "./components/DetailsPanel.tsx";
import { Header } from "./components/Header.tsx";
import { MachineList } from "./components/MachineList.tsx";
import { Tabs, type TabSpec } from "./components/Tabs.tsx";
import { TwinView } from "./components/TwinView.tsx";
import { useModel, useServices } from "./context.ts";
import { groupAlerts } from "./state/alerts.ts";
import { useStore } from "./state/store.ts";
import { useTheme } from "./theme.ts";
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
  useSourcesPolling();
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
      <ReplayBar replay={replay} />
      <div className="layout">
        <aside className="left">
          <MachineList />
        </aside>
        <main id="main" className="center" tabIndex={-1}>
          <TwinView />
          <section className="bottom panel" aria-label="Charts, history and alerts">
            <Tabs label="Panels" tabs={tabs} />
          </section>
        </main>
        <aside className="right">
          <DetailsPanel />
          <CommandPanel />
        </aside>
      </div>
    </div>
  );
}
