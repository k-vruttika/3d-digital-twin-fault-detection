/** The dashboard with its services in context (tests render it with fakes). */
import { App } from "./App.tsx";
import { DashboardContext, type DashboardServices } from "./context.ts";

export function Dashboard({ services }: { services: DashboardServices }) {
  return (
    <DashboardContext.Provider value={services}>
      <App />
    </DashboardContext.Provider>
  );
}
