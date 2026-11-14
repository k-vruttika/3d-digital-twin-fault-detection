/** React access to the dashboard model, session and injectable factories (tests swap them). */
import { createContext, useContext } from "react";

import type { RestClient } from "../api/rest.ts";
import type { ChartFactory } from "./charts/chartFactory.ts";
import type { EngineFactory } from "./engineFactory.ts";
import type { DashboardModel } from "./state/model.ts";
import type { ReplayController } from "./state/replay.ts";
import { useStore } from "./state/store.ts";
import type { Session } from "./state/session.ts";

export interface DashboardServices {
  model: DashboardModel;
  session: Session;
  chartFactory: ChartFactory;
  engineFactory: EngineFactory;
  replay: ReplayController;
}

export const DashboardContext = createContext<DashboardServices | null>(null);

export function useServices(): DashboardServices {
  const s = useContext(DashboardContext);
  if (s === null) throw new Error("DashboardContext missing");
  return s;
}

export function useModel(): DashboardModel {
  return useServices().model;
}

export function useRest(): RestClient {
  return useStore(useServices().session.rest);
}

/** Operator actions are possible: operator key, stream connected, live mode. */
export function useCanAct(): { canAct: boolean; operator: boolean; why: string | null } {
  const model = useModel();
  const auth = useStore(model.auth);
  const conn = useStore(model.connection);
  const mode = useStore(model.modeStore);
  const operator = auth.status === "ok" && auth.role === "operator";
  let why: string | null = null;
  if (!operator) why = "an operator API key is required";
  else if (mode === "replay") why = "not available in replay mode; return to live first";
  else if (conn.status !== "connected") why = "the stream is not connected";
  return { canAct: why === null, operator, why };
}
