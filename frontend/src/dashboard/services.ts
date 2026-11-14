/** Builds the dashboard's model, session and default (real) factories, and starts the session. */
import { chartJsFactory } from "./charts/chartFactory.ts";
import { twinEngineFactory } from "./engineFactory.ts";
import type { DashboardServices } from "./context.ts";
import { loadRememberedKey } from "./state/auth.ts";
import { DashboardModel } from "./state/model.ts";
import { ReplayController } from "./state/replay.ts";
import { Session, type SessionOptions } from "./state/session.ts";

export function createServices(opts: SessionOptions, start = true): DashboardServices {
  const model = new DashboardModel();
  const session = new Session(model, opts);
  if (start) {
    const remembered = loadRememberedKey(opts.storage);
    void session.applyKey(remembered, remembered !== null);
  }
  const replay = new ReplayController({
    model,
    rest: () => session.rest.get(),
    resubscribe: () => {
      session.resubscribe();
    },
    cfg: model.cfg,
  });
  return { model, session, chartFactory: chartJsFactory, engineFactory: twinEngineFactory, replay };
}
