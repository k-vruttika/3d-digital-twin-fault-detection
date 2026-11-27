/** The 3D engine as the dashboard sees it, and the default factory (engine in its own chunk). */
import type { SelectEvent } from "../engine/TwinEngine.ts";
import type { LabelMode } from "../engine/labelPolicy.ts";
import type { EngineLike } from "./state/model.ts";

/** What the dashboard needs from the engine on top of the model's sink interface. */
export interface EngineHandle extends EngineLike {
  on(event: "select", cb: (e: SelectEvent) => void): () => void;
  setTheme(theme: { background: number; replayBackground: number; floor: number; grid: number }): void;
  resize(): void;
  dispose(): void;
  readonly labelMode: LabelMode;
}

export type EngineFactory = (container: HTMLElement) => Promise<EngineHandle>;

export const twinEngineFactory: EngineFactory = async (container) => {
  const [{ TwinEngine }, { DEFAULT_CONFIG }, { sceneConfig }] = await Promise.all([
    import("../engine/TwinEngine.ts"),
    import("../engine/config.ts"),
    import("./sceneConfig.ts"),
  ]);
  return new TwinEngine(container, { config: sceneConfig(DEFAULT_CONFIG) });
};
