/** State badge: colour plus text (colour is never the only cue). */
import type { MachineState } from "../../api/types.ts";
import { stateTone } from "../format.ts";

export function StateBadge({ state }: { state: MachineState | null }) {
  return <span className={`badge tone-${stateTone(state)}`}>{state ?? "NO DATA"}</span>;
}
