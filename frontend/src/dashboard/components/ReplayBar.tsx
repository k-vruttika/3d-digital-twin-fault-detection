/**
 * The REPLAY banner and timeline: scrubber over the stored range, play / pause at 1x / 5x /
 * 20x, and one click back to live. The 3D view, list, details and alerts show the line as of
 * the scrubbed time; commands are disabled meanwhile.
 */
import { useModel } from "../context.ts";
import { formatDateTime } from "../format.ts";
import type { ReplayController } from "../state/replay.ts";
import { useStore } from "../state/store.ts";

export function ReplayButton({ replay }: { replay: ReplayController }) {
  const s = useStore(replay.state);
  const mode = useStore(useModel().modeStore);
  if (s.active || mode === "replay") return null;
  return (
    <button
      type="button"
      onClick={() => {
        void replay.enter();
      }}
      data-testid="enter-replay"
      title="Scrub back through the stored history"
    >
      Replay
    </button>
  );
}

export function ReplayBar({ replay }: { replay: ReplayController }) {
  const model = useModel();
  const s = useStore(replay.state);
  const speeds = model.cfg.replaySpeeds;
  if (!s.active) return null;
  const ready = s.start !== null && s.end !== null && s.t !== null;
  return (
    <div className="banner banner-replay" role="region" aria-label="Replay" data-testid="replay-banner">
      <span className="replay-tag">REPLAY</span>
      <span data-testid="replay-time" aria-live="polite">
        {s.t === null ? "loading…" : formatDateTime(s.t)}
      </span>
      {ready && (
        <>
          <button
            type="button"
            onClick={() => {
              if (s.playing) replay.pause();
              else replay.play();
            }}
            data-testid="replay-play"
          >
            {s.playing ? "Pause" : "Play"}
          </button>
          <span role="group" aria-label="Replay speed" className="seg">
            {speeds.map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={s.speed === v}
                onClick={() => {
                  replay.setSpeed(v);
                }}
              >
                {v}×
              </button>
            ))}
          </span>
          <input
            type="range"
            className="scrubber"
            aria-label="Replay time"
            min={s.start ?? 0}
            max={s.end ?? 0}
            step={1000}
            value={s.t ?? 0}
            onChange={(e) => {
              replay.seek(Number(e.target.value));
            }}
            data-testid="replay-scrubber"
          />
          <span className="small muted">
            {formatDateTime(s.start)} → {formatDateTime(s.end)}
          </span>
        </>
      )}
      {s.loading && <span className="small muted">loading…</span>}
      {s.error !== null && (
        <span className="error small" role="alert">
          {s.error}
        </span>
      )}
      <span className="spacer" />
      <button
        type="button"
        className="live-button"
        onClick={() => {
          replay.exit();
        }}
        data-testid="exit-replay"
      >
        Return to live
      </button>
    </div>
  );
}
