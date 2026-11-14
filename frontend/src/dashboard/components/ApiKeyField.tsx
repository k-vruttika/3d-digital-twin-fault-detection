/**
 * API key entry. The key is held in memory (or sessionStorage with "remember in this tab"),
 * never in localStorage or a URL. The current role is shown next to it.
 */
import { useState } from "react";

import { useModel, useServices } from "../context.ts";
import { roleText } from "../state/auth.ts";
import { useStore } from "../state/store.ts";

export function ApiKeyField() {
  const { session } = useServices();
  const auth = useStore(useModel().auth);
  const [draft, setDraft] = useState("");
  const [remember, setRemember] = useState(auth.remember);

  const submit = (): void => {
    void session.applyKey(draft.trim() === "" ? null : draft.trim(), remember);
    setDraft("");
  };

  return (
    <form
      className="keyfield"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      aria-label="API key"
    >
      <label className="sr-only" htmlFor="api-key">
        API key
      </label>
      <input
        id="api-key"
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder={auth.key !== null ? "key set (enter to replace)" : "API key"}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
        }}
      />
      <label className="check" title="Keep the key in this browser tab only (sessionStorage)">
        <input
          type="checkbox"
          checked={remember}
          onChange={(e) => {
            setRemember(e.target.checked);
          }}
        />
        remember in this tab
      </label>
      <button type="submit">Apply</button>
      {auth.key !== null && (
        <button
          type="button"
          onClick={() => {
            void session.applyKey(null, false);
          }}
        >
          Clear
        </button>
      )}
      <span className={`role role-${auth.role ?? "none"} auth-${auth.status}`} data-testid="role">
        {roleText(auth)}
      </span>
    </form>
  );
}
