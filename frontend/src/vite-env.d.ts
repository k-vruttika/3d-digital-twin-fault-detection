/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL, default http://127.0.0.1:8000. */
  readonly VITE_LINETWIN_API_URL?: string;
  /** API key (viewer is enough); put it in frontend/.env.local (gitignored), never in a URL. */
  readonly VITE_LINETWIN_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
