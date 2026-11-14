/**
 * Dashboard entry (index.html). The backend URL comes from VITE_LINETWIN_API_URL (default
 * http://127.0.0.1:8000). The API key is entered in the page, never configured in a URL.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./dashboard/styles.css";
import { Dashboard } from "./dashboard/Dashboard.tsx";
import { createServices } from "./dashboard/services.ts";

const rootElement = document.getElementById("root");
if (rootElement === null) {
  throw new Error("#root element not found");
}

const baseUrl = import.meta.env.VITE_LINETWIN_API_URL ?? "http://127.0.0.1:8000";
const services = createServices({ baseUrl });

createRoot(rootElement).render(
  <StrictMode>
    <Dashboard services={services} />
  </StrictMode>,
);
