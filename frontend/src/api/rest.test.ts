import { describe, expect, it } from "vitest";

import openapi from "./openapi.json";
import { ApiError, ENDPOINTS, RestClient } from "./rest.ts";

interface Call {
  url: string;
  init: RequestInit;
}

function fakeFetch(status: number, body: unknown, calls: Call[]): typeof fetch {
  return (input, init) => {
    calls.push({ url: input as string, init: init ?? {} });
    const text = typeof body === "string" ? body : JSON.stringify(body);
    return Promise.resolve(new Response(text, { status, headers: { "Content-Type": "application/json" } }));
  };
}

describe("RestClient", () => {
  it("every endpoint it uses exists in the committed OpenAPI schema", () => {
    const paths = openapi.paths as Record<string, Record<string, unknown>>;
    for (const [method, path] of ENDPOINTS) {
      expect(paths[path]?.[method], `${method.toUpperCase()} ${path}`).toBeDefined();
    }
  });

  it("sends the API key header and parses JSON", async () => {
    const calls: Call[] = [];
    const client = new RestClient({
      baseUrl: "http://127.0.0.1:8000/",
      apiKey: "secret",
      fetch: fakeFetch(200, { lineId: "line1" }, calls),
    });
    const line = await client.getLine();
    expect(line.lineId).toBe("line1");
    expect(calls[0]?.url).toBe("http://127.0.0.1:8000/line");
    expect(calls[0]?.init.method).toBe("GET");
    expect((calls[0]?.init.headers as Record<string, string>)["X-API-Key"]).toBe("secret");
  });

  it("omits the key when none is configured and never puts it in the URL", async () => {
    const calls: Call[] = [];
    const client = new RestClient({ baseUrl: "http://h", fetch: fakeFetch(200, {}, calls) });
    await client.getHealth();
    expect((calls[0]?.init.headers as Record<string, string>)["X-API-Key"]).toBeUndefined();
    const keyed = new RestClient({ baseUrl: "http://h", apiKey: "k1", fetch: fakeFetch(200, {}, calls) });
    await keyed.getMachines();
    expect(calls.every((c) => !c.url.includes("k1"))).toBe(true);
  });

  it("fills path parameters and query strings (arrays repeat, nulls are dropped)", async () => {
    const calls: Call[] = [];
    const client = new RestClient({ baseUrl: "http://h", fetch: fakeFetch(200, {}, calls) });
    await client.getMachineState("CNC 01");
    await client.getHistory({
      from: "2026-01-01T00:00:00Z",
      to: "2026-01-01T00:01:00Z",
      resolution: "1m",
      machines: ["CNC-01", "CNV-01"],
    });
    await client.getAlerts({ active: true, machine: null });
    expect(calls.map((c) => c.url)).toEqual([
      "http://h/machines/CNC%2001/state",
      "http://h/history?from=2026-01-01T00%3A00%3A00Z&to=2026-01-01T00%3A01%3A00Z&resolution=1m&machines=CNC-01&machines=CNV-01",
      "http://h/alerts?active=true",
    ]);
  });

  it("posts commands as JSON", async () => {
    const calls: Call[] = [];
    const client = new RestClient({
      baseUrl: "http://h",
      apiKey: "op",
      fetch: fakeFetch(200, { commandId: "c1", status: "EXECUTED", reason: null, ts: "2026-01-01T00:00:00Z" }, calls),
    });
    const ack = await client.postCommand({ machineId: "CNC-01", type: "STOP", params: {}, idempotencyKey: "k-1" });
    expect(ack.status).toBe("EXECUTED");
    expect(calls[0]?.init.method).toBe("POST");
    expect(JSON.parse(calls[0]?.init.body as string)).toEqual({
      machineId: "CNC-01",
      type: "STOP",
      params: {},
      idempotencyKey: "k-1",
    });
    expect((calls[0]?.init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    await client.ackAlert("CNC-01:line_stoppage:1");
    expect(calls[1]?.url).toBe("http://h/alerts/CNC-01%3Aline_stoppage%3A1/ack");
  });

  it("throws ApiError with status and detail", async () => {
    const client = new RestClient({ baseUrl: "http://h", fetch: fakeFetch(404, { detail: "unknown machine 'X'" }, []) });
    const err = await client.getMachineState("X").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 404, detail: "unknown machine 'X'", path: "/machines/{machine_id}/state" });
    const plain = new RestClient({ baseUrl: "http://h", fetch: fakeFetch(500, "boom", []) });
    await expect(plain.getLine()).rejects.toMatchObject({ status: 500, detail: "boom" });
  });
});
