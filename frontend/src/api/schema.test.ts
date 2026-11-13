/**
 * The generated types are up to date: regenerating schema.gen.ts from the committed
 * openapi.json gives the committed file (the backend side checks openapi.json itself).
 */
import { readFileSync } from "node:fs";

import openapiTS, { astToString, COMMENT_HEADER } from "openapi-typescript";
import { describe, expect, it } from "vitest";

describe("schema.gen.ts", () => {
  it("matches openapi-typescript's output for openapi.json", async () => {
    const schemaUrl = new URL("./openapi.json", import.meta.url);
    const expected = `${COMMENT_HEADER}${astToString(await openapiTS(schemaUrl, { silent: true }))}`;
    const committed = readFileSync(new URL("./schema.gen.ts", import.meta.url), "utf8").replace(/\r\n/g, "\n");
    expect(committed === expected, "src/api/schema.gen.ts is stale: run `uv run poe api-docs`").toBe(true);
  }, 30_000);
});
