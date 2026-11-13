"""Render the REST API reference into docs/api.md from the app's OpenAPI schema.

``uv run poe api-docs`` rewrites the part of ``docs/api.md`` between the BEGIN/END markers; the
hand-written sections around it (WebSocket protocol, auth, commands, alerts) stay as they are.
``tests`` check that the committed file is up to date, so the reference cannot drift from the
code. Output is deterministic (sorted, no timestamps).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

from twin_backend.api.deps import get_principal, require_operator

BEGIN = "<!-- BEGIN GENERATED: uv run poe api-docs -->"
END = "<!-- END GENERATED -->"
DOC_PATH = Path("docs/api.md")
METHOD_ORDER = {"get": 0, "post": 1, "put": 2, "delete": 3}


def build_app() -> FastAPI:
    """The app on the shipped configs (not started: nothing connects or listens)."""
    from twin_backend.api.app import create_app
    from twin_backend.runtime import build_runtime
    from twin_common.config import TwinSettings

    root = Path(__file__).resolve().parents[3]
    settings = TwinSettings(_env_file=None, config_dir=root / "config")
    return create_app(build_runtime(settings, root=root), host="127.0.0.1", manage_runtime=False)


def _calls(dependant: Any) -> set[Any]:
    out = {dependant.call}
    for dep in dependant.dependencies:
        out |= _calls(dep)
    return out


def _api_routes(routes: Any) -> list[APIRoute]:
    """Every APIRoute, also inside included routers (FastAPI wraps them in newer versions)."""
    out: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            out.append(route)
        else:
            inner = getattr(route, "original_router", None) or getattr(route, "router", None)
            if inner is not None and hasattr(inner, "routes"):
                out += _api_routes(inner.routes)
    return out


def roles(app: FastAPI) -> dict[tuple[str, str], str]:
    """(method, path) -> required role, from the routes' dependencies."""
    out: dict[tuple[str, str], str] = {}
    for route in _api_routes(app.routes):
        calls = _calls(route.dependant)
        role = (
            "operator"
            if require_operator in calls
            else "viewer"
            if get_principal in calls
            else "none (no key needed)"
        )
        for method in route.methods or ():
            out[(method.lower(), route.path)] = role
    return out


def _type(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return f"[{name}](#{name.lower()})"
    if "anyOf" in schema:
        return " \\| ".join(_type(s) for s in schema["anyOf"])
    if "enum" in schema:
        return " \\| ".join(f"`{v}`" for v in schema["enum"])
    if "const" in schema:
        return f"`{schema['const']}`"
    t = schema.get("type")
    if t == "array":
        return f"array of {_type(schema.get('items', {}))}"
    if t == "object" and "additionalProperties" in schema:
        extra = schema["additionalProperties"]
        return f"map of {_type(extra) if isinstance(extra, dict) else 'any'}"
    if t == "string" and schema.get("format"):
        return f"string ({schema['format']})"
    if t is None and "prefixItems" in schema:
        return "[" + ", ".join(_type(s) for s in schema["prefixItems"]) + "]"
    return str(t or "any")


def _cell(text: str | None) -> str:
    return (text or "").replace("\n", " ").replace("|", "\\|").strip()


def render(app: FastAPI) -> str:
    spec = app.openapi()
    role_of = roles(app)
    lines = [
        BEGIN,
        "",
        f"### REST API reference (OpenAPI {spec['openapi']}, "
        f"{spec['info']['title']} {spec['info']['version']})",
        "",
    ]
    lines.append(
        "Generated from the FastAPI app; do not edit by hand. Timestamps are ISO-8601 with a "
        'zone. Errors are `{"detail": "..."}`. A live schema is served at `/openapi.json` '
        "and interactive docs at `/docs`."
    )
    lines.append("")
    ops: list[tuple[str, str, str, dict[str, Any]]] = []
    for path, item in spec["paths"].items():
        for method, op in item.items():
            ops.append((op.get("tags", ["other"])[0], path, method, op))
    ops.sort(key=lambda o: (o[0], o[1], METHOD_ORDER.get(o[2], 9)))
    tag = None
    for op_tag, path, method, op in ops:
        if op_tag != tag:
            tag = op_tag
            lines += [f"#### {tag}", ""]
        lines += [f"##### `{method.upper()} {path}`", ""]
        lines.append(f"{_cell(op.get('summary'))}. Role: **{role_of.get((method, path), '?')}**.")
        lines.append("")
        if op.get("description"):
            lines += [op["description"].strip(), ""]
        params = [p for p in op.get("parameters", []) if p.get("in") != "header"]
        if params:
            lines += ["| Parameter | In | Required | Type | Description |", "|---|---|---|---|---|"]
            for p in params:
                lines.append(
                    f"| `{p['name']}` | {p['in']} | {'yes' if p.get('required') else 'no'} | "
                    f"{_type(p.get('schema', {}))} | {_cell(p.get('description'))} |"
                )
            lines.append("")
        body = op.get("requestBody", {}).get("content", {}).get("application/json")
        if body:
            lines += [f"Request body: {_type(body['schema'])}", ""]
        lines += ["| Status | Description | Body |", "|---|---|---|"]
        for code, resp in sorted(op.get("responses", {}).items()):
            content = resp.get("content", {})
            schema = content.get("application/json", {}).get("schema")
            if schema:
                shape = _type(schema) if schema else ""
            else:
                shape = ", ".join(f"`{m}`" for m in content if m != "application/json")
            lines.append(f"| {code} | {_cell(resp.get('description'))} | {shape} |")
        lines.append("")
    lines += ["#### Schemas", ""]
    for name, schema in sorted(spec.get("components", {}).get("schemas", {}).items()):
        lines += [f"##### {name}", ""]
        if schema.get("description"):
            lines += [_cell(schema["description"]), ""]
        props = schema.get("properties")
        if props:
            required = set(schema.get("required", []))
            lines += ["| Field | Type | Required | Description |", "|---|---|---|---|"]
            for field, fs in props.items():
                lines.append(
                    f"| `{field}` | {_type(fs)} | {'yes' if field in required else 'no'} | "
                    f"{_cell(fs.get('description'))} |"
                )
            lines.append("")
        elif "enum" in schema:
            lines += ["One of: " + ", ".join(f"`{v}`" for v in schema["enum"]), ""]
    lines.append(END)
    return "\n".join(lines) + "\n"


def splice(document: str, generated: str) -> str:
    if BEGIN not in document or END not in document:
        raise ValueError(f"docs/api.md needs the markers {BEGIN!r} and {END!r}")
    head = document.split(BEGIN, 1)[0]
    tail = document.split(END, 1)[1]
    return head + generated.rstrip("\n") + tail


def main() -> int:
    path = Path.cwd() / DOC_PATH
    current = path.read_text(encoding="utf-8")
    updated = splice(current, render(build_app()))
    if updated != current:
        path.write_text(updated, encoding="utf-8", newline="\n")
        print(f"updated {DOC_PATH}")
    else:
        print(f"{DOC_PATH} is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
