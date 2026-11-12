"""Build the adapter for an ``AdapterKind`` from ``sources.yaml``.

Network adapters are imported lazily, so minimal mode (file adapter) never imports aiomqtt,
asyncua, websockets or httpx.
"""

from __future__ import annotations

from pathlib import Path

from twin_backend.adapters.base import AdapterError, SourceAdapter
from twin_backend.clock import Clock
from twin_common.config.line import LineConfig
from twin_common.config.sources import FieldMapping, SourcesConfig
from twin_common.enums import AdapterKind


def mappings_from(sources: SourcesConfig) -> dict[AdapterKind, FieldMapping]:
    """Per-adapter field mappings for the ingest normalizer."""
    out: dict[AdapterKind, FieldMapping] = {}
    for kind in AdapterKind:
        section = getattr(sources, kind.value)
        if section is not None:
            out[kind] = section.mapping
    return out


def build_adapter(
    kind: AdapterKind,
    sources: SourcesConfig,
    line: LineConfig,
    *,
    root: Path,
    clock: Clock,
) -> SourceAdapter:
    try:
        sources.require(kind)
    except ValueError as exc:
        raise AdapterError(str(exc)) from exc
    adapter: SourceAdapter
    if kind is AdapterKind.FILE:
        from twin_backend.adapters.file import FileReplayAdapter

        assert sources.file is not None
        adapter = FileReplayAdapter(sources.file, line, root=root, clock=clock)
    elif kind is AdapterKind.MOCK:
        from twin_backend.adapters.mock import MockServerAdapter

        assert sources.mock is not None
        adapter = MockServerAdapter(
            sources.mock,
            line,
            reconnect=sources.reconnect,
            command_timeout_s=sources.command_timeout_s,
            clock=clock,
        )
    elif kind is AdapterKind.MQTT:
        from twin_backend.adapters.mqtt import MqttAdapter

        assert sources.mqtt is not None
        adapter = MqttAdapter(
            sources.mqtt,
            line,
            reconnect=sources.reconnect,
            command_timeout_s=sources.command_timeout_s,
            clock=clock,
        )
    else:
        from twin_backend.adapters.opcua import OpcUaAdapter

        assert sources.opcua is not None
        adapter = OpcUaAdapter(
            sources.opcua,
            line,
            reconnect=sources.reconnect,
            command_timeout_s=sources.command_timeout_s,
            clock=clock,
        )
    if sources.chaos.enabled:
        from twin_backend.adapters.chaos import ChaosAdapter

        adapter = ChaosAdapter(adapter, sources.chaos, clock)
    return adapter
