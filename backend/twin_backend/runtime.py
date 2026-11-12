"""Build the backend runtime from config: event bus, storage, source manager.

``build_runtime`` loads ``line.yaml``, ``sources.yaml`` and ``storage.yaml`` from the config
directory and wires everything together. The Phase 4 ``minimal`` and ``dev`` entry points use
it. With ``storage.backend: sqlite`` (the default) nothing beyond the standard library is
needed for persistence, so minimal mode keeps running without Docker, broker or ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twin_backend.adapters.manager import SourceManager
from twin_backend.bus import InMemoryEventBus
from twin_backend.clock import Clock, SystemClock
from twin_backend.ingest.metrics import IngestMetrics
from twin_backend.storage.service import StorageService, build_storage
from twin_common.config import (
    LineConfig,
    SourcesConfig,
    StorageConfig,
    load_line_config,
    load_sources_config,
    load_storage_config,
)
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings


@dataclass
class Runtime:
    settings: TwinSettings
    line: LineConfig
    sources: SourcesConfig
    storage_cfg: StorageConfig
    bus: InMemoryEventBus
    metrics: IngestMetrics
    storage: StorageService | None
    manager: SourceManager

    async def start(self) -> None:
        await self.manager.start()

    async def stop(self) -> None:
        await self.manager.stop()


def check_storage_against_ingest(storage: StorageConfig, sources: SourcesConfig) -> None:
    """Cross-file rule: snapshots must wait out the reorder window and one writer flush."""
    need = sources.ingest.lateness_window_s + storage.writer.flush_interval_s
    if storage.snapshots.enabled and storage.snapshots.settle_s < need:
        raise ConfigError(
            f"storage.yaml: snapshots.settle_s ({storage.snapshots.settle_s:g}) must be >= "
            f"sources.yaml ingest.lateness_window_s + writer.flush_interval_s ({need:g})"
        )


def build_runtime(
    settings: TwinSettings | None = None,
    *,
    root: Path | None = None,
    clock: Clock | None = None,
    line: LineConfig | None = None,
    sources: SourcesConfig | None = None,
    storage_cfg: StorageConfig | None = None,
    with_storage: bool = True,
) -> Runtime:
    settings = settings if settings is not None else TwinSettings()
    root = root if root is not None else Path.cwd()
    config_dir = (
        settings.config_dir if settings.config_dir.is_absolute() else root / settings.config_dir
    )
    line = line if line is not None else load_line_config(config_dir / "line.yaml")
    sources = sources if sources is not None else load_sources_config(config_dir / "sources.yaml")
    storage_cfg = (
        storage_cfg if storage_cfg is not None else load_storage_config(config_dir / "storage.yaml")
    )
    check_storage_against_ingest(storage_cfg, sources)
    clock = clock if clock is not None else SystemClock()
    bus = InMemoryEventBus(sources.bus.queue_size, clock=clock)
    metrics = IngestMetrics()
    storage = (
        build_storage(
            storage_cfg,
            bus,
            [m.id for m in line.machines],
            root=root,
            settings=settings,
            clock=clock,
            registry=metrics.registry,
        )
        if with_storage
        else None
    )
    manager = SourceManager(
        sources, line, bus, root=root, clock=clock, metrics=metrics, storage=storage
    )
    return Runtime(settings, line, sources, storage_cfg, bus, metrics, storage, manager)
