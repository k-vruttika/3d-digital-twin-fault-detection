"""Build the backend runtime from config: event bus, storage, source manager, twin engine.

``build_runtime`` loads the line file (``TwinSettings.line_file``, default ``line.yaml``),
``sources.yaml``, ``storage.yaml``, ``engine.yaml``, ``api.yaml`` and ``commands.yaml`` from
the config directory and wires everything together. The ``minimal`` and ``dev`` entry points
(``python -m twin_backend serve``) use it. With ``storage.backend: sqlite`` (the default)
nothing beyond the standard library is needed for persistence, so minimal mode keeps running
without Docker, broker or ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from twin_backend.adapters.manager import SourceManager
from twin_backend.bus import InMemoryEventBus
from twin_backend.clock import Clock, SystemClock
from twin_backend.commands.scenarios import ScenarioService
from twin_backend.commands.service import CommandService
from twin_backend.engine.twin import TwinEngine
from twin_backend.ingest.metrics import IngestMetrics
from twin_backend.storage.base import Stream
from twin_backend.storage.service import StorageService, build_storage
from twin_common.config import (
    ApiConfig,
    CommandsConfig,
    EngineConfig,
    LineConfig,
    SourcesConfig,
    StorageConfig,
    load_api_config,
    load_commands_config,
    load_engine_config,
    load_line_config,
    load_sources_config,
    load_storage_config,
)
from twin_common.config.inference import InferenceConfig, load_inference_config
from twin_common.config.loader import ConfigError
from twin_common.config.settings import TwinSettings

if TYPE_CHECKING:
    from twin_backend.inference.service import InferenceService


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
    engine_cfg: EngineConfig
    api_cfg: ApiConfig
    commands_cfg: CommandsConfig
    engine: TwinEngine
    clock: Clock
    root: Path
    commands: CommandService
    scenarios: ScenarioService
    # Phase 11: live inference (None unless enabled; never in minimal mode by default)
    inference_cfg: InferenceConfig | None = None
    inference: InferenceService | None = None

    async def start(self) -> None:
        """Storage, then the source, then the engine (after restoring its alerts).

        The engine subscribed to the bus when it was built, so nothing published in between
        is lost; it is applied once the engine starts.
        """
        await self.manager.start()
        if self.storage is not None:
            stored = await self.storage.store.fetch(Stream.ALERTS)
            self.engine.load_alerts([s.item for s in stored])
            await self.commands.load()
        self.engine.start()
        if self.inference is not None:
            await self.inference.start()

    async def stop(self) -> None:
        if self.inference is not None:
            await self.inference.stop()  # its last predictions are on the bus before storage stops
        await self.manager.stop(before_storage=self.engine.stop)


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
    engine_cfg: EngineConfig | None = None,
    api_cfg: ApiConfig | None = None,
    commands_cfg: CommandsConfig | None = None,
    inference_cfg: InferenceConfig | None = None,
    with_storage: bool = True,
) -> Runtime:
    settings = settings if settings is not None else TwinSettings()
    root = root if root is not None else Path.cwd()
    config_dir = (
        settings.config_dir if settings.config_dir.is_absolute() else root / settings.config_dir
    )
    line = line if line is not None else load_line_config(config_dir / settings.line_file)
    sources = sources if sources is not None else load_sources_config(config_dir / "sources.yaml")
    storage_cfg = (
        storage_cfg if storage_cfg is not None else load_storage_config(config_dir / "storage.yaml")
    )
    engine_cfg = (
        engine_cfg if engine_cfg is not None else load_engine_config(config_dir / "engine.yaml")
    )
    api_cfg = api_cfg if api_cfg is not None else load_api_config(config_dir / "api.yaml")
    commands_cfg = (
        commands_cfg
        if commands_cfg is not None
        else load_commands_config(config_dir / "commands.yaml")
    )
    if inference_cfg is None and (config_dir / "inference.yaml").is_file():
        inference_cfg = load_inference_config(config_dir / "inference.yaml")
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
    engine = TwinEngine(
        line,
        engine_cfg,
        bus,
        clock,
        registry=metrics.registry,
        queue_size=sources.bus.queue_size,
    )
    store = storage.store if storage is not None else None
    commands = CommandService(commands_cfg, line, manager, engine, bus, clock, store)
    scenarios = ScenarioService(manager, engine, bus, clock, store, metrics.registry)
    inference = None
    if inference_cfg is not None and inference_cfg.enabled:
        from twin_backend.inference.service import InferenceService  # no ML import (ADR 0030)

        inference = InferenceService(
            inference_cfg, line, bus, engine, clock, root, registry=metrics.registry
        )
    return Runtime(
        settings,
        line,
        sources,
        storage_cfg,
        bus,
        metrics,
        storage,
        manager,
        engine_cfg,
        api_cfg,
        commands_cfg,
        engine,
        clock,
        root,
        commands,
        scenarios,
        inference_cfg,
        inference,
    )
