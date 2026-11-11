"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
from typing import TextIO

import structlog


def configure_logging(
    level: str = "INFO", *, json: bool = True, stream: TextIO | None = None
) -> None:
    """Configure structlog process-wide. Call once at application start-up.

    ``stream`` selects the output (default: stdout); CLIs that stream data on stdout pass
    ``sys.stderr``.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"unknown log level: {level!r}")
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
