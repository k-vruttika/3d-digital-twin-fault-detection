"""Structured JSON logging via structlog, with secret redaction.

Secrets (API keys, passwords) must never reach a log line. Two safeguards run on every event:

- fields whose name looks like a credential (``api_key``, ``apiKey``, ``x-api-key``,
  ``password``, ``authorization``, ...) are replaced by ``"***"``;
- every registered secret value (:func:`register_secret`) is replaced wherever it appears in a
  string field, including the event message itself.

:class:`SecretFilter` applies the same value redaction to standard-library log records
(uvicorn's access and error logs), and also masks ``apiKey=...`` style query parameters.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog

REDACTED = "***"
_SENSITIVE_NAMES = frozenset(
    {"api_key", "apikey", "x-api-key", "x_api_key", "password", "authorization", "secret"}
)
_QUERY_SECRET = re.compile(r"(?i)\b(api[_-]?key|password|token)=([^&\s\"']+)")
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()
# Shorter values are not registered: masking them would mangle ordinary text.
_MIN_SECRET_LEN = 8


def register_secret(value: str) -> None:
    """Never log ``value`` (API keys, passwords): it is replaced by ``***`` everywhere."""
    if len(value) >= _MIN_SECRET_LEN:
        with _SECRETS_LOCK:
            _SECRETS.add(value)


def redact_text(text: str) -> str:
    with _SECRETS_LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for s in secrets:
        if s in text:
            text = text.replace(s, REDACTED)
    return _QUERY_SECRET.sub(lambda m: f"{m.group(1)}={REDACTED}", text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: _redact_field(k, v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_redact_value(v) for v in value)
    return value


def _redact_field(name: Any, value: Any) -> Any:
    if isinstance(name, str) and name.lower() in _SENSITIVE_NAMES:
        return REDACTED
    return _redact_value(value)


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: see the module docstring."""
    for key in list(event_dict):
        event_dict[key] = _redact_field(key, event_dict[key])
    return event_dict


class SecretFilter(logging.Filter):
    """Redacts secrets in standard-library log records (e.g. uvicorn access logs)."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Format first: arguments can be any object (e.g. an httpx.URL) whose str() holds a key.
        try:
            message = record.getMessage()
        except Exception:  # a malformed record: leave it to the logging machinery
            return True
        record.msg = redact_text(message)
        record.args = None
        return True


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
            redact_secrets,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
