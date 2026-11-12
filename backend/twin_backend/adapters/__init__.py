"""Source adapters: file replay, mock server, MQTT, OPC UA, and the chaos wrapper.

Only the protocol is imported here; concrete adapters are imported by
``twin_backend.adapters.registry`` on demand, so minimal mode loads no network library.
"""

from twin_backend.adapters.base import (
    AdapterError,
    AdapterHealth,
    Backoff,
    BaseAdapter,
    RawMessage,
    SourceAdapter,
)

__all__ = [
    "AdapterError",
    "AdapterHealth",
    "Backoff",
    "BaseAdapter",
    "RawMessage",
    "SourceAdapter",
]
