"""API-key authentication with two roles (viewer < operator), ADR 0013.

- Keys come from ``.env`` (``LINETWIN_API_KEYS_OPERATOR`` / ``LINETWIN_API_KEYS_VIEWER``), never
  from committed files. REST clients send the key in a header (``X-API-Key`` by default);
  WebSocket clients send it in their first message, so it never appears in a URL or an
  access log.
- **No keys configured:** read access (viewer) without a key, no writes, and only when the
  server is bound to a loopback address. A non-loopback bind without keys refuses to start.
- Keys are compared in constant time against every configured key and registered with the
  log redaction (``twin_common.logging.register_secret``).
"""

from __future__ import annotations

import hmac
import ipaddress
from dataclasses import dataclass

from twin_common.config.settings import ApiKey
from twin_common.enums import Role
from twin_common.logging import get_logger, register_secret

log = get_logger(__name__)

ANONYMOUS = "anonymous"
NO_OPERATOR_KEYS = (
    "no operator API keys are configured; set LINETWIN_API_KEYS_OPERATOR in .env "
    "(see .env.example) to enable commands, source switching and alert acknowledgement"
)
_RANK = {Role.VIEWER: 0, Role.OPERATOR: 1}


class StartupRefusedError(RuntimeError):
    """The server must not start with this configuration."""


@dataclass(frozen=True, slots=True)
class Principal:
    name: str
    role: Role
    anonymous: bool = False

    def has(self, role: Role) -> bool:
        return _RANK[self.role] >= _RANK[role]


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class Authenticator:
    def __init__(self, keys: list[ApiKey], *, bind_host: str) -> None:
        names = [k.principal for k in keys]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise StartupRefusedError(f"duplicate API key names: {dups}")
        self.keys = keys
        self.bind_host = bind_host
        if not keys and not is_loopback(bind_host):
            raise StartupRefusedError(
                f"refusing to bind to {bind_host!r} without API keys: anyone on the network "
                "could read the twin. Set LINETWIN_API_KEYS_OPERATOR and/or "
                "LINETWIN_API_KEYS_VIEWER in .env (format 'name:key,name2:key2', see "
                ".env.example), or bind to 127.0.0.1."
            )
        for k in keys:
            register_secret(k.secret.get_secret_value())
        self.anonymous_read = not keys
        self.operators_configured = any(k.role is Role.OPERATOR for k in keys)

    def log_startup(self) -> None:
        if not self.keys:
            log.warning(
                "api.no_api_keys",
                detail="read-only access without a key on a loopback address; writes disabled",
                bind=self.bind_host,
            )
        else:
            log.info(
                "api.auth",
                operators=sum(k.role is Role.OPERATOR for k in self.keys),
                viewers=sum(k.role is Role.VIEWER for k in self.keys),
            )

    def resolve(self, presented: str | None) -> Principal | None:
        """The principal for a presented key; None if the key is wrong or one is required.

        Without a key the result is the anonymous viewer when anonymous read is allowed.
        """
        if presented is None or presented == "":
            return (
                Principal(ANONYMOUS, Role.VIEWER, anonymous=True) if self.anonymous_read else None
            )
        given = presented.encode("utf-8")
        match: ApiKey | None = None
        for k in self.keys:  # check every key: no early exit on a match
            if hmac.compare_digest(given, k.secret.get_secret_value().encode("utf-8")):
                match = k
        return Principal(match.principal, match.role) if match is not None else None
