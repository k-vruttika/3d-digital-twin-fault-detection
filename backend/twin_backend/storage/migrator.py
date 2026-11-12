"""Versioned plain-SQL migrations (ADR 0009).

Files live in ``twin_backend/storage/migrations/<dialect>/NNNN_name.sql`` and are applied in
version order. The table ``schema_migrations(version, name, checksum, applied_at)`` records what
ran. An applied file whose content changed, or a version in the database that no file
provides, is an error: migrations are append-only.

A file runs in one transaction together with its ``schema_migrations`` row, unless its first
line is ``-- linetwin: no-transaction`` (TimescaleDB continuous aggregates cannot be created
inside a transaction block). Checksums are taken over the text with ``\\n`` line endings, so a
Windows checkout with CRLF line endings has the same checksums.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources

NO_TRANSACTION_MARKER = "-- linetwin: no-transaction"
_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """The migration files and the database disagree."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str
    transactional: bool


def parse_migration(filename: str, text: str) -> Migration:
    match = _FILE_RE.match(filename)
    if match is None:
        raise MigrationError(f"bad migration file name {filename!r} (expected NNNN_name.sql)")
    sql = text.replace("\r\n", "\n")
    first = sql.lstrip().splitlines()[0].strip() if sql.strip() else ""
    return Migration(
        version=int(match.group(1)),
        name=match.group(2),
        sql=sql,
        checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        transactional=first != NO_TRANSACTION_MARKER,
    )


def load_migrations(dialect: str) -> list[Migration]:
    """All migration files shipped for ``dialect`` (``sqlite`` or ``timescale``), by version."""
    folder = resources.files("twin_backend.storage") / "migrations" / dialect
    found = [
        parse_migration(entry.name, entry.read_text(encoding="utf-8"))
        for entry in folder.iterdir()
        if entry.name.endswith(".sql")
    ]
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    if len(set(versions)) != len(versions):
        raise MigrationError(f"duplicate migration versions for {dialect}: {versions}")
    return found


def pending(available: list[Migration], applied: dict[int, tuple[str, str]]) -> list[Migration]:
    """Migrations still to apply. ``applied`` maps version -> (name, checksum) from the DB."""
    by_version = {m.version: m for m in available}
    errors: list[str] = []
    for version, (name, checksum) in sorted(applied.items()):
        mig = by_version.get(version)
        if mig is None:
            errors.append(
                f"database has migration {version:04d}_{name} but no such file exists "
                "(database newer than this code?)"
            )
        elif mig.checksum != checksum:
            errors.append(
                f"migration {version:04d}_{mig.name} was changed after it was applied "
                "(add a new migration instead of editing an applied one)"
            )
    if errors:
        raise MigrationError("; ".join(errors))
    return [m for m in available if m.version not in applied]
