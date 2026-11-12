"""Plain-SQL migration runner: ordering, checksums, re-runs, no-transaction files."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from twin_backend.storage.migrator import (
    NO_TRANSACTION_MARKER,
    MigrationError,
    load_migrations,
    parse_migration,
    pending,
)
from twin_backend.storage.sqlite import SqliteEventStore


def test_shipped_migrations_load_in_order() -> None:
    migs = load_migrations("sqlite")
    assert [m.version for m in migs] == sorted(m.version for m in migs)
    assert migs[0].version == 1
    assert migs[0].name == "init"
    assert all(m.transactional for m in migs)


def test_checksum_ignores_line_endings() -> None:
    lf = parse_migration("0001_x.sql", "CREATE TABLE a (x INT);\nSELECT 1;\n")
    crlf = parse_migration("0001_x.sql", "CREATE TABLE a (x INT);\r\nSELECT 1;\r\n")
    assert lf.checksum == crlf.checksum


def test_no_transaction_marker() -> None:
    mig = parse_migration("0002_cagg.sql", f"{NO_TRANSACTION_MARKER}\nCREATE VIEW v AS SELECT 1;")
    assert not mig.transactional


@pytest.mark.parametrize("name", ["1_x.sql", "0001-x.sql", "0001_X.sql", "0001_x.txt"])
def test_bad_file_names_rejected(name: str) -> None:
    with pytest.raises(MigrationError, match="bad migration file name"):
        parse_migration(name, "SELECT 1;")


def test_pending_detects_edited_and_unknown_migrations() -> None:
    a = parse_migration("0001_a.sql", "SELECT 1;")
    b = parse_migration("0002_b.sql", "SELECT 2;")
    assert pending([a, b], {}) == [a, b]
    assert pending([a, b], {1: ("a", a.checksum)}) == [b]
    with pytest.raises(MigrationError, match="was changed after it was applied"):
        pending([a, b], {1: ("a", "0" * 64)})
    with pytest.raises(MigrationError, match="no such file exists"):
        pending([a], {1: ("a", a.checksum), 9: ("future", "f" * 64)})


async def test_fresh_database_then_rerun_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite"
    store = SqliteEventStore(path)
    await store.connect()
    assert await store.migrate() == ["0001_init"]
    assert await store.migrate() == []
    await store.close()
    # Reopening the file finds the recorded version and applies nothing.
    again = SqliteEventStore(path)
    await again.connect()
    assert await again.migrate() == []
    await again.close()
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT version, name FROM schema_migrations").fetchall()
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert rows == [(1, "init")]
    assert {"readings", "state_changes", "predictions", "commands", "command_acks"} <= tables
    assert {"alerts", "quarantine", "snapshots"} <= tables


async def test_tampered_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite"
    store = SqliteEventStore(path)
    await store.connect()
    await store.migrate()
    await store.close()
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("UPDATE schema_migrations SET checksum = 'x' WHERE version = 1")
    conn.close()
    store = SqliteEventStore(path)
    await store.connect()
    with pytest.raises(MigrationError, match="changed after it was applied"):
        await store.migrate()
    await store.close()
