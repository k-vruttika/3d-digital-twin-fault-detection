"""Dedup on (machineId, seq) and the reorder buffer's lateness window."""

from __future__ import annotations

import pytest

from twin_backend.ingest.dedup import Deduplicator, DedupResult
from twin_backend.ingest.reorder import ReorderBuffer

WINDOW_S = 2.0


# -- dedup -----------------------------------------------------------------------------------
def test_dedup_new_and_duplicate() -> None:
    d = Deduplicator(window=100, reset_threshold=50)
    assert d.check("A", 1) is DedupResult.NEW
    assert d.check("A", 2) is DedupResult.NEW
    assert d.check("A", 1) is DedupResult.DUPLICATE
    assert d.check("B", 1) is DedupResult.NEW  # per machine


def test_dedup_out_of_order_not_duplicate() -> None:
    d = Deduplicator(window=100, reset_threshold=50)
    for seq in (5, 3, 4, 1, 2):
        assert d.check("A", seq) is DedupResult.NEW
    assert d.check("A", 3) is DedupResult.DUPLICATE


def test_dedup_seq_reset_detected() -> None:
    d = Deduplicator(window=100, reset_threshold=50)
    for seq in range(1, 201):
        d.check("A", seq)
    assert d.check("A", 160) is DedupResult.DUPLICATE  # within the threshold: a duplicate
    assert d.check("A", 1) is DedupResult.RESET  # far below: the source restarted
    assert d.check("A", 2) is DedupResult.NEW
    assert d.check("A", 1) is DedupResult.DUPLICATE


def test_dedup_memory_bounded() -> None:
    d = Deduplicator(window=10, reset_threshold=10)
    for seq in range(1, 1000):
        d.check("A", seq)
    assert len(d._machines["A"].seen) <= 20


def test_dedup_threshold_must_fit_window() -> None:
    with pytest.raises(ValueError, match="reset_threshold"):
        Deduplicator(window=10, reset_threshold=11)


# -- reorder ---------------------------------------------------------------------------------
def released_seqs(rel: list) -> list[int]:  # type: ignore[type-arg]
    return [r.seq for r in rel]


def started(buf: ReorderBuffer[str], machine: str = "A", seq: int = 1) -> None:
    """Get past the initial hold: the first message waits one window."""
    assert buf.push(machine, seq, f"m{seq}", now=0.0) == []
    out = buf.tick(now=WINDOW_S)
    assert [(m, r.seq) for m, r in out] == [(machine, seq)]


def test_initial_hold_sorts_early_messages() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    assert buf.push("A", 2, "m2", now=0.0) == []
    assert buf.push("A", 1, "m1", now=0.5) == []
    assert buf.tick(now=1.9) == []
    out = buf.tick(now=2.0)
    assert [(r.seq, r.late, r.reordered) for _, r in out] == [(1, False, False), (2, False, False)]


def test_in_order_released_immediately() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf)
    for seq in range(2, 6):
        rel = buf.push("A", seq, f"m{seq}", now=3.0)
        assert released_seqs(rel) == [seq]
        assert not rel[0].reordered


def test_gap_filled_within_window() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf)
    assert buf.push("A", 3, "m3", now=3.0) == []
    assert buf.push("A", 4, "m4", now=3.1) == []
    rel = buf.push("A", 2, "m2", now=3.5)
    assert released_seqs(rel) == [2, 3, 4]
    assert [r.reordered for r in rel] == [False, True, True]
    assert [r.item for r in rel] == ["m2", "m3", "m4"]


def test_gap_given_up_after_window() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf)
    buf.push("A", 4, "m4", now=3.0)
    buf.push("A", 5, "m5", now=3.5)
    assert buf.tick(now=4.99) == []
    out = buf.tick(now=5.0)  # m4 waited exactly the window
    assert [(r.seq, r.gap_before, r.reordered) for _, r in out] == [(4, 2, False), (5, 0, False)]
    # seq 2 and 3 arrive after the window: late, released at once.
    late = buf.push("A", 2, "m2", now=6.0)
    assert [(r.seq, r.late) for r in late] == [(2, True)]
    assert buf.is_late("A", 3)


def test_several_gaps_released_in_order() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf)
    buf.push("A", 5, "m5", now=3.0)
    buf.push("A", 3, "m3", now=3.0)
    out = buf.tick(now=5.0)
    assert [(r.seq, r.gap_before) for _, r in out] == [(3, 1), (5, 1)]


def test_machines_independent() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf, "A")
    started(buf, "B")
    buf.push("A", 3, "a3", now=3.0)
    assert released_seqs(buf.push("B", 2, "b2", now=3.0)) == [2]
    assert buf.held("A") == 1


def test_flush_and_reset() -> None:
    buf: ReorderBuffer[str] = ReorderBuffer(WINDOW_S)
    started(buf)
    buf.push("A", 4, "m4", now=3.0)
    assert [(m, r.seq, r.gap_before) for m, r in buf.flush()] == [("A", 4, 2)]
    buf.push("A", 9, "m9", now=4.0)
    assert released_seqs(buf.reset("A")) == [9]
    assert not buf.is_late("A", 1)  # expected seq forgotten


def test_window_validated() -> None:
    with pytest.raises(ValueError, match="window_s"):
        ReorderBuffer(0)
