"""A child process spoken to with JSON lines on stdin / stdout (ADR 0030).

Used by the backend's inference service to talk to ``python -m twin_ml.serve worker`` and by
the live-latency measurement, so both see the same transport. Standard library only, and no
asyncio subprocess (the Windows selector loop cannot run one, ADR 0008): a reader thread
collects the replies; :meth:`JsonLinesProcess.arequest` waits on a worker thread.

Every request gets an increasing ``id``; the child answers each request with one line that
echoes the id. The child's stderr is inherited, so its logs appear in the parent's stderr.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class WorkerError(RuntimeError):
    """The child answered with an error, died, or did not answer in time."""


class JsonLinesProcess:
    def __init__(
        self,
        cmd: Sequence[str],
        *,
        name: str,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.cmd = list(cmd)
        self.name = name
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self._proc: subprocess.Popen[str] | None = None
        self._ids = itertools.count(1)
        self._write_lock = threading.Lock()
        self._cond = threading.Condition()
        self._replies: dict[int, dict[str, Any]] = {}
        self._reader: threading.Thread | None = None
        self._eof = False
        self._exit_code: int | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.alive:
            return
        self._eof = False
        self._replies.clear()
        self._proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=self.cwd,
            env={**os.environ, **self.env} if self.env is not None else None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read, args=(self._proc,), name=f"{self.name}-reader", daemon=True
        )
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        """Exit code (also after :meth:`close`); None while running or never started."""
        return self._exit_code if self._proc is None else self._proc.poll()

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    def close(self, timeout_s: float = 5.0) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            with contextlib.suppress(WorkerError, OSError):
                self.request({"op": "shutdown"}, timeout_s)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        self._exit_code = proc.poll()
        self._proc = None

    # ------------------------------------------------------------------ requests
    def _read(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # not a protocol line
            with self._cond:
                if isinstance(msg, dict) and isinstance(msg.get("id"), int):
                    self._replies[msg["id"]] = msg
                    self._cond.notify_all()
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def request(self, msg: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        """Send one request and wait for its reply; returns ``reply["result"]``."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise WorkerError(f"{self.name}: not running")
        rid = next(self._ids)
        line = json.dumps({**msg, "id": rid}) + "\n"
        with self._write_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise WorkerError(f"{self.name}: cannot write to the process ({exc})") from exc
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while rid not in self._replies:
                if self._eof:
                    raise WorkerError(
                        f"{self.name}: the process exited (code {proc.poll()}) before answering"
                    )
                left = deadline - time.monotonic()
                if left <= 0:
                    raise WorkerError(f"{self.name}: no answer within {timeout_s:g} s")
                self._cond.wait(left)
            reply = self._replies.pop(rid)
        if not reply.get("ok"):
            raise WorkerError(f"{self.name}: {reply.get('error')}\n{reply.get('trace', '')}")
        result: dict[str, Any] = reply.get("result") or {}
        return result

    async def arequest(self, msg: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
        return await asyncio.to_thread(self.request, msg, timeout_s)
