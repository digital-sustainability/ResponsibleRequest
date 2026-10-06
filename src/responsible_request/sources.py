"""Read request records back from the log while the client runs (cache lookups, spent cost)."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from pathlib import Path
from typing import IO, Any, Protocol

from .logfiles import compression_of, open_binary, rotated_segments


class _Target(Protocol):
    @property
    def sqlite(self) -> str | Path | None: ...
    @property
    def jsonl(self) -> str | Path | None: ...


def _cacheable(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("cache_key"))
        and row.get("status_code") == 200
        and not row.get("error")
        and not row.get("stream")
        and isinstance(row.get("response_body"), dict)
    )


class SQLiteSource:
    """Looks up responses in the ``requests`` table written by the SQLite log sink."""

    _QUERY = (
        "SELECT response_body FROM requests WHERE cache_key = ? AND status_code = 200 "
        "AND error IS NULL AND NOT stream AND response_body IS NOT NULL ORDER BY rowid LIMIT 1"
    )
    _COST = "SELECT SUM(cost_usd) FROM requests WHERE NOT cache_hit"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def _query(self, sql: str, *args: Any) -> Any:
        if self._conn is None:
            if not self.path.exists():  # the log sink creates it with the first record
                return None
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            return self._conn.execute(sql, args).fetchone()
        except sqlite3.OperationalError:  # table or column not created yet
            return None

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._query(self._QUERY, key)
        if row is None:
            return None
        with contextlib.suppress(ValueError):
            body = json.loads(row[0])
            if isinstance(body, dict):
                return body
        return None

    def total_cost(self) -> float:
        """Sum of ``cost_usd`` over all records that were not cache hits."""
        row = self._query(self._COST)
        return float(row[0]) if row is not None and row[0] is not None else 0.0

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class _Rotated(Exception):
    """The active file was replaced since it was last indexed."""


class JSONLSource:
    """Indexes a JSONL log by cache key and reads matching lines on demand.

    The segments loguru rotated away from the log are indexed too. Responses from compressed
    segments are kept in memory (only the first one per cache key); everything else is read back
    by byte offset. New lines in the active file are indexed incrementally on each lookup, and a
    rotation while the client runs triggers a full re-index.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._live = compression_of(self.path) is None  # a compressed file does not grow
        self._reset()

    def _reset(self) -> None:
        # cache key -> (plain file, byte offset of the first matching line) or response JSON
        self._index: dict[str, tuple[Path, int] | str] = {}
        self._cost = 0.0  # cost_usd of the indexed records that were not cache hits
        self._loaded = False  # rotated or compressed segments indexed
        self._file_id: tuple[int, int] | None = None  # (device, inode) of the active file
        self._offset = 0  # bytes of the active file indexed so far

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self._index:
            self._scan()
        for _ in range(2):
            entry = self._index.get(key)
            if entry is None:
                return None
            body = self._read(key, entry)
            if body is not None:
                return body
            # a rotation renamed, compressed or replaced the file since it was indexed
            self._reset()
            self._scan()
        return None

    @staticmethod
    def _read(key: str, entry: tuple[Path, int] | str) -> dict[str, Any] | None:
        if isinstance(entry, str):
            body: dict[str, Any] = json.loads(entry)
            return body
        path, pos = entry
        try:
            with path.open("rb") as fh:
                fh.seek(pos)
                row = json.loads(fh.readline())
        except (FileNotFoundError, ValueError):
            return None
        if not isinstance(row, dict) or row.get("cache_key") != key:
            return None
        body = row["response_body"]
        return body

    def _scan(self) -> None:
        for _ in range(5):
            try:
                if not self._loaded:
                    self._load_segments()
                if self._live:
                    self._scan_active()
                return
            except (FileNotFoundError, _Rotated):  # loguru renamed or compressed a file under us
                self._reset()

    def _load_segments(self) -> None:
        segments = rotated_segments(self.path) if self._live else [self.path]
        for segment in segments:
            with open_binary(segment) as fh:
                self._index_lines(fh, segment, 0, in_memory=compression_of(segment) is not None)
        self._loaded = True
        if self._live:
            self._scan_active()
            if rotated_segments(self.path) != segments:  # rotated while we were reading
                raise _Rotated

    def _scan_active(self) -> None:
        try:
            fh = self.path.open("rb")
        except FileNotFoundError:  # nothing written yet, or in the middle of a rotation
            return
        with fh:
            stat = os.fstat(fh.fileno())
            file_id = (stat.st_dev, stat.st_ino)
            if self._file_id is None:
                self._file_id = file_id
            elif file_id != self._file_id or stat.st_size < self._offset:
                raise _Rotated
            if stat.st_size > self._offset:
                fh.seek(self._offset)
                self._offset = self._index_lines(fh, self.path, self._offset, in_memory=False)

    def _index_lines(self, fh: IO[bytes], path: Path, pos: int, *, in_memory: bool) -> int:
        for line in fh:
            if not line.endswith(b"\n"):  # still being written
                break
            with contextlib.suppress(ValueError):
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if _cacheable(row) and row["cache_key"] not in self._index:
                    self._index[row["cache_key"]] = (
                        json.dumps(row["response_body"]) if in_memory else (path, pos)
                    )
                if not row.get("cache_hit"):
                    self._cost += float(row.get("cost_usd") or 0.0)
            pos += len(line)
        return pos

    def total_cost(self) -> float:
        """Sum of ``cost_usd`` over all records that were not cache hits."""
        self._scan()
        return self._cost

    def close(self) -> None:
        pass


RecordSource = SQLiteSource | JSONLSource


def open_source(target: _Target | None) -> RecordSource | None:
    """A reader for the SQLite database of ``target``, else its JSONL file, else None."""
    if target is not None and target.sqlite is not None:
        return SQLiteSource(target.sqlite)
    if target is not None and target.jsonl is not None:
        return JSONLSource(target.jsonl)
    return None
