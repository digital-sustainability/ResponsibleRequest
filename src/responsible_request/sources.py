"""Read request records back from the log while the client runs (cache lookups, spent cost)."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol


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


class JSONLSource:
    """Indexes a JSONL record file by cache key and reads matching lines on demand.

    New lines (e.g. written by this process) are indexed incrementally on each lookup.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._index: dict[str, int] = {}  # cache key -> byte offset of the first matching line
        self._offset = 0  # bytes indexed so far
        self._cost = 0.0  # cost_usd of the indexed records that were not cache hits

    def get(self, key: str) -> dict[str, Any] | None:
        if key not in self._index:
            self._scan()
        pos = self._index.get(key)
        if pos is None:
            return None
        with self.path.open("rb") as fh:
            fh.seek(pos)
            row = json.loads(fh.readline())
        body: dict[str, Any] = row["response_body"]
        return body

    def _scan(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self._offset:  # file was rotated or replaced
            self._index.clear()
            self._offset = 0
            self._cost = 0.0
        if size == self._offset:
            return
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            pos = self._offset
            for line in fh:
                if not line.endswith(b"\n"):  # still being written
                    break
                with contextlib.suppress(ValueError):
                    row = json.loads(line)
                    if isinstance(row, dict) and _cacheable(row):
                        self._index.setdefault(row["cache_key"], pos)
                    if isinstance(row, dict) and not row.get("cache_hit"):
                        self._cost += float(row.get("cost_usd") or 0.0)
                pos += len(line)
        self._offset = pos

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
