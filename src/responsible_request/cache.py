"""Serve requests from previously logged request records."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import CacheConfig, LogConfig


def cache_key(url: str, body: Any, salt: dict[str, Any] | None = None) -> str:
    """Hash of the request URL, its canonical JSON body and the salt (selected tag values)."""
    payload = {"url": url, "body": body}
    if salt:
        payload["salt"] = salt
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _cacheable(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("cache_key"))
        and row.get("status_code") == 200
        and not row.get("error")
        and not row.get("stream")
        and isinstance(row.get("response_body"), dict)
    )


class _SQLiteSource:
    """Looks up responses in the ``requests`` table written by the SQLite log sink."""

    _QUERY = (
        "SELECT response_body FROM requests WHERE cache_key = ? AND status_code = 200 "
        "AND error IS NULL AND NOT stream AND response_body IS NOT NULL ORDER BY rowid LIMIT 1"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def get(self, key: str) -> dict[str, Any] | None:
        if self._conn is None:
            if not self.path.exists():  # the log sink creates it with the first record
                return None
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            row = self._conn.execute(self._QUERY, (key,)).fetchone()
        except sqlite3.OperationalError:  # table or cache_key column not created yet
            return None
        if row is None:
            return None
        with contextlib.suppress(ValueError):
            body = json.loads(row[0])
            if isinstance(body, dict):
                return body
        return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class _JSONLSource:
    """Indexes a JSONL record file by cache key and reads matching lines on demand.

    New lines (e.g. written by this process) are indexed incrementally on each lookup.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._index: dict[str, int] = {}  # cache key -> byte offset of the first matching line
        self._offset = 0  # bytes indexed so far

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
                pos += len(line)
        self._offset = pos

    def close(self) -> None:
        pass


class ResponseCache:
    """Response lookup by cache key, backed by the records of a SQLite database or JSONL file."""

    def __init__(self, config: CacheConfig, log: LogConfig | None = None) -> None:
        self.config = config
        source: CacheConfig | LogConfig | None = config
        if config.sqlite is None and config.jsonl is None:  # read the records we write
            source = log
            if log is not None and not log.enabled("response_body"):
                raise ValueError("the cache reads responses from the log: enable response_body")
        if source is not None and source.sqlite is not None:
            self._source: _SQLiteSource | _JSONLSource = _SQLiteSource(source.sqlite)
        elif source is not None and source.jsonl is not None:
            self._source = _JSONLSource(source.jsonl)
        else:
            raise ValueError(
                "the cache needs records to read from: set LogConfig(sqlite=...) or "
                "LogConfig(jsonl=...), or CacheConfig(sqlite=...) or CacheConfig(jsonl=...)"
            )
        self.hits = 0
        self.misses = 0

    def salt(self, tags: dict[str, Any]) -> dict[str, Any]:
        return {name: tags[name] for name in self.config.key_tags if name in tags}

    def get(self, key: str) -> dict[str, Any] | None:
        body = self._source.get(key)
        if body is None:
            self.misses += 1
        else:
            self.hits += 1
        return body

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        self._source.close()
