"""Loguru-based logging: quiet console output plus structured per-request records.

Per-request records are emitted at loguru's ``TRACE`` level with the record attached as
``extra["rr_record"]``, so they never show up in a normal console handler but can be picked up
by any sink that filters for them (the JSONL and SQLite sinks below do exactly that).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from .config import LogConfig
from .records import JSON_FIELDS, record_columns

PACKAGE = "responsible_request"

_lock = threading.Lock()
_sinks: dict[tuple[str, str], int] = {}  # (kind, resolved path) -> loguru handler id
_console_handler: int | None = None


def _is_record(record: Any) -> bool:
    return "rr_record" in record["extra"]


def _is_console_message(record: Any) -> bool:
    return not _is_record(record)


class SQLiteSink:
    """Loguru sink that inserts request records into a ``requests`` table."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._columns = record_columns()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            cols = ", ".join(f'"{c}"' for c in self._columns)
            conn.execute(f"CREATE TABLE IF NOT EXISTS requests ({cols})")
            existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
            for col in self._columns:
                if col not in existing:  # schema grew in a newer version
                    conn.execute(f'ALTER TABLE requests ADD COLUMN "{col}"')
            conn.commit()
            self._conn = conn
        return self._conn

    def write(self, message: Any) -> None:
        data: dict[str, Any] = message.record["extra"]["rr_record"]
        row = {
            k: (json.dumps(v, default=str) if k in JSON_FIELDS and v is not None else v)
            for k, v in data.items()
            if k in self._columns
        }
        conn = self._connect()
        cols = ", ".join(f'"{c}"' for c in row)
        marks = ", ".join("?" for _ in row)
        conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
        conn.commit()

    def stop(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def setup_logging(config: LogConfig | None = None) -> None:
    """Enable this package's log output and add the sinks requested in ``config``.

    Safe to call repeatedly: a file or database is only attached once per process.
    """
    global _console_handler
    config = config or LogConfig()
    logger.enable(PACKAGE)
    with _lock:
        if config.console_level is not None and _console_handler is None:
            with contextlib.suppress(ValueError):
                logger.remove(0)  # loguru's default stderr handler, if still present
            _console_handler = logger.add(
                sys.stderr, level=config.console_level, filter=_is_console_message
            )
        if config.jsonl is not None:
            key = ("jsonl", str(Path(config.jsonl).resolve()))
            if key not in _sinks:
                _sinks[key] = logger.add(
                    config.jsonl,
                    level="TRACE",
                    filter=_is_record,
                    format="{extra[rr_json]}",
                    rotation=config.rotation,
                    enqueue=True,
                )
        if config.sqlite is not None:
            key = ("sqlite", str(Path(config.sqlite).resolve()))
            if key not in _sinks:
                _sinks[key] = logger.add(
                    SQLiteSink(config.sqlite),
                    level="TRACE",
                    filter=_is_record,
                    format="{message}",
                    enqueue=True,
                )


def remove_sinks() -> None:
    """Detach all JSONL/SQLite sinks added by :func:`setup_logging` (flushes pending writes)."""
    with _lock:
        for handler_id in _sinks.values():
            logger.remove(handler_id)
        _sinks.clear()


def emit_record(data: dict[str, Any]) -> None:
    logger.bind(rr_record=data, rr_json=json.dumps(data, default=str)).trace(
        "request {} {} {} -> {}",
        data.get("request_id"),
        data.get("model") or "",
        data.get("path"),
        data.get("status_code"),
    )
