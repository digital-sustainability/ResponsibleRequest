"""Loguru-based logging: quiet console output plus structured per-request records.

Per-request records are emitted at loguru's ``TRACE`` level with the record attached as
``extra["rr_record"]`` (run manifests as ``extra["rr_run"]``), so they never show up in a normal
console handler but can be picked up by any sink that filters for them (the JSONL and SQLite
sinks below do exactly that).
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
from .runs import RUN_COLUMNS, RUN_JSON_FIELDS

PACKAGE = "responsible_request"

_lock = threading.Lock()
_sinks: dict[tuple[str, str], int] = {}  # (kind, resolved path) -> loguru handler id
_console_handler: int | None = None


def _is_record(record: Any) -> bool:
    return "rr_record" in record["extra"]


def _is_run(record: Any) -> bool:
    return "rr_run" in record["extra"]


def _is_structured(record: Any) -> bool:
    return _is_record(record) or _is_run(record)


def _is_console_message(record: Any) -> bool:
    return not _is_structured(record)


def runs_path(jsonl: str | Path) -> Path:
    """Where run manifests go for a JSONL log: ``requests.jsonl`` -> ``requests.runs.jsonl``."""
    path = Path(jsonl)
    return path.with_name(f"{path.stem}.runs{path.suffix or '.jsonl'}")


def _ensure_table(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...] | list[str]
) -> None:
    cols = ", ".join(f'"{c}"' for c in columns)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col in columns:
        if col not in existing:  # schema grew in a newer version
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}"')


class SQLiteSink:
    """Loguru sink that inserts request records into a ``requests`` table and run manifests
    into a ``runs`` table."""

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
            _ensure_table(conn, "requests", self._columns)
            _ensure_table(conn, "runs", RUN_COLUMNS)
            conn.execute("CREATE INDEX IF NOT EXISTS requests_cache_key ON requests(cache_key)")
            conn.commit()
            self._conn = conn
        return self._conn

    def write(self, message: Any) -> None:
        extra = message.record["extra"]
        if "rr_run" in extra:
            self._insert("runs", extra["rr_run"], RUN_COLUMNS, RUN_JSON_FIELDS)
        else:
            self._insert("requests", extra["rr_record"], self._columns, JSON_FIELDS)

    def _insert(
        self,
        table: str,
        data: dict[str, Any],
        columns: tuple[str, ...] | list[str],
        json_fields: tuple[str, ...],
    ) -> None:
        row = {
            k: (json.dumps(v, default=str) if k in json_fields and v is not None else v)
            for k, v in data.items()
            if k in columns
        }
        conn = self._connect()
        cols = ", ".join(f'"{c}"' for c in row)
        marks = ", ".join("?" for _ in row)
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
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
            runs = runs_path(config.jsonl)
            key = ("jsonl", str(runs.resolve()))
            if key not in _sinks:
                _sinks[key] = logger.add(
                    runs,
                    level="TRACE",
                    filter=_is_run,
                    format="{extra[rr_json]}",
                    enqueue=True,
                )
        if config.sqlite is not None:
            key = ("sqlite", str(Path(config.sqlite).resolve()))
            if key not in _sinks:
                _sinks[key] = logger.add(
                    SQLiteSink(config.sqlite),
                    level="TRACE",
                    filter=_is_structured,
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
