"""Read logged request records back for analysis."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .logging import runs_path
from .records import JSON_FIELDS
from .runs import RUN_JSON_FIELDS

_JSONL_SUFFIXES = (".jsonl", ".json", ".log")


def load_records(path: str | Path, *, as_dataframe: bool | None = None) -> Any:
    """Load request records from a JSONL file or SQLite database written by this package.

    Returns a pandas DataFrame if pandas is installed (or ``as_dataframe=True``), otherwise a
    list of dicts.
    """
    path = Path(path)
    if path.suffix in _JSONL_SUFFIXES:
        rows = _read_jsonl(path)
    else:
        rows = _read_table(path, "requests", JSON_FIELDS)
    return _output(rows, as_dataframe)


def load_runs(path: str | Path, *, as_dataframe: bool | None = None) -> Any:
    """Load run manifests (see :class:`responsible_request.RunConfig`) from the SQLite database,
    or for a JSONL log (``requests.jsonl``) from ``requests.runs.jsonl`` next to it."""
    path = Path(path)
    if path.suffix in _JSONL_SUFFIXES:
        runs = path if path.stem.endswith(".runs") else runs_path(path)
        rows = _read_jsonl(runs) if runs.exists() else []
    else:
        rows = _read_table(path, "runs", RUN_JSON_FIELDS)
    return _output(rows, as_dataframe)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_table(path: Path, table: str, json_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
    except sqlite3.OperationalError:  # table does not exist (e.g. no runs in an old database)
        rows = []
    finally:
        conn.close()
    for row in rows:
        for key in json_fields:
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
    return rows


def _output(rows: list[dict[str, Any]], as_dataframe: bool | None) -> Any:
    if as_dataframe is False:
        return rows
    try:
        import pandas as pd
    except ImportError:
        if as_dataframe:
            raise ImportError("install pandas: pip install responsible-request[pandas]") from None
        return rows
    return pd.DataFrame(rows)
