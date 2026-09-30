"""Read logged request records back for analysis."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .records import JSON_FIELDS


def load_records(path: str | Path, *, as_dataframe: bool | None = None) -> Any:
    """Load request records from a JSONL file or SQLite database written by this package.

    Returns a pandas DataFrame if pandas is installed (or ``as_dataframe=True``), otherwise a
    list of dicts.
    """
    path = Path(path)
    if path.suffix in (".jsonl", ".json", ".log"):
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM requests")]
        finally:
            conn.close()
        for row in rows:
            for key in JSON_FIELDS:
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])

    if as_dataframe is False:
        return rows
    try:
        import pandas as pd
    except ImportError:
        if as_dataframe:
            raise ImportError("install pandas: pip install responsible-request[pandas]") from None
        return rows
    return pd.DataFrame(rows)
