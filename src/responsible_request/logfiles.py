"""The files of a JSONL log: the active file plus the segments loguru rotated away from it,
optionally compressed."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO, Any

COMPRESSIONS = ("gz", "bz2", "xz", "zst")
"""Values of ``LogConfig.compression``: formats that are read back as a single JSONL stream."""

# loguru names a rotated segment ``{stem}.{%Y-%m-%d_%H-%M-%S_%f}[.{n}]{suffix}``
_DATE = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}(?:\.\d+)?"


def _zstd() -> Any:
    try:
        from compression import zstd  # Python 3.14+

        return zstd
    except ImportError:
        pass
    try:
        import zstandard

        return zstandard
    except ImportError:
        raise ImportError(
            "zstd needs Python 3.14+ or: pip install responsible-request[zstd]"
        ) from None


def compression_of(path: Path) -> str | None:
    """The compression format of ``path`` from its extension, or None for a plain file."""
    ext = path.suffix.lstrip(".")
    return ext if ext in COMPRESSIONS else None


def strip_compression(path: Path) -> Path:
    """``requests.jsonl.gz`` -> ``requests.jsonl``."""
    return path.with_suffix("") if compression_of(path) else path


def open_binary(path: Path) -> IO[bytes]:
    """Open a plain or compressed log file for reading (decompressed bytes)."""
    fmt = compression_of(path)
    if fmt == "zst":  # zstandard's reader can't iterate over lines, a buffer on top of it can
        return io.BufferedReader(_zstd().open(path, "rb"))
    opener: Any = {"gz": gzip.open, "bz2": bz2.open, "xz": lzma.open}.get(fmt or "")
    stream: IO[bytes] = opener(path, "rb") if opener else path.open("rb")
    return stream


def rotated_segments(path: Path) -> list[Path]:
    """Segments rotated away from the log at ``path`` (and compressed copies of ``path`` itself,
    which loguru writes when a sink without rotation is removed), oldest first."""
    pattern = re.compile(
        rf"{re.escape(path.stem)}(?:\.{_DATE})?{re.escape(path.suffix)}"
        rf"(?:\.(?:{'|'.join(COMPRESSIONS)}))?"
    )
    try:
        names = {p.name for p in path.parent.iterdir()}
    except FileNotFoundError:
        return []
    found = []
    for name in names:
        if name == path.name or not pattern.fullmatch(name):
            continue
        plain = name.rsplit(".", 1)[0]
        if compression_of(Path(name)) and plain != path.name and plain in names:
            continue  # still being compressed: loguru removes the plain file once it is done
        found.append(path.parent / name)
    return sorted(found, key=lambda p: (_mtime(p), p.name))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def log_segments(path: Path) -> list[Path]:
    """All files of the log at ``path``, oldest first. A compressed ``path`` is read on its own."""
    if compression_of(path):
        return [path]
    return rotated_segments(path) + ([path] if path.exists() else [])


def _compress_zstd(path_in: str) -> None:
    """Compress a finished segment to ``{path_in}.zst``, the way loguru does for other formats."""
    path_out = Path(f"{path_in}.zst")
    if path_out.exists():  # keep the earlier archive under a dated name
        root, ext = os.path.splitext(path_in)
        date = datetime.fromtimestamp(path_out.stat().st_mtime).strftime("%Y-%m-%d_%H-%M-%S_%f")
        renamed, n = Path(f"{root}.{date}{ext}.zst"), 1
        while renamed.exists():
            n += 1
            renamed = Path(f"{root}.{date}.{n}{ext}.zst")
        path_out.rename(renamed)
    with open(path_in, "rb") as src, _zstd().open(path_out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path_in)


def loguru_compression(fmt: str | None) -> str | Callable[[str], None] | None:
    """The ``compression`` argument for loguru's ``logger.add``."""
    if fmt == "zst":
        _zstd()  # fail now rather than in the writer thread at the first rotation
        return _compress_zstd
    return fmt
