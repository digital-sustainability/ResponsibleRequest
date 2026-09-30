"""Run manifests: what is needed to reproduce the requests of one client."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from loguru import logger

from .config import RunConfig
from .records import utc_iso

RUN_COLUMNS = (
    "run_id",
    "name",
    "started_at",
    "endpoint",
    "metadata",
    "versions",
    "platform",
    "hostname",
    "argv",
    "cwd",
    "git",
    "config",
)
RUN_JSON_FIELDS = ("metadata", "versions", "argv", "git", "config")


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git(cwd: str) -> dict[str, Any] | None:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


def to_jsonable(value: Any) -> Any:
    """Dataclasses and mappings as plain JSON types (callables and other objects as ``repr``)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return json.loads(json.dumps(value, default=_default))


class Run:
    """The run of one client: its ``run_id`` and the manifest written on the first request."""

    def __init__(self, config: RunConfig | None = None, **configs: Any) -> None:
        self.config = config or RunConfig()
        self.run_id = uuid.uuid4().hex
        cwd = os.getcwd()
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "name": self.config.name,
            "started_at": utc_iso(time.time()),
            "endpoint": None,
            "metadata": to_jsonable(self.config.metadata),
            "versions": {
                "responsible_request": _version("responsible-request"),
                "openai": _version("openai"),
                "python": platform.python_version(),
            },
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "cwd": cwd,
            "git": _git(cwd) if self.config.capture_git else None,
            "config": to_jsonable(configs),
        }
        self.emitted = False

    def emit(self, endpoint: str) -> None:
        """Write the manifest (once), with the scheme and host of the first request."""
        if self.emitted:
            return
        self.emitted = True
        self.manifest["endpoint"] = endpoint
        logger.bind(rr_run=self.manifest, rr_json=json.dumps(self.manifest)).trace(
            "run {} started", self.run_id
        )
