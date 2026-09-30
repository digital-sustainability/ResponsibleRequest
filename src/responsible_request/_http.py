"""Resolve the HTTP library used by the installed ``openai`` SDK.

``openai<3`` is built on ``httpx``; ``openai>=3`` is built on its fork ``httpx2``. Both expose the
same transport API, so we pick whichever one ``openai.DefaultAsyncHttpxClient`` derives from.

The library's per-request ``INFO`` line ("HTTP Request: POST ...") is silenced, since every
request is already recorded by our transport. Lower the level again to get it back.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

import openai


def _resolve() -> ModuleType:
    for base in openai.DefaultAsyncHttpxClient.__mro__:
        root = base.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    return importlib.import_module("httpx")  # pragma: no cover


httpx = _resolve()

logging.getLogger(httpx.__name__).setLevel(logging.WARNING)

__all__ = ["httpx"]
