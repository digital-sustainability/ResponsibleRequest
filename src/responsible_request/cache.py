"""Serve requests from previously logged request records."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import CacheConfig, LogConfig
from .sources import RecordSource, open_source


def cache_key(url: str, body: Any, salt: dict[str, Any] | None = None) -> str:
    """Hash of the request URL, its canonical JSON body and the salt (selected tag values)."""
    payload = {"url": url, "body": body}
    if salt:
        payload["salt"] = salt
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    """Response lookup by cache key, backed by the records of a SQLite database or JSONL file."""

    def __init__(self, config: CacheConfig, log: LogConfig | None = None) -> None:
        self.config = config
        target: CacheConfig | LogConfig | None = config
        if config.sqlite is None and config.jsonl is None:  # read the records we write
            target = log
            if log is not None and not log.enabled("response_body"):
                raise ValueError("the cache reads responses from the log: enable response_body")
        source = open_source(target)
        if source is None:
            raise ValueError(
                "the cache needs records to read from: set LogConfig(sqlite=...) or "
                "LogConfig(jsonl=...), or CacheConfig(sqlite=...) or CacheConfig(jsonl=...)"
            )
        self._source: RecordSource = source
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
