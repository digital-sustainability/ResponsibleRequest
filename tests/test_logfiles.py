from __future__ import annotations

import json
import os
from typing import Any

import pytest
from loguru import logger

import responsible_request as rr
from responsible_request.logfiles import log_segments, rotated_segments
from responsible_request.sources import JSONLSource

from .conftest import chat_response, json_response, make_client

FORMATS = ["gz", "bz2", "xz", "zst"]


class Server:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: Any) -> Any:
        self.calls += 1
        body = chat_response(f"answer {self.calls}")
        body["usage"]["cost"] = 0.1
        return json_response(body)


async def ask(client: rr.AsyncOpenAI, prompt: str) -> str | None:
    msgs = [{"role": "user", "content": prompt}]
    resp = await client.chat.completions.create(model="m", messages=msgs)
    return resp.choices[0].message.content


@pytest.mark.parametrize("fmt", FORMATS)
async def test_compressed_segments_are_read_back(tmp_path, cleanup_sinks, fmt):
    path = tmp_path / "r.jsonl"
    # rotation=1 byte: every record after the first starts a new segment
    log = rr.LogConfig(jsonl=path, rotation=1, compression=fmt)
    server = Server()
    client = make_client(server, log=log, cache=True)
    for prompt in "abc":
        await ask(client, prompt)
    await client.close()
    rr.remove_sinks()  # flush the enqueued writers

    segments = rotated_segments(path)
    assert all(s.name.endswith(f".jsonl.{fmt}") for s in segments)
    rows = rr.load_records(path, as_dataframe=False)
    assert [r["response_body"]["choices"][0]["message"]["content"] for r in rows] == [
        "answer 1",
        "answer 2",
        "answer 3",
    ]
    # each segment can be read on its own; the last record is still in the active file
    assert sum(len(rr.load_records(s, as_dataframe=False)) for s in segments) == 2

    # a new client finds answers and spent cost in every segment
    client = make_client(server, log=log, cache=True, cost=rr.CostConfig(budget_usd=1.0))
    assert client.cost is not None and client.cost.spent_usd == pytest.approx(0.3)
    assert [await ask(client, p) for p in "abc"] == ["answer 1", "answer 2", "answer 3"]
    assert server.calls == 3


async def test_cache_follows_a_rotation_during_the_run(tmp_path, cleanup_sinks):
    log = rr.LogConfig(jsonl=tmp_path / "r.jsonl", rotation=1, compression="gz")
    server = Server()
    client = make_client(server, log=log, cache=True)
    await ask(client, "a")
    await logger.complete()
    await ask(client, "b")  # indexes "a" by its offset in the active file, then rotates it away
    await logger.complete()
    # the active file now holds "b" at that offset: the stale entry must not be served
    assert await ask(client, "a") == "answer 1"
    assert await ask(client, "b") == "answer 2"
    assert server.calls == 2


async def test_compression_without_rotation_applies_on_close(tmp_path, cleanup_sinks):
    path = tmp_path / "r.jsonl"
    log = rr.LogConfig(jsonl=path, rotation=None, compression="zst")
    server = Server()
    for prompt in "ab":  # two runs: the second keeps the first archive under a dated name
        client = make_client(server, log=log, cache=True)
        await ask(client, prompt)
        await client.close()
        rr.remove_sinks()
    assert not path.exists()
    assert (tmp_path / "r.jsonl.zst").exists()
    assert len(rotated_segments(path)) == 2
    rows = rr.load_records(path, as_dataframe=False)
    assert [r["response_body"]["choices"][0]["message"]["content"] for r in rows] == [
        "answer 1",
        "answer 2",
    ]
    # an archive can also be the cache source on its own
    client = make_client(server, cache=rr.CacheConfig(jsonl=tmp_path / "r.jsonl.zst"))
    assert await ask(client, "b") == "answer 2"
    assert server.calls == 2


def test_segment_discovery(tmp_path):
    path = tmp_path / "r.jsonl"
    names = [
        "r.2026-01-01_00-00-00_000000.jsonl.gz",
        "r.2026-01-01_00-00-01_000000.2.jsonl",
        "r.2026-01-01_00-00-02_000000.jsonl",
        "r.2026-01-01_00-00-02_000000.jsonl.gz",  # partial: its plain file still exists
        "r.jsonl.gz",
        "r.jsonl",
        "r.runs.jsonl",
        "other.jsonl",
        "r.2026-01-01_00-00-00_000000.jsonl.zip",
    ]
    for i, name in enumerate(names):
        (tmp_path / name).write_text("")
        os.utime(tmp_path / name, (1_700_000_000 + i,) * 2)  # listed in this order
    assert [p.name for p in rotated_segments(path)] == [
        "r.2026-01-01_00-00-00_000000.jsonl.gz",
        "r.2026-01-01_00-00-01_000000.2.jsonl",
        "r.2026-01-01_00-00-02_000000.jsonl",
        "r.jsonl.gz",
    ]
    assert log_segments(path)[-1] == path
    assert log_segments(tmp_path / "r.jsonl.gz") == [tmp_path / "r.jsonl.gz"]


def test_jsonl_source_ignores_partial_lines(tmp_path):
    path = tmp_path / "r.jsonl"
    row = {"cache_key": "k", "status_code": 200, "response_body": {"x": 1}, "cost_usd": 0.5}
    path.write_text(json.dumps(row) + "\n" + '{"cache_key": "k2"')
    source = JSONLSource(path)
    assert source.get("k") == {"x": 1}
    assert source.get("k2") is None
    assert source.total_cost() == pytest.approx(0.5)


def test_unknown_compression_is_rejected():
    with pytest.raises(ValueError, match="compression"):
        rr.LogConfig(jsonl="r.jsonl", compression="zip")
    with pytest.raises(ValueError, match="plain"):
        rr.LogConfig(jsonl="r.jsonl.gz")
