from __future__ import annotations

import json
from typing import Any

import openai
import pytest
from loguru import logger

import responsible_request as rr
from responsible_request._http import httpx

from .conftest import chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def cleanup_sinks():
    yield
    rr.remove_sinks()
    logger.disable("responsible_request")


class Server:
    """Answers with a new content per call, so cached and fresh answers can be told apart."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, req: Any) -> Any:
        self.calls += 1
        return json_response(chat_response(f"answer {self.calls}"))


async def ask(client: rr.AsyncOpenAI, **kwargs: Any) -> str | None:
    params = {"model": "m", "messages": MSG, **kwargs}
    resp = await client.chat.completions.create(**params)
    return resp.choices[0].message.content


@pytest.mark.parametrize("fmt", ["sqlite", "jsonl"])
async def test_repeated_request_is_served_from_the_log(tmp_path, cleanup_sinks, records, fmt):
    log = rr.LogConfig(**{fmt: tmp_path / f"r.{fmt}"})
    server = Server()
    client = make_client(server, log=log, cache=True)
    assert await ask(client) == "answer 1"
    await logger.complete()  # flush the enqueued writers
    assert await ask(client) == "answer 1"
    assert await ask(client, temperature=0.5) == "answer 2"  # different parameters: miss
    assert server.calls == 2
    assert client.cache is not None and client.cache.stats() == {"hits": 1, "misses": 2}

    first, hit, _ = records
    assert first["cache_key"] == hit["cache_key"] and not first["cache_hit"]
    assert hit["cache_hit"] and hit["status_code"] == 200 and hit["completion_tokens"] == 10
    assert hit["latency_s"] is None and hit["rpm"] is None
    assert client.throttle.stats()["m"]["requests"] == 2  # hits bypass the throttle


async def test_cache_survives_a_new_client(tmp_path, cleanup_sinks):
    log = rr.LogConfig(sqlite=tmp_path / "r.db")
    server = Server()
    client = make_client(server, log=log, cache=True)
    await ask(client)
    await client.close()

    # e.g. rerunning a script: a new client reads what the previous one recorded
    client = make_client(server, log=log, cache=True)
    assert await ask(client) == "answer 1"
    assert server.calls == 1


async def test_key_tags_salt_the_cache_key(tmp_path, cleanup_sinks):
    log = rr.LogConfig(sqlite=tmp_path / "r.db")
    server = Server()
    client = make_client(server, log=log, cache=rr.CacheConfig(key_tags=("sample",)))
    answers = []
    for i in range(3):
        with rr.tags(sample=i, run="first"):
            answers.append(await ask(client))
    await logger.complete()
    assert answers == ["answer 1", "answer 2", "answer 3"]
    for i in range(3):
        with rr.tags(sample=i, run="second"):  # other tags don't affect the key
            assert await ask(client) == answers[i]
    assert server.calls == 3


async def test_errors_and_streams_are_not_cached(tmp_path, cleanup_sinks):
    log = rr.LogConfig(sqlite=tmp_path / "r.db")
    calls = 0

    def handler(req: Any) -> Any:
        nonlocal calls
        calls += 1
        if json.loads(req.content).get("stream"):
            chunk = {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": "s"}}]}
            sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        if calls == 1:
            return json_response({"error": {"message": "bad"}}, 400)
        return json_response(chat_response())

    client = make_client(handler, log=log, cache=True, max_retries=0)
    with pytest.raises(openai.BadRequestError):
        await ask(client)
    await logger.complete()
    assert await ask(client) == "OK"
    for _ in range(2):
        stream = await client.chat.completions.create(model="m", messages=MSG, stream=True)
        [c async for c in stream]
        await logger.complete()
    assert calls == 4


async def test_default_params_are_part_of_the_key(tmp_path, cleanup_sinks):
    log = rr.LogConfig(sqlite=tmp_path / "r.db")
    server = Server()
    client = make_client(server, log=log, cache=True, default_params=rr.reproducible(seed=1))
    await ask(client)
    await client.close()
    client = make_client(server, log=log, cache=True, default_params=rr.reproducible(seed=2))
    assert await ask(client) == "answer 2"


def test_cache_needs_a_record_source():
    with pytest.raises(ValueError, match="records to read from"):
        rr.http_client(log=False, cache=True)
    with pytest.raises(ValueError, match="response_body"):
        rr.http_client(log=rr.LogConfig.minimal(sqlite="x.db"), cache=True)
