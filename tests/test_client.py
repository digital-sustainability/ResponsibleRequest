from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import openai
import pytest

import responsible_request as rr
from responsible_request._http import httpx
from responsible_request.records import RequestRecord

from .conftest import body_of, chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


async def test_drop_in_chat_completion_records_everything(records):
    client = make_client(lambda req: json_response(chat_response("hello")))
    resp = await client.chat.completions.create(model="m", messages=MSG, temperature=0.5)
    assert resp.choices[0].message.content == "hello"

    (r,) = records
    assert r["path"] == "/api/v1/chat/completions" and r["model"] == "m"
    assert r["status_code"] == 200 and r["error"] is None
    assert (r["prompt_tokens"], r["completion_tokens"], r["total_tokens"]) == (5, 10, 15)
    assert (r["cached_tokens"], r["reasoning_tokens"]) == (2, 3)
    assert r["latency_s"] >= 0 and r["wait_s"] >= 0 and r["ttfb_s"] is not None
    assert r["params"]["temperature"] == 0.5 and "messages" not in r["params"]
    assert r["request_body"]["messages"] == MSG
    assert r["response_body"]["choices"][0]["message"]["content"] == "hello"
    assert r["response_id"] == "chatcmpl-1" and r["finish_reason"] == "stop"
    assert r["state"] == "warmup" and r["rpm"] == 6000
    stats = client.throttle.stats()
    assert stats["m"]["requests"] == 1 and stats["m"]["completion_tokens"] == 10


async def test_field_groups_can_be_disabled(records):
    log = rr.LogConfig.minimal().with_fields(usage=False)
    client = make_client(lambda req: json_response(chat_response()), log=log)
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    for name in ("params", "request_body", "response_body", "prompt_tokens"):
        assert name not in r
    assert "latency_s" in r and "rpm" in r


async def test_default_params_only_fill_missing_keys():
    seen: list[dict[str, Any]] = []

    def handler(req):
        seen.append(body_of(req))
        return json_response(chat_response())

    client = make_client(handler, default_params=rr.reproducible(seed=7))
    await client.chat.completions.create(model="m", messages=MSG, temperature=0.9)
    assert seen[0]["temperature"] == 0.9
    assert seen[0]["seed"] == 7 and seen[0]["top_p"] == 1.0


async def test_streaming_measures_ttfb_and_usage(records):
    chunks = [
        {"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {
            "id": "c",
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        },
        {
            "id": "c",
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    seen: list[dict[str, Any]] = []

    def handler(req):
        seen.append(body_of(req))
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    stream = await client.chat.completions.create(model="m", messages=MSG, stream=True)
    text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    assert text == "Hello"
    assert seen[0]["stream_options"] == {"include_usage": True}
    (r,) = records
    assert r["stream"] is True and r["completion_tokens"] == 2
    assert r["response_body"]["content"] == "Hello" and r["finish_reason"] == "stop"
    assert r["ttfb_s"] is not None


async def test_429_throttles_and_sdk_retries_are_recorded(records):
    calls = 0

    def handler(req):
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response(
                {"error": {"message": "slow down"}}, 429, **{"retry-after-ms": "1"}
            )
        return json_response(chat_response())

    config = rr.ThrottleConfig(max_rpm=6000, min_rpm=3000, start_rpm=6000)
    client = make_client(handler, throttle=config, max_retries=2)
    await client.chat.completions.create(model="m", messages=MSG)
    assert [r["status_code"] for r in records] == [429, 200]
    assert [r["attempt"] for r in records] == [0, 1]
    assert records[0]["state"] == "throttled"
    assert client.throttle.stats()["m"]["rpm"] == 3000


async def test_embeddings_and_multipart_audio_pass_through(records):
    def handler(req):
        if req.url.path.endswith("/embeddings"):
            return json_response(
                {
                    "object": "list",
                    "model": "e",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            )
        return json_response({"text": "transcript"})

    client = make_client(handler)
    emb = await client.embeddings.create(model="e", input="hello")
    assert emb.data[0].embedding == [0.1]
    tr = await client.audio.transcriptions.create(model="whisperx", file=("a.wav", b"RIFF"))
    assert tr.text == "transcript"
    assert records[0]["prompt_tokens"] == 2
    assert records[1]["path"].endswith("/audio/transcriptions") and records[1]["model"] is None
    lanes = client.throttle.stats()
    assert set(lanes) == {"e", "/api/v1/audio/transcriptions"}


async def test_server_tools_are_not_used_for_load_estimation():
    client = make_client(
        lambda req: json_response(chat_response()),
        throttle=rr.ThrottleConfig(max_rpm=6000, start_rpm=6000, warmup_requests=1),
    )
    tools = [{"type": "mcp", "server_label": "x", "server_url": "litellm_proxy"}]
    await client.chat.completions.create(model="m", messages=MSG, tools=tools)
    assert client.throttle.stats()["m"]["baseline"] is None
    await client.chat.completions.create(model="m", messages=MSG)
    assert client.throttle.stats()["m"]["baseline"] is not None


async def test_connection_errors_are_recorded(records):
    def handler(req):
        raise httpx.ReadTimeout("boom", request=req)

    client = make_client(handler, max_retries=0)
    with pytest.raises(openai.APITimeoutError):
        await client.chat.completions.create(model="m", messages=MSG)
    assert records[0]["error"].startswith("ReadTimeout")
    assert records[0]["state"] == "throttled"
    assert client.throttle.stats()["m"]["in_flight"] == 0


async def test_plain_openai_client_with_http_client():
    http = rr.http_client(
        rr.ThrottleConfig(max_rpm=6000, start_rpm=6000),
        log=False,
        transport=httpx.MockTransport(lambda req: json_response(chat_response())),
    )
    client = openai.AsyncOpenAI(api_key="t", base_url="http://test/v1", http_client=http)
    await client.chat.completions.create(model="m", messages=MSG)
    assert rr.get_throttle(client).stats()["m"]["requests"] == 1
    copy = client.with_options(timeout=5)
    assert rr.get_throttle(copy) is rr.get_throttle(client)


async def test_rr_client_copy_keeps_throttle():
    client = make_client(lambda req: json_response(chat_response()))
    copy = client.with_options(max_retries=0)
    assert copy.throttle is client.throttle


def test_one_throttle_shared_by_clients_in_threads():
    throttle = rr.Throttle(rr.ThrottleConfig.fixed(60000, max_concurrency=4))
    errors: list[BaseException] = []

    def worker() -> None:
        async def main() -> None:
            client = make_client(lambda req: json_response(chat_response()), throttle=throttle)
            await asyncio.gather(
                *(client.chat.completions.create(model="m", messages=MSG) for _ in range(10))
            )
            await client.close()

        try:
            asyncio.run(main())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads) and not errors
    assert throttle.stats()["m"]["requests"] == 60


def _rate_limited(seconds: str | None = None) -> Any:
    headers = {} if seconds is None else {"retry-after": seconds}
    return json_response({"error": {"message": "slow down"}}, 429, **headers)


async def test_fixed_mode_obeys_retry_after(records):
    responses = iter([_rate_limited("0.3"), json_response(chat_response())])
    config = rr.ThrottleConfig.fixed(60000, backoff_jitter_s=0.05)
    client = make_client(lambda req: next(responses), throttle=config, max_retries=0)
    with pytest.raises(openai.RateLimitError):
        await client.chat.completions.create(model="m", messages=MSG)
    await client.chat.completions.create(model="m", messages=MSG)
    assert [r["status_code"] for r in records] == [429, 200]
    assert 0.29 <= records[1]["wait_s"] < 0.6
    assert records[0]["state"] == "fixed" and client.throttle.stats()["m"]["rpm"] == 60000


async def test_fixed_mode_pauses_only_the_affected_model(records):
    def handler(req):
        if body_of(req)["model"] == "a":
            return _rate_limited("30")
        return json_response(chat_response())

    client = make_client(handler, throttle=rr.ThrottleConfig.fixed(60000), max_retries=0)
    with pytest.raises(openai.RateLimitError):
        await client.chat.completions.create(model="a", messages=MSG)
    await client.chat.completions.create(model="b", messages=MSG)
    assert records[1]["wait_s"] < 0.1
    stats = client.throttle.stats()
    assert stats["a"]["paused_s"] > 29 and stats["b"]["paused_s"] == 0


def test_fixed_mode_backoff_without_retry_after():
    config = rr.ThrottleConfig.fixed(
        60000, backoff_initial_s=0.05, backoff_max_s=0.15, backoff_jitter_s=0
    )
    lane = rr.Throttle(config).lane("/v1/chat/completions", "m")

    def observe(status: int) -> float:
        lane.observe(RequestRecord("id", "t", "POST", "/", status_code=status), eligible=True)
        return lane.limiter.paused_s

    pauses = []
    for _ in range(4):
        pauses.append(observe(429))
        time.sleep(pauses[-1] + 0.01)  # the next request is sent after the pause
    assert pauses == pytest.approx([0.05, 0.1, 0.15, 0.15], abs=0.01)
    observe(429)  # a 429 of a request sent during the pause does not raise the level
    assert lane.limiter.paused_s == pytest.approx(0.15, abs=0.01)
    time.sleep(0.16)
    observe(200)
    assert observe(429) == pytest.approx(0.05, abs=0.01)  # success resets the backoff


def test_fixed_mode_jitter_differs_between_throttles():
    config = rr.ThrottleConfig.fixed(60000, backoff_jitter_s=10)
    pauses = set()
    for _ in range(2):
        lane = rr.Throttle(config).lane("/v1/chat/completions", "m")
        record = RequestRecord("id", "t", "POST", "/", status_code=429)
        lane.observe(record, eligible=True, retry_after=0)
        pauses.add(round(lane.limiter.paused_s, 3))
    assert len(pauses) == 2


def test_retry_after_parsing():
    from email.utils import formatdate

    from responsible_request.transport import retry_after_seconds

    assert retry_after_seconds(httpx.Headers({"retry-after-ms": "1500"})) == 1.5
    assert retry_after_seconds(httpx.Headers({"retry-after": "2"})) == 2.0
    assert retry_after_seconds(httpx.Headers({"retry-after": "-3"})) == 0.0
    date = formatdate(time.time() + 20, usegmt=True)
    assert 18 < retry_after_seconds(httpx.Headers({"retry-after": date})) <= 20  # type: ignore[operator]
    assert retry_after_seconds(httpx.Headers({"retry-after": "soon"})) is None
    assert retry_after_seconds(httpx.Headers()) is None
