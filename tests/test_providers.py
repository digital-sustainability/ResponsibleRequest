from __future__ import annotations

import json
from typing import Any

import responsible_request as rr
from responsible_request._http import httpx

from .conftest import chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


async def test_plain_response_leaves_provider_fields_empty(records):
    client = make_client(lambda req: json_response(chat_response()))
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    for name in ("provider", "cost_usd", "gateway_request_id", "upstream_duration_ms"):
        assert r[name] is None, name
    assert r["provider_meta"] is None
    assert r["run_id"] == client.run_id


async def test_litellm_headers(records):
    headers = {
        "x-litellm-call-id": "call-1",
        "x-litellm-response-duration-ms": "123.5",
        "x-litellm-response-cost": "0.002",
        "x-litellm-model-id": "abc",
        "x-litellm-model-api-base": "http://backend",
    }
    client = make_client(lambda req: json_response(chat_response(), **headers))
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    assert r["gateway_request_id"] == "call-1"
    assert r["upstream_duration_ms"] == 123.5 and r["cost_usd"] == 0.002
    assert r["provider_meta"] == {
        "litellm": {"model_id": "abc", "model_api_base": "http://backend"}
    }


def openrouter_response() -> dict[str, Any]:
    body = chat_response()
    body["provider"] = "DeepInfra"
    body["system_fingerprint"] = "fp_1"
    body["choices"][0]["native_finish_reason"] = "stop"
    body["usage"].update(cost=0.0005, is_byok=False)
    return body


async def test_openrouter_body(records):
    client = make_client(
        lambda req: json_response(openrouter_response(), **{"x-request-id": "req-9"})
    )
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    assert r["provider"] == "DeepInfra" and r["cost_usd"] == 0.0005
    assert r["system_fingerprint"] == "fp_1" and r["gateway_request_id"] == "req-9"
    assert r["provider_meta"] == {"openrouter": {"is_byok": False, "native_finish_reason": "stop"}}


async def test_openrouter_stream(records):
    chunks = [
        {
            "id": "gen-1",
            "model": "m",
            "provider": "Together",
            "system_fingerprint": "fp_2",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        },
        {
            "id": "gen-1",
            "model": "m",
            "provider": "Together",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "cost": 1e-5},
        },
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    client = make_client(
        lambda req: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    stream = await client.chat.completions.create(model="m", messages=MSG, stream=True)
    [c async for c in stream]
    (r,) = records
    assert r["provider"] == "Together" and r["cost_usd"] == 1e-5
    assert r["system_fingerprint"] == "fp_2"


async def test_custom_extractor_and_broken_extractor(records):
    def queue_time(record, headers, body):
        if "x-queue-ms" in headers:
            rr.providers.meta(record, "mine")["queue_ms"] = float(headers["x-queue-ms"])

    def broken(record, headers, body):
        raise RuntimeError("boom")

    client = make_client(
        lambda req: json_response(chat_response(), **{"x-queue-ms": "7"}),
        extractors=[broken, queue_time],
    )
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    assert r["status_code"] == 200
    assert r["provider_meta"] == {"mine": {"queue_ms": 7.0}}
