from __future__ import annotations

import pytest
from pydantic import BaseModel

import responsible_request as rr

from .conftest import body_of, chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


async def test_jsonl_and_sqlite_sinks(tmp_path, cleanup_sinks):
    log = rr.LogConfig(jsonl=tmp_path / "r.jsonl", sqlite=tmp_path / "r.db")
    client = make_client(lambda req: json_response(chat_response("x")), log=log)
    with rr.tags(experiment="e1"):
        await client.chat.completions.create(model="m", messages=MSG)
        await client.chat.completions.create(model="m", messages=MSG)
    await client.close()
    rr.remove_sinks()  # flush the enqueued writers

    for path in ("r.jsonl", "r.db"):
        rows = rr.load_records(tmp_path / path, as_dataframe=False)
        assert len(rows) == 2, path
        assert rows[0]["tags"] == {"experiment": "e1"}
        assert rows[0]["completion_tokens"] == 10
        assert rows[0]["request_body"]["messages"] == MSG


class Answer(BaseModel):
    city: str
    confidence: float = 0.5


def test_response_format_is_strict():
    rf = rr.response_format_from_model(Answer)
    schema = rf["json_schema"]["schema"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["city", "confidence"]


async def test_structured_retries_on_invalid_output():
    answers = iter(['{"town": "Bern"}', '```json\n{"city": "Bern", "confidence": 0.9}\n```'])
    seen = []

    def handler(req):
        seen.append(body_of(req))
        return json_response(chat_response(next(answers)))

    client = make_client(handler)
    result = await rr.structured(client, model="m", messages=MSG, schema=Answer)
    assert result == Answer(city="Bern", confidence=0.9)
    assert len(seen[1]["messages"]) == 3  # the error was fed back


async def test_structured_falls_back_to_json_object():
    seen = []

    def handler(req):
        body = body_of(req)
        seen.append(body)
        if body["response_format"]["type"] == "json_schema":
            return json_response({"error": {"message": "unsupported"}}, 400)
        return json_response(chat_response('{"city": "Bern", "confidence": 1}'))

    client = make_client(handler, max_retries=0)
    result = await rr.structured(client, model="m", messages=MSG, schema=Answer)
    assert result.city == "Bern"
    assert seen[1]["messages"][0]["role"] == "system"


async def test_structured_gives_up():
    client = make_client(lambda req: json_response(chat_response("nope")))
    with pytest.raises(rr.StructuredOutputError):
        await rr.structured(client, model="m", messages=MSG, schema=Answer, retries=1)


async def test_run_batch_keeps_order_and_tags(records):
    def handler(req):
        return json_response(chat_response(body_of(req)["messages"][0]["content"]))

    client = make_client(handler)
    reqs = [{"model": "m", "messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    results = await rr.run_batch(client, reqs, batch_id="b1")
    assert [r.choices[0].message.content for r in results] == ["0", "1", "2", "3", "4"]
    assert sorted(r["tags"]["item_index"] for r in records) == [0, 1, 2, 3, 4]
    assert {r["tags"]["batch_id"] for r in records} == {"b1"}


async def test_calibrate_pins_baseline():
    client = make_client(lambda req: json_response(chat_response(completion_tokens=1)))
    baseline = await rr.calibrate(client, "m", n=3)
    assert baseline > 0
    assert client.throttle.stats()["m"]["baseline"] == baseline


async def test_adaptive_throttling_end_to_end(clock):
    """Latency (here: a custom metric read from the response) drives the rate."""
    load = {"value": 1}

    def handler(req):
        return json_response(chat_response(completion_tokens=load["value"]))

    config = rr.ThrottleConfig(
        max_rpm=60000,
        min_rpm=6000,
        start_rpm=12000,
        warmup_requests=5,
        window=3,
        cooldown_s=0.0,
        ramp_interval_s=0.0,
        ramp_factor=2,
        metric=lambda r: float(r.completion_tokens or 1),
    )
    client = make_client(handler, throttle=config)

    async def send(n):
        for _ in range(n):
            await client.chat.completions.create(model="m", messages=MSG)

    await send(8)
    stats = client.throttle.stats()["m"]
    assert stats["state"] == "normal" and stats["rpm"] > 12000

    load["value"] = 5  # other users arrive: 5x baseline
    await send(2)
    assert client.throttle.stats()["m"]["state"] == "throttled"
    assert client.throttle.stats()["m"]["rpm"] == 6000

    load["value"] = 1  # they leave
    await send(10)
    stats = client.throttle.stats()["m"]
    assert stats["state"] in ("recovering", "normal") and stats["rpm"] > 6000
