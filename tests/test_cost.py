from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

import responsible_request as rr

from .conftest import chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


def priced_response(cost: float = 0.4) -> Any:
    body = chat_response()
    body["usage"]["cost"] = cost
    return json_response(body)


async def test_reported_cost_is_tracked(records):
    client = make_client(lambda req: priced_response(0.25), cost=True)
    await client.chat.completions.create(model="m", messages=MSG)
    await client.chat.completions.create(model="m", messages=MSG)
    assert client.cost is not None
    assert client.cost.stats()["spent_usd"] == 0.5 and client.cost.priced == 2
    assert client.throttle.stats()["m"]["cost_usd"] == 0.5
    assert [r["cost_usd"] for r in records] == [0.25, 0.25]


async def test_price_table_fallback(records):
    # 5 prompt tokens (2 cached), 10 completion tokens
    prices = {"m": rr.Price(input=1.0, output=2.0, cached_input=0.5)}
    client = make_client(
        lambda req: json_response(chat_response()), cost=rr.CostConfig(prices=prices)
    )
    await client.chat.completions.create(model="m", messages=MSG)
    (r,) = records
    assert r["cost_usd"] == pytest.approx((3 * 1.0 + 2 * 0.5 + 10 * 2.0) / 1e6)


async def test_budget_blocks_further_requests(records):
    calls = 0

    def handler(req: Any) -> Any:
        nonlocal calls
        calls += 1
        return priced_response(0.4)

    client = make_client(handler, cost=rr.CostConfig(budget_usd=1.0))
    for _ in range(3):
        await client.chat.completions.create(model="m", messages=MSG)
    with pytest.raises(rr.BudgetExceeded) as info:
        await client.chat.completions.create(model="m", messages=MSG)
    assert info.value.spent_usd == pytest.approx(1.2)
    assert calls == 3
    assert records[-1]["error"].startswith("BudgetExceeded") and records[-1]["status_code"] is None

    results = await rr.run_batch(client, [{"model": "m", "messages": MSG}] * 2)
    assert all(isinstance(r, rr.BudgetExceeded) for r in results)
    assert calls == 3


@pytest.mark.parametrize("fmt", ["sqlite", "jsonl"])
async def test_spent_cost_survives_a_restart_and_hits_are_free(tmp_path, cleanup_sinks, fmt):
    log = rr.LogConfig(**{fmt: tmp_path / f"r.{fmt}"})
    budget = rr.CostConfig(budget_usd=1.0)
    client = make_client(lambda req: priced_response(0.6), log=log, cost=budget, cache=True)
    await client.chat.completions.create(model="m", messages=MSG)
    await client.close()
    rr.remove_sinks()  # flush the enqueued writers

    client = make_client(lambda req: priced_response(0.6), log=log, cost=budget, cache=True)
    assert client.cost is not None and client.cost.spent_usd == pytest.approx(0.6)
    # a cache hit is served and costs nothing
    await client.chat.completions.create(model="m", messages=MSG)
    assert client.cost.spent_usd == pytest.approx(0.6)
    await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "new"}])
    with pytest.raises(rr.BudgetExceeded):
        await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "3"}])
    # the budget is exhausted, but answers we already paid for are still served
    await client.chat.completions.create(model="m", messages=MSG)
    await client.close()
    rr.remove_sinks()

    rows = rr.load_records(tmp_path / f"r.{fmt}", as_dataframe=False)
    assert [(r["cache_hit"], r["cost_usd"]) for r in rows if r["status_code"] == 200] == [
        (False, 0.6),
        (True, 0.0),
        (False, 0.6),
        (True, 0.0),
    ]


async def test_unpriced_requests_warn_once_with_a_budget():
    messages: list[str] = []
    logger.enable("responsible_request")
    handler_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        client = make_client(
            lambda req: json_response(chat_response()), cost=rr.CostConfig(budget_usd=1)
        )
        await client.chat.completions.create(model="m", messages=MSG)
        await client.chat.completions.create(model="m", messages=MSG)
    finally:
        logger.remove(handler_id)
        logger.disable("responsible_request")
    assert client.cost is not None and client.cost.unpriced == 2
    assert sum("no cost reported" in m for m in messages) == 1
