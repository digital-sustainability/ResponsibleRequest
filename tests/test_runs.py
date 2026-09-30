from __future__ import annotations

import json
import sqlite3

import pytest

import responsible_request as rr

from .conftest import chat_response, json_response, make_client

MSG = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("fmt", ["sqlite", "jsonl"])
async def test_run_manifest_is_written_once_and_linked(tmp_path, cleanup_sinks, fmt):
    path = tmp_path / f"r.{fmt}"
    client = make_client(
        lambda req: json_response(chat_response()),
        log=rr.LogConfig(**{fmt: path}),
        run=rr.RunConfig(name="ablation", metadata={"dataset": "v2"}),
        default_params=rr.reproducible(seed=3),
        cost=rr.CostConfig(prices={"m": rr.Price(1, 2)}),
    )
    await client.chat.completions.create(model="m", messages=MSG)
    await client.chat.completions.create(model="m", messages=MSG)
    await client.close()
    rr.remove_sinks()

    (run,) = rr.load_runs(path, as_dataframe=False)
    assert run["run_id"] == client.run_id and run["name"] == "ablation"
    assert run["metadata"] == {"dataset": "v2"}
    assert run["endpoint"] == "http://test"
    assert run["versions"]["openai"] and run["versions"]["python"]
    assert run["config"]["default_params"]["seed"] == 3
    assert run["config"]["throttle"]["max_rpm"] == 6000
    assert run["config"]["cost"]["prices"]["m"] == {"input": 1, "output": 2, "cached_input": None}
    assert "api_key" not in json.dumps(run)

    rows = rr.load_records(path, as_dataframe=False)
    assert {r["run_id"] for r in rows} == {client.run_id}


async def test_no_manifest_without_requests(tmp_path, cleanup_sinks):
    path = tmp_path / "r.jsonl"
    client = make_client(lambda req: json_response(chat_response()), log=rr.LogConfig(jsonl=path))
    await client.close()
    rr.remove_sinks()
    assert rr.load_runs(path, as_dataframe=False) == []


async def test_run_name_shortcut():
    client = make_client(lambda req: json_response(chat_response()), run="quick")
    assert client.run_id


async def test_old_database_is_migrated(tmp_path, cleanup_sinks):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE requests ("request_id", "path", "litellm_call_id")')
    conn.execute("INSERT INTO requests VALUES ('old', '/chat/completions', 'call-0')")
    conn.commit()
    conn.close()

    client = make_client(lambda req: json_response(chat_response()), log=rr.LogConfig(sqlite=path))
    await client.chat.completions.create(model="m", messages=MSG)
    await client.close()
    rr.remove_sinks()

    rows = rr.load_records(path, as_dataframe=False)
    assert [r["request_id"] for r in rows][0] == "old" and rows[0]["litellm_call_id"] == "call-0"
    assert rows[1]["run_id"] == client.run_id and rows[1]["litellm_call_id"] is None
    assert len(rr.load_runs(path, as_dataframe=False)) == 1
