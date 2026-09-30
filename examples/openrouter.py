"""An experiment against OpenRouter: fixed rate, cache, cost budget and a named run.

    export OPENROUTER_API_KEY=...        # or put it into .env
    uv run --env-file .env python examples/openrouter.py openai/gpt-4o-mini

Run it twice: the second run is answered from the cache and costs nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys

import responsible_request as rr


async def main(model: str) -> None:
    client = rr.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        throttle=rr.ThrottleConfig.fixed(120),  # OpenRouter enforces its own limits
        log=rr.LogConfig(sqlite="openrouter.db"),
        cache=True,
        cost=rr.CostConfig(budget_usd=0.10),  # counts what earlier runs in openrouter.db spent
        run=rr.RunConfig(name="openrouter-example", metadata={"note": "capitals"}),
        default_params=rr.reproducible(seed=42),
    )

    countries = ["France", "Italy", "Austria", "Germany", "Liechtenstein"]
    results = await rr.run_batch(
        client,
        [
            {"model": model, "messages": [{"role": "user", "content": f"Capital of {c}?"}]}
            for c in countries
        ],
    )
    for country, result in zip(countries, results, strict=True):
        text = result if isinstance(result, Exception) else result.choices[0].message.content
        print(f"{country}: {text}")

    print(client.cost.stats() if client.cost else None)
    print(client.cache.stats() if client.cache else None)
    await client.close()

    for record in rr.load_records("openrouter.db", as_dataframe=False)[-len(countries) :]:
        print(record["provider"], record["cost_usd"], record["cache_hit"], record["run_id"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"))
