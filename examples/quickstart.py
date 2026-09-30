"""Minimal usage against an OpenAI-compatible gateway.

    cp .env.example .env   # then fill in OPENAI_BASE_URL and OPENAI_API_KEY
    uv run --env-file .env python examples/quickstart.py gpt-oss:120b
"""

from __future__ import annotations

import asyncio
import sys

from pydantic import BaseModel

import responsible_request as rr


class Capital(BaseModel):
    country: str
    capital: str


async def main(model: str) -> None:
    client = rr.AsyncOpenAI(
        throttle=rr.ThrottleConfig(max_rpm=120),
        log=rr.LogConfig(sqlite="requests.db", jsonl="requests.jsonl"),
        default_params=rr.reproducible(seed=42),
    )

    # 1. Plain chat completion, exactly as with openai.AsyncOpenAI
    response = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say hello in Romansh."}]
    )
    print(response.choices[0].message.content)

    # 2. Structured output validated against a Pydantic model
    answer = await rr.structured(
        client,
        model=model,
        messages=[{"role": "user", "content": "What is the capital of Switzerland?"}],
        schema=Capital,
    )
    print(answer)

    # 3. Many requests, paced by the throttle, results in input order
    countries = ["France", "Italy", "Austria", "Germany", "Liechtenstein"]
    with rr.tags(experiment="quickstart"):
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

    print(client.throttle.stats())
    await client.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gpt-oss:120b"))
