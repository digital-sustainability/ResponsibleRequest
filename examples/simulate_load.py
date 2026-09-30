"""Simulate a shared inference server to watch the adaptive throttle at work.

The fake server batches up to ``BATCH`` requests without slowing down; beyond that, every
request slows down proportionally (like vLLM/Ollama under load). Between ``BUSY_FROM`` and
``BUSY_UNTIL`` seconds, other users keep ``OTHERS`` requests in flight.

Time is compressed: cooldowns and ramp intervals are seconds instead of minutes.

    uv run python examples/simulate_load.py            # prints a timeline
    uv run python examples/simulate_load.py --plot     # also writes simulate_load.png (matplotlib)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from loguru import logger

import responsible_request as rr
from responsible_request._http import httpx

BASE_LATENCY = 0.3  # seconds for a 100-token answer on an idle server
BATCH = 16  # requests served in parallel without slowdown
OTHERS = 48  # other users' in-flight requests while busy
DURATION, BUSY_FROM, BUSY_UNTIL = 45.0, 12.0, 24.0


class FakeServer:
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.active = 0

    def others(self) -> int:
        t = time.monotonic() - self.start
        return OTHERS if BUSY_FROM <= t < BUSY_UNTIL else 0

    async def handle(self, request: Any) -> Any:
        self.active += 1
        try:
            slowdown = max(1.0, (self.active + self.others()) / BATCH)
            await asyncio.sleep(BASE_LATENCY * slowdown)
        finally:
            self.active -= 1
        return httpx.Response(
            200,
            json={
                "id": "sim",
                "object": "chat.completion",
                "created": 0,
                "model": "sim-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            },
        )


async def main(plot: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> {message}")

    server = FakeServer()
    config = rr.ThrottleConfig(
        max_rpm=1800,  # 30 req/s
        min_rpm=60,
        start_rpm=300,
        max_concurrency=64,
        warmup_requests=15,
        window=8,
        cooldown_s=4,
        ramp_interval_s=1,
        ramp_factor=1.5,
    )
    client = rr.AsyncOpenAI(
        api_key="sim",
        base_url="http://sim/v1",
        http_client=rr.http_client(
            config,
            rr.LogConfig(summary_interval_s=None),
            transport=httpx.MockTransport(server.handle),
        ),
    )

    timeline: list[tuple[float, float, str, float | None, int]] = []
    stop = asyncio.Event()

    async def sample() -> None:
        while not stop.is_set():
            s = client.throttle.stats().get("sim-model")
            if s:
                t = time.monotonic() - server.start
                timeline.append((t, s["rpm"], s["state"], s["load_ratio"], server.others()))
            await asyncio.sleep(0.5)

    async def one() -> None:
        await client.chat.completions.create(
            model="sim-model", messages=[{"role": "user", "content": "hi"}]
        )

    async def producer() -> None:
        tasks: set[asyncio.Task[None]] = set()
        while time.monotonic() - server.start < DURATION:
            # keep a backlog so the limiter, not the producer, sets the pace
            while len(tasks) < 200:
                task = asyncio.create_task(one())
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            await asyncio.sleep(0.05)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    sampler = asyncio.create_task(sample())
    await producer()
    stop.set()
    await sampler
    await client.close()

    print(f"\n{'t [s]':>6} {'others':>6} {'RPM':>7} {'state':>11} {'load':>6}")
    for t, rpm, state, ratio, others in timeline[::2]:
        load = f"{ratio:.2f}" if ratio is not None else "-"
        print(f"{t:6.1f} {others:6d} {rpm:7.0f} {state:>11} {load:>6}")

    if plot:
        import matplotlib.pyplot as plt

        ts = [p[0] for p in timeline]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(ts, [p[1] for p in timeline], label="our RPM")
        ax.axvspan(BUSY_FROM, BUSY_UNTIL, alpha=0.15, color="red", label="other users active")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("requests / minute")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig("simulate_load.png", dpi=150)
        print("wrote simulate_load.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    asyncio.run(main(parser.parse_args().plot))
