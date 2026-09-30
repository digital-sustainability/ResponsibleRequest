from __future__ import annotations

import asyncio
import time

from responsible_request.limiter import AdaptiveLimiter


async def _times(limiter: AdaptiveLimiter, n: int) -> list[float]:
    stamps: list[float] = []

    async def one() -> None:
        async with limiter:
            stamps.append(time.monotonic())

    await asyncio.gather(*(one() for _ in range(n)))
    return sorted(stamps)


async def test_requests_are_evenly_spaced():
    stamps = await _times(AdaptiveLimiter(rpm=1200, max_concurrency=10), 5)  # every 50 ms
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert all(g > 0.035 for g in gaps), gaps
    assert stamps[-1] - stamps[0] < 0.4


async def test_rate_change_applies_to_waiters():
    limiter = AdaptiveLimiter(rpm=6, max_concurrency=10)  # one every 10 s
    start = time.monotonic()
    task = asyncio.ensure_future(_times(limiter, 3))
    await asyncio.sleep(0.05)
    limiter.set_rpm(1200)
    await asyncio.wait_for(task, timeout=2)
    assert time.monotonic() - start < 1


async def test_concurrency_cap():
    limiter = AdaptiveLimiter(rpm=60000, max_concurrency=2)
    active = peak = 0

    async def one() -> None:
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(one() for _ in range(8)))
    assert peak == 2
    assert limiter.in_flight == 0


def test_survives_multiple_event_loops():
    limiter = AdaptiveLimiter(rpm=60000, max_concurrency=2)
    for _ in range(2):
        asyncio.run(_times(limiter, 3))
