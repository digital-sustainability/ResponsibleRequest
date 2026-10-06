from __future__ import annotations

import asyncio
import threading
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


def _run_threads(target, n: int, timeout: float = 10) -> None:
    threads = [threading.Thread(target=target, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "a thread never got its requests through"


def test_shared_across_threads():
    """Many threads, each with its own event loop, share one rate and never hang."""
    limiter = AdaptiveLimiter(rpm=6000, max_concurrency=4)  # one every 10 ms
    stamps: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        mine = asyncio.run(_times(limiter, 10))  # all 10 requests at once
        with lock:
            stamps.extend(mine)

    _run_threads(worker, 8)
    assert len(stamps) == 80
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert min(gaps) > 0.007, min(gaps)
    assert limiter.in_flight == 0


def test_set_rpm_from_other_thread_wakes_waiters():
    limiter = AdaptiveLimiter(rpm=6, max_concurrency=10)  # one every 10 s
    done = threading.Event()

    def worker() -> None:
        asyncio.run(_times(limiter, 3))
        done.set()

    thread = threading.Thread(target=worker, daemon=True)
    start = time.monotonic()
    thread.start()
    time.sleep(0.05)
    limiter.set_rpm(6000)
    assert done.wait(2)
    assert time.monotonic() - start < 1


def test_concurrency_cap_across_threads():
    limiter = AdaptiveLimiter(rpm=600000, max_concurrency=3)
    active = peak = 0
    lock = threading.Lock()

    async def one() -> None:
        nonlocal active, peak
        async with limiter:
            with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            with lock:
                active -= 1

    async def many() -> None:
        await asyncio.gather(*(one() for _ in range(10)))

    _run_threads(lambda: asyncio.run(many()), 6)
    assert peak == 3
    assert limiter.in_flight == 0


async def test_pause_holds_back_requests():
    limiter = AdaptiveLimiter(rpm=60000, max_concurrency=10)
    limiter.pause(0.2)
    start = time.monotonic()
    await _times(limiter, 2)
    assert time.monotonic() - start >= 0.19


async def test_cancelled_waiters_do_not_block_others():
    limiter = AdaptiveLimiter(rpm=1200, max_concurrency=1)
    async with limiter:
        blocked = asyncio.ensure_future(limiter.acquire())
        await asyncio.sleep(0.01)
        blocked.cancel()
    await asyncio.wait_for(_times(limiter, 2), timeout=1)
    assert limiter.in_flight == 0
