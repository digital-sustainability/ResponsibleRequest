"""Evenly spaced rate limiting whose rate can change at runtime, plus a concurrency cap."""

from __future__ import annotations

import asyncio

from aiolimiter import AsyncLimiter


class _SpacedLimiter(AsyncLimiter):
    """An ``AsyncLimiter`` with capacity 1, i.e. one request every ``60 / rpm`` seconds."""

    def __init__(self, rpm: float) -> None:
        super().__init__(max_rate=1, time_period=60 / rpm)

    def set_rpm(self, rpm: float) -> None:
        if getattr(self, "_event_loop", None) is not None:
            self._leak()  # drain at the old rate up to now before switching
        self.time_period = 60 / rpm
        self._rate_per_sec = rpm / 60
        if self._waiters:
            self._wake_next()  # reschedule the pending wake-up for the new rate


class AdaptiveLimiter:
    """Paces acquisitions at ``rpm`` and caps the number of concurrent holders.

    Use as ``async with limiter: ...``. The concurrency slot is held for the duration of the
    block; the rate slot drains over time. The limiter re-creates its asyncio primitives if it
    is used from a new event loop (e.g. across ``asyncio.run`` calls).
    """

    def __init__(self, rpm: float, max_concurrency: int) -> None:
        self._rpm = rpm
        self.max_concurrency = max_concurrency
        self.in_flight = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._rate: _SpacedLimiter
        self._slots: asyncio.Semaphore

    @property
    def rpm(self) -> float:
        return self._rpm

    def set_rpm(self, rpm: float) -> None:
        self._rpm = rpm
        if self._loop is not None:
            self._rate.set_rpm(rpm)

    def _bind(self) -> None:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._rate = _SpacedLimiter(self._rpm)
            self._slots = asyncio.Semaphore(self.max_concurrency)
            self.in_flight = 0

    async def acquire(self) -> None:
        self._bind()
        # Concurrency first, then rate: requests leave the limiter evenly spaced.
        await self._slots.acquire()
        try:
            await self._rate.acquire()
        except BaseException:
            self._slots.release()
            raise
        self.in_flight += 1

    def release(self) -> None:
        self.in_flight -= 1
        self._slots.release()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()
