"""Evenly spaced rate limiting whose rate can change at runtime, plus a concurrency cap.

The limiter is safe to share between threads that each run their own event loop (e.g. several
threads calling ``asyncio.run``): its state is guarded by a ``threading.Lock``, and every waiter
sleeps in its own loop and is woken through ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque


class _Waiter:
    """A future in the caller's event loop that another thread can resolve."""

    __slots__ = ("future", "loop")

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.future: asyncio.Future[None] = self.loop.create_future()

    def renew(self) -> None:
        self.future = self.loop.create_future()

    def wake(self) -> bool:
        """Resolve the current future. Returns False if the waiter's loop is closed."""
        try:
            self.loop.call_soon_threadsafe(_resolve, self.future)
        except RuntimeError:
            return False
        return True


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


class AdaptiveLimiter:
    """Paces acquisitions at ``rpm`` and caps the number of concurrent holders.

    Use as ``async with limiter: ...``. The concurrency slot is held for the duration of the
    block. Requests leave the limiter at least ``60 / rpm`` seconds apart (no bursts), in the
    order in which they got a concurrency slot. ``set_rpm`` applies to requests that are already
    waiting, and ``pause`` holds back all requests for a while.
    """

    def __init__(self, rpm: float, max_concurrency: int) -> None:
        self._lock = threading.Lock()
        self._rpm = rpm
        self._interval = 60 / rpm
        self.max_concurrency = max_concurrency
        self.in_flight = 0
        self._slots_held = 0
        self._last_sent = -float("inf")
        self._paused_until = -float("inf")
        self._slot_waiters: deque[_Waiter] = deque()
        self._rate_waiters: deque[_Waiter] = deque()

    @property
    def rpm(self) -> float:
        return self._rpm

    @property
    def paused_s(self) -> float:
        """Remaining time of the current pause (0 if not paused)."""
        return max(0.0, self._paused_until - time.monotonic())

    def set_rpm(self, rpm: float) -> None:
        with self._lock:
            self._rpm = rpm
            self._interval = 60 / rpm
            self._wake_rate_head()  # it re-computes its wake-up time for the new rate

    def pause(self, seconds: float) -> None:
        """Send nothing for ``seconds`` (extends, never shortens, a running pause)."""
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)

    async def acquire(self) -> None:
        # Concurrency first, then rate: requests leave the limiter evenly spaced.
        await self._acquire_slot()
        try:
            await self._acquire_rate()
        except BaseException:
            self._release_slot()
            raise

    def release(self) -> None:
        with self._lock:
            self.in_flight -= 1
        self._release_slot()

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    # ------------------------------------------------------------------ internals

    async def _acquire_slot(self) -> None:
        with self._lock:
            if self._slots_held < self.max_concurrency and not self._slot_waiters:
                self._slots_held += 1
                return
            waiter = _Waiter()
            self._slot_waiters.append(waiter)
        try:
            await waiter.future  # resolved by _release_slot, which hands over its slot
        except BaseException:
            with self._lock:
                granted = waiter not in self._slot_waiters
                if not granted:
                    self._slot_waiters.remove(waiter)
            if granted:  # the slot was already handed to us: pass it on
                self._release_slot()
            raise

    def _release_slot(self) -> None:
        with self._lock:
            while self._slot_waiters:
                if self._slot_waiters.popleft().wake():
                    return  # the slot moves to the woken waiter
            self._slots_held -= 1

    async def _acquire_rate(self) -> None:
        waiter = _Waiter()
        with self._lock:
            self._rate_waiters.append(waiter)
        try:
            while True:
                with self._lock:
                    waiter.renew()  # wake-ups from now on resolve the new future
                    timeout: float | None = None
                    if self._rate_waiters[0] is waiter:
                        now = time.monotonic()
                        ready = max(self._last_sent + self._interval, self._paused_until)
                        if now >= ready:
                            self._last_sent = now
                            self._rate_waiters.popleft()
                            self._wake_rate_head()
                            self.in_flight += 1
                            return
                        timeout = ready - now
                # Sleep in this thread's own loop until the slot is due or we are woken.
                await asyncio.wait({waiter.future}, timeout=timeout)
        except BaseException:
            with self._lock:
                if waiter in self._rate_waiters:
                    was_head = self._rate_waiters[0] is waiter
                    self._rate_waiters.remove(waiter)
                    if was_head:
                        self._wake_rate_head()
            raise

    def _wake_rate_head(self) -> None:
        """Wake the first waiter for the rate slot (call with the lock held)."""
        while self._rate_waiters:
            if self._rate_waiters[0].wake():
                return
            self._rate_waiters.popleft()  # its loop is gone
