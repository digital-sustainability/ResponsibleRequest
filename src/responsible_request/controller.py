"""Decide the target request rate from the load signal."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .config import ThrottleConfig


class State(str, Enum):
    WARMUP = "warmup"  # establishing the baseline at start_rpm
    NORMAL = "normal"  # endpoint looks idle: ramp up to max_rpm
    THROTTLED = "throttled"  # other users detected: stay at min_rpm for at least the cooldown
    RECOVERING = "recovering"  # load cleared: ramping back up
    FIXED = "fixed"  # adaptation disabled


@dataclass(frozen=True)
class Transition:
    old_state: State
    new_state: State
    old_rpm: float
    new_rpm: float
    reason: str


class ThrottleController:
    """State machine with hysteresis mapping load ratios to a requests-per-minute target.

    * ratio >= ``high_ratio`` or a congestion error: drop to ``min_rpm`` immediately.
    * ``recover_ratio`` <= ratio < ``high_ratio``: hold the current rate (dead band).
    * ratio < ``recover_ratio``: multiply the rate by ``ramp_factor`` every ``ramp_interval_s``
      up to ``max_rpm`` (after the cooldown, if throttled).
    """

    def __init__(self, config: ThrottleConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._clock = clock
        self.state = State.WARMUP if config.adaptive else State.FIXED
        self.rpm = config.initial_rpm
        self._throttled_at = -float("inf")
        self._last_ramp = clock()

    def update(self, ratio: float | None, *, congested: bool = False) -> Transition | None:
        """Process one observation. Returns the transition if the state or the rate changed."""
        if self.state is State.FIXED:
            return None
        cfg, now = self.config, self._clock()
        old_state, old_rpm = self.state, self.rpm
        reason = ""

        if congested or (ratio is not None and ratio >= cfg.high_ratio):
            self._throttled_at = now
            self.state, self.rpm = State.THROTTLED, cfg.min_rpm
            reason = "server errors" if congested else f"load {ratio:.1f}x baseline"
        elif self.state is State.WARMUP:
            if ratio is not None:
                self.state, self._last_ramp = State.NORMAL, now
                reason = "baseline established"
        elif self.state is State.THROTTLED:
            cooled_down = now - self._throttled_at >= cfg.cooldown_s
            if cooled_down and ratio is not None and ratio < cfg.recover_ratio:
                self.state, self._last_ramp = State.RECOVERING, now
                reason = f"load {ratio:.1f}x baseline, recovering"
        elif ratio is not None and ratio < cfg.recover_ratio:
            if now - self._last_ramp >= cfg.ramp_interval_s and self.rpm < cfg.max_rpm:
                self.rpm = min(cfg.max_rpm, self.rpm * cfg.ramp_factor)
                self._last_ramp = now
                reason = f"load {ratio:.1f}x baseline, ramping up"
            if self.state is State.RECOVERING and self.rpm >= cfg.max_rpm:
                self.state = State.NORMAL
                reason = reason or "fully recovered"

        if self.state is old_state and self.rpm == old_rpm:
            return None
        return Transition(old_state, self.state, old_rpm, self.rpm, reason)
