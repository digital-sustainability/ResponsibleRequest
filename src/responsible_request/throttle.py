"""Per-model throttling state: limiter + load estimator + controller."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .config import ThrottleConfig
from .controller import State, ThrottleController, Transition
from .estimator import LatencyEstimator, LoadEstimator
from .limiter import AdaptiveLimiter
from .records import RequestRecord


@dataclass
class LaneStats:
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, record: RequestRecord) -> None:
        self.requests += 1
        if record.error or (record.status_code or 0) >= 400:
            self.errors += 1
        self.prompt_tokens += record.prompt_tokens or 0
        self.completion_tokens += record.completion_tokens or 0


class Lane:
    """Throttling state for one (endpoint, model) pair.

    The gateway routes every model to its own backend, so load is tracked separately per model.
    Lanes for endpoints that are not in ``observe_paths`` are paced and react to server errors,
    but they do not adapt to latency.
    """

    def __init__(
        self,
        name: str,
        config: ThrottleConfig,
        *,
        observed: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.config = config
        self.observed = observed
        self.limiter = AdaptiveLimiter(config.initial_rpm, config.max_concurrency)
        self.estimator: LoadEstimator = (
            config.estimator_factory(config)
            if config.estimator_factory is not None
            else LatencyEstimator(config, clock=clock)
        )
        self.controller = ThrottleController(config, clock=clock)
        self.totals = LaneStats()
        self.since_summary = LaneStats()
        self.last_record: RequestRecord | None = None

    @property
    def rpm(self) -> float:
        return self.controller.rpm

    @property
    def state(self) -> State:
        return self.controller.state

    def observe(self, record: RequestRecord, *, eligible: bool) -> Transition | None:
        """Update load estimate and rate after a request finished."""
        self.last_record = record
        self.totals.add(record)
        self.since_summary.add(record)
        congested = is_congestion(record)
        if not self.observed:
            ratio: float | None = 1.0  # non-latency lanes only react to server errors
        else:
            if eligible and not congested and record.error is None:
                self.estimator.observe(record, throttled=self.state is State.THROTTLED)
            ratio = self.estimator.load_ratio()
        transition = self.controller.update(ratio, congested=congested)
        if transition is not None:
            self.limiter.set_rpm(self.controller.rpm)
            _log_transition(self.name, transition)
        return transition

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "rpm": round(self.rpm, 2),
            "baseline": self.estimator.baseline,
            "load_ratio": self.estimator.load_ratio() if self.observed else None,
            "in_flight": self.limiter.in_flight,
            "requests": self.totals.requests,
            "errors": self.totals.errors,
            "prompt_tokens": self.totals.prompt_tokens,
            "completion_tokens": self.totals.completion_tokens,
        }


def is_congestion(record: RequestRecord) -> bool:
    """429, 5xx and timeouts mean the server is struggling."""
    status = record.status_code
    if status is not None and (status == 429 or status >= 500):
        return True
    return record.error is not None and "Timeout" in record.error


def _log_transition(name: str, t: Transition) -> None:
    if t.new_state is State.THROTTLED and t.old_state is not State.THROTTLED:
        logger.warning(
            "{}: {} -> throttling {:.0f} -> {:.0f} RPM", name, t.reason, t.old_rpm, t.new_rpm
        )
    elif t.old_rpm != t.new_rpm:
        logger.info(
            "{}: {} ({:.0f} -> {:.0f} RPM, {})",
            name,
            t.reason,
            t.old_rpm,
            t.new_rpm,
            t.new_state.value,
        )
    else:
        logger.info("{}: {} ({})", name, t.reason, t.new_state.value)


class Throttle:
    """All lanes of one client. Available as ``client.throttle`` on :class:`rr.AsyncOpenAI`."""

    def __init__(
        self, config: ThrottleConfig | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config or ThrottleConfig()
        self._clock = clock
        self._lanes: dict[tuple[str, str | None], Lane] = {}

    def lane(self, path: str, model: str | None) -> Lane:
        key = (path, model)
        lane = self._lanes.get(key)
        if lane is None:
            observed = self.config.adaptive and any(
                path.endswith(p) for p in self.config.observe_paths
            )
            name = model or path
            lane = Lane(name, self.config, observed=observed, clock=self._clock)
            self._lanes[key] = lane
        return lane

    def lanes(self) -> list[Lane]:
        return list(self._lanes.values())

    def find(self, model: str) -> list[Lane]:
        return [lane for (_, m), lane in self._lanes.items() if m == model]

    def stats(self) -> dict[str, dict[str, Any]]:
        """Current state, rate, load and totals per lane (keyed by model, or path)."""
        out: dict[str, dict[str, Any]] = {}
        for (path, model), lane in self._lanes.items():
            key = model if model is not None else path
            if key in out:
                key = f"{model} {path}"
            out[key] = lane.snapshot()
        return out

    def reset_baseline(self, model: str | None = None) -> None:
        """Forget learned baselines (of one model, or all)."""
        for lane in self._lanes.values():
            if model is None or lane.name == model:
                lane.estimator.reset()
