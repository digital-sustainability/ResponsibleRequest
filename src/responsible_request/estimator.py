"""Load estimation: turn observed request latencies into a load ratio (current / baseline)."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .config import ThrottleConfig
from .records import RequestRecord


@runtime_checkable
class LoadEstimator(Protocol):
    """Anything that can judge the endpoint's load from completed requests.

    Implement this to use a different signal (e.g. queue depth from a Prometheus endpoint) and
    pass it via ``ThrottleConfig(estimator_factory=...)``.
    """

    def observe(self, record: RequestRecord, *, throttled: bool) -> None:
        """Feed one successful, eligible request. ``throttled`` is True while at min RPM."""

    def load_ratio(self) -> float | None:
        """Current load relative to an idle endpoint (1.0 = idle); None while unknown."""

    @property
    def baseline(self) -> float | None: ...

    def reset(self) -> None:
        """Forget the learned baseline."""


def metric_value(record: RequestRecord, config: ThrottleConfig) -> float | None:
    """Compute the latency signal of one request according to ``config.metric``."""
    metric = config.metric
    if callable(metric):
        return metric(record)
    if metric == "latency":
        return record.latency_s
    if metric == "ttfb":
        return record.ttfb_s
    # latency_per_token: streamed requests expose the queueing delay directly as TTFB.
    if record.stream and record.ttfb_s is not None:
        return record.ttfb_s
    if record.latency_s is None:
        return None
    tokens = record.completion_tokens or 0
    return record.latency_s / max(tokens, config.min_tokens)


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class LatencyEstimator:
    """Compare the median of the last ``window`` observations to a low-percentile baseline.

    The baseline is the ``baseline_percentile``-th percentile of observations made within
    ``baseline_window_s``. Observations made while throttled are not added to the baseline, so a
    long busy period cannot turn into the new "normal". The window is measured relative to the
    newest baseline sample, which means the baseline freezes (rather than expires) meanwhile.
    """

    def __init__(self, config: ThrottleConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._clock = clock
        self._recent: deque[float] = deque(maxlen=config.window)
        self._samples: deque[tuple[float, float]] = deque(maxlen=5000)
        self._pinned: float | None = config.baseline

    def observe(self, record: RequestRecord, *, throttled: bool) -> None:
        value = metric_value(record, self.config)
        if value is None or value <= 0:
            return
        self._recent.append(value)
        if not throttled:
            now = self._clock()
            self._samples.append((now, value))
            while self._samples and self._samples[0][0] < now - self.config.baseline_window_s:
                self._samples.popleft()

    @property
    def baseline(self) -> float | None:
        if self._pinned is not None:
            return self._pinned
        if len(self._samples) < self.config.warmup_requests:
            return None
        return _percentile([v for _, v in self._samples], self.config.baseline_percentile)

    def current(self) -> float | None:
        return statistics.median(self._recent) if self._recent else None

    def load_ratio(self) -> float | None:
        baseline, current = self.baseline, self.current()
        if baseline is None or current is None or baseline <= 0:
            return None
        return current / baseline

    def pin_baseline(self, value: float | None) -> None:
        """Use ``value`` as the baseline (None returns to estimating it)."""
        self._pinned = value

    def reset(self) -> None:
        self._pinned = self.config.baseline
        self._samples.clear()
        self._recent.clear()
