from __future__ import annotations

import pytest

from responsible_request import LatencyEstimator, RequestRecord, ThrottleConfig
from responsible_request.estimator import metric_value


def rec(latency: float, tokens: int = 100, stream: bool = False, ttfb: float | None = None):
    return RequestRecord(
        request_id="x",
        timestamp=None,
        method="POST",
        path="/chat/completions",
        latency_s=latency,
        completion_tokens=tokens,
        stream=stream,
        ttfb_s=ttfb,
    )


def test_metric_variants():
    cfg = ThrottleConfig()
    assert metric_value(rec(2.0, 100), cfg) == pytest.approx(0.02)
    assert metric_value(rec(2.0, 1), cfg) == pytest.approx(2.0 / 16)  # min_tokens floor
    assert metric_value(rec(2.0, stream=True, ttfb=0.3), cfg) == 0.3
    assert metric_value(rec(2.0), ThrottleConfig(metric="latency")) == 2.0
    assert metric_value(rec(2.0), ThrottleConfig(metric=lambda r: 42.0)) == 42.0


def test_baseline_needs_warmup_then_ratio(clock):
    est = LatencyEstimator(ThrottleConfig(warmup_requests=5, window=3), clock=clock)
    for _ in range(4):
        est.observe(rec(1.0), throttled=False)
    assert est.baseline is None and est.load_ratio() is None
    est.observe(rec(1.0), throttled=False)
    assert est.load_ratio() == pytest.approx(1.0)
    for _ in range(3):
        est.observe(rec(4.0), throttled=False)
    assert est.load_ratio() == pytest.approx(4.0)


def test_throttled_samples_do_not_move_baseline(clock):
    est = LatencyEstimator(ThrottleConfig(warmup_requests=5, window=3), clock=clock)
    for _ in range(5):
        est.observe(rec(1.0), throttled=False)
    base = est.baseline
    for _ in range(100):
        clock.advance(60)
        est.observe(rec(5.0), throttled=True)
    assert est.baseline == base
    assert est.load_ratio() == pytest.approx(5.0)


def test_pinned_and_reset(clock):
    est = LatencyEstimator(ThrottleConfig(baseline=0.01, window=1), clock=clock)
    est.observe(rec(3.0), throttled=False)  # 0.03 per token
    assert est.load_ratio() == pytest.approx(3.0)
    est.pin_baseline(0.03)
    assert est.load_ratio() == pytest.approx(1.0)
    est.reset()
    assert est.baseline == 0.01 and est.load_ratio() is None
