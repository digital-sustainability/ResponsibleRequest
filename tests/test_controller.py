from __future__ import annotations

import pytest

from responsible_request import State, ThrottleConfig, ThrottleController

CFG = ThrottleConfig(
    max_rpm=100, min_rpm=5, start_rpm=20, cooldown_s=60, ramp_interval_s=10, ramp_factor=2
)


def test_warmup_until_ratio_known(clock):
    c = ThrottleController(CFG, clock=clock)
    assert (c.state, c.rpm) == (State.WARMUP, 20)
    assert c.update(None) is None
    t = c.update(1.0)
    assert t is not None and c.state is State.NORMAL and c.rpm == 20


def test_ramps_up_to_max(clock):
    c = ThrottleController(CFG, clock=clock)
    c.update(1.0)
    c.update(1.0)
    assert c.rpm == 20  # ramp interval not reached
    for expected in (40, 80, 100, 100):
        clock.advance(10)
        c.update(1.0)
        assert c.rpm == expected


def test_throttles_on_high_ratio_and_holds_in_dead_band(clock):
    c = ThrottleController(CFG, clock=clock)
    c.update(1.0)
    t = c.update(3.2)
    assert t is not None and c.state is State.THROTTLED and c.rpm == 5
    clock.advance(30)
    c.update(1.0)
    assert c.state is State.THROTTLED  # cooldown
    clock.advance(40)
    c.update(2.0)
    assert c.state is State.THROTTLED  # dead band: no recovery
    c.update(1.2)
    assert c.state is State.RECOVERING and c.rpm == 5
    for _ in range(5):
        clock.advance(10)
        c.update(1.2)
    assert c.state is State.NORMAL and c.rpm == 100


def test_dead_band_holds_rate(clock):
    c = ThrottleController(CFG, clock=clock)
    c.update(1.0)
    clock.advance(10)
    c.update(1.0)
    assert c.rpm == 40
    clock.advance(10)
    assert c.update(2.0) is None
    assert c.rpm == 40


def test_high_load_extends_cooldown(clock):
    c = ThrottleController(CFG, clock=clock)
    c.update(1.0)
    c.update(5.0)
    clock.advance(50)
    c.update(4.0)  # still loaded, cooldown restarts
    clock.advance(20)
    c.update(1.0)
    assert c.state is State.THROTTLED


def test_congestion_throttles(clock):
    c = ThrottleController(CFG, clock=clock)
    c.update(None, congested=True)
    assert c.state is State.THROTTLED and c.rpm == 5


def test_fixed_mode(clock):
    c = ThrottleController(ThrottleConfig.fixed(60), clock=clock)
    assert c.update(10.0, congested=True) is None
    assert (c.state, c.rpm) == (State.FIXED, 60)


def test_config_validation():
    with pytest.raises(ValueError):
        ThrottleConfig(min_rpm=10, max_rpm=5)
    with pytest.raises(ValueError):
        ThrottleConfig(recover_ratio=3, high_ratio=2)
