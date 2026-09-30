"""Sampling-parameter presets."""

from __future__ import annotations

from typing import Any


def reproducible(seed: int = 0, **overrides: Any) -> dict[str, Any]:
    """Parameters that make chat completions as repeatable as the backend allows.

    Greedy decoding (``temperature=0``, ``top_p=1``) plus a fixed ``seed``. Spread it into a call
    (``create(..., **rr.reproducible())``) or set it for every request with
    ``rr.AsyncOpenAI(default_params=rr.reproducible())``.

    Note: outputs can still differ between runs, because batched GPU inference is not bit-exact
    (the batch composition depends on the other requests the server is processing).
    """
    return {"temperature": 0.0, "top_p": 1.0, "seed": seed, **overrides}
