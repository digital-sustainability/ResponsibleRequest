"""Running many requests and calibrating the latency baseline."""

from __future__ import annotations

import asyncio
import statistics
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

import openai

from ..client import get_throttle
from ..config import ThrottleConfig
from ..estimator import LatencyEstimator, metric_value
from ..records import tags

Call = Callable[..., Awaitable[Any]]


async def run_batch(
    client: openai.AsyncOpenAI,
    requests: Iterable[dict[str, Any]],
    *,
    call: Call | None = None,
    progress: bool = False,
    return_exceptions: bool = True,
    batch_id: str | None = None,
) -> list[Any]:
    """Run many requests concurrently and return their results in input order.

    The throttle paces the requests, so all of them can be submitted at once. Each request
    record is tagged with ``batch_id`` and ``item_index`` for joining logs with results.

    :param requests: Keyword arguments for ``call``, one dict per request.
    :param call: Coroutine function to call, defaults to ``client.chat.completions.create``.
    :param progress: Show a tqdm progress bar (requires the ``progress`` extra).
    :param return_exceptions: Put exceptions into the result list instead of raising the first.
    :param batch_id: Tag value, a random id by default.
    """
    call = call or client.chat.completions.create
    items: Sequence[dict[str, Any]] = list(requests)
    batch_id = batch_id or uuid.uuid4().hex[:12]
    bar = _progress_bar(len(items)) if progress else None

    async def one(index: int, kwargs: dict[str, Any]) -> Any:
        try:
            with tags(batch_id=batch_id, item_index=index):
                return await call(**kwargs)
        finally:
            if bar is not None:
                bar.update(1)

    try:
        return await asyncio.gather(
            *(one(i, kw) for i, kw in enumerate(items)), return_exceptions=return_exceptions
        )
    finally:
        if bar is not None:
            bar.close()


def run_batch_sync(
    client: openai.AsyncOpenAI, requests: Iterable[dict[str, Any]], **kwargs: Any
) -> list[Any]:
    """Blocking version of :func:`run_batch` for scripts (runs its own event loop)."""

    async def main() -> list[Any]:
        try:
            return await run_batch(client, requests, **kwargs)
        finally:
            await client.close()

    return asyncio.run(main())


def _progress_bar(total: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "progress=True requires tqdm: pip install responsible-request[progress]"
        ) from exc
    return tqdm(total=total)


async def calibrate(
    client: openai.AsyncOpenAI,
    model: str,
    *,
    n: int = 5,
    messages: list[Any] | None = None,
    **kwargs: Any,
) -> float:
    """Measure the latency baseline of ``model`` with ``n`` sequential requests and pin it.

    Run this while the endpoint is idle (e.g. at night) or use a representative request of your
    experiment via ``messages``/``kwargs`` so that the baseline matches your workload. Returns
    the pinned baseline (in units of the configured metric).
    """
    throttle = get_throttle(client)
    config: ThrottleConfig = throttle.config
    messages = messages or [{"role": "user", "content": "Reply with the single word: OK"}]
    values: list[float] = []
    for _ in range(n):
        with tags(calibration=True):
            await client.chat.completions.create(model=model, messages=messages, **kwargs)
        # Use the transport's own measurement, which excludes time spent in the rate limiter.
        records = [lane.last_record for lane in throttle.find(model) if lane.last_record]
        value = metric_value(records[-1], config) if records else None
        if value is not None:
            values.append(value)
    if not values:
        raise RuntimeError("calibration produced no usable measurements")
    baseline = statistics.median(values)
    for lane in throttle.find(model):
        if isinstance(lane.estimator, LatencyEstimator):
            lane.estimator.pin_baseline(baseline)
    return baseline
