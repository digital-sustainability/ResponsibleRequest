"""Extract provider-specific response metadata into the generic record fields.

An extractor is called once per finished request with the record, the response headers and the
parsed response body (the JSON body, the merged summary of a stream, or None if the body was not
captured). It fills generic fields (``provider``, ``cost_usd``, ``gateway_request_id``,
``upstream_duration_ms``, ...) and puts anything else into ``record.provider_meta``.

All default extractors run on every response. Each one only fills fields when the headers or keys
it knows are present, so no provider has to be configured. Add your own with
``rr.AsyncOpenAI(extractors=[...])``::

    def my_gateway(record, headers, body):
        if "x-my-queue-ms" in headers:
            rr.providers.meta(record, "my_gateway")["queue_ms"] = float(headers["x-my-queue-ms"])
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from typing import Any

from .records import RequestRecord

Extractor = Callable[[RequestRecord, Mapping[str, str], "dict[str, Any] | None"], None]


def meta(record: RequestRecord, namespace: str) -> dict[str, Any]:
    """The ``provider_meta[namespace]`` dict of ``record``, created if missing."""
    if record.provider_meta is None:
        record.provider_meta = {}
    section: dict[str, Any] = record.provider_meta.setdefault(namespace, {})
    return section


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return None


def openai_compat(
    record: RequestRecord, headers: Mapping[str, str], body: dict[str, Any] | None
) -> None:
    """Request id headers of OpenAI and most OpenAI-compatible APIs."""
    for name in ("x-request-id", "request-id"):
        value = headers.get(name)
        if value and record.gateway_request_id is None:
            record.gateway_request_id = value
    duration = _float(headers.get("openai-processing-ms"))
    if duration is not None:
        record.upstream_duration_ms = duration


_LITELLM_PREFIX = "x-litellm-"


def litellm(record: RequestRecord, headers: Mapping[str, str], body: dict[str, Any] | None) -> None:
    """LiteLLM proxy headers: call id, duration, cost; all other ``x-litellm-*`` headers go to
    ``provider_meta["litellm"]``."""
    found = {
        k.lower()[len(_LITELLM_PREFIX) :]: v
        for k, v in headers.items()
        if k.lower().startswith(_LITELLM_PREFIX)
    }
    if not found:
        return
    if found.get("call-id"):
        record.gateway_request_id = found.pop("call-id")
    duration = _float(found.pop("response-duration-ms", None))
    if duration is not None:
        record.upstream_duration_ms = duration
    cost = _float(found.pop("response-cost", None))
    if cost is not None:
        record.cost_usd = cost
    if found:
        meta(record, "litellm").update({k.replace("-", "_"): v for k, v in found.items()})


def openrouter(
    record: RequestRecord, headers: Mapping[str, str], body: dict[str, Any] | None
) -> None:
    """OpenRouter and similar routers: upstream ``provider`` and ``usage.cost`` in the body."""
    if not isinstance(body, dict):
        return
    if isinstance(body.get("provider"), str):
        record.provider = body["provider"]
    usage = body.get("usage")
    if isinstance(usage, dict):
        cost = _float(usage.get("cost"))
        if cost is not None:
            record.cost_usd = cost
        extra = {k: usage[k] for k in ("is_byok", "cost_details") if usage.get(k) is not None}
        if extra:
            meta(record, "openrouter").update(extra)
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        native = choices[0].get("native_finish_reason")
        if native is not None:
            meta(record, "openrouter")["native_finish_reason"] = native


DEFAULT_EXTRACTORS: tuple[Extractor, ...] = (openai_compat, litellm, openrouter)
