"""The per-request record and helpers to fill it from HTTP requests and responses."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .config import LogConfig

_tags: ContextVar[dict[str, Any] | None] = ContextVar("rr_tags", default=None)


@contextmanager
def tags(**values: Any) -> Iterator[None]:
    """Attach ``values`` to the ``tags`` field of every request record created inside the block.

    Example::

        with rr.tags(experiment="baseline", item=17):
            await client.chat.completions.create(...)
    """
    token = _tags.set({**(_tags.get() or {}), **values})
    try:
        yield
    finally:
        _tags.reset(token)


def current_tags() -> dict[str, Any]:
    return dict(_tags.get() or {})


# Which record fields belong to which toggleable group. Fields not listed are core fields.
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "timing": ("sent_at", "first_byte_at", "finished_at", "wait_s", "ttfb_s", "latency_s"),
    "usage": (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cost_usd",
    ),
    "throttle": ("rpm", "state", "baseline", "load_ratio", "in_flight"),
    "response_meta": (
        "response_id",
        "response_model",
        "finish_reason",
        "system_fingerprint",
        "provider",
        "gateway_request_id",
        "upstream_duration_ms",
        "provider_meta",
    ),
    "params": ("params",),
    "request_body": ("request_body",),
    "response_body": ("response_body",),
}

JSON_FIELDS = ("params", "request_body", "response_body", "tags", "provider_meta")


def utc_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class RequestRecord:
    """Everything recorded about one HTTP request (one attempt, SDK retries are separate)."""

    request_id: str
    timestamp: str | None  # when the request was issued (queued), UTC ISO 8601
    method: str
    path: str
    model: str | None = None
    stream: bool = False
    attempt: int = 0
    status_code: int | None = None
    error: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None  # hash of URL, JSON body and key tags (see CacheConfig)
    cache_hit: bool = False  # served from the cache, not sent to the server
    run_id: str | None = None  # links the record to its run manifest (see RunConfig)

    # timing
    sent_at: str | None = None
    first_byte_at: str | None = None
    finished_at: str | None = None
    wait_s: float | None = None  # time spent waiting for the rate limiter
    ttfb_s: float | None = None  # time to first response byte (≈ TTFT when streaming)
    latency_s: float | None = None  # sent -> response fully received

    # usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None  # reported by the provider or computed from CostConfig.prices

    # throttle (state after this observation was processed)
    rpm: float | None = None  # rate at which this request was sent
    state: str | None = None
    baseline: float | None = None
    load_ratio: float | None = None
    in_flight: int | None = None

    # response metadata
    response_id: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    system_fingerprint: str | None = None
    provider: str | None = None  # upstream provider that served the request (e.g. via OpenRouter)
    gateway_request_id: str | None = None  # request/call id assigned by the gateway or API
    upstream_duration_ms: float | None = None  # processing time reported by the gateway
    provider_meta: dict[str, Any] | None = None  # anything else extractors found

    params: dict[str, Any] | None = None
    request_body: Any = None
    response_body: Any = None

    def to_dict(self, config: LogConfig | None = None) -> dict[str, Any]:
        data = asdict(self)
        if config is not None:
            for group, names in GROUP_FIELDS.items():
                if not config.enabled(group):
                    for name in names:
                        data.pop(name, None)
        return data


def record_columns() -> list[str]:
    return list(RequestRecord.__dataclass_fields__)


# --------------------------------------------------------------------------- request parsing

_BODY_KEYS = {"messages", "input", "prompt"}


@dataclass
class RequestInfo:
    """What we could learn from an outgoing request body."""

    model: str | None = None
    stream: bool = False
    uses_server_tools: bool = False
    body: Any = None  # parsed JSON body (None for non-JSON bodies)

    @property
    def params(self) -> dict[str, Any] | None:
        if not isinstance(self.body, dict):
            return None
        return {k: v for k, v in self.body.items() if k not in _BODY_KEYS}


def parse_request_body(content: bytes, content_type: str | None) -> RequestInfo:
    if not content or "json" not in (content_type or ""):
        return RequestInfo()
    try:
        body = json.loads(content)
    except ValueError:
        return RequestInfo()
    if not isinstance(body, dict):
        return RequestInfo(body=body)
    tools = body.get("tools") or []
    server_tools = any(
        isinstance(t, dict) and t.get("type") not in (None, "function", "custom") for t in tools
    )
    return RequestInfo(
        model=body.get("model") if isinstance(body.get("model"), str) else None,
        stream=bool(body.get("stream")),
        uses_server_tools=server_tools,
        body=body,
    )


# --------------------------------------------------------------------------- response parsing


def apply_usage(record: RequestRecord, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    record.prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    record.completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    record.total_tokens = usage.get("total_tokens")
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    completion_details = (
        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    )
    if isinstance(prompt_details, dict):
        record.cached_tokens = prompt_details.get("cached_tokens")
    if isinstance(completion_details, dict):
        record.reasoning_tokens = completion_details.get("reasoning_tokens")


def apply_json_response(record: RequestRecord, body: Any) -> None:
    if not isinstance(body, dict):
        return
    record.response_id = body.get("id") if isinstance(body.get("id"), str) else None
    record.response_model = body.get("model") if isinstance(body.get("model"), str) else None
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        record.finish_reason = choices[0].get("finish_reason")
    if isinstance(body.get("system_fingerprint"), str):
        record.system_fingerprint = body["system_fingerprint"]
    apply_usage(record, body.get("usage"))


def parse_sse(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse an OpenAI-style SSE stream into its JSON events and a merged response summary."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)

    summary: dict[str, Any] = {}
    content: list[str] = []
    reasoning: list[str] = []
    for event in events:
        for key in ("id", "model", "provider", "system_fingerprint"):
            if isinstance(event.get(key), str):
                summary[key] = event[key]
        if event.get("usage"):
            summary["usage"] = event["usage"]
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if choice.get("finish_reason"):
                summary["finish_reason"] = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                for key in ("reasoning_content", "reasoning"):
                    if isinstance(delta.get(key), str):
                        reasoning.append(delta[key])
    summary["content"] = "".join(content)
    if reasoning:
        summary["reasoning"] = "".join(reasoning)
    return events, summary


def apply_sse_response(record: RequestRecord, summary: dict[str, Any]) -> None:
    record.response_id = summary.get("id")
    record.response_model = summary.get("model")
    record.finish_reason = summary.get("finish_reason")
    record.system_fingerprint = summary.get("system_fingerprint")
    apply_usage(record, summary.get("usage"))
