"""An HTTP transport that paces, measures and records every request of an OpenAI client."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from loguru import logger

from ._http import httpx
from .config import LogConfig
from .logging import emit_record
from .records import (
    RequestInfo,
    RequestRecord,
    apply_json_response,
    apply_litellm_headers,
    apply_sse_response,
    current_tags,
    parse_request_body,
    parse_sse,
    utc_iso,
)
from .throttle import Lane, Throttle

MAX_CAPTURE_BYTES = 32 * 1024 * 1024


class ThrottledTransport(httpx.AsyncBaseTransport):  # type: ignore[misc, name-defined]
    """Wraps another async transport: waits for the rate limiter, forwards the request, measures
    timing and token usage from the response, updates the throttle and emits a request record.

    Pass it to the OpenAI SDK through ``http_client``; see :func:`responsible_request.http_client`.
    """

    def __init__(
        self,
        throttle: Throttle,
        *,
        log: LogConfig | None = None,
        default_params: Mapping[str, Any] | None = None,
        inject_stream_usage: bool = True,
        transport: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.throttle = throttle
        self.log = log
        self.default_params = dict(default_params or {})
        self.inject_stream_usage = inject_stream_usage
        self._inner = transport if transport is not None else httpx.AsyncHTTPTransport()
        self._clock = clock
        self._last_summary = clock()

    async def handle_async_request(self, request: Any) -> Any:
        request, info = self._prepare(request)
        path = request.url.path
        lane = self.throttle.lane(path, info.model)
        eligible = not (info.uses_server_tools and self.throttle.config.ignore_server_tools)

        record = RequestRecord(
            request_id=uuid.uuid4().hex,
            timestamp=utc_iso(time.time()),
            method=request.method,
            path=path,
            model=info.model,
            stream=info.stream,
            attempt=_int(request.headers.get("x-stainless-retry-count")) or 0,
            tags=current_tags(),
            params=info.params,
            request_body=info.body,
        )
        queued = self._clock()
        await lane.limiter.acquire()
        sent = self._clock()
        record.sent_at = utc_iso(time.time())
        record.wait_s = sent - queued
        record.rpm = lane.rpm
        record.in_flight = lane.limiter.in_flight

        try:
            response = await self._inner.handle_async_request(request)
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            self._finish(lane, record, sent, eligible, None)
            raise

        record.status_code = response.status_code
        apply_litellm_headers(record, response.headers)
        observed = _ObservedStream(
            response.stream,
            capture=_capturable(response.headers.get("content-type")),
            on_first_byte=lambda: self._first_byte(record, sent),
            on_done=lambda body, error: self._finish(
                lane, record, sent, eligible, (response.headers, body), error
            ),
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=observed,
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()

    # ------------------------------------------------------------------ helpers

    def _prepare(self, request: Any) -> tuple[Any, RequestInfo]:
        """Parse the JSON body and, if needed, rewrite it with default parameters."""
        content_type = request.headers.get("content-type") or ""
        if "json" not in content_type:  # e.g. streamed multipart uploads: leave untouched
            return request, RequestInfo()
        info = parse_request_body(request.content, content_type)
        body = info.body
        if not isinstance(body, dict):
            return request, info
        changed = False
        if self.default_params and request.url.path.endswith("/chat/completions"):
            for key, value in self.default_params.items():
                if key not in body:
                    body[key] = value
                    changed = True
        if (
            self.inject_stream_usage
            and info.stream
            and request.url.path.endswith("/chat/completions")
            and "stream_options" not in body
        ):
            body["stream_options"] = {"include_usage": True}
            changed = True
        if not changed:
            return request, info
        headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
        new_request = httpx.Request(
            request.method,
            request.url,
            headers=headers,
            content=json.dumps(body).encode(),
            extensions=request.extensions,
        )
        return new_request, info

    def _first_byte(self, record: RequestRecord, sent: float) -> None:
        record.first_byte_at = utc_iso(time.time())
        record.ttfb_s = self._clock() - sent

    def _finish(
        self,
        lane: Lane,
        record: RequestRecord,
        sent: float,
        eligible: bool,
        response: tuple[Any, bytes | None] | None,
        error: BaseException | None = None,
    ) -> None:
        lane.limiter.release()
        record.latency_s = self._clock() - sent
        record.finished_at = utc_iso(time.time())
        if error is not None and record.error is None:
            record.error = f"{type(error).__name__}: {error}"
        if response is not None:
            headers, raw = response
            if raw is not None:
                _apply_body(record, headers, raw)

        ok_status = record.status_code is not None and record.status_code < 400
        lane.observe(record, eligible=eligible and ok_status)
        record.state = lane.state.value
        record.baseline = lane.estimator.baseline
        record.load_ratio = lane.estimator.load_ratio() if lane.observed else None

        if record.status_code is not None and record.status_code >= 400:
            logger.debug("{}: HTTP {} on {}", lane.name, record.status_code, record.path)
        emit_record(record.to_dict(self.log))
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        interval = self.log.summary_interval_s if self.log is not None else None
        now = self._clock()
        if interval is None or now - self._last_summary < interval:
            return
        self._last_summary = now
        for lane in self.throttle.lanes():
            s = lane.since_summary
            if s.requests == 0:
                continue
            ratio = lane.estimator.load_ratio() if lane.observed else None
            logger.info(
                "{}: {} req ({} err), {} in / {} out tokens, {:.0f} RPM, {}{}",
                lane.name,
                s.requests,
                s.errors,
                s.prompt_tokens,
                s.completion_tokens,
                lane.rpm,
                lane.state.value,
                f", load {ratio:.2f}x" if ratio is not None else "",
            )
            lane.since_summary = type(s)()


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _capturable(content_type: str | None) -> bool:
    ct = content_type or ""
    return "json" in ct or "event-stream" in ct or ct.startswith("text/")


def _apply_body(record: RequestRecord, headers: Any, raw: bytes) -> None:
    """Decode the (possibly compressed) body and extract usage and metadata."""
    try:
        content = httpx.Response(200, headers=headers, content=raw).content
    except Exception:  # undecodable: keep the record without body details
        return
    content_type = headers.get("content-type") or ""
    text = content.decode("utf-8", errors="replace")
    if "event-stream" in content_type:
        _, summary = parse_sse(text)
        apply_sse_response(record, summary)
        record.response_body = summary
        return
    try:
        body = json.loads(text)
    except ValueError:
        record.response_body = text
        return
    apply_json_response(record, body)
    record.response_body = body


class StreamClosedEarly(Exception):
    pass


class _ObservedStream(httpx.AsyncByteStream):  # type: ignore[misc, name-defined]
    """Pass-through response stream that reports the first byte and the complete body."""

    def __init__(
        self,
        inner: Any,
        *,
        capture: bool,
        on_first_byte: Callable[[], None],
        on_done: Callable[[bytes | None, BaseException | None], None],
    ) -> None:
        self._inner = inner
        self._capture = capture
        self._chunks: list[bytes] = []
        self._size = 0
        self._on_first_byte = on_first_byte
        self._on_done = on_done
        self._started = False
        self._done = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                if not self._started and chunk:
                    self._started = True
                    self._on_first_byte()
                if self._capture:
                    self._size += len(chunk)
                    if self._size <= MAX_CAPTURE_BYTES:
                        self._chunks.append(chunk)
                    else:
                        self._capture, self._chunks = False, []
                yield chunk
        except BaseException as exc:
            self._complete(exc)
            raise
        self._complete(None)

    def _complete(self, error: BaseException | None) -> None:
        if self._done:
            return
        self._done = True
        body = b"".join(self._chunks) if self._capture else None
        self._chunks = []
        self._on_done(body, error)

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            if not self._done:  # closed before being fully read (e.g. abandoned stream)
                self._complete(StreamClosedEarly("response closed before it was fully read"))
