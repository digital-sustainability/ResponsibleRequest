"""User-facing configuration objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .estimator import LoadEstimator
    from .records import RequestRecord

Metric = Literal["latency_per_token", "latency", "ttfb"]


@dataclass(frozen=True)
class ThrottleConfig:
    """How requests are paced and how the pace adapts to the observed load.

    Requests are spaced evenly at the current RPM (no bursts). The RPM starts at ``start_rpm``,
    ramps up towards ``max_rpm`` while latency stays close to the baseline, and drops to
    ``min_rpm`` as soon as latency reaches ``high_ratio`` x baseline (or the server returns
    429/5xx/timeouts).
    """

    max_rpm: float = 15
    """Upper bound on requests per minute when the endpoint is idle."""
    min_rpm: float = 1
    """Requests per minute while other users are active. Also serves as probing rate."""
    start_rpm: float = 2
    """Initial rate while the latency baseline is being established (clamped to [min, max])."""
    max_concurrency: int = 32
    """Maximum number of in-flight requests per model, regardless of RPM."""

    high_ratio: float = 3.0
    """Throttle to ``min_rpm`` when current latency >= ``high_ratio`` x baseline."""
    recover_ratio: float = 1.5
    """Ramp up only while current latency < ``recover_ratio`` x baseline (dead band in between)."""
    cooldown_s: float = 120
    """Minimum time spent at ``min_rpm`` after the last high-load signal."""
    ramp_factor: float = 1.5
    """Multiplicative RPM increase per ramp step."""
    ramp_interval_s: float = 30
    """Minimum time between two ramp steps."""

    metric: Metric | Callable[[RequestRecord], float | None] = "latency_per_token"
    """Latency signal: ``latency_per_token`` (latency / max(completion tokens, ``min_tokens``);
    uses time-to-first-byte for streamed requests), ``latency`` (raw), ``ttfb``, or a callable."""
    min_tokens: int = 16
    """Floor for the token count used by the ``latency_per_token`` metric."""
    window: int = 10
    """Number of recent observations whose median forms the current latency signal."""
    warmup_requests: int = 20
    """Observations required before the baseline is trusted and the RPM may ramp up."""
    baseline: float | None = None
    """Pin the baseline (in units of ``metric``) instead of estimating it."""
    baseline_percentile: float = 10
    """Baseline = this percentile of the observations inside ``baseline_window_s``."""
    baseline_window_s: float = 1800
    """Time span of observations used for the baseline (samples taken while throttled are
    excluded, so the baseline freezes during high-load periods)."""

    observe_paths: tuple[str, ...] = ("/chat/completions", "/embeddings")
    """Endpoints whose latency feeds the load estimate. Other endpoints are paced and logged,
    and react to 429/5xx, but do not adapt to latency."""
    ignore_server_tools: bool = True
    """Exclude requests using server-side tools (e.g. MCP, web search) from the load estimate,
    since their latency includes external calls."""
    adaptive: bool = True
    """If False, requests are paced at a constant ``max_rpm``."""
    estimator_factory: Callable[[ThrottleConfig], LoadEstimator] | None = None
    """Custom load estimator (see :class:`responsible_request.estimator.LoadEstimator`)."""

    def __post_init__(self) -> None:
        if not 0 < self.min_rpm <= self.max_rpm:
            raise ValueError("require 0 < min_rpm <= max_rpm")
        if not 1 < self.recover_ratio < self.high_ratio:
            raise ValueError("require 1 < recover_ratio < high_ratio")
        if self.ramp_factor <= 1:
            raise ValueError("ramp_factor must be > 1")
        if self.max_concurrency < 1 or self.window < 1:
            raise ValueError("max_concurrency and window must be >= 1")
        if not 0 < self.baseline_percentile < 100:
            raise ValueError("baseline_percentile must be in (0, 100)")

    @property
    def initial_rpm(self) -> float:
        if not self.adaptive:
            return self.max_rpm
        return min(self.max_rpm, max(self.min_rpm, self.start_rpm))

    @classmethod
    def fixed(cls, rpm: float, max_concurrency: int = 32) -> ThrottleConfig:
        """A plain rate limiter at a constant ``rpm`` (no load adaptation)."""
        return cls(
            max_rpm=rpm,
            min_rpm=rpm,
            start_rpm=rpm,
            max_concurrency=max_concurrency,
            adaptive=False,
        )


FIELD_GROUPS = (
    "timing",
    "usage",
    "throttle",
    "response_meta",
    "params",
    "request_body",
    "response_body",
)
"""Record field groups that can be toggled in :class:`LogConfig`. Core fields (id, timestamp,
path, model, status, error, tags) are always logged."""


@dataclass(frozen=True)
class LogConfig:
    """Where and what to log.

    The console only receives throttle changes, a periodic summary, and warnings. Every request
    is written as one structured record to the configured JSONL file and/or SQLite database.
    """

    jsonl: str | Path | None = None
    """Append one JSON object per request to this file."""
    sqlite: str | Path | None = None
    """Insert one row per request into the ``requests`` table of this SQLite database."""
    fields: Mapping[str, bool] = field(default_factory=dict)
    """Toggle record field groups (see ``FIELD_GROUPS``); groups not mentioned are logged."""
    summary_interval_s: float | None = 60
    """Emit an INFO summary line per model at most this often (None disables)."""
    console_level: str | None = None
    """If set, replace loguru's default stderr handler with one at this level. If None, the
    console handlers are left untouched."""
    rotation: str | int | None = "500 MB"
    """Passed to loguru for the JSONL file."""

    def __post_init__(self) -> None:
        unknown = set(self.fields) - set(FIELD_GROUPS)
        if unknown:
            raise ValueError(f"unknown field groups {sorted(unknown)}; valid: {FIELD_GROUPS}")

    def enabled(self, group: str) -> bool:
        return self.fields.get(group, True)

    @classmethod
    def minimal(cls, **kwargs: Any) -> LogConfig:
        """Log timing, usage and throttle state, but no parameters or message contents."""
        off = {"params": False, "request_body": False, "response_body": False}
        return cls(fields=off, **kwargs)

    @classmethod
    def full(cls, **kwargs: Any) -> LogConfig:
        """Log everything (the default)."""
        return cls(**kwargs)

    def with_fields(self, **toggles: bool) -> LogConfig:
        return replace(self, fields={**self.fields, **toggles})


@dataclass(frozen=True)
class CacheConfig:
    """Answer requests from previously logged records instead of sending them again.

    A request is served from the cache if a successful (HTTP 200) record with the same cache key
    exists. The key is a hash of the URL and the complete JSON request body (model, messages and
    all parameters, after ``default_params`` are applied), plus the values of ``key_tags``.
    Streamed requests are never served from the cache. Cache hits are logged with
    ``cache_hit=True`` and neither wait for nor affect the throttle.
    """

    sqlite: str | Path | None = None
    """Read cached responses from this SQLite database (defaults to ``LogConfig.sqlite``)."""
    jsonl: str | Path | None = None
    """Read cached responses from this JSONL file (defaults to ``LogConfig.jsonl``)."""
    key_tags: tuple[str, ...] = ()
    """Tags (see :func:`responsible_request.tags`) that are part of the cache key, e.g.
    ``("sample",)`` to draw several independent samples for the same request."""


@dataclass(frozen=True)
class Price:
    """Token prices in USD per million tokens (used when the response reports no cost)."""

    input: float
    output: float
    cached_input: float | None = None
    """Price of cached prompt tokens; defaults to ``input``."""

    def cost(
        self, prompt_tokens: int | None, completion_tokens: int | None, cached_tokens: int | None
    ) -> float:
        prompt, cached = prompt_tokens or 0, min(cached_tokens or 0, prompt_tokens or 0)
        cached_price = self.input if self.cached_input is None else self.cached_input
        total = (prompt - cached) * self.input + cached * cached_price
        return (total + (completion_tokens or 0) * self.output) / 1_000_000


@dataclass(frozen=True)
class CostConfig:
    """Track the cost of every request and optionally stop sending requests above a budget.

    The cost of a request is the one reported by the provider (OpenRouter's ``usage.cost``, the
    LiteLLM ``x-litellm-response-cost`` header, or a custom extractor), else it is computed from
    ``prices`` and the token usage. Cache hits cost nothing.
    """

    budget_usd: float | None = None
    """Refuse new requests (raising :class:`responsible_request.BudgetExceeded`) once this much
    has been spent. Requests already in flight still complete, so the budget can be overshot by
    their cost. Cache hits are always served."""
    prices: Mapping[str, Price] = field(default_factory=dict)
    """Fallback prices per model name, for providers that don't report a cost."""
    include_logged: bool = True
    """Count the cost already recorded in the request log (``LogConfig.sqlite``/``jsonl``), so
    that restarting a script does not reset the budget."""

    def __post_init__(self) -> None:
        if self.budget_usd is not None and self.budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    """Describes a run: one client (or ``http_client``) from creation to close.

    Every request record carries the ``run_id`` of its run. On the first request a run manifest
    (versions, git commit, command line, all configurations, ``default_params``, ``metadata``) is
    written to the ``runs`` table of the SQLite log, or to ``<name>.runs.jsonl`` next to the
    JSONL log.
    """

    name: str | None = None
    """Human-readable name of the run, e.g. the experiment."""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Anything else worth recording (dataset version, hyperparameters, notes, ...)."""
    capture_git: bool = True
    """Record the git commit and dirty flag of the working directory."""
