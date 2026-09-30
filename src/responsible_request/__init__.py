"""Responsible LLM requests for research: polite (load-aware throttling), reproducible (request
logs, run manifests, decoding presets) and cost-safe (response cache, cost tracking, budgets).
Works with any OpenAI-compatible endpoint (LiteLLM, OpenRouter, vLLM, Ollama, OpenAI, ...).

Typical use::

    import responsible_request as rr

    client = rr.AsyncOpenAI(base_url=..., api_key=..., log=rr.LogConfig(sqlite="requests.db"))
    response = await client.chat.completions.create(model=..., messages=..., **rr.reproducible())
"""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from . import providers
from .analysis import load_records, load_runs
from .cache import ResponseCache
from .client import AsyncOpenAI, get_throttle, http_client
from .config import (
    FIELD_GROUPS,
    CacheConfig,
    CostConfig,
    LogConfig,
    Price,
    RunConfig,
    ThrottleConfig,
)
from .controller import State, ThrottleController
from .cost import BudgetExceeded, CostTracker
from .estimator import LatencyEstimator, LoadEstimator
from .helpers import (
    StructuredOutputError,
    calibrate,
    extract_json,
    reproducible,
    response_format_from_model,
    run_batch,
    run_batch_sync,
    structured,
)
from .logging import remove_sinks, setup_logging
from .records import RequestRecord, tags
from .throttle import Throttle
from .transport import ThrottledTransport

try:
    __version__ = version("responsible-request")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# Libraries should stay silent unless the application opts in (loguru convention).
logger.disable(__name__)

__all__ = [
    "FIELD_GROUPS",
    "AsyncOpenAI",
    "BudgetExceeded",
    "CacheConfig",
    "CostConfig",
    "CostTracker",
    "LatencyEstimator",
    "LoadEstimator",
    "LogConfig",
    "Price",
    "RequestRecord",
    "ResponseCache",
    "RunConfig",
    "State",
    "StructuredOutputError",
    "Throttle",
    "ThrottleConfig",
    "ThrottleController",
    "ThrottledTransport",
    "__version__",
    "calibrate",
    "extract_json",
    "get_throttle",
    "http_client",
    "load_records",
    "load_runs",
    "providers",
    "remove_sinks",
    "reproducible",
    "response_format_from_model",
    "run_batch",
    "run_batch_sync",
    "setup_logging",
    "structured",
    "tags",
]
