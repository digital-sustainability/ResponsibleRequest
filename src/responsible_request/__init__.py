"""Load-aware throttling and request logging for OpenAI-compatible LLM gateways.

Typical use::

    import responsible_request as rr

    client = rr.AsyncOpenAI(base_url=..., api_key=..., log=rr.LogConfig(sqlite="requests.db"))
    response = await client.chat.completions.create(model=..., messages=..., **rr.reproducible())
"""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from .analysis import load_records
from .client import AsyncOpenAI, get_throttle, http_client
from .config import FIELD_GROUPS, LogConfig, ThrottleConfig
from .controller import State, ThrottleController
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
    "LatencyEstimator",
    "LoadEstimator",
    "LogConfig",
    "RequestRecord",
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
    "remove_sinks",
    "reproducible",
    "response_format_from_model",
    "run_batch",
    "run_batch_sync",
    "setup_logging",
    "structured",
    "tags",
]
