"""Drop-in OpenAI client and an ``http_client`` factory for existing code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai
from loguru import logger

from .config import LogConfig, ThrottleConfig
from .logging import setup_logging
from .throttle import Throttle
from .transport import ThrottledTransport


def http_client(
    throttle: ThrottleConfig | Throttle | None = None,
    log: LogConfig | bool | None = True,
    *,
    default_params: Mapping[str, Any] | None = None,
    inject_stream_usage: bool = True,
    transport: Any = None,
    **client_kwargs: Any,
) -> Any:
    """Build an async HTTP client for the OpenAI SDK that paces, measures and logs requests.

    Use it to add throttling to existing code::

        client = openai.AsyncOpenAI(http_client=rr.http_client(rr.ThrottleConfig(max_rpm=120)))

    :param throttle: Throttle configuration (or an existing :class:`Throttle` to share state).
    :param log: ``True`` for console output only, a :class:`LogConfig` to also write request
        records to JSONL/SQLite, ``False``/``None`` to leave logging disabled.
    :param default_params: Chat completion parameters added to every request that does not set
        them itself, e.g. ``rr.reproducible()``.
    :param inject_stream_usage: Request token usage for streamed chat completions.
    :param transport: Underlying transport (defaults to a plain async HTTP transport).
    :param client_kwargs: Passed to ``openai.DefaultAsyncHttpxClient`` (e.g. ``timeout``).
    """
    state = throttle if isinstance(throttle, Throttle) else Throttle(throttle)
    log_config = LogConfig() if log is True else (log or None)
    if log_config is not None:
        setup_logging(log_config)
    rr_transport = ThrottledTransport(
        state,
        log=log_config,
        default_params=default_params,
        inject_stream_usage=inject_stream_usage,
        transport=transport,
    )
    client = openai.DefaultAsyncHttpxClient(transport=rr_transport, **client_kwargs)
    client.rr_throttle = state  # type: ignore[attr-defined]
    return client


def get_throttle(client: Any) -> Throttle:
    """Return the :class:`Throttle` behind an OpenAI client created with :func:`http_client`."""
    throttle = getattr(client, "throttle", None)
    if isinstance(throttle, Throttle):
        return throttle
    inner = getattr(client, "_client", client)  # the SDK keeps its http client in ``_client``
    throttle = getattr(inner, "rr_throttle", None)
    if not isinstance(throttle, Throttle):
        raise TypeError("client was not created with responsible_request")
    return throttle


class AsyncOpenAI(openai.AsyncOpenAI):
    """``openai.AsyncOpenAI`` with load-aware throttling and request logging.

    Accepts every argument of ``openai.AsyncOpenAI`` plus ``throttle``, ``log`` and
    ``default_params`` (see :func:`http_client`). The throttle state is available as
    ``client.throttle``; ``client.throttle.stats()`` shows the current rate and load per model.
    """

    throttle: Throttle

    def __init__(
        self,
        *,
        throttle: ThrottleConfig | Throttle | None = None,
        log: LogConfig | bool | None = True,
        default_params: Mapping[str, Any] | None = None,
        http_client: Any = None,
        **kwargs: Any,
    ) -> None:
        if http_client is None:
            http_client = _http_client(throttle, log, default_params=default_params)
        elif getattr(http_client, "rr_throttle", None) is None:
            raise TypeError(
                "pass throttle/log options instead of http_client, or build the http client "
                "with responsible_request.http_client()"
            )
        super().__init__(http_client=http_client, **kwargs)
        self.throttle = http_client.rr_throttle

    async def close(self) -> None:
        await super().close()
        await logger.complete()


_http_client = http_client
