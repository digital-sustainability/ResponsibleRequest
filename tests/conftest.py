from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from loguru import logger

import responsible_request as rr
from responsible_request._http import httpx


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def chat_response(content: str = "OK", completion_tokens: int = 10, **extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": extra.get("model", "test-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": completion_tokens,
            "total_tokens": 5 + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


Handler = Callable[[Any], Any]


def make_client(handler: Handler, **kwargs: Any) -> rr.AsyncOpenAI:
    """An rr.AsyncOpenAI whose HTTP traffic is answered by ``handler``."""
    transport = httpx.MockTransport(handler)
    throttle = kwargs.pop("throttle", rr.ThrottleConfig(max_rpm=6000, start_rpm=6000))
    log = kwargs.pop("log", False)
    default_params = kwargs.pop("default_params", None)
    rr_options = {k: kwargs.pop(k) for k in ("cache", "cost", "run", "extractors") if k in kwargs}
    http = rr.http_client(
        throttle, log, default_params=default_params, transport=transport, **rr_options
    )
    return rr.AsyncOpenAI(api_key="test", base_url="http://test/api/v1", http_client=http, **kwargs)


def json_response(data: Any, status: int = 200, **headers: str) -> Any:
    return httpx.Response(status, json=data, headers=headers)


@pytest.fixture
def records() -> Any:
    """Collect emitted request records."""
    collected: list[dict[str, Any]] = []
    logger.enable("responsible_request")
    handler_id = logger.add(
        lambda m: collected.append(m.record["extra"]["rr_record"]),
        level="TRACE",
        filter=lambda r: "rr_record" in r["extra"],
    )
    yield collected
    logger.remove(handler_id)
    logger.disable("responsible_request")


@pytest.fixture
def cleanup_sinks() -> Any:
    yield
    rr.remove_sinks()
    logger.disable("responsible_request")


def body_of(request: Any) -> dict[str, Any]:
    return json.loads(request.content)
