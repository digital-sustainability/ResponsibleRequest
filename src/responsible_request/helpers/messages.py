"""Message builders."""

from __future__ import annotations

from typing import Any


def cached_system_message(content: str) -> dict[str, Any]:
    """A system message marked for OpenRouter's prompt cache (``cache_control: ephemeral``).

    The provider caches the prompt prefix up to this message, so repeated requests with the same
    system prompt pay less for input tokens; the model still runs. This is unrelated to the
    response cache (``cache=True``), which replays stored responses without sending a request.
    """
    return {
        "role": "system",
        "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
    }
