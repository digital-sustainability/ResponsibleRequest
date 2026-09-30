"""Structured output: request JSON matching a Pydantic model and validate the answer."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model did not produce valid output within the allowed number of attempts."""

    def __init__(self, message: str, *, last_content: str | None, errors: list[Exception]) -> None:
        super().__init__(message)
        self.last_content = last_content
        self.errors = errors


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        node.pop("default", None)
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)


def response_format_from_model(model: type[BaseModel], *, strict: bool = True) -> dict[str, Any]:
    """Build a ``json_schema`` ``response_format`` from a Pydantic model.

    With ``strict=True`` every object forbids additional properties and marks all properties as
    required, as strict structured-output backends demand.
    """
    schema = copy.deepcopy(model.model_json_schema())
    if strict:
        _strictify(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": schema, "strict": strict},
    }


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip Markdown code fences and surrounding prose around a JSON object."""
    match = _FENCE.match(text)
    if match:
        return match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start and not text.lstrip().startswith("{"):
        return text[start : end + 1]
    return text


async def structured(
    client: openai.AsyncOpenAI,
    *,
    model: str,
    messages: Iterable[Any],
    schema: type[T],
    retries: int = 2,
    strict: bool = True,
    fallback: bool = True,
    **kwargs: Any,
) -> T:
    """Ask for JSON matching ``schema`` and return it as a validated ``schema`` instance.

    If the answer does not validate, the model is shown the validation error and asked again
    (up to ``retries`` times). If the backend rejects ``json_schema`` response formats and
    ``fallback`` is set, it switches to ``json_object`` mode with the schema in the prompt.

    Extra keyword arguments are passed to ``client.chat.completions.create``.
    """
    msgs: list[Any] = list(messages)
    response_format: dict[str, Any] = response_format_from_model(schema, strict=strict)
    errors: list[Exception] = []
    content: str | None = None

    attempt = 0
    while attempt <= retries:
        try:
            params: dict[str, Any] = {
                "model": model,
                "messages": msgs,
                "response_format": response_format,
                **kwargs,
            }
            response = await client.chat.completions.create(**params)
        except openai.BadRequestError as exc:
            if not fallback or response_format["type"] != "json_schema":
                raise
            errors.append(exc)
            schema_text = json.dumps(response_format["json_schema"]["schema"])
            msgs = [
                {
                    "role": "system",
                    "content": f"Respond only with a JSON object matching this schema:\n"
                    f"{schema_text}",
                },
                *msgs,
            ]
            response_format = {"type": "json_object"}
            continue  # the fallback does not count as a retry

        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(extract_json(content))
        except ValidationError as exc:
            errors.append(exc)
            msgs = [
                *msgs,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "Your answer did not match the required JSON schema:\n"
                    f"{exc}\nRespond again with only the corrected JSON object.",
                },
            ]
        attempt += 1

    raise StructuredOutputError(
        f"no valid {schema.__name__} after {retries + 1} attempts",
        last_content=content,
        errors=errors,
    )
