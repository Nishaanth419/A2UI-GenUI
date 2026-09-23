"""The streaming loop: free-text in, a stream of A2UI v0.9 events out.

Groq's GPT-OSS structured-output endpoint does not support response streaming,
so the model is asked for one strict JSON-schema completion. Once it arrives,
the resulting A2UI messages are still streamed over SSE to the browser.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from pydantic import ValidationError

import a2ui
from agent.config import MODEL, get_client
from agent.emit import emit_blocks
from agent.events import Event
from agent.fallbacks import fallback
from agent.follow_ups import clean_follow_ups
from agent.prompts import SYSTEM_PROMPT
from schemas import AgentTurn, Block, Turn

logger = logging.getLogger("genui.agent")


def _groq_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a Pydantic schema valid for Groq's strict JSON Schema mode.

    Groq requires every object node, including nested definitions under
    ``$defs``, to explicitly forbid undeclared properties. Pydantic's default
    schema leaves that unconstrained, so normalise the generated schema at the
    provider boundary rather than duplicating provider-specific config across
    every model in ``schemas/``.
    """
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    normalised = schema.copy()
    visit(normalised)
    return normalised


def _messages(message: str, history: list[Turn]) -> list[dict]:
    replayed = [{"role": turn.role, "content": turn.content} for turn in history]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *replayed,
        {"role": "user", "content": message},
    ]


def _generate(message: str, history: list[Turn]) -> AgentTurn:
    """Request and validate one strict JSON-schema completion from Groq."""
    completion = get_client().chat.completions.create(
        model=MODEL,
        messages=_messages(message, history),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_turn",
                "strict": True,
                "schema": _groq_strict_schema(AgentTurn.model_json_schema()),
            },
        },
        temperature=0.2,
    )

    choice = completion.choices[0]
    if choice.message.refusal:
        raise ValueError(f"model refused: {choice.message.refusal}")
    if choice.message.content is None:
        raise ValueError("model returned no parseable output")
    return AgentTurn.model_validate_json(choice.message.content)


def _usable_blocks(blocks: list[Block]) -> list[Block]:
    """Sanitise a turn's blocks, dropping any that cannot be drawn.

    One unusable block does not sink the turn -- a bad donut alongside a good
    chart should cost the donut, not the answer.
    """
    usable: list[Block] = []
    for block in blocks[: a2ui.MAX_BLOCKS]:
        try:
            usable.append(a2ui.sanitize(block))
        except ValueError as exc:
            logger.warning("dropping unusable block: %s", exc)
    return usable


def run(message: str, history: list[Turn], surface_id: str) -> Iterator[Event]:
    """Stream one turn. Never raises: every failure path yields a text_note."""
    yield Event("a2ui", a2ui.create_surface(surface_id))
    yield Event("a2ui", a2ui.update_data_model(surface_id, "/", {"blocks": []}))

    if not os.getenv("GROQ_API_KEY"):
        yield from fallback(surface_id, "missing_api_key")
        return

    try:
        turn = _generate(message, history)
    except ValidationError as exc:
        logger.warning("model output failed validation: %s", exc)
        yield from fallback(surface_id, "schema_validation_failed")
        return
    except Exception:  # network, auth, rate limit, refusal, anything upstream
        logger.exception("generation failed")
        yield from fallback(surface_id, "upstream_error")
        return

    final = _usable_blocks(turn.blocks)
    if not final:
        yield from fallback(surface_id, "unusable_blocks")
        return

    yield from emit_blocks(surface_id, final)
    yield Event(
        "meta",
        {
            "ok": True,
            "explanation": turn.explanation,
            "follow_ups": clean_follow_ups(turn.follow_ups, message),
            "fallback": False,
            "error": None,
            # What went on screen, for the transcript. The next turn needs this
            # to resolve "that chart" without replaying the whole data model.
            "rendered": [f"{block.title} ({block.type})" for block in final],
        },
    )
