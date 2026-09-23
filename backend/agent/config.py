"""Groq model configuration via its OpenAI-compatible API."""

from __future__ import annotations

import os

from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_FOLLOW_UPS = 3
MAX_FOLLOW_UP_CHARS = 70

DEFAULT_FOLLOW_UPS = [
    "Show revenue as a line chart",
    "Expenses by category as a bar chart",
    "What was net cash flow in August?",
]

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Build an OpenAI-compatible client for Groq only when it is needed."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
        )
    return _client
