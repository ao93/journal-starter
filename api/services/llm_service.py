"""Task 4: Implement analyze_journal_entry using any OpenAI-compatible API."""

import json

from openai import AsyncOpenAI

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyze a journal entry using an OpenAI-compatible LLM."""
    if client is None:
        client = _default_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a learning journal analyst. Analyze the journal entry and respond "
                "with a JSON object containing exactly these fields: "
                "sentiment (one of: positive, negative, neutral), "
                "summary (2 sentences), "
                "topics (list of 2-4 key topics as strings). "
                "Respond with only valid JSON, no extra text."
            ),
        },
        {
            "role": "user",
            "content": entry_text,
        },
    ]

    response = await client.chat.completions.create(
        model=get_settings().openai_model,
        messages=messages,
    )

    result = json.loads(response.choices[0].message.content)
    result["entry_id"] = entry_id
    return result
