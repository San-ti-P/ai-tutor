"""Per-segment LLM topic extraction (TXR-03).

Uses ``get_llm()`` with a Pydantic JSON schema embedded in the system
prompt — same pattern as ``_ollama_json_mode_chain`` (``llm.py:45-103``).
No ``with_structured_output()``.

Processes segments sequentially (one ``await`` per segment).  Parse
failures are logged and the segment is skipped; the pipeline continues.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("tutor.topic_extraction.extract")

# ── Pydantic schema (embedded in system prompt as JSON string) ────────────────


class TopicItem(BaseModel):
    """A single academic topic with a one-sentence description."""

    topic: str = Field(description="Academic topic phrase (3-8 words)")
    description: str = Field(description="One-sentence description, max 15 words, academic Spanish")


class SegmentTopics(BaseModel):
    """Expected LLM response shape for per-segment topic extraction."""

    topics: list[TopicItem] = Field(
        description="3-8 academic topics detected in this segment, each with a description",
        min_length=1,
        max_length=12,
    )


_TOPIC_SCHEMA_JSON = json.dumps(SegmentTopics.model_json_schema(), indent=2)

# ── System prompt template ────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Respondé únicamente con un objeto JSON que coincida exactamente con este esquema. "
    "No incluyas ningún otro texto:\n"
    f"{_TOPIC_SCHEMA_JSON}\n\n"
    "Extraé entre 3 y 8 temas académicos concretos del fragmento de texto a continuación. "
    "Los temas deben ser frases específicas, no palabras genéricas sueltas. "
    "Para cada tema, agregá una breve descripción (máximo 15 palabras) que "
    "explique el concepto. IMPORTANTE: usá la terminología y vocabulario "
    "exacto del texto fuente en las descripciones, no parafrasees con "
    "sinónimos. Devolvé solo JSON válido."
)

# ── Regex for extracting JSON from LLM output ─────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _parse_json_response(text: str) -> SegmentTopics:
    """Extract and validate JSON from an LLM response string.

    Raises:
        ValueError: If no JSON object is found or validation fails.
    """
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:300]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}") from exc
    return SegmentTopics.model_validate(data)


# ── Public extraction function ────────────────────────────────────────────────


async def _extract_segment_topics(
    segment: str,
    llm: Any,  # BaseChatModel — passed in to avoid re-creating per call
    segment_index: int = 0,
    total: int = 1,
) -> list[TopicItem]:
    """Extract topics from a single text segment via LLM.

    Args:
        segment: Text segment (up to ``topic_segment_size`` chars).
        llm: A configured ``BaseChatModel`` from ``get_llm()``.
        segment_index: Zero-based segment number (for logging).
        total: Total segment count (for logging).

    Returns:
        List of TopicItem objects, or empty list on failure.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    human_msg = f"Fragmento de texto ({segment_index + 1} de {total}):\n\n{segment}"

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ]

    for attempt in range(2):  # 1 retry
        try:
            response = await llm.ainvoke(messages)
            response_text: str = response.content if hasattr(response, "content") else str(response)
            result = _parse_json_response(response_text)
            return list(result.topics)
        except Exception as exc:
            if attempt == 0:
                logger.debug(
                    "Segment %d/%d parse attempt %d failed: %s — retrying",
                    segment_index + 1,
                    total,
                    attempt + 1,
                    exc,
                )
                continue
            logger.warning(
                "Segment %d/%d failed after %d attempts: %s",
                segment_index + 1,
                total,
                attempt + 1,
                exc,
            )

    return []
