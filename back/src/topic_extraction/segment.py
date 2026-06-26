"""Markdown-aware text segmentation for topic extraction.

Splits academic markdown on heading boundaries (`#`, `##`, `###`, `####`)
so each segment sent to the LLM represents a coherent topic unit (TXR-02).
"""

from __future__ import annotations

import logging
import re

from src.config import settings

logger = logging.getLogger("tutor.topic_extraction.segment")

# Matches markdown ATX headings: # through #### at line start
_HEADING_RE = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)


def segment_text(
    text: str,
    min_section: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Split *text* into segments bounded by markdown heading boundaries.

    Strategy (in order):
    1. Split on ``#``, ``##``, ``###``, ``####`` headings.
    2. Merge adjacent sections shorter than *min_section* chars.
    3. If no headings found, fall back to splitting on ``\\n\\n`` (paragraphs).
    4. If *text* is shorter than *max_chars*, return ``[text]`` (passthrough).
    5. Whitespace-only text returns ``[]``.

    Args:
        text: Raw markdown text from ``markitdown`` conversion.
        min_section: Adjacent sections below this character count are
            merged into the next section.  Defaults to
            ``settings.topic_min_section_chars`` (200).
        max_chars: Maximum segment size.  Text shorter than this is
            returned as a single segment.  Defaults to
            ``settings.topic_segment_size`` (6000).

    Returns:
        List of text segments, each beginning at a heading boundary.
        Empty list if *text* is empty or whitespace-only.
    """
    if min_section is None:
        min_section = settings.topic_min_section_chars
    if max_chars is None:
        max_chars = settings.topic_segment_size

    if not text or not text.strip():
        return []

    # Find heading positions
    headings = list(_HEADING_RE.finditer(text))

    if not headings:
        # Fallback: no headings → split on double newline (paragraphs)
        if len(text) <= max_chars:
            return [text]
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) <= 1:
            return [text]
        return parts

    # Split at heading boundaries
    segments: list[str] = []
    for i, match in enumerate(headings):
        start = match.start()
        if i + 1 < len(headings):
            end = headings[i + 1].start()
        else:
            end = len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

    # Merge adjacent sections shorter than min_section
    merged: list[str] = []
    i = 0
    while i < len(segments):
        current = segments[i]
        j = i + 1
        while j < len(segments) and len(current) < min_section:
            current = current + "\n\n" + segments[j]
            j += 1
        merged.append(current.strip())
        i = j

    # Passthrough: if single segment and text is short enough
    if len(merged) == 1 and len(text) <= max_chars:
        return [text.strip()]

    logger.debug(
        "Segmented %d chars into %d sections (headings=%d, merged=%d)",
        len(text),
        len(merged),
        len(headings),
        len(merged),
    )
    return merged
