"""Topic Extraction Pipeline — public API.

``extract_topics_pipeline(text)`` is the single entry point for full-document
topic extraction. Callable by any agent or tool (TXR-09).
"""

from __future__ import annotations

import json
import logging

from src.llm import get_llm
from src.topic_extraction.extract import _extract_segment_topics
from src.topic_extraction.segment import segment_text
from src.topic_extraction.tree import build_topic_tree
from src.config import settings
from src.topic_extraction.unify import reconcile_topics as reconcile_topics, unify_topics as unify_topics

from typing import Any

logger = logging.getLogger("tutor.topic_extraction")


async def extract_topics_pipeline(text: str) -> dict[str, Any]:
    """Extract hierarchical topics from a full academic document.

    Flow: segment → extract per segment (sequential await) → unify → build tree.

    Args:
        text: Raw markdown text (e.g. from ``markitdown`` conversion).

    Returns:
        Dict with keys:
        - ``summary`` (str): One-sentence summary (empty if no topics).
        - ``topics`` (list[str]): Unified topic list, deduplicated.
        - ``topic_tree`` (str): JSON-serialized nested topic hierarchy.
        - ``segment_count`` (int): Number of text segments processed.
        - ``failed_segments`` (list[int]): Indices of failed segments.

        TXR-10: If *text* is shorter than ``topic_segment_size`` chars,
        the pipeline processes it as a single segment with zero
        segmentation overhead.
    """
    if not text or not text.strip():
        return {
            "summary": "",
            "topics": [],
            "topic_tree": "{}",
            "segment_count": 0,
            "failed_segments": [],
        }

    # 1. Segment
    segments = segment_text(text)
    segment_count = len(segments)

    # 2. Extract per segment (sequential await)
    llm = get_llm(temperature=settings.topic_extraction_temperature)
    all_topics: list[str] = []
    failed_segments: list[int] = []

    for i, segment in enumerate(segments):
        try:
            segment_topics = await _extract_segment_topics(
                segment, llm, segment_index=i, total=segment_count
            )
            all_topics.extend(segment_topics)
        except Exception as exc:
            logger.warning("Segment %d/%d failed: %s", i + 1, segment_count, exc)
            failed_segments.append(i)

    # 3. Unify (skip if single segment — already one coherent set)
    if segment_count <= 1:
        unified = list(set(all_topics))
    else:
        unified = unify_topics(all_topics)

    # 4. Build tree
    tree_dict = await build_topic_tree(unified)

    topic_tree_str = json.dumps(tree_dict, ensure_ascii=False)

    # 5. Summary from first topic (or empty)
    summary = unified[0] if unified else ""

    logger.info(
        "Pipeline complete: %d segments → %d topics (tree=%d keys, %d failures)",
        segment_count,
        len(unified),
        len(tree_dict),
        len(failed_segments),
    )

    return {
        "summary": summary,
        "topics": unified,
        "topic_tree": topic_tree_str,
        "segment_count": segment_count,
        "failed_segments": failed_segments,
    }
