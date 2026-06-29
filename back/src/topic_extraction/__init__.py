"""Topic Extraction Pipeline — public API.

``extract_topics_pipeline(text)`` is the single entry point for full-document
topic extraction. Callable by any agent or tool (TXR-09).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import settings
from src.llm import get_llm
from src.topic_extraction.extract import TopicItem, _extract_segment_topics
from src.topic_extraction.segment import segment_text
from src.topic_extraction.tree import build_topic_tree
from src.topic_extraction.unify import (
    reconcile_topics as reconcile_topics,
)
from src.topic_extraction.unify import (
    unify_topics as unify_topics,
)

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
        - ``topic_descriptions`` (dict[str, str]): Per-topic descriptions
          in academic Spanish (TDR-01). Empty dict if no topics.
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
            "topic_descriptions": {},
            "segment_count": 0,
            "failed_segments": [],
        }

    # 1. Segment
    segments = segment_text(text)
    segment_count = len(segments)

    # 2. Extract per segment (sequential await)
    llm = get_llm(temperature=settings.topic_extraction_temperature)
    all_topic_items: list[TopicItem] = []
    raw_topic_descriptions: dict[str, str] = {}
    failed_segments: list[int] = []

    for i, segment in enumerate(segments):
        try:
            segment_items = await _extract_segment_topics(
                segment, llm, segment_index=i, total=segment_count
            )
            for item in segment_items:
                all_topic_items.append(item)
                # Keep the richest description per topic across segments
                existing = raw_topic_descriptions.get(item.topic, "")
                if len(item.description) > len(existing):
                    raw_topic_descriptions[item.topic] = item.description
        except Exception as exc:
            logger.warning("Segment %d/%d failed: %s", i + 1, segment_count, exc)
            failed_segments.append(i)

    # Extract just the topic strings for unification (backward-compatible)
    all_topic_strings: list[str] = [item.topic for item in all_topic_items]

    # 3. Unify (skip if single segment — already one coherent set)
    if segment_count <= 1:
        unified = list(set(all_topic_strings))
    else:
        unified = unify_topics(all_topic_strings, descriptions=raw_topic_descriptions)

    # 4. Rebuild topic_descriptions dict for unified topics.
    # For each unified (canonical) topic, pick the richest description
    # among all original TopicItem objects that map to it.
    topic_descriptions: dict[str, str] = {}
    for canonical in unified:
        best_desc = raw_topic_descriptions.get(canonical, "")
        # Also check merged-away topics that may have richer descriptions
        for item in all_topic_items:
            if item.topic != canonical and item.topic not in unified:
                # Check if this item was likely merged into the canonical
                from src.topic_extraction.preprocess import jaccard_similarity, stem_topic

                try:
                    sim = jaccard_similarity(stem_topic(item.topic), stem_topic(canonical))
                    if sim >= settings.topic_similarity_threshold:
                        if len(item.description) > len(best_desc):
                            best_desc = item.description
                except Exception:
                    pass
        topic_descriptions[canonical] = best_desc

    # 5. Build tree (renumbered from 4 above)
    tree_dict = await build_topic_tree(unified)

    topic_tree_str = json.dumps(tree_dict, ensure_ascii=False)

    # 6. Summary from first topic (or empty)
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
        "topic_descriptions": topic_descriptions,
        "segment_count": segment_count,
        "failed_segments": failed_segments,
    }
