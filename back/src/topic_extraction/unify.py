"""Topic unification via Jaccard clustering (TXR-05).

Stems each topic → pairwise Jaccard similarity → union-find clusters →
picks the longest string per cluster as canonical name.

Deterministic — same input always produces same output. No LLM calls.
"""

from __future__ import annotations

import logging

from src.config import settings
from src.topic_extraction.preprocess import jaccard_similarity, stem_topic

logger = logging.getLogger("tutor.topic_extraction.unify")


def unify_topics(
    all_topics: list[str],
    threshold: float | None = None,
    max_count: int | None = None,
) -> list[str]:
    """Merge near-duplicate topics using Jaccard similarity on stemmed keywords.

    1. Stem each topic to ``set[str]`` of keyword stems.
    2. Pairwise Jaccard ≥ *threshold* → put in same cluster (union-find).
    3. Canonical name = longest string in cluster.
    4. Cap output to *max_count* (largest clusters first).
    5. Sort alphabetically.

    Args:
        all_topics: Raw topic strings from per-segment LLM extraction.
        threshold: Jaccard similarity threshold for merging.  Defaults to
            ``settings.topic_similarity_threshold`` (0.6).
        max_count: Maximum number of unified topics.  Defaults to
            ``settings.max_topics_per_document`` (30).

    Returns:
        Sorted list of canonical topic strings.
    """
    if threshold is None:
        threshold = settings.topic_similarity_threshold
    if max_count is None:
        max_count = settings.max_topics_per_document

    if not all_topics:
        return []

    n = len(all_topics)
    if n == 1:
        return list(all_topics)

    # 1. Stem all topics
    stem_map: list[set[str]] = []
    for t in all_topics:
        stem_map.append(stem_topic(t))

    # 2. Union-find clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_similarity(stem_map[i], stem_map[j]) >= threshold:
                union(i, j)

    # 3. Group by root, canonical = longest string
    clusters: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(all_topics[i])

    unified: list[tuple[str, int]] = []
    for cluster in clusters.values():
        canonical = max(cluster, key=len)
        unified.append((canonical, len(cluster)))

    # 4. Cap by max_count — keep largest clusters first
    unified.sort(key=lambda x: x[1], reverse=True)
    unified = unified[:max_count]

    # 5. Sort alphabetically
    result = sorted(t[0] for t in unified)

    logger.debug(
        "Unified %d topics → %d (threshold=%.2f, max=%d)",
        n,
        len(result),
        threshold,
        max_count,
    )
    return result


def reconcile_topics(
    new_topics: list[str],
    existing_topics: list[str],
    threshold: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Map new topics to existing topics in the session if they are semantically similar.

    This prevents duplicate/near-duplicate topics across different files in the same session.

    Args:
        new_topics: Newly extracted topic strings from the current document.
        existing_topics: Already stored topics in other documents of the same session.
        threshold: Jaccard similarity threshold for merging. Defaults to settings.topic_similarity_threshold (0.6).

    Returns:
        A tuple containing:
            - A list of reconciled topic strings.
            - A dictionary mapping original topic names to their reconciled names.
    """
    if threshold is None:
        threshold = settings.topic_similarity_threshold

    if not existing_topics or not new_topics:
        return list(new_topics), {t: t for t in new_topics}

    # Pre-stem existing topics
    existing_topics_unique = list(dict.fromkeys(existing_topics))
    existing_stems = [(t, stem_topic(t)) for t in existing_topics_unique]

    reconciled_topics = []
    topic_map = {}

    for topic in new_topics:
        topic_stem = stem_topic(topic)
        best_match = None
        best_sim = -1.0

        for ext_topic, ext_stem in existing_stems:
            sim = jaccard_similarity(topic_stem, ext_stem)
            if sim >= threshold and sim > best_sim:
                best_match = ext_topic
                best_sim = sim

        if best_match:
            reconciled_topics.append(best_match)
            topic_map[topic] = best_match
            logger.info(
                "Reconciled topic '%s' -> '%s' (similarity: %.2f)",
                topic,
                best_match,
                best_sim,
            )
        else:
            reconciled_topics.append(topic)
            topic_map[topic] = topic

    # Deduplicate while preserving order
    unique_reconciled = []
    seen = set()
    for t in reconciled_topics:
        if t not in seen:
            seen.add(t)
            unique_reconciled.append(t)

    return unique_reconciled, topic_map

