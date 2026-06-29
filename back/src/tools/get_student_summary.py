"""LangChain tool for read-only student profile aggregation."""

from __future__ import annotations

from typing import Any
from langchain_core.tools import tool
from langfuse import observe


@tool
@observe(name="get_student_summary", as_type="tool")
async def get_student_summary(student_id: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Return an aggregated read-only summary of a student's profile.

    Performs zero writes. Aggregates identity, preferences, per-topic
    scores, weak topics, recent session history, and session count.

    Args:
        student_id: Unique student identifier.
        session_id: Optional session scope. When provided, topic_scores
            and weak_topics are filtered to that session only.

    Returns:
        A dict with keys ``id``, ``preferences``, ``topic_scores``,
        ``weak_topics``, ``session_history``, ``session_count``.
        Returns ``None`` when the student ID is unknown.
    """
    from src.memory.schema import (
        compute_weak_topics,
        get_recent_sessions,
        get_student_profile,
        get_topic_scores,
    )

    profile = await get_student_profile(student_id, session_id)
    if profile is None:
        return None

    topic_scores_list = await get_topic_scores(student_id, session_id)
    weak_topics = await compute_weak_topics(student_id, session_id)
    session_history = await get_recent_sessions(student_id)

    return {
        "id": profile["id"],
        "preferences": profile.get("preferences", {}),
        "topic_scores": topic_scores_list,
        "weak_topics": weak_topics,
        "session_history": session_history,
        "session_count": profile.get("session_count", 0),
    }
