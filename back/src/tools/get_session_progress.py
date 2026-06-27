"""Tool: return per-session progress (topic scores, weak topics, exam count)."""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory import schema as _schema


@tool
async def get_session_progress(session_id: str) -> dict:
    """Return topic scores, weak topics, exam count, and average score for
    the current session.

    Data is aggregated from the ``evaluations`` table, not the global
    ``topic_scores`` table, so it reflects only work done in this session.

    Args:
        session_id: Current session ID.

    Returns:
        A dict with keys: session_id, topic_scores, weak_topics, exam_count,
        average_score. Returns empty/defaults when the session has no
        evaluations or does not exist.
    """
    profile = await _schema.get_session_profile(session_id)

    if profile is None:
        return {
            "session_id": session_id,
            "topic_scores": {},
            "weak_topics": [],
            "exam_count": 0,
            "average_score": None,
        }

    return {
        "session_id": profile.get("session_id", session_id),
        "topic_scores": profile.get("topic_scores", {}),
        "weak_topics": profile.get("weak_topics", []),
        "exam_count": profile.get("exam_count", 0),
        "average_score": profile.get("average_score"),
    }
