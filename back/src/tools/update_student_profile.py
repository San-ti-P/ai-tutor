"""LangChain tool for updating student preferences and topic scores."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
async def update_student_profile(
    student_id: str,
    topic_scores: dict[str, float],
    preferences: dict | None = None,
) -> dict:
    """Upsert student preferences and per-topic scores into SQLite.

    Args:
        student_id: Unique student identifier.
        topic_scores: Mapping of topic name → score (0-10).
        preferences: Optional dict of exam preferences (difficulty, question
            types, count, include/exclude topics). Defaults are used if
            omitted.

    Returns:
        A dict with ``status``, ``student_id``, ``upserted_topics``
        count, and any ``errors``.
    """
    from src.memory.schema import upsert_student_profile, upsert_topic_scores

    default_prefs = {
        "difficulty": "medium",
        "question_types": ["mcq"],
        "question_count": 5,
        "include_topics": [],
        "exclude_topics": [],
    }

    errors: list[str] = []

    # Store preferences (use defaults if none provided)
    merged_prefs = {**default_prefs, **(preferences or {})}
    try:
        await upsert_student_profile(student_id, merged_prefs)
    except Exception as exc:
        errors.append(f"Failed to upsert preferences: {exc}")

    # Convert dict[str, float] to list[dict] for upsert_topic_scores
    scores_list = [{"topic": topic, "score": score} for topic, score in topic_scores.items()]

    try:
        await upsert_topic_scores(student_id, scores_list)
    except Exception as exc:
        errors.append(f"Failed to upsert topic scores: {exc}")

    if errors:
        return {
            "status": "partial",
            "student_id": student_id,
            "upserted_topics": len(scores_list),
            "errors": errors,
        }

    return {
        "status": "ok",
        "student_id": student_id,
        "upserted_topics": len(scores_list),
        "errors": [],
    }
