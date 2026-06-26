"""Tool: list files uploaded to the current session."""

from __future__ import annotations

from langchain_core.tools import tool

from src.memory import schema as _schema


@tool
async def list_session_files(session_id: str) -> dict:
    """Return the list of files uploaded to the current session.

    Args:
        session_id: Current session ID.

    Returns:
        A dict with keys: session_id, files (list of file dicts), count.
    """
    rows = await _schema.list_session_files(session_id)
    files = [
        {
            "id": row["id"],
            "file_name": row["file_name"],
            "classification": row.get("classification") or "",
            "topics": _parse_topics(row.get("topics_json")),
            "chunks_count": row.get("chunks_count", 0),
            "ingested_at": row["ingested_at"],
        }
        for row in rows
    ]
    return {"session_id": session_id, "files": files, "count": len(files)}


def _parse_topics(topics_json: str | None) -> list[str]:
    """Parse a JSON topic list, returning an empty list on failure."""
    import json

    if not topics_json:
        return []
    try:
        topics = json.loads(topics_json)
        return topics if isinstance(topics, list) else []
    except Exception:
        return []
