"""SQLite schema for student profiles, sessions, and evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime

import aiosqlite

from src.config import settings
from src.rag import get_chroma_client


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    """Return True if *column* already exists in *table*."""
    # PRAGMA does not accept parameter binding for the table name.
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return any(row[1] == column for row in rows)


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Apply additive schema migrations idempotently."""
    if not await _column_exists(db, "sessions", "name"):
        await db.execute(
            "ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''"
        )
    if not await _column_exists(db, "sessions", "description"):
        await db.execute("ALTER TABLE sessions ADD COLUMN description TEXT DEFAULT ''")
    if not await _column_exists(db, "ingested_documents", "session_id"):
        await db.execute(
            "ALTER TABLE ingested_documents ADD COLUMN session_id TEXT "
            "REFERENCES sessions(id)"
        )


async def init_db() -> None:
    """Initialize the SQLite database and create tables if they don't exist."""
    db_dir = os.path.dirname(settings.sqlite_db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                preferences_json TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                intent TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                topic TEXT,
                score REAL,
                feedback_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS topic_scores (
                topic TEXT NOT NULL,
                student_id TEXT NOT NULL,
                score REAL NOT NULL,
                evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (topic, student_id),
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS ingested_documents (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                classification TEXT,
                topics_json TEXT,
                chunks_count INTEGER DEFAULT 0,
                ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        await _run_migrations(db)
        await db.commit()


async def resolve_student_id(session_id: str, student_id: str | None = None) -> str:
    """Resolve the student_id for a given session_id.

    Priority:
        1. Explicit *student_id* override if provided.
        2. ``sessions.student_id`` looked up by *session_id*.
        3. Fall back to *session_id* itself when no row exists (backward
           compatibility for anonymous sessions).
    """
    if student_id:
        return student_id

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT student_id FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return row["student_id"] if row else session_id


async def get_student_profile(student_id: str) -> dict | None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        student = dict(row)
        preferences = json.loads(student["preferences_json"]) if student["preferences_json"] else {}

        cursor = await db.execute(
            "SELECT topic, score, evaluated_at FROM topic_scores WHERE student_id = ?",
            (student_id,),
        )
        topic_rows = await cursor.fetchall()
        topic_scores: dict[str, list[float]] = {}
        for t in topic_rows:
            topic = dict(t)
            topic_scores.setdefault(topic["topic"], []).append(topic["score"])

        cursor = await db.execute(
            "SELECT COUNT(*) as count FROM sessions WHERE student_id = ?",
            (student_id,),
        )
        count_row = await cursor.fetchone()
        session_count = dict(count_row)["count"] if count_row else 0

        return {
            "id": student["id"],
            "preferences": preferences,
            "topic_scores": topic_scores,
            "session_count": session_count,
        }


async def upsert_student_profile(student_id: str, preferences: dict) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO students (id, preferences_json)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET preferences_json = excluded.preferences_json
            """,
            (student_id, json.dumps(preferences)),
        )
        await db.commit()


async def save_session(session_id: str, student_id: str, intent: str, status: str) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "INSERT INTO sessions (id, student_id, intent, status) VALUES (?, ?, ?, ?)",
            (session_id, student_id, intent, status),
        )
        await db.commit()


async def save_evaluation(evaluation: dict) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO evaluations (id, session_id, student_id, question_id, topic, score,
                                     feedback_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation["id"],
                evaluation["session_id"],
                evaluation["student_id"],
                evaluation["question_id"],
                evaluation.get("topic"),
                evaluation.get("score"),
                evaluation.get("feedback_json"),
            ),
        )
        await db.commit()


async def get_topic_scores(student_id: str) -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT topic, score, evaluated_at FROM topic_scores WHERE student_id = ?",
            (student_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def upsert_topic_scores(student_id: str, scores: list[dict]) -> None:
    """Upsert per-topic scores for a student.

    Uses INSERT OR REPLACE so each (topic, student_id) row keeps only the
    latest score.  ``scores`` is a list of dicts, each with ``topic``
    (str) and ``score`` (float).
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.executemany(
            """
            INSERT OR REPLACE INTO topic_scores (topic, student_id, score, evaluated_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            [(s["topic"], student_id, s["score"]) for s in scores],
        )
        await db.commit()


async def compute_weak_topics(student_id: str, threshold: float = 6.0, limit: int = 3) -> list[str]:
    """Return topic names whose latest score is below *threshold*.

    Reads from ``topic_scores`` (fast, indexed lookup).  Results are
    sorted by score ascending so the weakest topics appear first.
    Only the first *limit* results are returned.
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT topic, score
               FROM topic_scores
               WHERE student_id = ? AND score < ?
               ORDER BY score ASC
               LIMIT ?""",
            (student_id, threshold, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r)["topic"] for r in rows]


async def get_recent_sessions(student_id: str, limit: int = 10) -> list[dict]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE student_id = ? ORDER BY started_at DESC LIMIT ?",
            (student_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_enriched_session_history(student_id: str, limit: int = 10) -> list[dict]:
    """Return recent sessions enriched with questions_answered and average_score."""
    sessions = await get_recent_sessions(student_id, limit)
    enriched: list[dict] = []
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        for ses in sessions:
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt, AVG(score) as avg_score "
                "FROM evaluations WHERE session_id = ? AND score IS NOT NULL",
                (ses["id"],),
            )
            row = await cursor.fetchone()
            questions_answered = row["cnt"] if row else 0
            avg_score = round(row["avg_score"], 2) if row and row["avg_score"] is not None else None
            enriched.append(
                {
                    "id": ses["id"],
                    "started_at": ses["started_at"],
                    "ended_at": ses["ended_at"],
                    "intent": ses["intent"],
                    "status": ses["status"],
                    "questions_answered": questions_answered,
                    "average_score": avg_score,
                }
            )
    return enriched


# ── Session lifecycle (Epic 9) ───────────────────────────────────────────────


async def create_session(
    student_id: str, name: str, description: str = "", session_id: str | None = None
) -> dict:
    """Create a named session and return its metadata."""
    session_id = session_id or str(uuid.uuid4())
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO sessions (id, student_id, name, description, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            (session_id, student_id, name, description),
        )
        await db.commit()
    return {
        "id": session_id,
        "student_id": student_id,
        "name": name,
        "description": description,
        "created_at": None,  # populated by get_session if needed
    }


async def list_sessions(student_id: str) -> list[dict]:
    """Return all sessions for a student ordered by most recent first."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, student_id, name, description, started_at as created_at, status "
            "FROM sessions WHERE student_id = ? ORDER BY started_at DESC, rowid DESC",
            (student_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_session(session_id: str) -> dict | None:
    """Return session details including file_count and progress summary."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, student_id, name, description, started_at as created_at, status "
            "FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        session = dict(row)

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM ingested_documents WHERE session_id = ?",
            (session_id,),
        )
        session["file_count"] = (await cursor.fetchone())["cnt"]

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt, AVG(score) as avg_score "
            "FROM evaluations WHERE session_id = ? AND score IS NOT NULL",
            (session_id,),
        )
        eval_row = await cursor.fetchone()
        session["exam_count"] = eval_row["cnt"] if eval_row else 0
        avg = eval_row["avg_score"] if eval_row else None
        session["average_score"] = round(avg, 2) if avg is not None else None

        return session


async def delete_session(session_id: str) -> None:
    """Delete a session, cascade its ingested_documents, and drop Chroma collection."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("DELETE FROM ingested_documents WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

    try:
        client = get_chroma_client()
        await asyncio.to_thread(client.delete_collection, f"session_{session_id}")
    except Exception:
        # Collection may not exist; don't fail the delete.
        logger = logging.getLogger(__name__)
        logger.warning("Could not drop ChromaDB collection for session %s", session_id)


async def insert_ingested_document(doc: dict) -> None:
    """Persist file metadata after a successful ingest."""
    ingested_at = doc.get("ingested_at") or datetime.now(UTC).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO ingested_documents
            (id, file_name, classification, topics_json, chunks_count, session_id, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["id"],
                doc["file_name"],
                doc.get("classification"),
                doc.get("topics_json"),
                doc.get("chunks_count", 0),
                doc.get("session_id"),
                ingested_at,
            ),
        )
        await db.commit()


async def list_session_files(session_id: str) -> list[dict]:
    """Return files for a session ordered by most recent first."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, file_name, classification, topics_json, chunks_count, "
            "ingested_at, session_id FROM ingested_documents "
            "WHERE session_id = ? ORDER BY ingested_at DESC, rowid DESC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_session_profile(session_id: str) -> dict | None:
    """Return per-session progress aggregated from evaluations.

    Includes topic scores (latest per topic within the session), weak topics
    (score < threshold), exam count, and average score.
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        if await cursor.fetchone() is None:
            return None

        cursor = await db.execute(
            """
            SELECT topic, score
            FROM evaluations
            WHERE session_id = ? AND topic IS NOT NULL AND score IS NOT NULL
            ORDER BY created_at DESC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()

    topic_scores: dict[str, list[float]] = {}
    latest_score: dict[str, float] = {}
    for row in rows:
        topic = row["topic"]
        score = row["score"]
        topic_scores.setdefault(topic, []).append(score)
        # First row per topic is the latest because of ORDER BY created_at DESC
        if topic not in latest_score:
            latest_score[topic] = score

    weak_topics = [
        topic for topic, score in sorted(latest_score.items(), key=lambda x: x[1])
        if score < 6.0
    ][:3]

    all_scores = [s for scores in topic_scores.values() for s in scores]
    average_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    return {
        "session_id": session_id,
        "topic_scores": topic_scores,
        "weak_topics": weak_topics,
        "exam_count": len(all_scores),
        "average_score": average_score,
    }
