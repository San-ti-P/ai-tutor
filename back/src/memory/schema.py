"""SQLite schema for student profiles, sessions, and evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from src.config import settings
from src.rag import get_chroma_client


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    """Return True if *column* already exists in *table*."""
    # PRAGMA does not accept parameter binding for the table name.
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return any(row[1] == column for row in rows)


async def _table_exists(db: aiosqlite.Connection, table: str) -> bool:
    """Return True if *table* exists in the database."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    row = await cursor.fetchone()
    return row is not None


async def _run_migrations(db: aiosqlite.Connection) -> None:
    """Apply additive schema migrations idempotently."""
    if not await _column_exists(db, "sessions", "name"):
        await db.execute("ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT ''")
    if not await _column_exists(db, "sessions", "description"):
        await db.execute("ALTER TABLE sessions ADD COLUMN description TEXT DEFAULT ''")
    if not await _column_exists(db, "ingested_documents", "session_id"):
        await db.execute(
            "ALTER TABLE ingested_documents ADD COLUMN session_id TEXT REFERENCES sessions(id)"
        )
    if not await _column_exists(db, "ingested_documents", "topic_tree_json"):
        await db.execute(
            "ALTER TABLE ingested_documents ADD COLUMN topic_tree_json TEXT DEFAULT '{}'"
        )
    if not await _column_exists(db, "evaluations", "exam_id"):
        await db.execute("ALTER TABLE evaluations ADD COLUMN exam_id TEXT DEFAULT ''")
    if not await _column_exists(db, "ingested_documents", "topic_descriptions_json"):
        await db.execute(
            "ALTER TABLE ingested_documents ADD COLUMN topic_descriptions_json TEXT DEFAULT '{}'"
        )
    await db.execute("""
        CREATE TABLE IF NOT EXISTS exams (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    # Migration: chat_messages table (idempotent)
    if not await _table_exists(db, "chat_messages"):
        await db.execute("""
            CREATE TABLE chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        await db.execute("""
            CREATE INDEX idx_chat_messages_session_created
                ON chat_messages(session_id, created_at DESC)
        """)
    # Migration: topic_scores.session_id (per-session isolation)
    if not await _column_exists(db, "topic_scores", "session_id"):
        await db.execute("""
            CREATE TABLE topic_scores_new (
                topic TEXT NOT NULL,
                student_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL,
                evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (topic, student_id, session_id),
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
        await db.execute("""
            INSERT INTO topic_scores_new (topic, student_id, session_id, score, evaluated_at)
            SELECT topic, student_id, '', score, evaluated_at FROM topic_scores
        """)
        await db.execute("DROP TABLE topic_scores")
        await db.execute("ALTER TABLE topic_scores_new RENAME TO topic_scores")


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

            CREATE TABLE IF NOT EXISTS exams (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
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
                session_id TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL,
                evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (topic, student_id, session_id),
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
        cursor = await db.execute("SELECT student_id FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return row["student_id"] if row else session_id


async def get_student_profile(student_id: str, session_id: str | None = None) -> dict[str, Any] | None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        student = dict(row)
        preferences = json.loads(student["preferences_json"]) if student["preferences_json"] else {}

        if session_id is not None:
            cursor = await db.execute(
                "SELECT topic, score, evaluated_at FROM topic_scores "
                "WHERE student_id = ? AND session_id = ?",
                (student_id, session_id),
            )
        else:
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


async def upsert_student_profile(student_id: str, preferences: dict[str, Any]) -> None:
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


async def save_evaluation(evaluation: dict[str, Any]) -> None:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO evaluations (id, session_id, student_id, question_id, exam_id,
                                     topic, score, feedback_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation["id"],
                evaluation["session_id"],
                evaluation["student_id"],
                evaluation["question_id"],
                evaluation.get("exam_id", ""),
                evaluation.get("topic"),
                evaluation.get("score"),
                evaluation.get("feedback_json", ""),
            ),
        )
        await db.commit()


async def get_session_evaluations(session_id: str) -> list[dict[str, Any]]:
    """Return evaluations for a session grouped by exam_id, newest first.

    Each group contains the exam_id, created_at of the first question evaluated,
    average score, and the list of per-question feedback dicts parsed from
    feedback_json.
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT id, question_id, exam_id, topic, score, feedback_json, created_at
            FROM evaluations
            WHERE session_id = ?
            ORDER BY exam_id, created_at ASC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()

    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        exam_id = row_dict["exam_id"] or "unknown"
        if exam_id not in groups:
            groups[exam_id] = {
                "exam_id": exam_id,
                "created_at": row_dict["created_at"],
                "results": [],
            }
        feedback = {}
        if row_dict["feedback_json"]:
            try:
                feedback = json.loads(row_dict["feedback_json"])
            except (json.JSONDecodeError, ValueError):
                feedback = {}
        groups[exam_id]["results"].append({
            "questionId": row_dict["question_id"],
            "score": row_dict["score"] or 0.0,
            "justification": feedback.get("justification", ""),
            "conceptualErrors": feedback.get("conceptual_errors", []),
            "suggestions": feedback.get("suggestions", []),
            "isEvaluable": feedback.get("is_evaluable", True),
            "nonEvaluableReason": feedback.get("non_evaluable_reason", ""),
            "requiresReview": feedback.get("requires_review", False),
            "judgeScore": feedback.get("judge_score"),
            "questionText": feedback.get("question_text", ""),
            "userAnswer": feedback.get("student_answer", ""),
            "sourceChunks": feedback.get("source_chunks", []),
        })

    result_list = list(groups.values())
    for g in result_list:
        scores = [r["score"] for r in g["results"] if r["isEvaluable"]]
        g["averageScore"] = round(sum(scores) / len(scores), 2) if scores else None

    result_list.sort(key=lambda g: g["created_at"], reverse=True)
    return result_list


async def get_topic_scores(student_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        if session_id is not None:
            cursor = await db.execute(
                "SELECT topic, score, evaluated_at FROM topic_scores "
                "WHERE student_id = ? AND session_id = ?",
                (student_id, session_id),
            )
        else:
            cursor = await db.execute(
                "SELECT topic, score, evaluated_at FROM topic_scores WHERE student_id = ?",
                (student_id,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def upsert_topic_scores(student_id: str, session_id: str, scores: list[dict[str, Any]]) -> None:
    """Upsert per-topic scores for a student within a session.

    Uses INSERT OR REPLACE so each (topic, student_id, session_id) row keeps
    only the latest score.  ``scores`` is a list of dicts, each with ``topic``
    (str) and ``score`` (float).
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.executemany(
            """
            INSERT OR REPLACE INTO topic_scores (topic, student_id, session_id, score, evaluated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            [(s["topic"], student_id, session_id, s["score"]) for s in scores],
        )
        await db.commit()


async def compute_weak_topics(
    student_id: str, session_id: str | None = None, threshold: float = 6.0, limit: int = 3
) -> list[str]:
    """Return topic names whose latest score is below *threshold*.

    Reads from ``topic_scores`` (fast, indexed lookup).  Results are
    sorted by score ascending so the weakest topics appear first.
    Only the first *limit* results are returned.

    When *session_id* is provided, only scores from that session are
    considered.  When None (default), aggregates across all sessions.
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        if session_id is not None:
            cursor = await db.execute(
                """SELECT topic, score
                   FROM topic_scores
                   WHERE student_id = ? AND session_id = ? AND score < ?
                   ORDER BY score ASC
                   LIMIT ?""",
                (student_id, session_id, threshold, limit),
            )
        else:
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


async def get_recent_sessions(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE student_id = ? ORDER BY started_at DESC LIMIT ?",
            (student_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_enriched_session_history(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return recent sessions enriched with questions_answered and average_score."""
    sessions = await get_recent_sessions(student_id, limit)
    enriched: list[dict[str, Any]] = []
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


async def ensure_student_exists(student_id: str) -> None:
    """Create a student row if one does not already exist (idempotent)."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(
            "INSERT OR IGNORE INTO students (id) VALUES (?)",
            (student_id,),
        )
        await db.commit()


async def create_session(
    student_id: str, name: str, description: str = "", session_id: str | None = None
) -> dict[str, Any]:
    """Create a named session and return its metadata."""
    session_id = session_id or str(uuid.uuid4())
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO sessions (id, student_id, name, description, status)
            VALUES (?, ?, ?, ?, 'empty')
            """,
            (session_id, student_id, name, description),
        )
        await db.commit()
    return {
        "id": session_id,
        "student_id": student_id,
        "name": name,
        "description": description,
        "created_at": datetime.now(UTC).isoformat() + "Z",
        "status": "empty",
        "file_count": 0,
        "exam_count": 0,
        "average_score": None,
    }


async def list_sessions(student_id: str) -> list[dict[str, Any]]:
    """Return all sessions for a student ordered by most recent first."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT 
                s.id, 
                s.student_id, 
                s.name, 
                s.description, 
                s.started_at as created_at, 
                s.status,
                COALESCE(d.file_cnt, 0) as file_count,
                COALESCE(e.exam_cnt, 0) as exam_count,
                e.avg_score as average_score
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) as file_cnt 
                FROM ingested_documents 
                GROUP BY session_id
            ) d ON s.id = d.session_id
            LEFT JOIN (
                SELECT session_id, COUNT(DISTINCT exam_id) as exam_cnt, AVG(score) as avg_score
                FROM (
                    SELECT session_id, COALESCE(NULLIF(exam_id, ''), id) AS exam_id, score FROM evaluations WHERE score IS NOT NULL
                    UNION
                    SELECT session_id, id AS exam_id, NULL AS score FROM exams
                )
                GROUP BY session_id
            ) e ON s.id = e.session_id
            WHERE s.student_id = ?
            ORDER BY s.started_at DESC, s.rowid DESC
            """,
            (student_id,),
        )
        rows = await cursor.fetchall()
        sessions = []
        for row in rows:
            session = dict(row)
            if session["average_score"] is not None:
                session["average_score"] = round(session["average_score"], 2)
            sessions.append(session)
        return sessions


async def get_session(session_id: str) -> dict[str, Any] | None:
    """Return session details including file_count and progress summary."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT 
                s.id, 
                s.student_id, 
                s.name, 
                s.description, 
                s.started_at as created_at, 
                s.status,
                COALESCE(d.file_cnt, 0) as file_count,
                COALESCE(e.exam_cnt, 0) as exam_count,
                e.avg_score as average_score
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) as file_cnt 
                FROM ingested_documents 
                GROUP BY session_id
            ) d ON s.id = d.session_id
            LEFT JOIN (
                SELECT session_id, COUNT(DISTINCT exam_id) as exam_cnt, AVG(score) as avg_score
                FROM (
                    SELECT session_id, COALESCE(NULLIF(exam_id, ''), id) AS exam_id, score FROM evaluations WHERE score IS NOT NULL
                    UNION
                    SELECT session_id, id AS exam_id, NULL AS score FROM exams
                )
                GROUP BY session_id
            ) e ON s.id = e.session_id
            WHERE s.id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        session = dict(row)
        if session["average_score"] is not None:
            session["average_score"] = round(session["average_score"], 2)
        return session


async def delete_session(session_id: str) -> None:
    """Delete a session, cascade its ingested_documents, exams, evaluations, and drop Chroma collection."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("DELETE FROM ingested_documents WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM exams WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM evaluations WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()

    try:
        client = get_chroma_client()
        await asyncio.to_thread(client.delete_collection, f"session_{session_id}")
    except Exception:
        # Collection may not exist; don't fail the delete.
        logger = logging.getLogger(__name__)
        logger.warning("Could not drop ChromaDB collection for session %s", session_id)


async def rename_session(session_id: str, name: str, description: str | None = None) -> dict[str, Any] | None:
    """Rename a session and optionally update its description. Returns updated row or None."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        if description is not None:
            await db.execute(
                "UPDATE sessions SET name = ?, description = ? WHERE id = ?",
                (name, description, session_id),
            )
        else:
            await db.execute(
                "UPDATE sessions SET name = ? WHERE id = ?",
                (name, session_id),
            )
        await db.commit()
    return await get_session(session_id)


async def update_session_status(session_id: str, status: str) -> None:
    """Update the status of a session."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            "UPDATE sessions SET status = ? WHERE id = ?",
            (status, session_id),
        )
        await db.commit()


async def insert_generated_exam(exam_id: str, session_id: str, topic: str, difficulty: str) -> None:
    """Insert a generated exam record."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO exams (id, session_id, topic, difficulty)
            VALUES (?, ?, ?, ?)
            """,
            (exam_id, session_id, topic, difficulty),
        )
        await db.commit()


async def insert_ingested_document(doc: dict[str, Any]) -> None:
    """Persist file metadata after a successful ingest."""
    ingested_at = doc.get("ingested_at") or datetime.now(UTC).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO ingested_documents
            (id, file_name, classification, topics_json, topic_tree_json,
             topic_descriptions_json, chunks_count, session_id, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["id"],
                doc["file_name"],
                doc.get("classification"),
                doc.get("topics_json"),
                doc.get("topic_tree_json", "{}"),
                doc.get("topic_descriptions_json", "{}"),
                doc.get("chunks_count", 0),
                doc.get("session_id"),
                ingested_at,
            ),
        )
        await db.commit()


async def list_session_files(session_id: str) -> list[dict[str, Any]]:
    """Return files for a session ordered by most recent first."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, file_name, classification, topics_json, topic_tree_json, "
            "topic_descriptions_json, chunks_count, "
            "ingested_at, session_id FROM ingested_documents "
            "WHERE session_id = ? ORDER BY ingested_at DESC, rowid DESC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_session_profile(session_id: str) -> dict[str, Any] | None:
    """Return per-session progress aggregated from evaluations.

    Includes topic scores (all scores per topic within the session), weak topics
    (latest score < threshold), exam count, and average score.
    """
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        if await cursor.fetchone() is None:
            return None

        cursor = await db.execute(
            """
            SELECT COUNT(DISTINCT exam_id) as cnt FROM (
                SELECT COALESCE(NULLIF(exam_id, ''), id) AS exam_id FROM evaluations WHERE session_id = ?
                UNION
                SELECT id AS exam_id FROM exams WHERE session_id = ?
            )
            """,
            (session_id, session_id),
        )
        row = await cursor.fetchone()
        exam_count = row["cnt"] if row else 0

        cursor = await db.execute(
            "SELECT AVG(score) as avg_score FROM evaluations WHERE session_id = ? AND score IS NOT NULL",
            (session_id,),
        )
        summary = await cursor.fetchone()
        average_score = (
            round(summary["avg_score"], 2) if summary and summary["avg_score"] is not None else None
        )

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
        topic for topic, score in sorted(latest_score.items(), key=lambda x: x[1]) if score < 6.0
    ][:3]

    return {
        "session_id": session_id,
        "topic_scores": topic_scores,
        "weak_topics": weak_topics,
        "exam_count": exam_count,
        "average_score": average_score,
    }


# ── Chat message history ─────────────────────────────────────────────────────


async def save_chat_message(message: dict[str, Any]) -> None:
    """Persist a single chat message (user or assistant) for a session.

    ``message`` must contain: id, session_id, role, content.
    Optional: metadata_json (JSON string), created_at (ISO string).
    Silently skips insertion when session_id does not exist.
    """
    created_at = message.get("created_at") or datetime.now(UTC).isoformat()
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        # Verify session exists before inserting to avoid FK violations
        cursor = await db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (message["session_id"],)
        )
        if await cursor.fetchone() is None:
            return
        await db.execute(
            """
            INSERT OR IGNORE INTO chat_messages
                (id, session_id, role, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message["id"],
                message["session_id"],
                message["role"],
                message["content"],
                message.get("metadata_json", "{}"),
                created_at,
            ),
        )
        await db.commit()


async def get_chat_messages(
    session_id: str,
    limit: int = 10,
    before_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return *limit* messages for *session_id*, ordered newest-first.

    Cursor-based pagination: when *before_id* is provided, returns messages
    whose ``created_at`` is strictly older than the message with that id.
    Returns at most ``min(limit, 50)`` rows.
    """
    limit = min(max(1, limit), 50)
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        if before_id:
            cursor = await db.execute(
                """
                SELECT id, session_id, role, content, metadata_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                  AND created_at < (
                      SELECT created_at FROM chat_messages WHERE id = ?
                  )
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, before_id, limit),
            )
        else:
            cursor = await db.execute(
                """
                SELECT id, session_id, role, content, metadata_json, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_chat_message_count(session_id: str) -> int:
    """Return the total number of chat messages for *session_id*."""
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
