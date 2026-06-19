"""SQLite schema for student profiles, sessions, and evaluations."""

from __future__ import annotations

import json

import aiosqlite

from src.config import settings


async def init_db() -> None:
    """Initialize the SQLite database and create tables if they don't exist."""
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
        await db.commit()


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


async def compute_weak_topics(
    student_id: str, threshold: float = 6.0, limit: int = 3
) -> list[str]:
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
