"""Tests for per-session profile endpoint (T-016..T-017)."""

import uuid
from unittest.mock import patch

import pytest

from src.config import settings
from src.memory.schema import create_session, get_session_profile


@pytest.fixture
async def db_session(tmp_path):
    """Yield an isolated in-memory DB session with migrations applied."""
    db_path = tmp_path / "session_profile.db"
    with patch.object(settings, "sqlite_db_path", str(db_path)):
        from src.memory.schema import init_db

        await init_db()
        session_id = str(uuid.uuid4())
        await create_session(session_id, "Materia", "", session_id=session_id)
        yield session_id


class TestGetSessionProfile:
    """T-016: get_session_profile aggregation."""

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_session(self):
        """GIVEN unknown session_id → THEN return None (maps to 404)."""
        result = await get_session_profile("no-such-session")
        assert result is None

    @pytest.mark.asyncio
    async def test_aggregates_topic_scores_and_weak_topics(self, db_session):
        """GIVEN evaluations for session → THEN topic_scores, weak_topics, avg."""
        session_id = db_session
        await _insert_evaluations(
            session_id,
            [
                {"topic": "cálculo", "score": 8.0, "created_at": "2026-06-26T10:00:00"},
                {"topic": "cálculo", "score": 7.0, "created_at": "2026-06-26T10:01:00"},
                {"topic": "álgebra", "score": 4.5, "created_at": "2026-06-26T10:02:00"},
            ],
        )

        profile = await get_session_profile(session_id)

        assert profile["topic_scores"]["cálculo"] == [7.0, 8.0]
        assert profile["topic_scores"]["álgebra"] == [4.5]
        assert profile["weak_topics"] == ["álgebra"]
        assert profile["exam_count"] == 3
        assert profile["average_score"] == pytest.approx(6.5, 0.01)

    @pytest.mark.asyncio
    async def test_filters_by_session_only(self, db_session):
        """GIVEN evaluations in two sessions → THEN only current session aggregated."""
        session_a = db_session
        session_b = str(uuid.uuid4())
        await create_session(session_b, "Otra", "", session_id=session_b)

        await _insert_evaluations(
            session_a,
            [{"topic": "cálculo", "score": 9.0, "created_at": "2026-06-26T10:00:00"}],
        )
        await _insert_evaluations(
            session_b,
            [{"topic": "física", "score": 3.0, "created_at": "2026-06-26T10:00:00"}],
        )

        profile = await get_session_profile(session_a)
        assert profile["topic_scores"] == {"cálculo": [9.0]}
        assert profile["exam_count"] == 1

    @pytest.mark.asyncio
    async def test_weak_topics_sorted_and_capped_at_three(self, db_session):
        """GIVEN >3 weak topics → THEN return 3 lowest."""
        session_id = db_session
        await _insert_evaluations(
            session_id,
            [
                {"topic": "a", "score": 5.0, "created_at": "2026-06-26T10:00:00"},
                {"topic": "b", "score": 3.0, "created_at": "2026-06-26T10:01:00"},
                {"topic": "c", "score": 4.0, "created_at": "2026-06-26T10:02:00"},
                {"topic": "d", "score": 2.0, "created_at": "2026-06-26T10:03:00"},
            ],
        )

        profile = await get_session_profile(session_id)
        assert profile["weak_topics"] == ["d", "b", "c"]

    @pytest.mark.asyncio
    async def test_no_evaluations_returns_empty_profile(self, db_session):
        """GIVEN session with no evaluations → THEN empty metrics."""
        profile = await get_session_profile(db_session)
        assert profile["topic_scores"] == {}
        assert profile["weak_topics"] == []
        assert profile["exam_count"] == 0
        assert profile["average_score"] is None


async def _insert_evaluations(session_id: str, rows: list[dict]) -> None:
    """Helper to insert evaluation rows for a session."""
    import aiosqlite

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        for row in rows:
            await db.execute(
                """
                INSERT INTO evaluations
                (id, session_id, student_id, question_id, topic, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_id,
                    session_id,
                    str(uuid.uuid4()),
                    row["topic"],
                    row["score"],
                    row["created_at"],
                ),
            )
        await db.commit()
