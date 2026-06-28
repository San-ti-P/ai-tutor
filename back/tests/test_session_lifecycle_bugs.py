"""Tests for session lifecycle, status updates, exam counting, and cascading deletes."""

import uuid
import pytest
import aiosqlite
from unittest.mock import patch

from src.config import settings
from src.memory.schema import (
    create_session,
    get_session,
    insert_ingested_document,
    insert_generated_exam,
    update_session_status,
    delete_session,
    get_session_profile
)

@pytest.fixture
async def db_session(tmp_path):
    """Yield an isolated in-memory DB session with schema and migrations applied."""
    db_path = tmp_path / "session_lifecycle.db"
    with patch.object(settings, "sqlite_db_path", str(db_path)):
        from src.memory.schema import init_db
        await init_db()
        session_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        # create_session creates the session as 'empty'
        await create_session(student_id, "Sesión de prueba", "", session_id=session_id)
        yield session_id, student_id


@pytest.mark.asyncio
async def test_session_starts_empty_and_becomes_active_on_status_update(db_session):
    """Verify that a session starts as 'empty' and can be updated to 'active'."""
    session_id, student_id = db_session
    
    # 1. Check initial session state
    session = await get_session(session_id)
    assert session is not None
    assert session["status"] == "empty"
    assert session["file_count"] == 0
    assert session["exam_count"] == 0
    assert session["average_score"] is None

    # 2. Add ingested document and update status
    await insert_ingested_document({
        "id": str(uuid.uuid4()),
        "file_name": "apunte.pdf",
        "classification": "apunte",
        "topics_json": "[]",
        "chunks_count": 5,
        "session_id": session_id
    })
    await update_session_status(session_id, "active")

    # 3. Verify updated details
    session = await get_session(session_id)
    assert session["status"] == "active"
    assert session["file_count"] == 1


@pytest.mark.asyncio
async def test_exam_count_increments_on_generation_and_aggregates_correctly(db_session):
    """Verify that generated exams increment exam_count and are tracked correctly."""
    session_id, student_id = db_session

    # 1. Insert a generated exam
    exam_id = str(uuid.uuid4())
    await insert_generated_exam(exam_id, session_id, "Redes Neuronales", "difícil")

    # 2. Get session details and profile, check count
    session = await get_session(session_id)
    assert session["exam_count"] == 1

    profile = await get_session_profile(session_id)
    assert profile["exam_count"] == 1

    # 3. Try inserting the same exam (idempotent)
    await insert_generated_exam(exam_id, session_id, "Redes Neuronales", "difícil")
    session = await get_session(session_id)
    assert session["exam_count"] == 1


@pytest.mark.asyncio
async def test_average_score_calculated_from_multiple_evaluations(db_session):
    """Verify that evaluations under a single exam are grouped and average scores are computed correctly."""
    session_id, student_id = db_session

    # 1. Insert a generated exam
    exam_id = str(uuid.uuid4())
    await insert_generated_exam(exam_id, session_id, "Lógica", "medio")

    # 2. Insert question-level evaluations under the same exam
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        evals = [
            (str(uuid.uuid4()), session_id, student_id, "q1", "Lógica", 9.0, exam_id),
            (str(uuid.uuid4()), session_id, student_id, "q2", "Lógica", 7.0, exam_id),
        ]
        for ev in evals:
            await db.execute(
                """
                INSERT INTO evaluations (id, session_id, student_id, question_id, topic, score, exam_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ev
            )
        await db.commit()

    # 3. Retrieve session metrics: exam_count should be 1 (grouped), average_score should be 8.0
    session = await get_session(session_id)
    assert session["exam_count"] == 1
    assert session["average_score"] == 8.0

    profile = await get_session_profile(session_id)
    assert profile["exam_count"] == 1
    assert profile["average_score"] == 8.0


@pytest.mark.asyncio
async def test_delete_session_cascades_properly(db_session):
    """Verify that deleting a session cascades and deletes its documents, exams, and evaluations."""
    session_id, student_id = db_session

    # 1. Insert a document, an exam, and an evaluation
    doc_id = str(uuid.uuid4())
    await insert_ingested_document({
        "id": doc_id,
        "file_name": "apunte.pdf",
        "classification": "apunte",
        "topics_json": "[]",
        "chunks_count": 5,
        "session_id": session_id
    })

    exam_id = str(uuid.uuid4())
    await insert_generated_exam(exam_id, session_id, "IA", "fácil")

    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        await db.execute(
            """
            INSERT INTO evaluations (id, session_id, student_id, question_id, topic, score, exam_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), session_id, student_id, "q1", "IA", 10.0, exam_id)
        )
        await db.commit()

    # 2. Check counts inside the database before delete
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT COUNT(*) as c FROM ingested_documents")).fetchone()
        assert row["c"] == 1
        row = await (await db.execute("SELECT COUNT(*) as c FROM exams")).fetchone()
        assert row["c"] == 1
        row = await (await db.execute("SELECT COUNT(*) as c FROM evaluations")).fetchone()
        assert row["c"] == 1

    # 3. Delete session
    await delete_session(session_id)

    # 4. Check counts inside the database after delete - all should be 0
    async with aiosqlite.connect(settings.sqlite_db_path) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute("SELECT COUNT(*) as c FROM sessions WHERE id = ?", (session_id,))).fetchone()
        assert row["c"] == 0
        row = await (await db.execute("SELECT COUNT(*) as c FROM ingested_documents WHERE session_id = ?", (session_id,))).fetchone()
        assert row["c"] == 0
        row = await (await db.execute("SELECT COUNT(*) as c FROM exams WHERE session_id = ?", (session_id,))).fetchone()
        assert row["c"] == 0
        row = await (await db.execute("SELECT COUNT(*) as c FROM evaluations WHERE session_id = ?", (session_id,))).fetchone()
        assert row["c"] == 0
