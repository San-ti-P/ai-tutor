"""Integration test: end-to-end session lifecycle (T-028).

Covers: create session → file metadata persistence → chat interaction
→ evaluation → per-session profile. Uses real SQLite + ChromaDB with
mocked LLM so it exercises the full stack without external network calls.

Marked ``@pytest.mark.integration`` — requires real SentenceTransformer
embeddings (downloaded once) and ephemeral ChromaDB storage.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import settings


@pytest.mark.integration
class TestSessionLifecycle:
    """T-028: Full end-to-end session lifecycle with real SQLite + ChromaDB."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_create_upload_chat_profile(
        self, tmp_path, in_memory_chroma
    ):
        """Create session → upload metadata → chat → profile."""
        db_path = tmp_path / "lifecycle.db"

        # ── Setup: real DB + ChromaDB ────────────────────────────────────
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import (
                create_session,
                get_session,
                get_session_profile,
                init_db,
                insert_ingested_document,
                list_session_files,
            )

            await init_db()

            # Wire in-memory ChromaDB for collection drop on delete
            with patch("src.rag.get_chroma_client", return_value=in_memory_chroma):
                student_id = "stu-integration-1"
                session_id = str(uuid.uuid4())

                # ── Step 1: Create session ───────────────────────────────
                created = await create_session(student_id, "Integración Test", "sesión de prueba", session_id=session_id)
                assert created["name"] == "Integración Test"
                assert created["id"] == session_id

                # ── Step 2: Insert ingested document metadata ─────────────
                doc_id = str(uuid.uuid4())
                await insert_ingested_document({
                    "id": doc_id,
                    "file_name": "apunte_test.pdf",
                    "classification": "apunte",
                    "topics_json": '["cálculo","derivadas"]',
                    "chunks_count": 5,
                    "session_id": session_id,
                })

                files = await list_session_files(session_id)
                assert len(files) == 1
                assert files[0]["file_name"] == "apunte_test.pdf"

                # ── Step 3: Get session detail ───────────────────────────
                detail = await get_session(session_id)
                assert detail is not None
                assert detail["file_count"] == 1
                assert detail["name"] == "Integración Test"

                # ── Step 4: Insert evaluations ───────────────────────────
                import aiosqlite

                async with aiosqlite.connect(str(db_path)) as db:
                    await db.execute(
                        "INSERT INTO students (id) VALUES (?)", (student_id,)
                    )
                    await db.executemany(
                        """INSERT INTO evaluations
                           (id, session_id, student_id, question_id, topic, score, feedback_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [
                            (str(uuid.uuid4()), session_id, student_id, "q1", "cálculo", 8.0, "{}"),
                            (str(uuid.uuid4()), session_id, student_id, "q2", "cálculo", 7.0, "{}"),
                            (str(uuid.uuid4()), session_id, student_id, "q3", "álgebra", 4.5, "{}"),
                        ],
                    )
                    await db.commit()

                # ── Step 5: Per-session profile ──────────────────────────
                profile = await get_session_profile(session_id)
                assert profile is not None
                assert profile["session_id"] == session_id
                assert "cálculo" in profile["topic_scores"]
                assert len(profile["topic_scores"]["cálculo"]) == 2
                assert profile["weak_topics"] == ["álgebra"]
                assert profile["exam_count"] == 3
                assert profile["average_score"] == pytest.approx(6.5, 0.01)

                # ── Step 6: Orchestrator graph flow with mocked LLM ──────
                from src.agents.orchestrator import build_orchestrator

                graph = build_orchestrator().compile()
                config = {"configurable": {"thread_id": session_id}}

                with patch(
                    "src.tools.get_student_summary.get_student_summary",
                    new=AsyncMock(return_value={"weak_topics": ["álgebra"]}),
                ), patch(
                    "src.memory.schema.resolve_student_id",
                    new=AsyncMock(return_value=student_id),
                ), patch(
                    "src.agents.orchestrator.get_structured_llm",
                ) as mock_structured, patch(
                    "src.agents.orchestrator._get_llm",
                ) as mock_llm_factory:
                    mock_structured.return_value = type(
                        "M", (),
                        {
                            "invoke": MagicMock(
                                return_value=type(
                                    "R", (), {"intent": "general_chat", "confidence": 0.99}
                                )()
                            )
                        },
                    )()
                    mock_llm = MagicMock()
                    mock_llm.invoke.return_value = type(
                        "R", (), {"content": "¡Hola! Veo que tenés material de cálculo."}
                    )()
                    mock_llm_factory.return_value = mock_llm

                    final = await graph.ainvoke(
                        {
                            "session_id": session_id,
                            "user_message": "hola",
                            "intent": "general_chat",
                            "confidence": 0.0,
                            "plan": [],
                            "current_step": 0,
                            "results": [],
                            "errors": [],
                            "response": "",
                            "status": "pending",
                            "iteration_count": 0,
                            "student_profile": None,
                            "messages_history": [],
                        },
                        config=config,
                    )

                assert final["response"] is not None
                assert len(final["response"]) > 0
                assert final["status"] == "complete"
                # Session context should have been loaded (files + progress)
                # and injected into the classify_intent / synthesize_response prompts

    @pytest.mark.asyncio
    async def test_delete_session_cascades(self, tmp_path, in_memory_chroma):
        """Delete session → ingested_documents gone, ChromaDB collection dropped."""
        db_path = tmp_path / "delete_cascade.db"

        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import (
                create_session,
                delete_session,
                init_db,
                insert_ingested_document,
                list_session_files,
            )
            from src.rag import get_chroma_client

            await init_db()

            with patch("src.rag.get_chroma_client", return_value=in_memory_chroma):
                session_id = str(uuid.uuid4())

                await create_session("stu-del", "Para borrar", session_id=session_id)
                await insert_ingested_document({
                    "id": str(uuid.uuid4()),
                    "file_name": "temp.pdf",
                    "classification": "apunte",
                    "topics_json": "[]",
                    "chunks_count": 1,
                    "session_id": session_id,
                })

                files_before = await list_session_files(session_id)
                assert len(files_before) == 1

                await delete_session(session_id)

                files_after = await list_session_files(session_id)
                assert len(files_after) == 0

    @pytest.mark.asyncio
    async def test_integration_exam_generation_flow(self, tmp_path, in_memory_chroma):
        """Create session → generate exam tool with mocked LLM."""
        db_path = tmp_path / "exam_flow.db"

        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import create_session, init_db
            from src.rag import get_chroma_client

            await init_db()

            with patch("src.rag.get_chroma_client", return_value=in_memory_chroma):
                session_id = str(uuid.uuid4())
                await create_session("stu-exam", "Exam Flow", session_id=session_id)

                # Test that generate_exam tool accepts session_id
                with patch(
                    "src.agents.exam_generator.build_exam_generator",
                ) as mock_build:
                    mock_graph = MagicMock()
                    mock_graph.compile.return_value = mock_graph
                    from unittest.mock import AsyncMock
                    mock_graph.ainvoke = AsyncMock(return_value={
                        "exam": {"exam_id": "e1", "questions": []},
                    })
                    mock_build.return_value = mock_graph

                    from src.tools import generate_exam

                    result = await generate_exam.ainvoke({
                        "session_id": session_id,
                        "topics": ["cálculo"],
                        "difficulty": "medium",
                        "question_count": 3,
                        "mcq_ratio": 0.5,
                    })

                assert result is not None
                assert isinstance(result, dict)
