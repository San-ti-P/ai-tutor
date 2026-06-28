"""TDD tests for Epic 9 Phase 2 — Session CRUD API (US-9.6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestSessionSchemaCrud:
    """T-007: schema CRUD functions."""

    async def test_create_session(self, tmp_path):
        """create_session inserts a row and returns session dict."""
        from src.config import settings
        from src.memory.schema import create_session, init_db

        db_path = tmp_path / "session_crud.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            session = await create_session("stu-1", "Cálculo", "Repaso final")

        assert session["name"] == "Cálculo"
        assert session["description"] == "Repaso final"
        assert session["student_id"] == "stu-1"
        assert "id" in session
        assert "created_at" in session

    async def test_list_sessions_ordered_by_recent(self, tmp_path):
        """list_sessions returns sessions for a student ordered by most recent."""
        from src.config import settings
        from src.memory.schema import create_session, init_db, list_sessions

        db_path = tmp_path / "session_list.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            s1 = await create_session("stu-1", "Álgebra", "")
            s2 = await create_session("stu-1", "Física", "")
            s3 = await create_session("stu-2", "Química", "")

            rows = await list_sessions("stu-1")

        assert len(rows) == 2
        assert rows[0]["id"] == s2["id"]
        assert rows[1]["id"] == s1["id"]
        assert s3["id"] not in {r["id"] for r in rows}

    async def test_get_session_includes_file_count_and_progress(self, tmp_path):
        """get_session returns file_count and progress summary."""
        from src.config import settings
        from src.memory.schema import (
            create_session,
            get_session,
            init_db,
            insert_ingested_document,
            save_evaluation,
        )

        db_path = tmp_path / "session_get.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            session = await create_session("stu-1", "Cálculo", "")
            await insert_ingested_document(
                {
                    "id": "doc-1",
                    "file_name": "apunte.pdf",
                    "classification": "apunte_teorico",
                    "topics_json": '["derivadas"]',
                    "chunks_count": 5,
                    "session_id": session["id"],
                }
            )
            await save_evaluation(
                {
                    "id": "eval-1",
                    "session_id": session["id"],
                    "student_id": "stu-1",
                    "question_id": "q-1",
                    "topic": "derivadas",
                    "score": 7.5,
                }
            )

            detail = await get_session(session["id"])

        assert detail is not None
        assert detail["id"] == session["id"]
        assert detail["file_count"] == 1
        assert detail["exam_count"] == 1
        assert detail["average_score"] == 7.5

    async def test_get_session_unknown_returns_none(self, tmp_path):
        """get_session returns None for unknown session id."""
        from src.config import settings
        from src.memory.schema import get_session, init_db

        db_path = tmp_path / "session_get_none.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            result = await get_session("no-such-session")
        assert result is None

    async def test_delete_session_cascade_files_and_drop_collection(self, tmp_path):
        """delete_session removes row, ingested_documents, and drops Chroma collection."""
        from src.config import settings
        from src.memory.schema import (
            create_session,
            delete_session,
            get_session,
            init_db,
            insert_ingested_document,
        )

        db_path = tmp_path / "session_delete.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            session = await create_session("stu-1", "Borrar", "")
            await insert_ingested_document(
                {
                    "id": "doc-del",
                    "file_name": "tmp.pdf",
                    "classification": "apunte_teorico",
                    "topics_json": "[]",
                    "chunks_count": 1,
                    "session_id": session["id"],
                }
            )

            mock_client = MagicMock()
            with patch("src.memory.schema.get_chroma_client") as mock_get_client:
                mock_get_client.return_value = mock_client
                await delete_session(session["id"])
                mock_client.delete_collection.assert_called_once_with(
                    f"session_{session['id']}"
                )

            assert await get_session(session["id"]) is None


class TestSessionApiEndpoints:
    """T-009: API endpoints."""

    def _client(self, tmp_path, **mocks):
        """Build a TestClient with router function mocks and a temp DB."""
        from src.config import settings

        db_path = tmp_path / "api_sessions.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.main import app

            client = TestClient(app)
            with patch.multiple("src.api.router", **mocks):
                yield client

    def test_create_session_endpoint(self, tmp_path):
        """POST /api/sessions creates a session."""
        mock_create = AsyncMock(
            return_value={
                "id": "sess-new",
                "name": "Cálculo",
                "description": "",
                "student_id": "stu-1",
                "created_at": "2026-06-26T12:00:00",
            }
        )
        mock_get = AsyncMock(
            return_value={
                "id": "sess-new",
                "name": "Cálculo",
                "description": "",
                "student_id": "stu-1",
                "created_at": "2026-06-26T12:00:00",
                "file_count": 0,
                "exam_count": 0,
                "average_score": None,
            }
        )
        for client in self._client(
            tmp_path, _create_session=mock_create, _get_session=mock_get
        ):
            response = client.post(
                "/api/sessions",
                json={"name": "Cálculo", "description": "", "student_id": "stu-1"},
            )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["id"] == "sess-new"
        assert data["name"] == "Cálculo"

    def test_list_sessions_endpoint(self, tmp_path):
        """GET /api/sessions?student_id= returns sessions."""
        mock_list = AsyncMock(
            return_value=[
                {
                    "id": "s1",
                    "name": "A",
                    "description": "",
                    "created_at": "2026-06-26",
                    "student_id": "stu-1",
                    "status": "active",
                },
            ]
        )
        for client in self._client(tmp_path, _list_sessions=mock_list):
            response = client.get("/api/sessions?student_id=stu-1")

        assert response.status_code == 200
        assert len(response.json()["data"]) == 1

    def test_get_session_endpoint(self, tmp_path):
        """GET /api/sessions/{id} returns session detail."""
        mock_get = AsyncMock(
            return_value={
                "id": "s1",
                "name": "A",
                "description": "",
                "student_id": "stu-1",
                "created_at": "2026-06-26",
                "file_count": 2,
                "exam_count": 1,
                "average_score": 8.0,
            }
        )
        for client in self._client(tmp_path, _get_session=mock_get):
            response = client.get("/api/sessions/s1")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["fileCount"] == 2

    def test_get_session_endpoint_404(self, tmp_path):
        """GET /api/sessions/{id} returns 404 when session missing."""
        mock_get = AsyncMock(return_value=None)
        for client in self._client(tmp_path, _get_session=mock_get):
            response = client.get("/api/sessions/missing")

        assert response.status_code == 404

    def test_delete_session_endpoint(self, tmp_path):
        """DELETE /api/sessions/{id} removes session."""
        mock_get = AsyncMock(
            return_value={
                "id": "s1",
                "name": "A",
                "description": "",
                "student_id": "stu-1",
                "created_at": "2026-06-26",
                "file_count": 0,
                "exam_count": 0,
                "average_score": None,
            }
        )
        mock_delete = AsyncMock(return_value=None)
        for client in self._client(
            tmp_path, _get_session=mock_get, _delete_session=mock_delete
        ):
            response = client.delete("/api/sessions/s1")

        assert response.status_code == 200

    def test_evaluate_endpoint_resolves_student_id_from_session(self, tmp_path):
        """POST /api/evaluate resolves student_id from session when not provided."""
        mock_get_session = AsyncMock(
            return_value={
                "id": "s-eval-1",
                "name": "Eval Session",
                "description": "",
                "student_id": "stu-eval-1",
                "created_at": "2026-06-26T12:00:00",
                "file_count": 0,
                "exam_count": 0,
                "average_score": None,
            }
        )
        mock_ensure_student = AsyncMock(return_value=None)

        mock_eval_results = [
            {
                "question_id": "q-1",
                "score": 8.0,
                "justification": "Correcta comprensión del concepto.",
                "conceptual_errors": [],
                "suggestions": [],
                "is_evaluable": True,
                "non_evaluable_reason": "",
                "requires_review": False,
            }
        ]

        # Patch the evaluator graph compilation to return mock results
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"evaluation_results": mock_eval_results}

        with (
            patch("src.api.router._get_session", mock_get_session),
            patch("src.api.router._ensure_student_exists", mock_ensure_student),
            patch(
                "src.tools._get_or_compile",
                return_value=mock_graph,
            ) as mock_get_or_compile,
        ):
            from src.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "s-eval-1",
                    "exam_id": "exam-1",
                    "answers": {"q-1": "La derivada es el límite..."},
                    "exam_questions": [
                        {
                            "id": "q-1",
                            "type": "open",
                            "prompt": "Define derivada.",
                            "topic": "cálculo",
                            "difficulty": "medium",
                            "baseAnswer": "Límite del cociente incremental.",
                        }
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert len(data["data"]) == 1
        # Verify graph was invoked with resolved student_id, not ""
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["student_id"] == "stu-eval-1"

    def test_evaluate_endpoint_uses_explicit_student_id(self, tmp_path):
        """POST /api/evaluate uses explicit studentId when provided in request."""
        mock_get_session = AsyncMock(
            return_value={
                "id": "s-eval-2",
                "name": "Eval Session 2",
                "description": "",
                "student_id": "stu-from-session",
                "created_at": "2026-06-26T12:00:00",
                "file_count": 0,
                "exam_count": 0,
                "average_score": None,
            }
        )
        mock_ensure_student = AsyncMock(return_value=None)

        mock_eval_results = [
            {
                "question_id": "q-1",
                "score": 7.0,
                "justification": "Bien pero incompleto.",
                "conceptual_errors": [],
                "suggestions": [],
                "is_evaluable": True,
                "non_evaluable_reason": "",
                "requires_review": False,
            }
        ]

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"evaluation_results": mock_eval_results}

        with (
            patch("src.api.router._get_session", mock_get_session),
            patch("src.api.router._ensure_student_exists", mock_ensure_student),
            patch("src.tools._get_or_compile", return_value=mock_graph),
        ):
            from src.main import app
            from fastapi.testclient import TestClient

            client = TestClient(app)
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "s-eval-2",
                    "exam_id": "exam-2",
                    "answers": {"q-1": "La derivada es..."},
                    "studentId": "stu-explicit",
                },
            )

        assert response.status_code == 200
        # Explicit student_id must win over session's student_id
        call_args = mock_graph.invoke.call_args[0][0]
        assert call_args["student_id"] == "stu-explicit"
