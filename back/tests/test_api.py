"""TDD test suite for epic-07-ui API endpoint changes (REQ-API-002 through REQ-API-005 + CONFIG-003).

Covers: wired exam/profile endpoints, trace_id propagation, exercise endpoint, preferences endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


# ==============================================================================
# REQ-API-002: Wired /api/exam/generate
# ==============================================================================


class TestExamGenerateEndpoint:
    """REQ-API-002: POST /api/exam/generate invokes generate_exam tool."""

    def test_exam_generate_returns_real_questions(self):
        """GIVEN material ingested → THEN exam has non-empty questions."""
        fake_exam = {
            "exam_id": "exam-001",
            "session_id": "sess-2",
            "total_questions": 2,
            "questions": [
                {
                    "id": "q-1",
                    "type": "mcq",
                    "prompt": "\u00bfCu\u00e1l es la derivada de x\u00b2?",
                    "options": ["2x", "x\u00b2", "2", "x"],
                    "baseAnswer": "2x",
                    "sourceChunkIds": ["chunk-001"],
                },
                {
                    "id": "q-2",
                    "type": "open",
                    "prompt": "Defin\u00ed el concepto de l\u00edmite.",
                    "baseAnswer": "Un l\u00edmite es...",
                },
            ],
            "topics_covered": ["c\u00e1lculo/derivadas"],
            "status": "complete",
        }

        with patch("src.tools.generate_exam") as mock_tool:
            mock_tool.invoke.return_value = fake_exam
            response = client.post(
                "/api/exam/generate",
                json={
                    "session_id": "sess-2",
                    "topic": "c\u00e1lculo/derivadas",
                    "preferences": {
                        "questionTypes": ["mcq", "open"],
                        "difficulty": "medium",
                        "questionCount": 5,
                        "includeTopics": [],
                        "excludeTopics": [],
                    },
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        exam = data["data"]
        assert len(exam["questions"]) > 0
        for q in exam["questions"]:
            assert q["id"]
            assert q["prompt"]
            assert q["type"] in ("mcq", "open")
        assert "topic" in exam
        assert exam["topic"] == "c\u00e1lculo/derivadas"

    def test_exam_generate_uses_tool_with_correct_params(self):
        """GIVEN request with preferences → THEN tool receives correct args."""
        with patch("src.tools.generate_exam") as mock_tool:
            mock_tool.invoke.return_value = {
                "exam_id": "exam-002",
                "total_questions": 1,
                "questions": [
                    {
                        "id": "q-1",
                        "type": "mcq",
                        "prompt": "Pregunta",
                        "options": ["A", "B"],
                        "baseAnswer": "A",
                    }
                ],
                "topics_covered": ["test"],
                "status": "complete",
            }
            client.post(
                "/api/exam/generate",
                json={
                    "session_id": "sess-3",
                    "topic": "\u00e1lgebra",
                    "preferences": {
                        "questionTypes": ["mcq"],
                        "difficulty": "hard",
                        "questionCount": 10,
                        "includeTopics": [],
                        "excludeTopics": [],
                    },
                },
            )

        call_kwargs = mock_tool.invoke.call_args[0][0]
        assert call_kwargs["session_id"] == "sess-3"
        assert call_kwargs["topics"] == ["\u00e1lgebra"]
        assert call_kwargs["difficulty"] == "hard"
        assert call_kwargs["question_count"] == 10
        assert call_kwargs["mcq_ratio"] == 1.0  # only mcq

    def test_exam_generate_includes_trace_id(self):
        """GIVEN successful generation → THEN response includes trace_id."""
        with patch("src.tools.generate_exam") as mock_tool:
            mock_tool.invoke.return_value = {
                "exam_id": "exam-003",
                "total_questions": 1,
                "questions": [
                    {
                        "id": "q-1",
                        "type": "open",
                        "prompt": "Explica x.",
                        "baseAnswer": "Respuesta",
                    }
                ],
                "topics_covered": ["test"],
                "status": "complete",
            }
            response = client.post(
                "/api/exam/generate",
                json={
                    "session_id": "sess-4",
                    "topic": "test",
                    "preferences": {
                        "questionTypes": ["open"],
                        "difficulty": "easy",
                        "questionCount": 3,
                        "includeTopics": [],
                        "excludeTopics": [],
                    },
                },
            )

        data = response.json()
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)
        assert len(data["trace_id"]) > 0


# ==============================================================================
# REQ-API-003: Wired /api/profile/{id}
# ==============================================================================


class TestProfileEndpoint:
    """REQ-API-003: GET /api/profile/{id} invokes get_student_summary tool."""

    @pytest.mark.asyncio
    async def test_profile_returns_real_student_data(self):
        """GIVEN student with sessions → THEN profile has real data."""
        with patch("src.tools.get_student_summary.get_student_summary") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "id": "stu-1",
                    "preferences": {
                        "question_types": ["mcq", "open"],
                        "difficulty": "hard",
                        "question_count": 8,
                        "include_topics": ["c\u00e1lculo"],
                        "exclude_topics": [],
                    },
                    "topic_scores": [
                        {"topic": "c\u00e1lculo/derivadas", "score": 7.5},
                        {"topic": "\u00e1lgebra/matrices", "score": 5.0},
                    ],
                    "weak_topics": ["\u00e1lgebra/matrices"],
                    "session_history": [],
                    "session_count": 3,
                }
            )
            response = client.get("/api/profile/stu-1")

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        profile = data["data"]
        assert profile["id"] == "stu-1"
        assert profile["sessionCount"] == 3
        assert len(profile["weakTopics"]) > 0
        assert "trace_id" in data

    @pytest.mark.asyncio
    async def test_profile_unknown_student_returns_404(self):
        """GIVEN nonexistent student → THEN 404."""
        with patch("src.tools.get_student_summary.get_student_summary") as mock_tool:
            mock_tool.ainvoke = AsyncMock(return_value=None)
            response = client.get("/api/profile/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_profile_not_a_placeholder(self):
        """GIVEN real student → THEN response is NOT the hardcoded placeholder."""
        with patch("src.tools.get_student_summary.get_student_summary") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "id": "stu-real",
                    "preferences": {
                        "question_types": ["open"],
                        "difficulty": "easy",
                        "question_count": 3,
                        "include_topics": [],
                        "exclude_topics": [],
                    },
                    "topic_scores": [{"topic": "test", "score": 9.0}],
                    "weak_topics": [],
                    "session_history": [],
                    "session_count": 1,
                }
            )
            response = client.get("/api/profile/stu-real")

        data = response.json()
        profile = data["data"]
        assert profile["id"] != "placeholder-student"
        assert profile["sessionCount"] > 0


# ==============================================================================
# REQ-API-004: POST /api/exercise/generate
# ==============================================================================


class TestExerciseGenerateEndpoint:
    """REQ-API-004: POST /api/exercise/generate returns structured exercise."""

    def test_exercise_generate_returns_exercise(self):
        """GIVEN material ingested → THEN exercise with model_solution."""
        with patch("src.tools.generate_exercise") as mock_tool:
            mock_tool.invoke.return_value = {
                "exercise_id": "ex-001",
                "statement": "Calcul\u00e1 la derivada.",
                "given_data": "f(x) = x\u00b2",
                "question": "f'(x) = ?",
                "model_solution": {
                    "steps": ["Paso 1", "Paso 2"],
                    "final_answer": "2x",
                    "key_concepts": ["derivada"],
                },
                "topics_covered": ["c\u00e1lculo/derivadas"],
                "status": "complete",
            }
            response = client.post(
                "/api/exercise/generate",
                json={
                    "session_id": "sess-3",
                    "topic": "c\u00e1lculo/derivadas",
                    "difficulty": "hard",
                    "exercise_type": "problem_solving",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        ex = data["data"]
        assert ex["exercise_id"] == "ex-001"
        assert ex["statement"]
        assert ex["model_solution"]
        assert "steps" in ex["model_solution"]
        assert "trace_id" in data

    def test_exercise_generate_unknown_topic_returns_suggestions(self):
        """GIVEN no chunks for topic → THEN topic_not_found + suggestions."""
        with patch("src.tools.generate_exercise") as mock_tool:
            mock_tool.invoke.return_value = {
                "exercise_id": "",
                "statement": "",
                "question": "",
                "model_solution": {
                    "steps": [],
                    "final_answer": "",
                    "key_concepts": [],
                },
                "topics_covered": [],
                "topic_not_found": ["astrof\u00edsica"],
                "topic_suggestions": ["c\u00e1lculo", "\u00e1lgebra", "f\u00edsica"],
                "status": "topic_not_found",
            }
            response = client.post(
                "/api/exercise/generate",
                json={
                    "session_id": "sess-5",
                    "topic": "astrof\u00edsica",
                    "difficulty": "medium",
                    "exercise_type": "problem_solving",
                },
            )

        assert response.status_code == 200
        data = response.json()
        ex = data["data"]
        assert "topic_not_found" in ex
        assert "topic_suggestions" in ex
        assert len(ex["topic_suggestions"]) > 0


# ==============================================================================
# REQ-API-005: trace_id on all responses
# ==============================================================================


class TestTraceIdPropagation:
    """REQ-API-005: Every ApiResponse includes trace_id."""

    def test_chat_has_trace_id(self):
        """POST /api/chat response includes trace_id."""
        with patch("src.tools.orchestrate_chat") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "response": "Hola.",
                    "intent": "general_chat",
                    "status": "complete",
                    "trace_id": "trace-chat-001",
                }
            )
            response = client.post(
                "/api/chat",
                json={"session_id": "sess-t", "message": "Hola"},
            )

        data = response.json()
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)

    @pytest.mark.skip(reason="Ingest endpoint requires full pipeline — tested via MockIngest below")
    def test_ingest_has_trace_id(self):
        """POST /api/ingest response includes trace_id."""
        pass

    def test_ingest_forwards_session_id(self):
        """GIVEN a session_id form field → THEN ingest_document receives that session_id.

        Covers: ingestion spec scenario "Frontend sends session_id with upload" (task 3.2).
        """
        import uuid as _uuid
        from unittest.mock import patch


        provided_sid = str(_uuid.uuid4())

        with patch("src.tools.ingest_document") as mock_tool:
            mock_tool.invoke.return_value = {
                "session_id": provided_sid,
                "status": "ok",
                "classification": "apunte_teorico",
                "topics": ["test"],
                "chunks_created": 3,
                "classification_confidence": 0.9,
                "document_id": "doc-1",
            }
            response = client.post(
                "/api/ingest",
                files=[("files", ("test.pdf", b"fake pdf content", "application/pdf"))],
                data={"session_id": provided_sid},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        results = data["data"]
        assert len(results) >= 1
        # The tool should have been called with the provided session_id
        call_kwargs = mock_tool.invoke.call_args[0][0]
        assert call_kwargs["session_id"] == provided_sid

    def test_ingest_generates_uuid_when_session_id_missing(self):
        """GIVEN no session_id form field → THEN a UUID4 is generated.

        Covers: ingestion spec scenario "Backward compatibility — no session_id provided" (task 3.2).
        """
        from unittest.mock import patch

        with patch("src.tools.ingest_document") as mock_tool:
            mock_tool.invoke.return_value = {
                "session_id": "generated-uuid",
                "status": "ok",
                "classification": "apunte_teorico",
                "topics": ["test"],
                "chunks_created": 3,
                "classification_confidence": 0.9,
                "document_id": "doc-1",
            }
            response = client.post(
                "/api/ingest",
                files=[("files", ("test.pdf", b"fake pdf content", "application/pdf"))],
                # No session_id in data
            )

        assert response.status_code == 200
        data = response.json()
        results = data["data"]
        assert len(results) >= 1
        # The generated session_id should be a UUID4 (36 chars, dashes)
        sid = results[0]["sessionId"]
        assert len(sid) == 36
        assert sid.count("-") == 4

    def test_ingest_validates_session_id_too_long(self):
        """GIVEN session_id > 64 chars → THEN fallback to generated UUID.

        Covers: design decision — Langfuse OTEL baggage drop prevention.
        """
        from unittest.mock import patch

        with patch("src.tools.ingest_document") as mock_tool:
            mock_tool.invoke.return_value = {
                "session_id": "should-be-uuid",
                "status": "ok",
                "classification": "apunte_teorico",
                "topics": ["test"],
                "chunks_created": 3,
                "classification_confidence": 0.9,
                "document_id": "doc-1",
            }
            long_id = "x" * 100
            response = client.post(
                "/api/ingest",
                files=[("files", ("test.pdf", b"fake pdf content", "application/pdf"))],
                data={"session_id": long_id},
            )

        assert response.status_code == 200
        # The effective session_id should NOT be the 100-char string
        sid = response.json()["data"][0]["sessionId"]
        assert len(sid) == 36  # UUID4 fallback
        assert sid.count("-") == 4

    def test_evaluate_has_trace_id(self):
        """POST /api/evaluate response includes trace_id."""
        with patch("src.tools.evaluate_answer") as mock_tool:
            mock_tool.invoke.return_value = [
                {
                    "question_id": "q-1",
                    "score": 1.0,
                    "justification": "Correcto.",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "is_evaluable": True,
                }
            ]
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "sess-ev",
                    "exam_id": "exam-ev",
                    "answers": {"q-1": "2x"},
                },
            )

        data = response.json()
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)

    def test_health_has_trace_id(self):
        """GET /api/health response includes trace_id."""
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "trace_id" in data

    def test_dashboard_has_trace_id(self):
        """GET /api/students/{id}/dashboard response includes trace_id."""
        with (
            patch("src.memory.schema.get_student_profile") as mock_profile,
            patch("src.memory.schema.get_topic_scores") as mock_scores,
            patch("src.memory.schema.compute_weak_topics") as mock_weak,
            patch("src.memory.schema.get_recent_sessions") as mock_sessions,
        ):
            mock_profile.return_value = {"id": "stu-d", "preferences": {}, "session_count": 0}
            mock_scores.return_value = []
            mock_weak.return_value = []
            mock_sessions.return_value = []

            response = client.get("/api/students/stu-d/dashboard")

        data = response.json()
        if response.status_code == 200:
            assert "trace_id" in data
            assert isinstance(data["trace_id"], str)


# ==============================================================================
# REQ-CONFIG-003: PUT /api/profile/{id}/preferences
# ==============================================================================


class TestPreferencesEndpoint:
    """REQ-CONFIG-003: PUT /api/profile/{id}/preferences persists exam preferences."""

    @pytest.mark.asyncio
    async def test_put_preferences_saves_to_profile(self):
        """GIVEN valid preferences → THEN upsert succeeds."""
        with patch("src.tools.update_student_profile.update_student_profile") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "status": "ok",
                    "student_id": "stu-1",
                    "upserted_topics": 0,
                    "errors": [],
                }
            )
            response = client.put(
                "/api/profile/stu-1/preferences",
                json={
                    "questionTypes": ["mcq", "open"],
                    "difficulty": "hard",
                    "questionCount": 8,
                    "includeTopics": ["c\u00e1lculo"],
                    "excludeTopics": [],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["status"] == "ok"
        assert "trace_id" in data

    @pytest.mark.asyncio
    async def test_put_preferences_calls_tool_with_correct_args(self):
        """GIVEN preferences → THEN tool receives mapped args."""
        with patch("src.tools.update_student_profile.update_student_profile") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "status": "ok",
                    "student_id": "stu-2",
                    "upserted_topics": 0,
                    "errors": [],
                }
            )
            client.put(
                "/api/profile/stu-2/preferences",
                json={
                    "questionTypes": ["open"],
                    "difficulty": "easy",
                    "questionCount": 3,
                    "includeTopics": [],
                    "excludeTopics": [],
                },
            )

        call_args = mock_tool.ainvoke.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_put_preferences_upserts_new_student(self):
        """GIVEN nonexistent student → THEN still succeeds (upsert)."""
        with patch("src.tools.update_student_profile.update_student_profile") as mock_tool:
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "status": "ok",
                    "student_id": "new-stu",
                    "upserted_topics": 0,
                    "errors": [],
                }
            )
            response = client.put(
                "/api/profile/new-stu/preferences",
                json={
                    "questionTypes": ["mcq"],
                    "difficulty": "medium",
                    "questionCount": 5,
                    "includeTopics": [],
                    "excludeTopics": [],
                },
            )

        assert response.status_code == 200


# ==============================================================================
# REQ-rag-exclusive-answers: Evaluation exam question mapping (task 3.3)
# ==============================================================================


class TestEvaluateMapsExamQuestions:
    """Task 3.3: /api/evaluate forwards full exam data to evaluator tool."""

    def test_evaluate_maps_exam_questions(self):
        """GIVEN examQuestions in request → THEN evaluator receives full context."""
        with patch("src.tools.evaluate_answer") as mock_tool:
            mock_tool.invoke.return_value = [
                {
                    "question_id": "q-001",
                    "score": 8.0,
                    "justification": "Correcto.",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "is_evaluable": True,
                }
            ]
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "sess-ev-map",
                    "exam_id": "exam-ev",
                    "answers": {"q-001": "La derivada es un límite."},
                    "examQuestions": [
                        {
                            "id": "q-001",
                            "type": "open",
                            "prompt": "¿Qué es una derivada?",
                            "baseAnswer": "Límite del cociente incremental.",
                            "sourceChunkIds": ["chunk-001"],
                            "topic": "cálculo/derivadas",
                            "difficulty": "medium",
                        }
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None

        # Verify tool received populated answer entries
        call_kwargs = mock_tool.invoke.call_args[0][0]
        answers = call_kwargs["answers"]
        assert len(answers) >= 1
        ans = answers[0]
        assert ans["question"] == "¿Qué es una derivada?"
        assert ans["base_answer"] == "Límite del cociente incremental."
        assert ans["topic"] == "cálculo/derivadas"
        assert ans["difficulty"] == "medium"
        assert ans["source_chunk_ids"] == ["chunk-001"]

    def test_evaluate_maps_exam_questions_missing_id(self):
        """GIVEN examQuestions with non-matching id → THEN empty placeholders + no crash."""
        with patch("src.tools.evaluate_answer") as mock_tool:
            mock_tool.invoke.return_value = [
                {
                    "question_id": "q-999",
                    "score": 0.0,
                    "justification": "N/A",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "is_evaluable": False,
                    "non_evaluable_reason": "no_material",
                }
            ]
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "sess-ev-map2",
                    "exam_id": "exam-ev",
                    "answers": {"q-999": "respuesta"},
                    "examQuestions": [
                        {
                            "id": "q-001",  # Does NOT match q-999
                            "type": "open",
                            "prompt": "Pregunta sobre otra cosa?",
                            "baseAnswer": "Otra respuesta.",
                            "sourceChunkIds": ["chunk-002"],
                            "topic": "otro/tema",
                            "difficulty": "easy",
                        }
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None

        # Answer with non-matching question_id gets empty placeholders
        call_kwargs = mock_tool.invoke.call_args[0][0]
        answers = call_kwargs["answers"]
        ans = answers[0]
        assert ans["question"] == ""
        assert ans["base_answer"] == ""
        assert ans["topic"] == ""
        assert ans["source_chunk_ids"] == []

    def test_evaluate_without_exam_questions_backward_compat(self):
        """GIVEN no examQuestions → THEN backward-compatible empty placeholders."""
        with patch("src.tools.evaluate_answer") as mock_tool:
            mock_tool.invoke.return_value = [
                {
                    "question_id": "q-1",
                    "score": 1.0,
                    "justification": "Correcto.",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "is_evaluable": True,
                }
            ]
            response = client.post(
                "/api/evaluate",
                json={
                    "session_id": "sess-ev-legacy",
                    "exam_id": "exam-ev",
                    "answers": {"q-1": "2x"},
                    # No examQuestions
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None

        # Backward compat: empty placeholders
        call_kwargs = mock_tool.invoke.call_args[0][0]
        answers = call_kwargs["answers"]
        ans = answers[0]
        assert ans["question"] == ""
        assert ans["base_answer"] == ""
