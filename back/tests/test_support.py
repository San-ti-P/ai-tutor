"""Tests for the Support Agent, schema helpers, and tools.

Covers PRD cases:
  - SUP-01 (auto-create student profile)
  - SUP-02 (upsert topic scores)
  - SUP-03 (evaluator → topic_scores sync)
  - SUP-04 (compute weak topics)
  - SUP-05 (get_student_summary tool)
  - SUP-06 (update_student_profile tool)
  - SUP-07 (multi-session weak topic adaptation)
  - SUP-08 (dashboard endpoint)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures for support tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Return a temporary SQLite DB path for isolated testing."""
    return str(tmp_path / "test_tutor.db")


@pytest.fixture
async def populated_db(test_db_path: str) -> str:
    """Initialise DB schema and insert a student for profile tests."""
    import aiosqlite

    async with aiosqlite.connect(test_db_path) as db:
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
        """)
        await db.execute(
            "INSERT INTO students (id, preferences_json) VALUES (?, ?)",
            ("test-student-001", json.dumps({"difficulty": "medium"})),
        )
        await db.commit()

    return test_db_path


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.1: upsert_topic_scores
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertTopicScores:
    """SUP-02: store per-topic latest score in topic_scores."""

    @pytest.mark.asyncio
    async def test_upsert_topic_scores_inserts_and_updates(self, populated_db):
        """GIVEN no row → WHEN upsert → THEN row inserted.
        GIVEN existing row → WHEN overwritten → THEN latest score wins.
        """
        import aiosqlite

        from src.memory.schema import upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            # First insert
            await upsert_topic_scores(
                "test-student-001",
                [{"topic": "cálculo", "score": 7.5}],
            )

            # Verify insert
            async with aiosqlite.connect(populated_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT topic, score FROM topic_scores WHERE student_id = ? AND topic = ?",
                    ("test-student-001", "cálculo"),
                )
                row = await cursor.fetchone()
                assert row is not None
                assert dict(row)["score"] == 7.5

            # Update (overwrite)
            await upsert_topic_scores(
                "test-student-001",
                [{"topic": "cálculo", "score": 9.0}],
            )

            # Verify overwrite
            async with aiosqlite.connect(populated_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT topic, score FROM topic_scores WHERE student_id = ? AND topic = ?",
                    ("test-student-001", "cálculo"),
                )
                row = await cursor.fetchone()
                assert dict(row)["score"] == 9.0

    @pytest.mark.asyncio
    async def test_upsert_multiple_topics(self, populated_db):
        """GIVEN multiple topic/score pairs → WHEN upsert → THEN all stored."""
        import aiosqlite

        from src.memory.schema import upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "cálculo", "score": 4.0},
                    {"topic": "álgebra", "score": 8.0},
                    {"topic": "física", "score": 5.5},
                ],
            )

            async with aiosqlite.connect(populated_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT topic, score FROM topic_scores WHERE student_id = ? ORDER BY topic",
                    ("test-student-001",),
                )
                rows = await cursor.fetchall()
                scores = {dict(r)["topic"]: dict(r)["score"] for r in rows}
                assert scores == {"álgebra": 8.0, "cálculo": 4.0, "física": 5.5}


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.2: compute_weak_topics
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeWeakTopics:
    """SUP-04: compute weak_topics from topic_scores (latest score < 6)."""

    @pytest.mark.asyncio
    async def test_weak_topics_below_threshold(self, populated_db):
        """GIVEN scores {Cálculo:4, Álgebra:8, Física:5.5} → THEN weak = ['cálculo','física']."""

        from src.memory.schema import compute_weak_topics, upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "cálculo", "score": 4.0},
                    {"topic": "álgebra", "score": 8.0},
                    {"topic": "física", "score": 5.5},
                ],
            )

            weak = await compute_weak_topics("test-student-001")
            assert weak == ["cálculo", "física"]  # sorted asc: 4.0 < 5.5

    @pytest.mark.asyncio
    async def test_no_weak_topics_all_above_threshold(self, populated_db):
        """GIVEN all scores ≥ 6 → THEN empty list."""
        from src.memory.schema import compute_weak_topics, upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "cálculo", "score": 8.0},
                    {"topic": "álgebra", "score": 7.5},
                    {"topic": "física", "score": 9.0},
                ],
            )

            weak = await compute_weak_topics("test-student-001")
            assert weak == []

    @pytest.mark.asyncio
    async def test_weak_topics_no_scores_returns_empty(self, populated_db):
        """GIVEN student with no topic_scores → THEN empty list."""
        from src.memory.schema import compute_weak_topics

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            weak = await compute_weak_topics("test-student-001")
            assert weak == []

    @pytest.mark.asyncio
    async def test_weak_topics_respects_limit(self, populated_db):
        """GIVEN 5 topics with scores < 6 → WHEN limit=3 → THEN top 3 weakest."""
        from src.memory.schema import compute_weak_topics, upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "t1", "score": 1.0},
                    {"topic": "t2", "score": 2.0},
                    {"topic": "t3", "score": 3.0},
                    {"topic": "t4", "score": 4.0},
                    {"topic": "t5", "score": 5.0},
                ],
            )

            weak = await compute_weak_topics("test-student-001", limit=3)
            assert weak == ["t1", "t2", "t3"]  # lowest scores first


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.3: update_student_profile tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateStudentProfileTool:
    """SUP-06: @tool for upserting preferences and topic scores."""

    @pytest.mark.asyncio
    async def test_update_student_profile_updates_db(self, populated_db):
        """GIVEN a student → WHEN tool called with scores + prefs → THEN DB updated."""
        import aiosqlite

        from src.tools.update_student_profile import update_student_profile

        with patch("src.memory.schema.settings") as mock_schema_settings:
            mock_schema_settings.sqlite_db_path = populated_db

            result = await update_student_profile.ainvoke(
                {
                    "student_id": "test-student-001",
                    "topic_scores": {"cálculo": 7.5, "álgebra": 6.0},
                    "preferences": {"difficulty": "hard", "question_types": ["open"]},
                }
            )

            assert result["status"] == "ok"
            assert result["student_id"] == "test-student-001"

            # Verify topic_scores
            async with aiosqlite.connect(populated_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT topic, score FROM topic_scores WHERE student_id = ? ORDER BY topic",
                    ("test-student-001",),
                )
                rows = await cursor.fetchall()
                scores = {dict(r)["topic"]: dict(r)["score"] for r in rows}
                assert scores.get("cálculo") == 7.5
                assert scores.get("álgebra") == 6.0

                # Verify preferences
                cursor = await db.execute(
                    "SELECT preferences_json FROM students WHERE id = ?",
                    ("test-student-001",),
                )
                row = await cursor.fetchone()
                prefs = json.loads(dict(row)["preferences_json"])
                assert prefs["difficulty"] == "hard"

    @pytest.mark.asyncio
    async def test_update_student_profile_no_preferences_uses_defaults(self, populated_db):
        """GIVEN no preferences → THEN tool stores defaults in DB."""
        import json

        import aiosqlite

        from src.tools.update_student_profile import update_student_profile

        with patch("src.memory.schema.settings") as mock_schema_settings:
            mock_schema_settings.sqlite_db_path = populated_db

            result = await update_student_profile.ainvoke(
                {
                    "student_id": "test-student-001",
                    "topic_scores": {"cálculo": 8.0},
                }
            )

            assert result["status"] == "ok"

            # Verify defaults were persisted in DB
            async with aiosqlite.connect(populated_db) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT preferences_json FROM students WHERE id = ?",
                    ("test-student-001",),
                )
                row = await cursor.fetchone()
                prefs = json.loads(dict(row)["preferences_json"])
                assert prefs["difficulty"] == "medium"
                assert prefs["question_types"] == ["mcq"]
                assert prefs["question_count"] == 5
                assert prefs["include_topics"] == []
                assert prefs["exclude_topics"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.4: get_student_summary tool
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetStudentSummaryTool:
    """SUP-05: read-only tool returning aggregated student profile."""

    @pytest.mark.asyncio
    async def test_get_student_summary_returns_full_profile(self, populated_db):
        """GIVEN valid student with scores → WHEN tool called → THEN full dict."""

        from src.memory.schema import upsert_topic_scores
        from src.tools.get_student_summary import get_student_summary

        with patch("src.memory.schema.settings") as mock_schema_settings:
            mock_schema_settings.sqlite_db_path = populated_db

            # Seed topic scores
            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "cálculo", "score": 4.0},
                    {"topic": "álgebra", "score": 8.0},
                    {"topic": "física", "score": 3.0},
                ],
            )

            result = await get_student_summary.ainvoke(
                {"student_id": "test-student-001"}
            )

            assert result is not None
            assert result["id"] == "test-student-001"
            assert "topic_scores" in result
            assert "preferences" in result
            assert "weak_topics" in result
            assert "session_count" in result

    @pytest.mark.asyncio
    async def test_get_student_summary_unknown_id_returns_none(self, populated_db):
        """GIVEN unknown student ID → THEN tool returns None."""
        from src.tools.get_student_summary import get_student_summary

        with patch("src.memory.schema.settings") as mock_schema_settings:
            mock_schema_settings.sqlite_db_path = populated_db

            result = await get_student_summary.ainvoke(
                {"student_id": "nonexistent"}
            )

            assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.6: Support Agent graph nodes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportAgentNodes:
    """Verify support agent nodes return partial state dicts."""

    def test_fetch_student_profile_returns_state(self, populated_db):
        """fetch_student_profile populates profile_data and topic_scores."""
        from src.agents.support import SupportState, fetch_student_profile

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            state: SupportState = {
                "session_id": "sess-001",
                "student_id": "test-student-001",
                "query_type": "query",
                "profile_data": None,
                "session_history": [],
                "topic_scores": [],
                "weak_topics": [],
                "preferences": None,
                "response": "",
                "status": "pending",
            }

            result = fetch_student_profile(state)
            assert isinstance(result, dict)
            assert result.get("profile_data") is not None
            assert result["profile_data"]["id"] == "test-student-001"
            assert "topic_scores" in result

    def test_fetch_session_history_returns_state(self, populated_db):
        """fetch_session_history populates session_history."""
        from src.agents.support import SupportState, fetch_session_history

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            state: SupportState = {
                "session_id": "sess-001",
                "student_id": "test-student-001",
                "query_type": "query",
                "profile_data": None,
                "session_history": [],
                "topic_scores": [],
                "weak_topics": [],
                "preferences": None,
                "response": "",
                "status": "pending",
            }

            result = fetch_session_history(state)
            assert isinstance(result, dict)
            assert "session_history" in result

    def test_compute_progress_summary_presence(self):
        """compute_progress_summary returns weak_topics and status."""
        from src.agents.support import SupportState, compute_progress_summary

        state: SupportState = {
            "session_id": "sess-001",
            "student_id": "test-student-001",
            "query_type": "query",
            "profile_data": {
                "id": "test-student-001",
                "preferences": {},
                "topic_scores": {},
                "session_count": 0,
            },
            "session_history": [],
            "topic_scores": [
                {"topic": "cálculo", "score": 4.0},
                {"topic": "álgebra", "score": 8.0},
            ],
            "weak_topics": [],
            "preferences": {},
            "response": "",
            "status": "pending",
        }

        result = compute_progress_summary(state)
        assert isinstance(result, dict)
        # weak_topics should be computed from topic_scores
        assert "weak_topics" in result
        assert result["weak_topics"] == ["cálculo"]  # score 4.0 < 6

    def test_generate_response_query_type(self):
        """generate_response produces a natural-language response for query type."""
        from src.agents.support import SupportState, generate_response

        state: SupportState = {
            "session_id": "sess-001",
            "student_id": "test-student-001",
            "query_type": "query",
            "profile_data": {
                "id": "test-student-001",
                "preferences": {},
                "topic_scores": {},
                "session_count": 3,
            },
            "session_history": [],
            "topic_scores": [{"topic": "cálculo", "score": 4.0}],
            "weak_topics": ["cálculo"],
            "preferences": {},
            "response": "",
            "status": "processing",
        }

        result = generate_response(state)
        assert isinstance(result, dict)
        assert "response" in result
        assert len(result["response"]) > 0

    def test_generate_response_update_type(self):
        """generate_response for update type confirms profile update."""
        from src.agents.support import SupportState, generate_response

        state: SupportState = {
            "session_id": "sess-001",
            "student_id": "test-student-001",
            "query_type": "update",
            "profile_data": {
                "id": "test-student-001",
                "preferences": {},
                "topic_scores": {},
                "session_count": 1,
            },
            "session_history": [],
            "topic_scores": [],
            "weak_topics": [],
            "preferences": {"difficulty": "hard"},
            "response": "",
            "status": "processing",
        }

        result = generate_response(state)
        assert isinstance(result, dict)
        assert "response" in result
        assert "actualiz" in result["response"].lower()


class TestSupportAgentRouting:
    """Conditional routing based on query_type."""

    def test_route_after_fetch_query(self):
        """When query_type is 'query', route to fetch_session_history."""
        from src.agents.support import _route_after_fetch

        state = {"query_type": "query"}
        assert _route_after_fetch(state) == "fetch_session_history"

    def test_route_after_fetch_update(self):
        """When query_type is 'update', skip history, go to compute_progress_summary."""
        from src.agents.support import _route_after_fetch

        state = {"query_type": "update"}
        assert _route_after_fetch(state) == "compute_progress_summary"


class TestSupportGraphBuilder:
    """Verify build_support_agent compiles correctly."""

    def test_build_support_agent_compiles(self):
        """Graph compiles without errors and has expected nodes."""
        from src.agents.support import build_support_agent

        builder = build_support_agent()
        graph = builder.compile()

        nodes = graph.get_graph().nodes
        node_names = {n for n in nodes if n not in ("__start__", "__end__")}
        assert "fetch_student_profile" in node_names
        assert "fetch_session_history" in node_names
        assert "compute_progress_summary" in node_names
        assert "generate_response" in node_names

    def test_graph_invocation_query(self, populated_db):
        """Full graph invocation for query type returns a response."""
        from src.agents.support import build_support_agent

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            graph = build_support_agent().compile()

            initial_state = {
                "session_id": "sess-001",
                "student_id": "test-student-001",
                "query_type": "query",
                "profile_data": None,
                "session_history": [],
                "topic_scores": [],
                "weak_topics": [],
                "preferences": None,
                "response": "",
                "status": "pending",
            }

            result = graph.invoke(initial_state)
            assert result["status"] == "done"
            assert len(result["response"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.9: Multi-session adaptation integration test (PRD case 4)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMultiSessionAdaptation:
    """SUP-07 / PRD case 4: weak topics are detected across sessions."""

    def test_second_session_prioritizes_weak_topics(self, populated_db, evaluator_state):
        """GIVEN first session with low scores on topic X → WHEN compute_weak_topics
        → THEN X appears in weak_topics for the second session.
        """
        from src.agents.evaluator import sync_scores
        from src.memory.schema import compute_weak_topics

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            # Session 1: student answers poorly on cálculo, well on álgebra
            state = dict(evaluator_state)
            state["student_id"] = "test-student-001"
            state["evaluation_results"] = [
                {
                    "question_id": "q-001",
                    "topic": "cálculo",
                    "score": 3.0,
                    "justification": "Poor understanding of derivatives",
                    "conceptual_errors": ["Wrong definition"],
                    "suggestions": ["Review limits"],
                    "source_chunk_ids": [],
                    "status": "evaluated",
                },
                {
                    "question_id": "q-002",
                    "topic": "álgebra",
                    "score": 8.0,
                    "justification": "Good matrix understanding",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "source_chunk_ids": [],
                    "status": "evaluated",
                },
            ]

            result = sync_scores(state)
            assert result["scores_synced"] is True

            # Session 2: compute weak topics — cálculo should appear (score 3.0 < 6.0)
            import asyncio

            async def _check():
                return await compute_weak_topics("test-student-001")

            weak = asyncio.run(_check())
            assert "cálculo" in weak, f"Expected 'cálculo' in weak topics, got {weak}"
            assert "álgebra" not in weak, (
                f"'álgebra' with score 8.0 should not be weak, got {weak}"
            )


@pytest.mark.integration
class TestEvaluatorTopicScoresSync:
    """SUP-03: Evaluator sync_scores calls upsert_topic_scores after save_evaluation."""

    def test_evaluator_updates_topic_scores(self, populated_db, evaluator_state):
        """GIVEN eval results → WHEN sync_scores runs → THEN topic_scores table updated."""
        import asyncio

        import aiosqlite

        from src.agents.evaluator import sync_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            # Simulate having evaluation results ready
            state = dict(evaluator_state)
            state["student_id"] = "test-student-001"
            state["evaluation_results"] = [
                {
                    "question_id": "q-001",
                    "topic": "cálculo/derivadas",
                    "score": 8.0,
                    "justification": "Great understanding",
                    "conceptual_errors": [],
                    "suggestions": [],
                    "source_chunk_ids": [],
                    "status": "evaluated",
                },
                {
                    "question_id": "q-002",
                    "topic": "álgebra/matrices",
                    "score": 4.0,
                    "justification": "Poor understanding of matrices",
                    "conceptual_errors": ["Matrix rank calculation is wrong"],
                    "suggestions": ["Review Gaussian elimination"],
                    "source_chunk_ids": [],
                    "status": "evaluated",
                },
            ]

            result = sync_scores(state)
            assert result["scores_synced"] is True
            assert result["status"] == "synced"

            # The function uses asyncio.ensure_future or event loop — need to wait
            import time
            time.sleep(0.5)

            # Verify topic_scores were upserted
            async def _check():
                async with aiosqlite.connect(populated_db) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(
                        "SELECT topic, score FROM topic_scores WHERE student_id = ?",
                        ("test-student-001",),
                    )
                    rows = await cursor.fetchall()
                    return {dict(r)["topic"]: dict(r)["score"] for r in rows}

            scores = asyncio.run(_check())
            assert scores.get("cálculo/derivadas") == 8.0
            assert scores.get("álgebra/matrices") == 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# T-6.8: Dashboard API endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardEndpoint:
    """SUP-08: GET /students/{id}/dashboard returns aggregated data."""

    @pytest.mark.asyncio
    async def test_dashboard_returns_aggregated_data(self, populated_db):
        """GIVEN 5 evals, 3 sessions, 2 weak topics → THEN endpoint returns StudentProfile."""
        import aiosqlite
        from fastapi.testclient import TestClient

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            # Seed sessions
            async with aiosqlite.connect(populated_db) as db:
                for i in range(3):
                    await db.execute(
                        "INSERT INTO sessions (id, student_id, intent, status) VALUES (?, ?, ?, ?)",
                        (f"sess-{i:03d}", "test-student-001", "exam", "completed"),
                    )
                await db.commit()

            # Seed evaluations and topic_scores via upsert_topic_scores
            from src.memory.schema import upsert_topic_scores
            await upsert_topic_scores(
                "test-student-001",
                [
                    {"topic": "cálculo", "score": 4.0},
                    {"topic": "álgebra", "score": 3.5},
                    {"topic": "física", "score": 7.0},
                ],
            )

            # Import after patching so it gets the mocked settings
            from src.main import app

            client = TestClient(app)

            response = client.get("/students/test-student-001/dashboard")
            assert response.status_code == 200

            data = response.json()
            assert data["error"] is None
            # ApiResponse wraps data
            profile = data["data"]
            assert profile["id"] == "test-student-001"
            assert profile["sessionCount"] == 3
            assert "cálculo" in profile["weakTopics"]
            assert "álgebra" in profile["weakTopics"]

    @pytest.mark.asyncio
    async def test_dashboard_unknown_student_returns_404(self, populated_db):
        """GIVEN unknown ID → THEN 404."""
        from fastapi.testclient import TestClient

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            from src.main import app

            client = TestClient(app)

            response = client.get("/students/nonexistent/dashboard")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_dashboard_latency_p95_under_300ms(self, populated_db):
        """SUP-NFR-02: Dashboard endpoint responds within 300ms p95."""
        import time

        from fastapi.testclient import TestClient

        from src.memory.schema import upsert_topic_scores

        with patch("src.memory.schema.settings") as mock_settings:
            mock_settings.sqlite_db_path = populated_db

            # Seed realistic data volume
            topics = [
                {"topic": f"topic_{i}", "score": float(i % 10)}
                for i in range(20)
            ]
            await upsert_topic_scores("test-student-001", topics)

            from src.main import app
            client = TestClient(app)

            # Warm-up
            client.get("/students/test-student-001/dashboard")

            # Measure 20 samples, check p95
            latencies: list[float] = []
            for _ in range(20):
                start = time.perf_counter()
                response = client.get("/students/test-student-001/dashboard")
                elapsed = (time.perf_counter() - start) * 1000  # ms
                latencies.append(elapsed)
                assert response.status_code == 200

            latencies.sort()
            p95_index = int(len(latencies) * 0.95)
            p95_latency = latencies[p95_index]

            assert p95_latency < 300, (
                f"Dashboard p95 latency {p95_latency:.1f}ms exceeds 300ms budget"
            )
