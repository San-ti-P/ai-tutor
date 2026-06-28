"""TDD tests for Epic 9 Profile Bootstrap (US-9.1..US-9.5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


class TestResolveStudentId:
    """T-004: resolve_student_id helper in schema.py."""

    async def test_override_wins(self):
        """Explicit student_id override bypasses DB lookup."""
        from src.memory.schema import resolve_student_id

        result = await resolve_student_id("sess-1", "stu-override")
        assert result == "stu-override"

    async def test_resolves_from_session_row(self, tmp_path):
        """Existing session row returns its student_id."""
        from src.config import settings
        from src.memory.schema import init_db, resolve_student_id

        db_path = tmp_path / "resolve.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            async with __import__("aiosqlite").connect(str(db_path)) as db:
                await db.execute(
                    "INSERT INTO students (id) VALUES (?)", ("stu-1",)
                )
                await db.execute(
                    "INSERT INTO sessions (id, student_id) VALUES (?, ?)",
                    ("sess-1", "stu-1"),
                )
                await db.commit()

            result = await resolve_student_id("sess-1")
            assert result == "stu-1"

    async def test_fallback_to_session_id_when_no_row(self, tmp_path):
        """No session row → fallback to session_id itself."""
        from src.config import settings
        from src.memory.schema import init_db, resolve_student_id

        db_path = tmp_path / "resolve_fallback.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            await init_db()
            result = await resolve_student_id("unknown-sess")
            assert result == "unknown-sess"


class TestLoadProfile:
    """T-001/T-002: load_profile node."""

    async def test_load_profile_success(self):
        """Profile load returns summary dict in student_profile."""
        from src.agents.orchestrator import load_profile

        state = {
            "session_id": "sess-1",
            "student_id": "stu-1",
            "user_message": "hola",
        }
        profile = {
            "id": "stu-1",
            "weak_topics": ["cálculo"],
            "preferences": {"difficulty": "medium"},
        }

        with patch(
            "src.agents.orchestrator.resolve_student_id",
            new=AsyncMock(return_value="stu-1"),
        ):
            with patch(
                "src.tools.get_student_summary.get_student_summary",
            ) as mock_summary:
                mock_summary.ainvoke = AsyncMock(return_value=profile)
                result = await load_profile(state)

        mock_summary.ainvoke.assert_awaited_once_with({"student_id": "stu-1"})
        assert result["student_profile"] == profile

    async def test_load_profile_exception_fallback_empty(self, caplog):
        """Tool exception → empty profile + WARNING log."""
        from src.agents.orchestrator import load_profile

        state = {"session_id": "sess-1", "student_id": "stu-1"}

        with patch(
            "src.agents.orchestrator.resolve_student_id",
            new=AsyncMock(return_value="stu-1"),
        ):
            with patch(
                "src.tools.get_student_summary.get_student_summary",
            ) as mock_summary:
                mock_summary.ainvoke = AsyncMock(side_effect=RuntimeError("DB down"))
                with caplog.at_level("WARNING"):
                    result = await load_profile(state)

        assert result["student_profile"] == {}
        assert "DB down" in caplog.text
        # Phase 3 robustness: will set profile_load_error on DB failure.
        if "profile_load_error" in result:
            assert result["profile_load_error"] is not None

    async def test_load_profile_none_fallback_empty(self):
        """Tool returns None (unknown student) → empty profile."""
        from src.agents.orchestrator import load_profile

        state = {"session_id": "sess-1", "student_id": "stu-1"}

        with patch(
            "src.agents.orchestrator.resolve_student_id",
            new=AsyncMock(return_value="stu-1"),
        ):
            with patch(
                "src.tools.get_student_summary.get_student_summary",
            ) as mock_summary:
                mock_summary.ainvoke = AsyncMock(return_value=None)
                result = await load_profile(state)

        assert result["student_profile"] == {}
        # No error when profile is legitimately not found (not a DB failure)
        assert result.get("profile_load_error") is None


class TestClassifyIntentProfileEnrichment:
    """T-003: classify_intent prompt includes profile context."""

    def test_prompt_includes_weak_topics_and_preferences(self):
        """Non-empty profile → prompt mentions weak topics and preferences."""
        from src.agents.orchestrator import classify_intent

        state = {
            "session_id": "sess-1",
            "user_message": "Generame un examen",
            "student_profile": {
                "weak_topics": ["cálculo/derivadas"],
                "preferences": {"difficulty": "hard"},
            },
        }

        captured_prompt = None

        class _FakeStructured:
            def invoke(self, prompt, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                from src.agents.orchestrator import IntentClassification

                return IntentClassification(intent="generate_exam", confidence=0.95)

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured(),
        ):
            classify_intent(state)

        assert captured_prompt is not None
        assert "cálculo/derivadas" in captured_prompt
        assert "hard" in captured_prompt

    def test_prompt_works_without_profile(self):
        """Empty profile → prompt still classifies without profile section."""
        from src.agents.orchestrator import classify_intent

        state = {
            "session_id": "sess-1",
            "user_message": "Hola",
            "student_profile": {},
        }

        captured_prompt = None

        class _FakeStructured:
            def invoke(self, prompt, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                from src.agents.orchestrator import IntentClassification

                return IntentClassification(intent="general_chat", confidence=0.99)

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured(),
        ):
            result = classify_intent(state)

        assert result["intent"] == "general_chat"
        assert "Contexto del estudiante:" not in captured_prompt
