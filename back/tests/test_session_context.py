"""TDD tests for Phase 6: Session Context for Agent (T-018..T-021)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from src.agents.orchestrator import (
    OrchestratorState,
    build_orchestrator,
    synthesize_response,
)


# ── T-018: Tools ────────────────────────────────────────────────────────────


class TestListSessionFilesTool:
    """T-018a: list_session_files tool exports and returns correct shape."""

    def test_tool_is_importable(self):
        """GIVEN tools/__init__.py → THEN list_session_files is exported."""
        from src.tools import list_session_files

        assert list_session_files is not None
        assert isinstance(list_session_files, StructuredTool)

    @pytest.mark.asyncio
    async def test_returns_files_and_count(self):
        """GIVEN files in session → THEN tool returns structured dict.

        Mock target is ``src.memory.schema.list_session_files`` because
        the tool module imports the schema module (not the function directly)
        to allow tests to intercept the call.
        """
        from src.tools import list_session_files

        with patch(
            "src.memory.schema.list_session_files",
            new=AsyncMock(
                return_value=[
                    {
                        "id": "doc-1",
                        "file_name": "apunte.pdf",
                        "classification": "apunte",
                        "topics_json": '["cálculo","derivadas"]',
                        "chunks_count": 5,
                        "ingested_at": "2026-06-26T10:00:00",
                        "session_id": "sess-1",
                    },
                    {
                        "id": "doc-2",
                        "file_name": "examen.pdf",
                        "classification": "examen",
                        "topics_json": '["álgebra"]',
                        "chunks_count": 3,
                        "ingested_at": "2026-06-26T09:00:00",
                        "session_id": "sess-1",
                    },
                ]
            ),
        ):
            result = await list_session_files.ainvoke({"session_id": "sess-1"})

        assert result["session_id"] == "sess-1"
        assert result["count"] == 2
        assert len(result["files"]) == 2
        assert result["files"][0]["file_name"] == "apunte.pdf"
        assert result["files"][0]["topics"] == ["cálculo", "derivadas"]
        assert result["files"][1]["file_name"] == "examen.pdf"

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty_list(self):
        """GIVEN no files in session → THEN tool returns empty files list."""
        from src.tools import list_session_files

        with patch(
            "src.memory.schema.list_session_files",
            new=AsyncMock(return_value=[]),
        ):
            result = await list_session_files.ainvoke({"session_id": "sess-empty"})

        assert result["session_id"] == "sess-empty"
        assert result["count"] == 0
        assert result["files"] == []


class TestGetSessionProgressTool:
    """T-018b: get_session_progress tool exports and returns correct shape."""

    def test_tool_is_importable(self):
        """GIVEN tools/__init__.py → THEN get_session_progress is exported."""
        from src.tools import get_session_progress

        assert get_session_progress is not None
        assert isinstance(get_session_progress, StructuredTool)

    @pytest.mark.asyncio
    async def test_returns_progress_with_topic_scores(self):
        """GIVEN evaluations in session → THEN tool returns progress dict."""
        from src.tools import get_session_progress

        with patch(
            "src.memory.schema.get_session_profile",
            new=AsyncMock(
                return_value={
                    "session_id": "sess-1",
                    "topic_scores": {"cálculo": [8.0, 7.0], "álgebra": [4.5]},
                    "weak_topics": ["álgebra"],
                    "exam_count": 3,
                    "average_score": 6.5,
                }
            ),
        ):
            result = await get_session_progress.ainvoke({"session_id": "sess-1"})

        assert result["session_id"] == "sess-1"
        assert result["topic_scores"] == {"cálculo": [8.0, 7.0], "álgebra": [4.5]}
        assert result["weak_topics"] == ["álgebra"]
        assert result["exam_count"] == 3
        assert result["average_score"] == 6.5

    @pytest.mark.asyncio
    async def test_missing_session_returns_empty_progress(self):
        """GIVEN unknown session_id → THEN tool returns empty progress dict."""
        from src.tools import get_session_progress

        with patch(
            "src.memory.schema.get_session_profile",
            new=AsyncMock(return_value=None),
        ):
            result = await get_session_progress.ainvoke({"session_id": "no-such"})

        assert result["session_id"] == "no-such"
        assert result["topic_scores"] == {}
        assert result["weak_topics"] == []
        assert result["exam_count"] == 0


# ── T-019: load_session_context node ────────────────────────────────────────


class TestLoadSessionContextNode:
    """T-019: load_session_context node populates session_context state field."""

    def test_node_is_defined(self):
        """GIVEN orchestrator module → THEN load_session_context function exists."""
        from src.agents.orchestrator import load_session_context

        assert load_session_context is not None
        assert callable(load_session_context)

    @pytest.mark.asyncio
    async def test_populates_files_and_progress(self):
        """GIVEN session with files and profile → THEN session_context populated.

        The ``load_session_context`` node imports schema functions inside its
        body, so we patch at ``src.memory.schema.*``.
        """
        from src.agents.orchestrator import load_session_context

        state: OrchestratorState = {
            "session_id": "sess-1",
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
        }

        mock_files = [
            {"id": "d1", "file_name": "apunte.pdf", "classification": "apunte",
             "topics_json": '["cálculo"]', "chunks_count": 3,
             "ingested_at": "2026-06-26", "session_id": "sess-1"},
        ]
        mock_profile = {
            "session_id": "sess-1",
            "topic_scores": {"cálculo": [7.0]},
            "weak_topics": ["álgebra"],
            "exam_count": 1,
            "average_score": 7.0,
        }

        with patch(
            "src.memory.schema.list_session_files",
            new=AsyncMock(return_value=mock_files),
        ), patch(
            "src.memory.schema.get_session_profile",
            new=AsyncMock(return_value=mock_profile),
        ):
            result = await load_session_context(state)

        assert "session_context" in result
        ctx = result["session_context"]
        assert "files" in ctx
        assert len(ctx["files"]) == 1
        assert ctx["files"][0]["file_name"] == "apunte.pdf"
        assert "progress" in ctx
        assert ctx["progress"]["weak_topics"] == ["álgebra"]
        assert ctx["progress"]["exam_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_session_produces_default_context(self):
        """GIVEN session with no files and no profile → THEN default context."""
        from src.agents.orchestrator import load_session_context

        state: OrchestratorState = {
            "session_id": "sess-empty",
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
        }

        with patch(
            "src.memory.schema.list_session_files",
            new=AsyncMock(return_value=[]),
        ), patch(
            "src.memory.schema.get_session_profile",
            new=AsyncMock(return_value=None),
        ):
            result = await load_session_context(state)

        ctx = result["session_context"]
        assert ctx["files"] == []
        assert ctx["progress"] == {
            "topic_scores": {}, "weak_topics": [], "exam_count": 0, "average_score": None,
        }

    def test_node_is_wired_in_graph(self):
        """GIVEN build_orchestrator → THEN load_session_context node added to graph."""
        builder = build_orchestrator()
        graph = builder.compile()
        node_names = list(graph.nodes.keys())
        assert "load_session_context" in node_names


# ── T-020: synthesize_response enrichment ────────────────────────────────────


class TestSynthesizeResponseEnrichment:
    """T-020: synthesize_response prompt uses profile + session_context + messages_history."""

    def test_prompt_includes_profile_weak_topics(self):
        """GIVEN student_profile with weak_topics → THEN prompt includes them.

        Uses a non-academic message ("Hola cómo estás") to avoid the
        academic-probe branch which would trigger ``query_material``.
        """
        state: OrchestratorState = {
            "session_id": "sess-1",
            "user_message": "Hola cómo estás",
            "intent": "general_chat",
            "confidence": 0.95,
            "plan": [],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": {
                "weak_topics": ["cálculo", "álgebra"],
                "preferences": {"difficulty": "medium"},
            },
            "messages_history": [],
        }

        with patch("src.agents.orchestrator._get_llm") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type("R", (), {"content": "ok"})()
            mock_factory.return_value = mock_llm

            synthesize_response(state, RunnableConfig())

        # The LLM should have been called
        assert mock_llm.invoke.called, "LLM was not invoked — academic probe may have interfered"
        prompt = mock_llm.invoke.call_args[0][0]
        has_profile_context = any(
            term in prompt.lower() for term in ["cálculo", "álgebra", "débil", "debil", "perfil"]
        )
        assert has_profile_context, f"Prompt missing profile context: {prompt[:400]}"

    def test_prompt_includes_session_context(self):
        """GIVEN session_context with files and progress → THEN prompt includes them."""
        state: OrchestratorState = {
            "session_id": "sess-1",
            "user_message": "Hola buenos días",
            "intent": "general_chat",
            "confidence": 0.95,
            "plan": [],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": None,
            "session_context": {
                "files": [
                    {"file_name": "apunte.pdf", "topics": ["cálculo"]},
                    {"file_name": "examen.pdf", "topics": ["álgebra"]},
                ],
                "progress": {
                    "topic_scores": {"cálculo": [8.0]},
                    "weak_topics": ["álgebra"],
                    "exam_count": 1,
                    "average_score": 8.0,
                },
            },
            "messages_history": [],
        }

        with patch("src.agents.orchestrator._get_llm") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type("R", (), {"content": "ok"})()
            mock_factory.return_value = mock_llm

            synthesize_response(state, RunnableConfig())

        assert mock_llm.invoke.called, "LLM was not invoked"
        prompt = mock_llm.invoke.call_args[0][0]
        has_session_context = any(
            term in prompt.lower()
            for term in ["apunte.pdf", "examen.pdf", "sesión", "sesion", "archivo", "progreso"]
        )
        assert has_session_context, f"Prompt missing session context: {prompt[:400]}"

    def test_prompt_includes_messages_history(self):
        """GIVEN messages_history with prior exchanges → THEN prompt references them."""
        state: OrchestratorState = {
            "session_id": "sess-1",
            "user_message": "Gracias por la ayuda",
            "intent": "general_chat",
            "confidence": 0.95,
            "plan": [],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": None,
            "messages_history": [
                {"role": "user", "content": "¿Qué es un límite?"},
                {"role": "assistant", "content": "Un límite es..."},
            ],
        }

        with patch("src.agents.orchestrator._get_llm") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type("R", (), {"content": "ok"})()
            mock_factory.return_value = mock_llm

            synthesize_response(state, RunnableConfig())

        assert mock_llm.invoke.called, "LLM was not invoked"
        prompt = mock_llm.invoke.call_args[0][0]
        assert "límite" in prompt.lower() or "limite" in prompt.lower(), (
            f"Prompt missing messages_history: {prompt[:400]}"
        )

    def test_synthesize_still_returns_messages_history(self):
        """GIVEN synthesize_response → THEN still appends to messages_history."""
        state: OrchestratorState = {
            "session_id": "sess-1",
            "user_message": "Hola qué tal",
            "intent": "general_chat",
            "confidence": 0.95,
            "plan": [],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": {"weak_topics": ["cálculo"]},
            "session_context": {
                "files": [],
                "progress": {
                    "topic_scores": {}, "weak_topics": [], "exam_count": 0, "average_score": None,
                },
            },
            "messages_history": [],
        }

        with patch("src.agents.orchestrator._get_llm") as mock_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type("R", (), {"content": "Hola!"})()
            mock_factory.return_value = mock_llm

            result = synthesize_response(state, RunnableConfig())

        assert "messages_history" in result
        assert len(result["messages_history"]) == 2


# ── T-021: Integration of Phase 6 components ────────────────────────────────


class TestSessionContextIntegration:
    """T-021: End-to-end wiring: graph node + tools + synthesize enrichment."""

    def test_graph_includes_load_session_context(self):
        """GIVEN compiled graph → THEN load_session_context runs before classify_intent."""
        node_names = list(build_orchestrator().compile().nodes.keys())
        assert "load_session_context" in node_names

    @pytest.mark.asyncio
    async def test_graph_flow_with_session_context(self):
        """GIVEN full graph with checkpointer → THEN session_context flows end-to-end."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = f"/tmp/sc_integration_{uuid.uuid4().hex}.db"
        conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(conn)
        graph = build_orchestrator().compile(checkpointer=saver)

        session_id = "sc-session-1"
        config = {"configurable": {"thread_id": session_id}}

        # Use real async functions instead of AsyncMock to avoid LangGraph
        # checkpointer serialization warnings (AsyncMockMixin._execute_mock_call)
        async def _get_summary(*a, **kw):
            return {"weak_topics": ["cálculo"]}

        async def _resolve_id(*a, **kw):
            return "stu-1"

        async def _list_files(*a, **kw):
            return [
                {
                    "id": "d1",
                    "file_name": "apunte.pdf",
                    "classification": "apunte",
                    "topics_json": '["cálculo"]',
                    "chunks_count": 3,
                    "ingested_at": "2026-06-26",
                    "session_id": session_id,
                }
            ]

        async def _get_profile(*a, **kw):
            return {
                "session_id": session_id,
                "topic_scores": {"cálculo": [8.0]},
                "weak_topics": [],
                "exam_count": 1,
                "average_score": 8.0,
            }

        with (
            patch("src.tools.get_student_summary.get_student_summary", new=_get_summary),
            patch("src.memory.schema.resolve_student_id", new=_resolve_id),
            patch("src.memory.schema.list_session_files", new=_list_files),
            patch("src.memory.schema.get_session_profile", new=_get_profile),
            patch("src.agents.orchestrator.get_structured_llm") as mock_structured,
            patch("src.agents.orchestrator._get_llm") as mock_llm_factory,
        ):
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
                "R", (), {"content": "¡Hola! Veo que tenés apunte.pdf cargado."}
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

        await conn.close()


# ── T-018c: Tool exported from __init__.py ──────────────────────────────────


class TestToolExports:
    """T-018c: Both tools are properly exported from src.tools."""

    def test_init_exports_list_session_files(self):
        """GIVEN tools/__init__.py → THEN list_session_files is importable."""
        from src.tools import list_session_files

        assert list_session_files is not None

    def test_init_exports_get_session_progress(self):
        """GIVEN tools/__init__.py → THEN get_session_progress is importable."""
        from src.tools import get_session_progress

        assert get_session_progress is not None
