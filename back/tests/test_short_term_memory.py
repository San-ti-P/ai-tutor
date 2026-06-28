"""Tests for short-term conversation memory (T-014..T-015)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig

from src.agents.orchestrator import (
    OrchestratorState,
    build_orchestrator,
    classify_intent,
    synthesize_response,
)


class TestMessagesHistoryState:
    """T-014: messages_history field exists and synthesize appends."""

    def test_state_accepts_messages_history(self):
        """GIVEN messages_history key → THEN OrchestratorState accepts it."""
        state: OrchestratorState = {
            "session_id": "s1",
            "user_message": "hola",
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
            "messages_history": [{"role": "user", "content": "prev"}],
        }
        assert state["messages_history"][0]["content"] == "prev"

    def test_synthesize_appends_exchange(self):
        """GIVEN synthesize_response → THEN it returns new messages_history entries."""
        state: OrchestratorState = {
            "session_id": "s1",
            "user_message": "Hola, ¿cómo estás?",
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
            "messages_history": [],
        }

        with patch("src.agents.orchestrator._get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type("R", (), {"content": "Un límite es..."})()
            mock_llm_factory.return_value = mock_llm

            result = synthesize_response(state, RunnableConfig())

        assert "messages_history" in result
        assert len(result["messages_history"]) == 2
        assert result["messages_history"][0] == {
            "role": "user",
            "content": "Hola, ¿cómo estás?",
        }
        assert result["messages_history"][1]["role"] == "assistant"
        assert "Un límite es..." in result["messages_history"][1]["content"]

    def test_classify_uses_last_n_history(self):
        """GIVEN >10 messages_history → THEN classify prompt only mentions last 10."""
        history = [
            {"role": "user", "content": f"message-{i:02d}"} for i in range(12)
        ]
        state: OrchestratorState = {
            "session_id": "s1",
            "user_message": "último",
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
            "messages_history": history,
        }

        with patch("src.agents.orchestrator.get_structured_llm") as mock_structured:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type(
                "R", (), {"intent": "general_chat", "confidence": 0.99}
            )()
            mock_structured.return_value = mock_llm

            classify_intent(state)

        prompt = mock_llm.invoke.call_args[0][0]
        assert "message-02" in prompt  # last 10: message-02..message-11
        assert "message-00" not in prompt
        assert "message-01" not in prompt


class TestShortTermMemoryAcrossTurns:
    """T-015: history restored from checkpointer across two turns."""

    @pytest.mark.asyncio
    async def test_history_restored_across_invocations(self):
        """GIVEN two sequential graph invocations → THEN second sees first history."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = f"/tmp/stm_test_{uuid.uuid4().hex}.db"
        conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(conn)
        graph = build_orchestrator().compile(checkpointer=saver)

        session_id = "stm-session-1"
        config = {"configurable": {"thread_id": session_id}}

        with patch("src.agents.orchestrator._get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = type(
                "R", (), {"content": "Respuesta del turno 1"}
            )()
            mock_llm_factory.return_value = mock_llm

            # Turn 1
            await graph.ainvoke(
                {
                    "session_id": session_id,
                    "user_message": "hola",
                    "intent": "general_chat",
                    "confidence": 0.99,
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

            # Turn 2: check that messages_history was restored
            final = await graph.ainvoke(
                {
                    "session_id": session_id,
                    "user_message": "otra pregunta",
                    "intent": "general_chat",
                    "confidence": 0.99,
                    "plan": [],
                    "current_step": 0,
                    "results": [],
                    "errors": [],
                    "response": "",
                    "status": "pending",
                    "iteration_count": 0,
                    "student_profile": None,
                },
                config=config,
            )

        assert final["messages_history"]
        user_contents = [
            h["content"] for h in final["messages_history"] if h["role"] == "user"
        ]
        assert "hola" in user_contents
        assert "otra pregunta" in user_contents

        await conn.close()

    @pytest.mark.asyncio
    async def test_first_message_empty_history(self):
        """GIVEN first invocation → THEN classify works with empty history."""
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = f"/tmp/stm_empty_{uuid.uuid4().hex}.db"
        conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(conn)
        graph = build_orchestrator().compile(checkpointer=saver)

        session_id = "stm-session-2"
        config = {"configurable": {"thread_id": session_id}}

        async def _get_summary(*a, **kw):
            return {}

        with patch("src.agents.orchestrator._get_llm") as mock_llm_factory, patch(
            "src.agents.orchestrator.get_structured_llm"
        ) as mock_structured, patch(
            "src.tools.get_student_summary.get_student_summary", new=_get_summary
        ):
            mock_structured.return_value = type(
                "M",
                (),
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
                "R", (), {"content": "Respuesta inicial"}
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

        assert final["response"] == "Respuesta inicial"
        assert len(final["messages_history"]) == 2
        await conn.close()
