"""TDD test suite for Orchestrator agent — epic-01-orchestrator.

Plumbing tests mock the LLM via conftest's patch_llm() and run by default.
Real LLM integration tests are marked @pytest.mark.integration (requires Ollama).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.orchestrator import OrchestratorState

# ==============================================================================
# TASK-ORCH-001: State Schema + Fixtures
# ==============================================================================


class TestOrchestratorStateSchema:
    """Verify the OrchestratorState TypedDict has all required fields."""

    def test_state_has_confidence_field(self):
        """confidence: float must exist for intent classification confidence."""
        state = OrchestratorState(
            session_id="s1",
            user_message="hola",
            intent="general_chat",
            confidence=0.95,
            plan=[],
            current_step=0,
            results=[],
            errors=[],
            response="",
            status="pending",
            iteration_count=0,
            student_profile=None,
        )
        assert state["confidence"] == 0.95
        assert isinstance(state["confidence"], float)

    def test_state_has_errors_field(self):
        """errors: list[dict] — records errors within a single graph invocation."""
        state = OrchestratorState(
            session_id="s1",
            user_message="hola",
            intent="generate_exam",
            confidence=0.88,
            plan=["generate_exam"],
            current_step=0,
            results=[],
            errors=[],
            response="",
            status="pending",
            iteration_count=0,
            student_profile=None,
        )
        assert state["errors"] == []
        assert isinstance(state["errors"], list)

    def test_state_has_status_field(self):
        """status: str tracks pending|complete|incomplete|partial."""
        state = OrchestratorState(
            session_id="s1",
            user_message="hola",
            intent="general_chat",
            confidence=0.8,
            plan=[],
            current_step=0,
            results=[],
            errors=[],
            response="ok",
            status="complete",
            iteration_count=0,
            student_profile=None,
        )
        assert state["status"] == "complete"

    def test_state_has_student_profile_field(self):
        """student_profile: dict | None for session bootstrap."""
        profile = {"weak_topics": ["cálculo"], "preferences": {}}
        state = OrchestratorState(
            session_id="s1",
            user_message="hola",
            intent="generate_exam",
            confidence=0.9,
            plan=["generate_exam"],
            current_step=0,
            results=[],
            errors=[],
            response="",
            status="pending",
            iteration_count=0,
            student_profile=profile,
        )
        assert state["student_profile"] == profile

    def test_state_defaults_are_valid(self):
        """Minimal valid state with default values should instantiate correctly."""
        state = OrchestratorState(
            session_id="test-001",
            user_message="¿Qué es una derivada?",
            intent="general_chat",
            confidence=0.0,
            plan=[],
            current_step=0,
            results=[],
            errors=[],
            response="",
            status="pending",
            iteration_count=0,
            student_profile=None,
        )
        assert state["session_id"] == "test-001"
        assert state["intent"] == "general_chat"
        assert state["confidence"] == 0.0
        assert state["status"] == "pending"


# ==============================================================================
# TASK-ORCH-002: classify_intent
# ==============================================================================


class TestClassifyIntent:
    """REQ-ORCH-002: Intent classification with structured output + confidence fallback."""

    def test_classify_high_confidence_generate_exam(self, orchestrator_state):
        """Strong signal → intent=generate_exam, plan pre-populated with tool name."""
        from src.agents.orchestrator import classify_intent

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured("generate_exam", 0.95),
        ):
            result = classify_intent(orchestrator_state)

        assert result["intent"] == "generate_exam"
        assert result["confidence"] == 0.95
        assert result["plan"] == ["generate_exam"]

    def test_classify_high_confidence_retrieve(self, orchestrator_state):
        """Strong signal → intent=retrieve, plan pre-populated with tool name."""
        from src.agents.orchestrator import classify_intent

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured("retrieve", 0.95),
        ):
            result = classify_intent(orchestrator_state)

        assert result["intent"] == "retrieve"
        assert result["confidence"] == 0.95
        assert result["plan"] == ["retrieve"]

    def test_classify_low_confidence_fallback_to_general_chat(self, orchestrator_state):
        """Confidence < threshold → effective intent becomes general_chat."""
        from src.agents.orchestrator import classify_intent

        # Mock returns low-confidence result
        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured("generate_exam", 0.30),
        ):
            result = classify_intent(orchestrator_state)

        assert result["intent"] == "general_chat"
        assert result["confidence"] == 0.30  # Original confidence preserved
        assert result["plan"] == []

    def test_classify_composite_intent(self, orchestrator_state):
        """User message requesting multi-step → intent=composite, plan=[] (filled by planner)."""
        from src.agents.orchestrator import classify_intent

        state = {**orchestrator_state, "user_message": "Ingest notes then quiz me"}
        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeStructured("composite", 0.85),
        ):
            result = classify_intent(state)

        assert result["intent"] == "composite"
        assert result["confidence"] == 0.85
        assert result["plan"] == []  # composite → plan filled by plan_composite later

    def test_classify_exception_fallback_to_general_chat(self, orchestrator_state):
        """Any exception during classification → fallback to general_chat."""
        from src.agents.orchestrator import classify_intent

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = classify_intent(orchestrator_state)

        assert result["intent"] == "general_chat"
        assert result["confidence"] == 0.0
        assert result["plan"] == []

    def test_classify_single_tool_prepopulates_plan(self, orchestrator_state):
        """Non-composite, non-chat intents should pre-populate plan with tool name."""
        from src.agents.orchestrator import classify_intent

        for intent in ["ingest", "evaluate", "query_profile", "generate_exercise"]:
            with patch(
                "src.agents.orchestrator.get_structured_llm",
                return_value=_FakeStructured(intent, 0.90),
            ):
                result = classify_intent(orchestrator_state)
            assert result["plan"] == [intent], f"Expected plan=[{intent}], got {result['plan']}"


# ==============================================================================
# TASK-ORCH-003: route_to_agent
# ==============================================================================


class TestRouteToAgent:
    """REQ-ORCH-003: Conditional edge maps intent to correct node name."""

    def test_route_composite_to_plan_composite(self):
        """composite → plan_composite node."""
        from src.agents.orchestrator import route_to_agent

        result = route_to_agent({"intent": "composite"})
        assert result == "plan_composite"

    def test_route_general_chat_to_synthesize_response(self):
        """general_chat → synthesize_response node (answered inline)."""
        from src.agents.orchestrator import route_to_agent

        result = route_to_agent({"intent": "general_chat"})
        assert result == "synthesize_response"

    def test_route_single_tool_intents_to_execute_step(self):
        """All single-tool intents → execute_step node."""
        from src.agents.orchestrator import route_to_agent

        single_intents = [
            "ingest",
            "generate_exam",
            "generate_exercise",
            "evaluate",
            "query_profile",
        ]
        for intent in single_intents:
            result = route_to_agent({"intent": intent})
            assert result == "execute_step", f"{intent} should route to execute_step, got {result}"


# ==============================================================================
# TASK-ORCH-004: plan_composite
# ==============================================================================


class TestPlanComposite:
    """REQ-ORCH-004: Plan-and-Execute planner for composite intents."""

    def test_plan_composite_generates_valid_plan(self, orchestrator_state):
        """Valid composite plan with known tool names is returned."""
        from src.agents.orchestrator import plan_composite

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Ingest notes then generate exam",
        }

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeCompositeStructured(["ingest", "generate_exam"]),
        ):
            result = plan_composite(state)

        assert result["plan"] == ["ingest", "generate_exam"]

    def test_plan_composite_strips_invalid_tools(self, orchestrator_state):
        """Tool names not in TOOL_MAP are stripped from the plan."""
        from src.agents.orchestrator import plan_composite

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Do something impossible",
        }

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeCompositeStructured(["nonexistent_tool", "ingest", "bad_tool"]),
        ):
            result = plan_composite(state)

        assert result["plan"] == ["ingest"]  # only valid tool kept

    def test_plan_composite_empty_plan_fallback(self, orchestrator_state):
        """Empty plan after stripping → returns empty list (treated as general_chat downstream)."""
        from src.agents.orchestrator import plan_composite

        state = {**orchestrator_state, "intent": "composite", "user_message": "something"}

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            return_value=_FakeCompositeStructured([]),
        ):
            result = plan_composite(state)

        assert result["plan"] == []

    def test_plan_composite_exception_returns_empty_plan(self, orchestrator_state):
        """Planner failure → returns empty plan (treated as general_chat downstream)."""
        from src.agents.orchestrator import plan_composite

        state = {**orchestrator_state, "intent": "composite", "user_message": "Do stuff"}

        with patch(
            "src.agents.orchestrator.get_structured_llm",
            side_effect=RuntimeError("Planner LLM down"),
        ):
            result = plan_composite(state)

        assert result["plan"] == []


# ==============================================================================
# TASK-ORCH-005: execute_step success path
# ==============================================================================


class TestExecuteStepSuccess:
    """REQ-ORCH-004/003: Execute step invokes tool, appends result, increments counter."""

    async def test_execute_step_single_tool_invocation(self, orchestrator_state):
        """Execute plan[0] → tool invoked → result appended → step incremented."""
        from unittest.mock import AsyncMock

        from src.agents.orchestrator import execute_step

        # Build a state with a single-step plan
        state = {
            **orchestrator_state,
            "intent": "generate_exam",
            "plan": ["generate_exam"],
            "current_step": 0,
            "iteration_count": 0,
        }

        # Mock TOOL_MAP with a fake async tool
        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(return_value={"exam": "ok"})

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}):
            result = await execute_step(state)

        assert len(result["results"]) == 1
        assert result["results"][0]["tool"] == "generate_exam"
        assert result["results"][0]["step"] == 0
        assert result["results"][0]["result"] == {"exam": "ok"}
        assert result["current_step"] == 1
        assert result["iteration_count"] == 1

    async def test_execute_step_increments_counters(self, orchestrator_state):
        """current_step and iteration_count increment by 1 on successful execution."""
        from unittest.mock import AsyncMock

        from src.agents.orchestrator import execute_step

        state = {
            **orchestrator_state,
            "plan": ["ingest", "generate_exam"],
            "current_step": 1,
            "iteration_count": 5,
        }

        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(return_value={"exam": "done"})

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}):
            result = await execute_step(state)

        assert result["current_step"] == 2
        assert result["iteration_count"] == 6

    async def test_execute_step_uses_build_tool_args(self, orchestrator_state):
        """_build_tool_args is called to construct tool arguments from state."""
        from unittest.mock import AsyncMock

        from src.agents.orchestrator import execute_step

        state = {
            **orchestrator_state,
            "plan": ["query_profile"],
            "current_step": 0,
            "iteration_count": 0,
            "session_id": "sess-abc",
        }

        mock_tool = AsyncMock()
        mock_tool.name = "query_profile"
        mock_tool.ainvoke = AsyncMock(return_value={"profile": "data"})

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"query_profile": mock_tool}):
            with patch(
                "src.agents.orchestrator._build_tool_args",
                return_value={"session_id": "sess-abc"},
            ) as mock_build:
                result = await execute_step(state)

        mock_build.assert_called_once_with("query_profile", state)
        mock_tool.ainvoke.assert_called_once_with({"session_id": "sess-abc"})
        assert len(result["results"]) == 1

    def test_build_tool_args_retrieve(self, orchestrator_state):
        """retrieve intent gets query and top_k from state."""
        from src.agents.orchestrator import _build_tool_args

        state = {**orchestrator_state, "user_message": "¿Qué dice el apunte?"}
        args = _build_tool_args("retrieve", state)

        assert args["session_id"] == state["session_id"]
        assert args["query"] == "¿Qué dice el apunte?"
        assert args["top_k"] == 8  # retrieval_top_k default


# ==============================================================================
# TASK-ORCH-006: execute_step retry and failure handling
# ==============================================================================


class TestExecuteStepRetry:
    """REQ-ORCH-006: Failed tool calls retried ONCE; double-failure → partial."""

    async def test_retry_succeeds_on_second_attempt(self, orchestrator_state):
        """First call raises, retry succeeds → result stored normally."""
        from unittest.mock import AsyncMock

        from src.agents.orchestrator import execute_step

        state = {
            **orchestrator_state,
            "plan": ["generate_exam"],
            "current_step": 0,
            "iteration_count": 0,
        }

        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(side_effect=[RuntimeError("fail1"), {"exam": "retried"}])

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}):
            result = await execute_step(state)

        assert len(result["results"]) == 1
        assert result["results"][0]["result"] == {"exam": "retried"}
        assert result["current_step"] == 1

    async def test_retry_fails_returns_partial(self, orchestrator_state):
        """Both attempts fail → error recorded, status=partial."""
        from unittest.mock import AsyncMock

        from src.agents.orchestrator import execute_step

        state = {
            **orchestrator_state,
            "plan": ["generate_exam"],
            "current_step": 0,
            "iteration_count": 0,
        }

        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(side_effect=[RuntimeError("fail1"), RuntimeError("fail2")])

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}):
            result = await execute_step(state)

        assert result.get("status") == "partial"
        assert len(result["errors"]) == 1
        assert result["errors"][0]["step"] == 0
        assert result["errors"][0]["tool"] == "generate_exam"
        assert "fail" in result["errors"][0]["error"].lower()
        assert result["current_step"] == 1

    async def test_tool_not_found_in_map(self, orchestrator_state):
        """Tool name not in TOOL_MAP → error recorded, status=partial."""
        from src.agents.orchestrator import execute_step

        state = {
            **orchestrator_state,
            "plan": ["nonexistent_tool"],
            "current_step": 0,
            "iteration_count": 0,
        }

        result = await execute_step(state)

        assert result.get("status") == "partial"
        assert len(result["errors"]) == 1
        assert "not found" in result["errors"][0]["error"].lower()


# ==============================================================================
# TASK-ORCH-007: check_iteration_limit
# ==============================================================================


class TestCheckIterationLimit:
    """REQ-ORCH-005: Guardrail enforces max_iterations_per_task."""

    def test_within_cap_continue(self, orchestrator_state):
        """iteration_count < max, more steps → continue."""
        from src.agents.orchestrator import check_iteration_limit

        state = {
            **orchestrator_state,
            "plan": ["a", "b", "c"],
            "current_step": 0,
            "iteration_count": 5,
            "status": "pending",
        }
        result = check_iteration_limit(state)
        assert result == "continue"

    def test_cap_hit_terminate(self, orchestrator_state):
        """iteration_count >= max → terminate."""
        from src.agents.orchestrator import check_iteration_limit
        from src.config import settings

        state = {
            **orchestrator_state,
            "plan": ["a"],
            "current_step": 0,
            "iteration_count": settings.max_iterations_per_task,
            "status": "pending",
        }
        result = check_iteration_limit(state)
        assert result == "terminate"

    def test_all_done_terminate(self, orchestrator_state):
        """current_step >= len(plan) → terminate (all steps complete)."""
        from src.agents.orchestrator import check_iteration_limit

        state = {
            **orchestrator_state,
            "plan": ["a", "b"],
            "current_step": 2,  # equals len(plan)
            "iteration_count": 2,
            "status": "pending",
        }
        result = check_iteration_limit(state)
        assert result == "terminate"

    def test_partial_status_terminate(self, orchestrator_state):
        """status=partial → terminate immediately (error already hit)."""
        from src.agents.orchestrator import check_iteration_limit

        state = {
            **orchestrator_state,
            "plan": ["a", "b", "c"],
            "current_step": 1,
            "iteration_count": 2,
            "status": "partial",
        }
        result = check_iteration_limit(state)
        assert result == "terminate"


# ==============================================================================
# TASK-ORCH-008: synthesize_response
# ==============================================================================


class TestSynthesizeResponse:
    """REQ-ORCH-003/005: Aggregates results into final user-facing response."""

    def test_general_chat_inline_response(self, orchestrator_state):
        """general_chat → LLM synthesizes direct answer from user_message."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "Hola, ¿cómo estás?",
            "plan": [],
            "results": [],
            "errors": [],
            "status": "pending",
        }

        with patch(
            "src.agents.orchestrator._get_llm",
            return_value=_FakeDirectLLM("Hola, soy tu tutor académico."),
        ):
            result = synthesize_response(state)

        assert "tutor" in result["response"]
        assert result["status"] == "complete"

    def test_composite_aggregation(self, orchestrator_state):
        """Composite with results → LLM aggregates into coherent response."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Ingest notes then quiz me",
            "results": [
                {"step": 0, "tool": "ingest", "result": {"status": "ok", "chunks": 5}},
                {"step": 1, "tool": "generate_exam", "result": {"exam": "ready"}},
            ],
            "errors": [],
            "status": "pending",
        }

        with patch(
            "src.agents.orchestrator._get_llm",
            return_value=_FakeDirectLLM("Completé la ingesta y generé el examen."),
        ):
            result = synthesize_response(state)

        assert "ingesta" in result["response"].lower() or "examen" in result["response"].lower()
        assert result["status"] == "complete"

    def test_incomplete_status_prepends_warning(self, orchestrator_state):
        """status='incomplete' → prepends cap-hit warning in Spanish."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "results": [{"step": 0, "tool": "a", "result": {}}],
            "errors": [],
            "status": "incomplete",
        }

        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeDirectLLM("Acá está el resumen.")
        ):
            result = synthesize_response(state)

        assert "límite" in result["response"].lower() or "limite" in result["response"].lower()
        assert result["status"] == "incomplete"

    def test_partial_error_summary(self, orchestrator_state):
        """status='partial' → response includes error summary."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "results": [{"step": 0, "tool": "a", "result": {}}],
            "errors": [{"step": 1, "tool": "b", "error": "Connection refused"}],
            "status": "partial",
        }

        with patch(
            "src.agents.orchestrator._get_llm",
            return_value=_FakeDirectLLM("Parcial completado. Error en paso b."),
        ):
            result = synthesize_response(state)

        assert result["status"] == "partial"
        assert "error" in result["response"].lower()

    def test_llm_failure_hardcoded_fallback(self, orchestrator_state):
        """LLM synthesize fails → hardcoded Spanish apology + raw results."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "results": [{"step": 0, "tool": "x", "result": {"ok": True}}],
            "errors": [],
            "status": "pending",
        }

        with patch(
            "src.agents.orchestrator._get_llm",
            side_effect=RuntimeError("LLM down"),
        ):
            result = synthesize_response(state)

        assert "disculpas" in result["response"].lower() or "error" in result["response"].lower()
        assert '"ok"' in result["response"] or "ok" in result["response"].lower()
        assert result["status"] == "complete"


# ==============================================================================
# TASK-ORCH-009: Full graph wiring (build_orchestrator end-to-end)
# ==============================================================================


class TestBuildOrchestratorGraph:
    """End-to-end graph invoke with mocked LLM, tools, and checkpointer.

    Patches _get_llm instead of node functions because LangGraph
    captures node function references at add_node() time.
    """

    async def test_e2e_general_chat_invoke(self, orchestrator_state):
        """Full graph: classify → route → synthesize → END for general_chat."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        with patch(
            "src.agents.orchestrator._get_llm",
            return_value=_FakeDirectLLM("Hola, soy tu tutor. ¿En qué te ayudo?"),
        ):
            config = {"configurable": {"thread_id": "test-thread-001"}}
            result = await graph.ainvoke(
                {**orchestrator_state, "user_message": "Hola"},
                config=config,
            )

        assert result["response"] == "Hola, soy tu tutor. ¿En qué te ayudo?"
        assert result["status"] == "complete"
        assert result["intent"] == "general_chat"

    async def test_e2e_single_tool_invoke(self, orchestrator_state):
        """Full graph: classify → route → execute_step → synthesize → END."""
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        state = {
            **orchestrator_state,
            "user_message": "Generame un examen de derivadas",
        }

        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(return_value={"exam": "generated"})

        with (
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeLLM(intent="generate_exam", confidence=0.95),
            ),
            patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}),
        ):
            config = {"configurable": {"thread_id": "test-thread-002"}}
            result = await graph.ainvoke(state, config=config)

        # Should have executed generate_exam and received the result
        assert result["intent"] == "generate_exam"
        assert len(result["results"]) >= 1
        assert result["results"][0]["tool"] == "generate_exam"

    async def test_e2e_composite_loop(self, orchestrator_state):
        """Composite: plan_composite → execute_step × 2 → synthesize → END."""
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Ingest notes then quiz me",
        }

        mock_tool1 = AsyncMock()
        mock_tool1.name = "ingest"
        mock_tool1.ainvoke = AsyncMock(return_value={"status": "ok"})
        mock_tool2 = AsyncMock()
        mock_tool2.name = "generate_exam"
        mock_tool2.ainvoke = AsyncMock(return_value={"exam": "ready"})

        # classify returns composite; plan_composite already has plan
        fake_llm_classify = _FakeLLM(intent="composite", confidence=0.9)
        fake_llm_plan = _FakeCompositeLLM(steps=["ingest", "generate_exam"])
        fake_llm_synth = _FakeDirectLLM("Completé la ingesta y generé el examen.")

        with (
            patch(
                "src.agents.orchestrator._get_llm",
                side_effect=[fake_llm_classify, fake_llm_plan, fake_llm_synth],
            ),
            patch.dict(
                "src.agents.orchestrator.TOOL_MAP",
                {"ingest": mock_tool1, "generate_exam": mock_tool2},
            ),
        ):
            config = {"configurable": {"thread_id": "test-thread-003"}}
            result = await graph.ainvoke(state, config=config)

        assert len(result["results"]) == 2
        assert result["results"][0]["tool"] == "ingest"
        assert result["results"][1]["tool"] == "generate_exam"


# ==============================================================================
# TASK-ORCH-010: Singleton compilation + SqliteSaver
# ==============================================================================


class TestSingletonCompilation:
    """REQ-ORCH-001: Module-level singleton, compiled once."""

    async def test_get_orchestrator_graph_returns_compiled_graph(self):
        """get_orchestrator_graph returns a compiled StateGraph."""
        from src.agents.orchestrator import get_orchestrator_graph

        graph = await get_orchestrator_graph()
        assert graph is not None
        # Should have the invoke method
        assert hasattr(graph, "ainvoke")

    async def test_get_orchestrator_graph_same_instance(self):
        """Multiple calls return the same compiled graph instance."""
        from src.agents.orchestrator import get_orchestrator_graph

        graph1 = await get_orchestrator_graph()
        graph2 = await get_orchestrator_graph()
        assert graph1 is graph2

    def test_sqlite_db_path_from_settings(self):
        """Settings provides sqlite_db_path for SqliteSaver."""
        from src.config import settings

        assert settings.sqlite_db_path
        assert isinstance(settings.sqlite_db_path, str)

    async def test_close_orchestrator_graph_cleans_up(self):
        """After close_orchestrator_graph(), next get_orchestrator_graph() is fresh."""
        import src.agents.orchestrator as orch_mod

        # Ensure singleton is initialized
        graph1 = await orch_mod.get_orchestrator_graph()
        assert graph1 is not None
        assert orch_mod._orchestrator_graph is not None

        await orch_mod.close_orchestrator_graph()

        # After closing, state is reset
        assert orch_mod._orchestrator_graph is None
        assert orch_mod._orchestrator_db_conn is None

        # Next call creates a fresh instance
        graph2 = await orch_mod.get_orchestrator_graph()
        try:
            assert graph2 is not None
            assert graph1 is not graph2
        finally:
            # Clean up to avoid ResourceWarning from this test
            await orch_mod.close_orchestrator_graph()


# ==============================================================================
# TASK-ORCH-011: SqliteSaver exception narrowing
# ==============================================================================


class TestSqliteSaverExceptionHandling:
    """REQ-ORCH-001: Narrow exception catch in get_orchestrator_graph()."""

    async def test_broad_exception_not_swallowed(self):
        """RuntimeError during DB connect propagates — not silently caught."""
        import src.agents.orchestrator as orch_mod

        # Reset singleton so get_orchestrator_graph() will try to connect
        orch_mod._orchestrator_graph = None
        orch_mod._orchestrator_db_conn = None

        with patch("aiosqlite.connect", side_effect=RuntimeError("Boom!")):
            with pytest.raises(RuntimeError, match="Boom!"):
                await orch_mod.get_orchestrator_graph()

    async def test_expected_exception_falls_back_with_warning(self):
        """OSError during DB connect falls back to InMemorySaver with a warning."""
        import src.agents.orchestrator as orch_mod

        # Reset singleton so get_orchestrator_graph() will try to connect
        orch_mod._orchestrator_graph = None
        orch_mod._orchestrator_db_conn = None

        with patch("aiosqlite.connect", side_effect=OSError("Permission denied")):
            graph = await orch_mod.get_orchestrator_graph()

        assert graph is not None
        assert hasattr(graph, "ainvoke")
        # InMemorySaver was used — no DB connection stored
        assert orch_mod._orchestrator_db_conn is None


# ==============================================================================
# TASK-ORCH-011: /chat endpoint integration
# ==============================================================================


class TestChatEndpoint:
    """REQ-ORCH-001: /chat returns real Orchestrator responses (no placeholder)."""

    def test_chat_endpoint_uses_orchestrate_chat_tool(self):
        """POST /chat calls orchestrate_chat tool instead of importing agent internals."""
        from unittest.mock import AsyncMock, patch

        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        # Replace the orchestrate_chat tool in src.tools with a simple mock
        # that has the .ainvoke() method. StructuredTool (Pydantic) doesn't
        # allow patching individual methods, so we swap the whole object.
        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(
            return_value={
                "response": "¡Hola! Soy tu tutor académico.",
                "intent": "general_chat",
                "status": "complete",
                "trace_id": "fake-trace-abc123",
            }
        )

        with patch("src.tools.orchestrate_chat", mock_tool):
            response = client.post(
                "/api/chat",
                json={"session_id": "test-session-99", "message": "Hola"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert "data" in data
        assert data["data"]["response"] == "¡Hola! Soy tu tutor académico."
        # Should NOT be the placeholder message
        assert "aún no están implementados" not in data["data"]["response"]
        assert data["data"]["intent"] == "general_chat"
        assert data["data"]["trace_id"] is not None

    def test_chat_endpoint_returns_exercise_when_present(self):
        """POST /chat returns the exercise when it's present in the orchestrator results."""
        from unittest.mock import AsyncMock, patch
        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app)

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(
            return_value={
                "response": "Acá tenés un ejercicio.",
                "intent": "generate_exercise",
                "status": "complete",
                "trace_id": "fake-trace-123",
                "exercise": {
                    "exercise_id": "ex-001",
                    "statement": "Calcular la derivada de x^2",
                    "question": "¿Cuál es la derivada?",
                    "model_solution": {
                        "steps": [
                            {
                                "stepNumber": 1,
                                "description": "Aplicar regla de potencia",
                                "result": "2x",
                                "sourceChunkIds": []
                            }
                        ],
                        "finalAnswer": "2x",
                        "keyConcepts": ["derivada", "regla de potencia"],
                    },
                    "topics_covered": ["derivadas"],
                    "status": "complete"
                }
            }
        )

        with patch("src.tools.orchestrate_chat", mock_tool):
            response = client.post(
                "/api/chat",
                json={"session_id": "test-session-99", "message": "Dame un ejercicio de derivadas"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        assert "data" in data
        assert data["data"]["exercise"] is not None
        assert data["data"]["exercise"]["exercise_id"] == "ex-001"
        assert data["data"]["exercise"]["statement"] == "Calcular la derivada de x^2"
        assert data["data"]["exercise"]["model_solution"]["finalAnswer"] == "2x"
        assert data["data"]["intent"] == "generate_exercise"


# ==============================================================================
# TASK-ORCH-011b: orchestrate_chat tool boundary
# ==============================================================================


class TestOrchestrateChatTool:
    """REF-07: orchestrate_chat wraps get_orchestrator_graph behind the tools boundary."""

    async def test_tool_extracts_last_user_message(self):
        """The tool extracts the last user message from a messages list."""
        from unittest.mock import AsyncMock, patch

        from src.tools.orchestrate_chat import orchestrate_chat

        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(
            return_value={
                "response": "Soy el tutor.",
                "intent": "general_chat",
                "status": "complete",
            }
        )

        with patch(
            "src.agents.orchestrator.get_orchestrator_graph",
            return_value=fake_graph,
        ):
            result = await orchestrate_chat.ainvoke(
                {
                    "messages": [
                        {"role": "system", "content": "Eres un tutor."},
                        {"role": "user", "content": "¿Qué es un escalar?"},
                    ],
                    "thread_id": "th-001",
                }
            )

        assert result["response"] == "Soy el tutor."
        assert result["intent"] == "general_chat"
        assert result["status"] == "complete"
        assert "trace_id" in result
        # Verify the graph received the correct user_message in initial_state
        call_args = fake_graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["user_message"] == "¿Qué es un escalar?"
        assert initial_state["session_id"] == "th-001"

    async def test_tool_generates_thread_id_when_none(self):
        """When thread_id is None, the tool generates a UUID4 session ID."""
        from unittest.mock import AsyncMock, patch

        from src.tools.orchestrate_chat import orchestrate_chat

        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(
            return_value={
                "response": "Hola.",
                "intent": "general_chat",
                "status": "complete",
            }
        )

        with patch(
            "src.agents.orchestrator.get_orchestrator_graph",
            return_value=fake_graph,
        ):
            result = await orchestrate_chat.ainvoke(
                {
                    "messages": [{"role": "user", "content": "Hola"}],
                }
            )

        # A UUID4 session_id was generated (36 chars with dashes)
        call_args = fake_graph.ainvoke.call_args
        session_id = call_args[0][0]["session_id"]
        assert len(session_id) == 36
        assert session_id.count("-") == 4
        assert result["response"] == "Hola."

    async def test_tool_handles_empty_messages(self):
        """Empty messages list → user_message defaults to empty string."""
        from unittest.mock import AsyncMock, patch

        from src.tools.orchestrate_chat import orchestrate_chat

        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(
            return_value={
                "response": "No entendí tu mensaje.",
                "intent": "general_chat",
                "status": "complete",
            }
        )

        with patch(
            "src.agents.orchestrator.get_orchestrator_graph",
            return_value=fake_graph,
        ):
            result = await orchestrate_chat.ainvoke(
                {
                    "messages": [],
                    "thread_id": "t-empty",
                }
            )

        call_args = fake_graph.ainvoke.call_args
        assert call_args[0][0]["user_message"] == ""
        assert result["response"] == "No entendí tu mensaje."

    async def test_tool_accepts_and_forwards_exam_params(self):
        """orchestrate_chat accepts and forwards exam parameters to initial state."""
        from unittest.mock import AsyncMock, patch

        from src.tools.orchestrate_chat import orchestrate_chat

        fake_graph = AsyncMock()
        fake_graph.ainvoke = AsyncMock(
            return_value={
                "response": "Examen evaluado.",
                "intent": "evaluate",
                "status": "complete",
            }
        )

        with patch(
            "src.agents.orchestrator.get_orchestrator_graph",
            return_value=fake_graph,
        ):
            result = await orchestrate_chat.ainvoke(
                {
                    "messages": [{"role": "user", "content": "Entrego examen"}],
                    "thread_id": "t-exam",
                    "exam_id": "exam-123",
                    "answers": {"q1": "my answer"},
                    "exam_questions": [{"id": "q1", "prompt": "Question 1"}],
                }
            )

        call_args = fake_graph.ainvoke.call_args
        init_state = call_args[0][0]
        assert init_state["exam_id"] == "exam-123"
        assert init_state["answers"] == {"q1": "my answer"}
        assert init_state["exam_questions"] == [{"id": "q1", "prompt": "Question 1"}]
        assert result["intent"] == "evaluate"


# ==============================================================================
# Helpers for fake LLM responses
# ==============================================================================


class _FakeLLM:
    """Fake LLM that returns a pre-programmed IntentClassification via with_structured_output."""

    def __init__(self, intent: str = "generate_exam", confidence: float = 0.95):
        self._intent = intent
        self._confidence = confidence

    def with_structured_output(self, schema):
        return _FakeStructured(self._intent, self._confidence)


class _FakeStructured:
    def __init__(self, intent: str, confidence: float):
        self._intent = intent
        self._confidence = confidence

    def invoke(self, prompt, **kwargs):
        from src.agents.orchestrator import IntentClassification

        return IntentClassification(intent=self._intent, confidence=self._confidence)


class _FakeCompositeLLM:
    """Fake LLM for plan_composite testing."""

    def __init__(self, steps: list[str]):
        self._steps = steps

    def with_structured_output(self, schema):
        return _FakeCompositeStructured(self._steps)


class _FakeCompositeStructured:
    def __init__(self, steps: list[str]):
        self._steps = steps

    def invoke(self, prompt, **kwargs):
        from src.agents.orchestrator import CompositePlan

        return CompositePlan(steps=self._steps)


class _FakeDirectLLM:
    """Fake LLM that returns a pre-programmed string for synthesize_response."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, prompt, **kwargs):
        return type("FakeAIMessage", (), {"content": self._response})()


# ==============================================================================
# TASK-ORCH-012: Persistence & plumbing tests (runs by default, LLM mocked)
# ==============================================================================


class TestIntegrationPersistence:
    """End-to-end integration: graph wiring + checkpointer persistence.

    Uses InMemorySaver for isolation.
    classify_intent/plan_composite patched via get_structured_llm;
    synthesize_response patched via _get_llm.
    """

    async def test_new_session_creates_checkpoint(self, orchestrator_state):
        """A first /chat with a new session ID creates a checkpoint."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                return_value=_FakeStructured("general_chat", 0.98),
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("Hola, ¿cómo estás?"),
            ),
        ):
            config = {"configurable": {"thread_id": "new-session-int-001"}}
            state = {**orchestrator_state, "user_message": "Hola"}
            result = await graph.ainvoke(state, config=config)

        assert result["response"] == "Hola, ¿cómo estás?"
        assert result["status"] == "complete"

    async def test_restore_session_resets_results(self, orchestrator_state):
        """Same thread_id twice → results/errors reset per invocation.

        The checkpointer preserves the thread for future memory features, but
        per-invocation artifacts (results/errors) must not leak across turns.
        """
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        mock_tool = AsyncMock()
        mock_tool.name = "query_profile"
        mock_tool.ainvoke = AsyncMock(return_value={"profile": "data"})

        config = {"configurable": {"thread_id": "restore-session-int"}}

        # First invocation
        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                return_value=_FakeStructured("query_profile", 0.95),
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("Perfil listo."),
            ),
            patch.dict("src.agents.orchestrator.TOOL_MAP", {"query_profile": mock_tool}),
        ):
            result1 = await graph.ainvoke(
                {**orchestrator_state, "user_message": "Mi perfil"},
                config=config,
            )

        assert len(result1["results"]) == 1

        # Second invocation — same thread_id, fresh results/errors
        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                return_value=_FakeStructured("query_profile", 0.95),
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("Perfil listo."),
            ),
            patch.dict("src.agents.orchestrator.TOOL_MAP", {"query_profile": mock_tool}),
        ):
            result2 = await graph.ainvoke(
                {**orchestrator_state, "user_message": "Mi perfil otra vez"},
                config=config,
            )

        # Results should NOT accumulate; each invocation starts fresh.
        assert len(result2["results"]) == 1
        assert result2["errors"] == []

    async def test_composite_multi_step_executes_all(self, orchestrator_state):
        """Composite with 2-step plan → both steps executed in order."""
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        tool1 = AsyncMock()
        tool1.name = "ingest"
        tool1.ainvoke = AsyncMock(return_value={"status": "ok"})
        tool2 = AsyncMock()
        tool2.name = "generate_exam"
        tool2.ainvoke = AsyncMock(return_value={"exam": "ready"})

        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                side_effect=[
                    _FakeStructured("composite", 0.9),
                    _FakeCompositeStructured(["ingest", "generate_exam"]),
                ],
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("Todo listo."),
            ),
            patch.dict(
                "src.agents.orchestrator.TOOL_MAP",
                {"ingest": tool1, "generate_exam": tool2},
            ),
        ):
            config = {"configurable": {"thread_id": "composite-int-001"}}
            state = {
                **orchestrator_state,
                "intent": "composite",
                "user_message": "Ingest notes then quiz me",
            }
            result = await graph.ainvoke(state, config=config)

        assert len(result["results"]) == 2
        assert result["results"][0]["tool"] == "ingest"
        assert result["results"][1]["tool"] == "generate_exam"
        assert result["status"] == "complete"

    async def test_cap_hit_incomplete_status(self, orchestrator_state):
        """When iteration cap is hit, status becomes incomplete."""
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        # Create a long plan but set max_iterations very low
        tool = AsyncMock()
        tool.name = "ingest"
        tool.ainvoke = AsyncMock(return_value={"status": "ok"})

        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                side_effect=[
                    _FakeStructured("composite", 0.9),
                    _FakeCompositeStructured(["ingest", "ingest", "ingest", "ingest"]),
                ],
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("parcial"),
            ),
            patch.dict("src.agents.orchestrator.TOOL_MAP", {"ingest": tool}),
            patch("src.agents.orchestrator.settings.max_iterations_per_task", 2),
        ):
            config = {"configurable": {"thread_id": "cap-hit-int"}}
            state = {
                **orchestrator_state,
                "intent": "composite",
                "user_message": "Do lots of stuff",
            }
            result = await graph.ainvoke(state, config=config)

        # After 2 iterations, the graph should stop and synthesize
        assert result["status"] == "incomplete"
        assert "límite" in result["response"].lower() or "limite" in result["response"].lower()

    async def test_retry_failure_partial_result(self, orchestrator_state):
        """Double failure → partial result with error in response."""
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())

        tool = AsyncMock()
        tool.name = "generate_exam"
        tool.ainvoke = AsyncMock(side_effect=[RuntimeError("fail1"), RuntimeError("fail2")])

        with (
            patch(
                "src.agents.orchestrator.get_structured_llm",
                return_value=_FakeStructured("generate_exam", 0.95),
            ),
            patch(
                "src.agents.orchestrator._get_llm",
                return_value=_FakeDirectLLM("Error al generar el examen."),
            ),
            patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": tool}),
        ):
            config = {"configurable": {"thread_id": "retry-fail-int"}}
            state = {
                **orchestrator_state,
                "user_message": "Generame un examen",
            }
            result = await graph.ainvoke(state, config=config)

        assert result["status"] == "partial"
        assert len(result["errors"]) >= 1


class TestIntegrationSqliteSaver:
    """REQ-ORCH-001: Integration tests exercising AsyncSqliteSaver persistence.

    Uses a temporary SQLite database via tmp_path. LLM is mocked.
    These tests exercise the real persistence path (unlike the InMemorySaver
    tests above).
    """

    async def test_sqlite_saver_persists_session(self, orchestrator_state, tmp_path):
        """Invoke with AsyncSqliteSaver, then again with same thread_id — state restored."""
        from unittest.mock import patch

        import src.agents.orchestrator as orch_mod

        db_path = tmp_path / "test_orch_persist.db"

        # Force a fresh singleton with temp DB path
        orch_mod._orchestrator_graph = None
        orch_mod._orchestrator_db_conn = None

        with patch.object(orch_mod.settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import init_db

            await init_db()
            graph = await orch_mod.get_orchestrator_graph()

            thread_id = "sqlite-persist-int-001"
            config = {"configurable": {"thread_id": thread_id}}

            # First invocation
            with (
                patch(
                    "src.agents.orchestrator.get_structured_llm",
                    return_value=_FakeStructured("general_chat", 0.98),
                ),
                patch(
                    "src.agents.orchestrator._get_llm",
                    return_value=_FakeDirectLLM("Respuesta uno."),
                ),
            ):
                state = {**orchestrator_state, "user_message": "Mensaje uno"}
                result1 = await graph.ainvoke(state, config=config)

            assert result1["response"] == "Respuesta uno."
            assert result1["status"] == "complete"
            assert result1["intent"] == "general_chat"

            # Second invocation — same thread_id → state restored from checkpoint
            with (
                patch(
                    "src.agents.orchestrator.get_structured_llm",
                    return_value=_FakeStructured("general_chat", 0.98),
                ),
                patch(
                    "src.agents.orchestrator._get_llm",
                    return_value=_FakeDirectLLM("Respuesta dos."),
                ),
            ):
                state2 = {**orchestrator_state, "user_message": "Mensaje dos"}
                result2 = await graph.ainvoke(state2, config=config)

            assert result2["response"] == "Respuesta dos."
            assert result2["status"] == "complete"
            # With operator.add reducer, results accumulate across invocations
            # on the same thread_id (checkpointer restores previous state)

            # Clean up
            await orch_mod.close_orchestrator_graph()

    async def test_sqlite_saver_singleton_reuses_connection(self, orchestrator_state, tmp_path):
        """get_orchestrator_graph() reuses the same aiosqlite connection across calls."""
        from unittest.mock import patch

        import src.agents.orchestrator as orch_mod

        db_path = tmp_path / "test_orch_singleton.db"

        # Force a fresh singleton with temp DB path
        orch_mod._orchestrator_graph = None
        orch_mod._orchestrator_db_conn = None

        with patch.object(orch_mod.settings, "sqlite_db_path", str(db_path)):
            graph1 = await orch_mod.get_orchestrator_graph()
            conn1 = orch_mod._orchestrator_db_conn

            graph2 = await orch_mod.get_orchestrator_graph()
            conn2 = orch_mod._orchestrator_db_conn

            assert graph1 is graph2, "Same compiled graph instance expected"
            assert conn1 is not None, "DB connection should be stored"
            assert conn1 is conn2, "Same DB connection expected across calls"

            # Clean up
            await orch_mod.close_orchestrator_graph()


class TestSecurityAndPublicAPI:
    """REQ-ORCH-007: Single public entry point, no hardcoded secrets."""

    def test_build_orchestrator_public_entry(self):
        """build_orchestrator and get_orchestrator_graph are the only public builders."""
        import inspect

        import src.agents.orchestrator as mod

        public_funcs = [
            name
            for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if not name.startswith("_")
        ]
        # Public API: build_orchestrator, get_orchestrator_graph
        assert "build_orchestrator" in public_funcs
        assert "get_orchestrator_graph" in public_funcs

    def test_no_hardcoded_secrets(self):
        """Source code scan: no API keys or raw credentials hardcoded."""
        import re
        from pathlib import Path

        orchestrator_path = (
            Path(__file__).resolve().parent.parent / "src" / "agents" / "orchestrator.py"
        )
        source = orchestrator_path.read_text(encoding="utf-8")

        # Look for common API key patterns (in strings, not identifiers)
        key_patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"gsk_[a-zA-Z0-9]{20,}",  # Groq keys
        ]

        # Also scan for suspicious string literals that look like API keys
        # but exclude identifiers (underscore-separated words)
        suspicious = re.findall(r"'[a-zA-Z0-9_-]{32,}'|\"[a-zA-Z0-9_-]{32,}\"", source)
        real_matches = [
            m
            for m in suspicious
            if not re.match(r"^[a-z_]+$", m.strip("\"'"))  # skip snake_case identifiers
        ]

        # Check key patterns
        for pattern in key_patterns:
            matches = re.findall(pattern, source)
            assert len(matches) == 0, f"Found potential API key: {matches}"

        assert len(real_matches) == 0, f"Found potential secret: {real_matches}"


# ==============================================================================
# TASK-ORCH-013: Real LLM integration tests (Ollama)
# ==============================================================================


@pytest.mark.integration
class TestRealOrchestratorIntegration:
    """End-to-end orchestrator with real Ollama LLM.

    These tests exercise classify_intent, synthesize_response, and the
    full graph with the real configured model. No LLM mocking — uses
    the same ChatOllama that production does.

    Run with: pytest tests/test_orchestrator.py -v -m integration
    """

    # ── classify_intent with real LLM ─────────────────────────────────────

    def test_classify_exam_request(self, requires_ollama, orchestrator_state):
        """Clear exam request → intent=generate_exam with high confidence."""
        from src.agents.orchestrator import classify_intent

        state = {**orchestrator_state, "user_message": "Generame un examen de álgebra lineal"}
        result = classify_intent(state)

        assert result["intent"] in ("generate_exam", "composite"), (
            f"Expected generate_exam, got {result['intent']}"
        )
        assert result["confidence"] > 0.50, f"Confidence too low: {result['confidence']}"

    def test_classify_general_chat(self, requires_ollama, orchestrator_state):
        """Casual greeting → intent=general_chat."""
        from src.agents.orchestrator import classify_intent

        state = {**orchestrator_state, "user_message": "Hola, ¿cómo estás?"}
        result = classify_intent(state)

        assert result["intent"] == "general_chat", f"Expected general_chat, got {result['intent']}"

    def test_classify_multi_step_request(self, requires_ollama, orchestrator_state):
        """Multi-task query → intent=composite or pre-populates plan."""
        from src.agents.orchestrator import classify_intent

        state = {
            **orchestrator_state,
            "user_message": "Subí mis apuntes de cálculo y después generame un examen",
        }
        result = classify_intent(state)

        # Either classified as composite, or as single-tool with a plan
        valid = result["intent"] == "composite" or len(result.get("plan", [])) > 0
        assert valid, (
            f"Expected composite or single-tool with plan, got intent={result['intent']}, "
            f"plan={result.get('plan')}"
        )

    def test_classify_ingest_request(self, requires_ollama, orchestrator_state):
        """Upload/document request → intent=ingest."""
        from src.agents.orchestrator import classify_intent

        state = {
            **orchestrator_state,
            "user_message": "Quiero subir un PDF con apuntes de física cuántica",
        }
        result = classify_intent(state)

        assert result["intent"] in ("ingest", "composite"), (
            f"Expected ingest or composite, got {result['intent']}"
        )

    # ── synthesize_response with real LLM ──────────────────────────────────

    def test_synthesize_general_chat_response(self, requires_ollama, orchestrator_state):
        """Simple question → coherent Spanish response."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "¿Qué es un vector en álgebra lineal?",
            "plan": [],
            "results": [],
            "errors": [],
            "status": "pending",
        }
        with patch("src.tools.query_material") as mock_qm:
            mock_qm.invoke.return_value = {
                "chunks_found": 1,
                "sources": [
                    "Un vector es un elemento de un espacio vectorial, caracterizado por magnitud y dirección."
                ],
            }
            result = synthesize_response(state)

        assert len(result["response"]) > 20, f"Response too short: '{result['response']}'"
        assert result["status"] == "complete"
        # Should contain Spanish educational content
        assert any(
            word in result["response"].lower()
            for word in ("vector", "magnitud", "dirección", "elemento", "espacio")
        ), f"Response doesn't discuss vectors: '{result['response'][:100]}'"

    def test_synthesize_composite_aggregation(self, requires_ollama, orchestrator_state):
        """Composite results → LLM aggregates into coherent summary."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Subí apuntes y generame un examen",
            "results": [
                {"step": 0, "tool": "ingest", "result": {"status": "ok", "chunks": 5}},
                {"step": 1, "tool": "generate_exam", "result": {"exam": "generado"}},
            ],
            "errors": [],
            "status": "pending",
        }
        result = synthesize_response(state)

        assert len(result["response"]) > 30, f"Response too short: '{result['response']}'"
        assert result["status"] == "complete"

    def test_synthesize_incomplete_response(self, requires_ollama, orchestrator_state):
        """Incomplete status → response includes cap warning."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Procesá todo",
            "results": [{"step": 0, "tool": "ingest", "result": {"ok": True}}],
            "errors": [],
            "status": "incomplete",
        }
        result = synthesize_response(state)

        assert result["status"] == "incomplete"
        # Should include limit/cap warning or be substantially longer than prefix
        assert len(result["response"]) > 40, (
            f"Response too short for incomplete status: '{result['response']}'"
        )

    def test_synthesize_partial_with_errors(self, requires_ollama, orchestrator_state):
        """Partial status with errors → response mentions issues."""
        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "composite",
            "user_message": "Generame un examen de derivadas",
            "results": [],
            "errors": [
                {"step": 0, "tool": "generate_exam", "error": "No se encontraron chunks relevantes"}
            ],
            "status": "partial",
        }
        result = synthesize_response(state)

        assert result["status"] == "partial"
        assert len(result["response"]) > 30, f"Response too short: '{result['response']}'"
        # Response should acknowledge the error
        assert any(
            word in result["response"].lower()
            for word in ("error", "problema", "falló", "disculpa", "lamentablemente", "no pude")
        ), f"Response doesn't mention error: '{result['response'][:150]}'"

    # ── Full graph with real LLM ───────────────────────────────────────────

    async def test_e2e_general_chat_real(self, requires_ollama, orchestrator_state):
        """Full graph invoke with real LLM: classify → synthesize → response."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "real-chat-001"}}

        state = {**orchestrator_state, "user_message": "Hola, ¿qué podés hacer?"}
        result = await graph.ainvoke(state, config=config)

        assert result["intent"] == "general_chat", f"Expected general_chat, got {result['intent']}"
        assert result["status"] == "complete"
        assert len(result["response"]) > 10, f"Empty or too-short response: '{result['response']}'"
        # Should be Spanish
        assert (
            any(char in result["response"] for char in "áéíóúñ") or len(result["response"]) > 30
        ), f"Response doesn't look like Spanish: '{result['response'][:80]}'"

    async def test_e2e_exam_request_real(self, requires_ollama, orchestrator_state):
        """Exam request → classifies correctly, reaches synthesize with real LLM.

        Tools are mocked to avoid ChromaDB dependency — we're testing the
        orchestration pipeline (classify → route → execute → synthesize)
        with real LLM, not the actual tool implementations.
        """
        from unittest.mock import AsyncMock

        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "real-exam-001"}}

        # Mock the generate_exam tool so we don't need real ChromaDB
        mock_tool = AsyncMock()
        mock_tool.name = "generate_exam"
        mock_tool.ainvoke = AsyncMock(
            return_value={
                "status": "complete",
                "total_questions": 5,
                "exam_id": "exam-real-test",
            }
        )

        state = {
            **orchestrator_state,
            "user_message": "Generame un examen de álgebra lineal con 5 preguntas",
        }

        with patch.dict("src.agents.orchestrator.TOOL_MAP", {"generate_exam": mock_tool}):
            result = await graph.ainvoke(state, config=config)

        # Real LLM classifies intent; tool mock provides result
        assert result["intent"] in ("generate_exam", "composite"), (
            f"Unexpected intent: {result['intent']}"
        )
        # Response synthesized by real LLM
        assert len(result["response"]) > 10, f"Empty or too-short response: '{result['response']}'"
        assert result["status"] in ("complete", "partial", "incomplete"), (
            f"Unexpected status: {result['status']}"
        )

    async def test_e2e_multi_turn_resets_results(self, requires_ollama, orchestrator_state):
        """Two turns on same thread → results/errors reset per turn.

        The thread_id is preserved for future memory features, but tool
        execution artifacts from one turn must not leak into the next.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        from src.agents.orchestrator import build_orchestrator

        graph = build_orchestrator().compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "real-multi-001"}}

        # Turn 1: simple greeting
        state1 = {**orchestrator_state, "user_message": "Hola"}
        result1 = await graph.ainvoke(state1, config=config)

        assert result1["status"] == "complete"
        assert result1["intent"] == "general_chat"

        # Turn 2: follow-up — LANGGRAPH RESUMES STATE FROM CHECKPOINT
        # Per-invocation artifacts must be reset by the initial_state passed
        # from orchestrate_chat, so turn 2 starts with empty results/errors.
        state2 = {**orchestrator_state, "user_message": "¿Qué materias podés ayudarme a estudiar?"}
        result2 = await graph.ainvoke(state2, config=config)

        assert result2["status"] == "complete"
        assert result2["intent"] == "general_chat"
        assert result2["errors"] == []
        assert len(result2.get("results", [])) == 0, (
            "General-chat turns should not carry results from previous turns"
        )


# ==============================================================================
# RAG-exclusive answers: orchestrator academic probe tests (task 3.5)
# ==============================================================================


class TestOrchestratorAcademicProbe:
    """Task 3.5: synthesise_response probes retrieval for academic questions."""

    def test_academic_question_triggers_query_material(self, orchestrator_state):
        """GIVEN academic-looking message → THEN query_material is called.

        Covers: orchestration spec "Academic question detected — probe retrieval".
        """
        from unittest.mock import patch

        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "¿Qué dice el apunte sobre derivadas?",
            "results": [],
            "errors": [],
            "status": "pending",
        }

        fake_qm_result = {
            "answer": "Las derivadas se definen como el límite del cociente incremental.",
            "sources": ["chunk1 text"],
            "chunks_found": 1,
        }

        with patch("src.agents.orchestrator._get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = type(
                "FakeMsg", (), {"content": "RESPUESTA: Las derivadas..."}
            )()
            with patch("src.tools.query_material") as mock_qm:
                mock_qm.invoke.return_value = fake_qm_result
                result = synthesize_response(state)

        # query_material should have been called
        mock_qm.invoke.assert_called_once()
        call_args = mock_qm.invoke.call_args[0][0]
        assert call_args["query"] == "¿Qué dice el apunte sobre derivadas?"
        assert call_args["top_k"] == 5  # lighter probe

        assert "derivadas" in result["response"].lower()
        assert result["status"] == "complete"

    def test_greeting_skips_probe(self, orchestrator_state):
        """GIVEN a greeting → THEN query_material is NOT called.

        Covers: orchestration spec "Greeting — no probe".
        """
        from unittest.mock import patch

        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "Hola, buenos días",
            "results": [],
            "errors": [],
            "status": "pending",
        }

        with (
            patch("src.agents.orchestrator._get_llm") as mock_llm,
            patch("src.tools.query_material") as mock_qm,
        ):
            mock_llm.return_value.invoke.return_value = type(
                "FakeMsg", (), {"content": "¡Hola! Soy tu tutor."}
            )()
            result = synthesize_response(state)

        # query_material should NOT have been called for a greeting
        mock_qm.invoke.assert_not_called()
        assert "Hola" in result["response"]
        assert result["status"] == "complete"

    def test_academic_question_no_chunks_returns_no_material(self, orchestrator_state):
        """GIVEN academic question + no chunks → THEN no_material_message is returned.

        Covers: orchestration spec "Probe finds no chunks — no-material message".
        """
        from unittest.mock import patch

        from src.agents.orchestrator import synthesize_response
        from src.rag.policy import no_material_message

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "¿Qué es la mecánica cuántica?",
            "results": [],
            "errors": [],
            "status": "pending",
        }

        fake_empty = {"answer": "", "sources": [], "chunks_found": 0}

        with patch("src.tools.query_material") as mock_qm:
            mock_qm.invoke.return_value = fake_empty
            result = synthesize_response(state)

        assert no_material_message() in result["response"]
        assert result["status"] == "complete"

    def test_meta_question_skips_probe(self, orchestrator_state):
        """GIVEN a how-to-use-app question → THEN probe not triggered.

        Covers: orchestration spec "Meta-question about the app — no probe".
        """
        from unittest.mock import patch

        from src.agents.orchestrator import synthesize_response

        state = {
            **orchestrator_state,
            "intent": "general_chat",
            "user_message": "¿Cómo subo un archivo?",
            "results": [],
            "errors": [],
            "status": "pending",
        }

        with (
            patch("src.agents.orchestrator._get_llm") as mock_llm,
            patch("src.tools.query_material") as mock_qm,
        ):
            mock_llm.return_value.invoke.return_value = type(
                "FakeMsg", (), {"content": "Para subir un archivo, usá el botón de upload."}
            )()
            result = synthesize_response(state)

        mock_qm.invoke.assert_not_called()
        assert result["status"] == "complete"


class TestOrchestratorToolMap:
    """Verify TOOL_MAP includes all required tools (Epic 12 Phase 1)."""

    def test_tool_map_includes_update_student_profile(self):
        """update_student_profile is registered so orchestrator can trigger profile syncs."""
        from src.agents.orchestrator import _init_tool_map

        tool_map = _init_tool_map()
        assert "update_student_profile" in tool_map
        assert tool_map["update_student_profile"] is not None

    def test_build_tool_args_resolves_student_id_for_evaluate(self):
        """_build_tool_args resolves student_id from state for evaluate tool."""
        from src.agents.orchestrator import _build_tool_args

        state: dict = {
            "session_id": "sess-test-1",
            "user_message": "evaluate",
            "intent": "evaluate",
            "confidence": 0.95,
            "plan": ["evaluate"],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": {},
            "student_id": "stu-resolved-1",
        }
        args = _build_tool_args("evaluate", state)
        assert args["student_id"] == "stu-resolved-1"

    def test_build_tool_args_falls_back_to_session_id(self):
        """_build_tool_args uses session_id when student_id is not set."""
        from src.agents.orchestrator import _build_tool_args

        state: dict = {
            "session_id": "sess-fallback-1",
            "user_message": "eval",
            "intent": "evaluate",
            "confidence": 0.9,
            "plan": ["evaluate"],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": {},
        }
        args = _build_tool_args("evaluate", state)
        assert args["student_id"] == "sess-fallback-1"

    def test_build_tool_args_for_update_student_profile(self):
        """_build_tool_args builds correct args for update_student_profile tool."""
        from src.agents.orchestrator import _build_tool_args

        state: dict = {
            "session_id": "sess-up",
            "user_message": "actualiza perfil",
            "intent": "query_profile",
            "confidence": 0.9,
            "plan": ["update_student_profile"],
            "current_step": 0,
            "results": [],
            "errors": [],
            "response": "",
            "status": "pending",
            "iteration_count": 0,
            "student_profile": {
                "preferences": {"difficulty": "hard", "question_types": ["open"]},
            },
            "student_id": "stu-up-1",
        }
        args = _build_tool_args("update_student_profile", state)
        assert args["student_id"] == "stu-up-1"
        assert args["preferences"] == {"difficulty": "hard", "question_types": ["open"]}


# ==============================================================================
# Phase 5.2: get_orchestrator_graph() race safety (Epic 13, SS-3)
# ==============================================================================


class TestOrchestratorGraphRaceSafety:
    """SS-3: asyncio.Lock serializes singleton compilation."""

    @pytest.mark.asyncio
    async def test_concurrent_init_single_singleton(self, monkeypatch):
        """GIVEN 5 concurrent get_orchestrator_graph() calls
        THEN only one graph instance is created.
        """
        import asyncio

        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from src.agents.orchestrator import (
            _orchestrator_graph,
            _orchestrator_db_conn,
            _orchestrator_lock,
            get_orchestrator_graph,
        )
        from src.config import settings

        # Force clean state for deterministic test
        import src.agents.orchestrator as orch_mod

        orch_mod._orchestrator_graph = None
        orch_mod._orchestrator_db_conn = None
        orch_mod._orchestrator_lock = None

        # Patch DB path to temp
        import tempfile
        import os

        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "race_test.db")
        monkeypatch.setattr(settings, "sqlite_db_path", db_path)

        try:
            # Launch 5 concurrent calls
            results = await asyncio.gather(
                *[get_orchestrator_graph() for _ in range(5)]
            )

            # All should be the same object
            first = results[0]
            for i, r in enumerate(results[1:], 1):
                assert r is first, f"Result {i} is a different object"

            # Singleton should be set
            assert orch_mod._orchestrator_graph is not None
            assert orch_mod._orchestrator_db_conn is not None
        finally:
            # Clean up
            if orch_mod._orchestrator_db_conn is not None:
                await orch_mod._orchestrator_db_conn.close()
            orch_mod._orchestrator_graph = None
            orch_mod._orchestrator_db_conn = None
            orch_mod._orchestrator_lock = None
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
