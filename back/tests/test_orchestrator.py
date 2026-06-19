"""TDD test suite for Orchestrator agent — epic-01-orchestrator.

All unit tests mock the LLM via conftest's patch_llm().
Integration tests are marked @pytest.mark.integration (skipped by default).
"""

from __future__ import annotations

from unittest.mock import patch

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
        """errors: Annotated[list[dict], operator.add] — accumulates across nodes."""
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
            "src.agents.orchestrator._get_llm", return_value=_FakeLLM()
        ):
            result = classify_intent(orchestrator_state)

        assert result["intent"] == "generate_exam"
        assert result["confidence"] == 0.95
        assert result["plan"] == ["generate_exam"]

    def test_classify_low_confidence_fallback_to_general_chat(self, orchestrator_state):
        """Confidence < threshold → effective intent becomes general_chat."""
        from src.agents.orchestrator import classify_intent

        # Mock LLM returns low-confidence result
        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeLLM(
                intent="generate_exam", confidence=0.30
            )
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
            "src.agents.orchestrator._get_llm", return_value=_FakeLLM(
                intent="composite", confidence=0.85
            )
        ):
            result = classify_intent(state)

        assert result["intent"] == "composite"
        assert result["confidence"] == 0.85
        assert result["plan"] == []  # composite → plan filled by plan_composite later

    def test_classify_exception_fallback_to_general_chat(self, orchestrator_state):
        """Any exception during classification → fallback to general_chat."""
        from src.agents.orchestrator import classify_intent

        with patch(
            "src.agents.orchestrator._get_llm",
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
                "src.agents.orchestrator._get_llm", return_value=_FakeLLM(
                    intent=intent, confidence=0.90
                )
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

        state = {**orchestrator_state, "intent": "composite",
                 "user_message": "Ingest notes then generate exam"}

        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeCompositeLLM(
                steps=["ingest", "generate_exam"]
            )
        ):
            result = plan_composite(state)

        assert result["plan"] == ["ingest", "generate_exam"]

    def test_plan_composite_strips_invalid_tools(self, orchestrator_state):
        """Tool names not in TOOL_MAP are stripped from the plan."""
        from src.agents.orchestrator import plan_composite

        state = {**orchestrator_state, "intent": "composite",
                 "user_message": "Do something impossible"}

        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeCompositeLLM(
                steps=["nonexistent_tool", "ingest", "bad_tool"]
            )
        ):
            result = plan_composite(state)

        assert result["plan"] == ["ingest"]  # only valid tool kept

    def test_plan_composite_empty_plan_fallback(self, orchestrator_state):
        """Empty plan after stripping → returns empty list (treated as general_chat downstream)."""
        from src.agents.orchestrator import plan_composite

        state = {**orchestrator_state, "intent": "composite",
                 "user_message": "something"}

        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeCompositeLLM(steps=[])
        ):
            result = plan_composite(state)

        assert result["plan"] == []

    def test_plan_composite_exception_returns_empty_plan(self, orchestrator_state):
        """Planner failure → returns empty plan (treated as general_chat downstream)."""
        from src.agents.orchestrator import plan_composite

        state = {**orchestrator_state, "intent": "composite",
                 "user_message": "Do stuff"}

        with patch(
            "src.agents.orchestrator._get_llm",
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
        mock_tool.ainvoke = AsyncMock(
            side_effect=[RuntimeError("fail1"), RuntimeError("fail2")]
        )

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
            "user_message": "¿Qué es un vector?",
            "plan": [],
            "results": [],
            "errors": [],
            "status": "pending",
        }

        with patch(
            "src.agents.orchestrator._get_llm", return_value=_FakeDirectLLM(
                "Un vector es un elemento de un espacio vectorial."
            )
        ):
            result = synthesize_response(state)

        assert "vector" in result["response"]
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
            "src.agents.orchestrator._get_llm", return_value=_FakeDirectLLM(
                "Completé la ingesta y generé el examen."
            )
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
            "src.agents.orchestrator._get_llm", return_value=_FakeDirectLLM(
                "Acá está el resumen."
            )
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
            "src.agents.orchestrator._get_llm", return_value=_FakeDirectLLM(
                "Parcial completado. Error en paso b."
            )
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


class _FakeDirectLLM:
    """Fake LLM that returns a pre-programmed string for synthesize_response."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, prompt):
        return type("FakeAIMessage", (), {"content": self._response})()


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

    def invoke(self, prompt):
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

    def invoke(self, prompt):
        from src.agents.orchestrator import CompositePlan
        return CompositePlan(steps=self._steps)
