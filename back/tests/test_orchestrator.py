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
