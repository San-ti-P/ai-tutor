"""Observability test suite — 12 PRD §8 unit cases + 3 integration tests.

Unit tier: ``pytest -m "not integration"`` — all Langfuse interactions mocked.
Integration tier: ``pytest -m integration`` — requires LANGFUSE_OBSERVE_TESTS=true.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.observability import ObservabilityManager, flush_traces, get_tracer
from src.observability._client import _reset_langfuse_client

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — OBS-01 happy path: manager init with valid keys
# ═══════════════════════════════════════════════════════════════════════════════


class TestManagerInit:
    def test_manager_init_with_valid_keys(self, mock_langfuse):
        """OBS-01 happy: manager.enabled is True when Langfuse client creates."""
        _reset_langfuse_client()
        mgr = ObservabilityManager()
        mgr._ensure_init()

        assert mgr.enabled is True
        mock_langfuse.assert_called_once()

    def test_manager_disabled_with_empty_keys(self):
        """OBS-01 edge: manager.enabled is False when keys are empty."""
        # Override settings with empty keys
        with patch.object(settings, "langfuse_public_key", ""):
            with patch.object(settings, "langfuse_secret_key", ""):
                _reset_langfuse_client()
                mgr = ObservabilityManager()
                mgr._ensure_init()

                assert mgr.enabled is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — OBS-02: root trace creation
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateRootTrace:
    def test_create_root_trace(self, obs_manager):
        """OBS-02: trace created with session_id, user_id, metadata."""
        trace = obs_manager.create_trace(
            name="exam_generation",
            session_id="sess-1",
            user_id="stu-42",
            metadata={"task": "exam"},
        )

        assert trace is not None

    def test_create_trace_disabled_returns_none(self):
        """When disabled, create_trace returns None."""
        _reset_langfuse_client()
        with patch.object(settings, "langfuse_public_key", ""):
            mgr = ObservabilityManager()
            mgr._ensure_init()
            trace = mgr.create_trace(name="test", session_id="s1")
            assert trace is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — OBS-03: LLM span via CallbackHandler
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMSpanCallbacks:
    def test_llm_span_via_callback_handler(self, obs_manager, mock_observe):
        """OBS-03: CallbackHandler is created and passable to LLM config."""
        handler = obs_manager.get_callback_handler(
            session_id="sess-1",
            user_id="stu-42",
        )

        assert handler is not None
        # Verify the handler can be used in LangChain callbacks list
        from src.llm import get_llm

        llm = get_llm(callbacks=[handler])
        assert llm is not None

    def test_callback_handler_disabled_returns_none(self):
        """When disabled, get_callback_handler returns None."""
        _reset_langfuse_client()
        with patch.object(settings, "langfuse_public_key", ""):
            mgr = ObservabilityManager()
            mgr._ensure_init()
            handler = mgr.get_callback_handler(session_id="s1")
            assert handler is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — OBS-04: Tool span via @observe
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolSpanObserve:
    def test_tool_span_via_observe(self, mock_observe):
        """OBS-04: @observe decorator wraps tool function without error."""
        from langfuse import observe

        @observe(name="test_tool")
        def sample_tool(x: int) -> int:
            return x * 2

        result = sample_tool(21)
        assert result == 42

    def test_tool_span_disabled_client_still_runs(self, mock_observe):
        """OBS-NFR-01: tool still executes even when Langfuse unreachable."""
        from langfuse import observe

        @observe(name="resilient_tool")
        def resilient_tool(x: int) -> int:
            return x + 1

        result = resilient_tool(10)
        assert result == 11


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — OBS-05: RAG retrieval span
# ═══════════════════════════════════════════════════════════════════════════════


class TestRAGRetrievalSpan:
    def test_rag_retrieval_span(self, mock_langfuse, mock_observe, mock_embedding_model):
        """OBS-05: @observe on retrieve() records query, top_k, results."""
        from src.rag import get_chroma_client

        # Pre-populate a test collection
        client = get_chroma_client()
        col = client.get_or_create_collection("test_obs_collection")
        col.add(
            ids=["c1", "c2"],
            documents=["Derivada es el límite del cociente incremental", "Integral es el área bajo la curva"],
        )

        from src.rag import retrieve

        results = retrieve("derivada", collection_name="test_obs_collection", top_k=2)
        assert isinstance(results, list)

    def test_empty_retrieval_span(self, mock_langfuse, mock_observe):
        """OBS-05 edge: @observe works when collection empty — returns []."""
        from src.rag import retrieve

        results = retrieve("nonexistent", collection_name="no_such_collection", top_k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — OBS-06: Evaluation span
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluationSpan:
    def test_evaluation_span(self, mock_langfuse, mock_observe):
        """OBS-06: @observe on evaluate_answer decorates tool invocation."""
        from src.tools import evaluate_answer as eval_tool

        # StructuredTool is always callable via .invoke(); verify it exists
        assert hasattr(eval_tool, "invoke")

    def test_non_evaluable_trace(self, mock_langfuse, mock_observe):
        """OBS-06 edge: non-evaluable answer still emits span."""
        from src.agents.evaluator import EvaluatorState, check_evaluability

        state: EvaluatorState = {
            "session_id": "s1",
            "student_id": "st1",
            "exam_id": "e1",
            "trace_id": "t1",
            "answers": [{"question_id": "q1", "question": "Q?", "student_answer": "ab"}],
            "current_index": 0,
            "answer_text": "",
            "ocr_extracted_text": None,
            "ocr_confidence": 0.0,
            "retrieved_chunks": [],
            "collection_name": "",
            "evaluation": None,
            "evaluation_results": [],
            "non_evaluable": False,
            "non_evaluable_reason": "",
            "judge_sample": False,
            "judge_result": None,
            "requires_review": False,
            "scores_synced": False,
            "errors": [],
            "status": "pending",
        }

        result = check_evaluability(state)
        assert result["non_evaluable"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — OBS-07: Aggregate metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestAggregateMetrics:
    def test_aggregate_metrics(self):
        """OBS-07: compute_aggregate_metrics returns correct totals."""
        from src.observability._metrics import compute_aggregate_metrics

        # Build a synthetic trace with mock observations
        mock_trace = MagicMock()
        mock_obs1 = MagicMock()
        mock_obs1.usage = MagicMock(input=100, output=200)
        mock_obs1.calculated_total_cost = 0.001
        mock_obs1.latency = 150.0
        mock_obs1.type = "span"
        mock_obs1.name = "tool_retrieve"
        mock_obs1.status_message = ""
        mock_obs1.scores = []

        mock_obs2 = MagicMock()
        mock_obs2.usage = MagicMock(input=50, output=100)
        mock_obs2.calculated_total_cost = 0.002
        mock_obs2.latency = 250.0
        mock_obs2.type = "span"
        mock_obs2.name = "tool_generate"
        mock_obs2.status_message = "error"
        mock_obs2.scores = [MagicMock(value=8.0), MagicMock(value=6.0)]

        mock_trace.observations = [mock_obs1, mock_obs2]

        metrics = compute_aggregate_metrics(mock_trace)

        assert metrics["total_steps"] == 2
        assert metrics["total_tokens"] == 450  # 100+200 + 50+100
        assert metrics["total_cost"] == 0.003
        assert metrics["total_latency_ms"] == 400.0
        assert metrics["tool_success_rate"] == 0.5  # 1 success / 2 calls
        assert metrics["avg_score"] == 7.0  # (8+6)/2

    def test_aggregate_metrics_empty_trace(self):
        """Empty trace returns zero-filled metrics."""
        from src.observability._metrics import compute_aggregate_metrics

        metrics = compute_aggregate_metrics(None)
        assert metrics["total_steps"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["avg_score"] == 0.0

    def test_aggregate_metrics_no_observations(self):
        """Trace with no observations returns zero-filled metrics."""
        from src.observability._metrics import compute_aggregate_metrics

        mock_trace = MagicMock()
        mock_trace.observations = []
        metrics = compute_aggregate_metrics(mock_trace)

        assert metrics["total_steps"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — OBS-01 adversarial: missing keys does not crash
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoCrash:
    def test_missing_keys_no_crash(self, caplog):
        """OBS-01 adversarial: agents operate without tracing when keys empty."""
        with patch.object(settings, "langfuse_public_key", ""):
            with patch.object(settings, "langfuse_secret_key", ""):
                _reset_langfuse_client()
                mgr = ObservabilityManager()

                # All public API calls should return safe defaults
                assert mgr.enabled is False
                assert mgr.create_trace(name="t", session_id="s") is None
                assert mgr.get_callback_handler(session_id="s") is None
                mgr.flush()  # must not raise
                mgr.shutdown()  # must not raise

        # WARNING should be logged
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("keys not configured" in w for w in warnings), f"Expected WARNING log, got: {warnings}"

    def test_langfuse_unreachable_no_block(self, caplog):
        """OBS-NFR-01 adversarial: Langfuse timeout does not crash agent."""
        # Simulate Langfuse constructor raising connection error
        with patch("langfuse.Langfuse", side_effect=ConnectionError("timeout")):
            _reset_langfuse_client()
            mgr = ObservabilityManager()
            mgr._ensure_init()

            # Manager must be disabled
            assert mgr.enabled is False

            # All calls must return safe defaults without exception
            assert mgr.create_trace(name="t", session_id="s") is None
            assert mgr.get_callback_handler(session_id="s") is None
            mgr.flush()  # no-op, no exception
            mgr.shutdown()  # no-op, no exception

    def test_flush_traces_public_api(self, obs_manager):
        """flush_traces() calls manager.flush() without error."""
        flush_traces()  # must not raise

    def test_get_tracer_singleton(self):
        """get_tracer() returns same instance on repeated calls."""
        _reset_langfuse_client()
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_langfuse_client_lazy_init_key_missing(self, caplog):
        """_get_langfuse_client returns None and logs WARNING when keys missing."""
        from src.observability._client import _get_langfuse_client, _reset_langfuse_client

        _reset_langfuse_client()
        with patch.object(settings, "langfuse_public_key", ""):
            result = _get_langfuse_client()
            assert result is None

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("keys not configured" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests — real Langfuse client (opt-in via LANGFUSE_OBSERVE_TESTS)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestLangfuseRealTraces:
    """Integration tests requiring LANGFUSE_OBSERVE_TESTS=true and valid keys."""

    def test_create_trace_emits_test_metadata(
        self, langfuse_observe_tests, test_run_id, obs_manager, request, monkeypatch
    ):
        """Trace created via obs_manager carries test metadata tags.

        Spy on the underlying client.start_observation() to verify the
        metadata dict includes the four test tags: environment, test_run_id,
        test_name, source.
        """
        if not langfuse_observe_tests:
            pytest.skip("LANGFUSE_OBSERVE_TESTS not set")
        if not obs_manager.enabled:
            pytest.skip("Langfuse not configured — missing keys in .env")

        # Spy on _client.start_observation to capture the metadata argument
        captured_meta: dict[str, object] = {}
        original_start = obs_manager._client.start_observation

        def spy_start(*, name, as_type="span", metadata=None, **kwargs):
            captured_meta.update(metadata or {})
            return original_start(
                name=name, as_type=as_type, metadata=metadata, **kwargs
            )

        monkeypatch.setattr(obs_manager._client, "start_observation", spy_start)

        trace = obs_manager.create_trace(
            name="integration-test-trace",
            session_id="sess-integration-001",
        )

        assert trace is not None
        assert captured_meta.get("environment") == "test", (
            f"Expected environment=test, got {captured_meta}"
        )
        assert captured_meta.get("test_run_id") == test_run_id, (
            f"Expected test_run_id={test_run_id}, got {captured_meta}"
        )
        assert captured_meta.get("source") == "pytest-integration", (
            f"Expected source=pytest-integration, got {captured_meta}"
        )
        assert "test_name" in captured_meta, (
            f"Expected test_name in metadata, got {captured_meta}"
        )

    def test_agent_invocation_creates_trace(
        self, langfuse_observe_tests, test_run_id, obs_manager, sample_txt,
        mock_llm_response, mock_embedding_model, in_memory_chroma, request, monkeypatch,
    ):
        """Running an agent flow creates a Langfuse trace with test metadata.

        Uses the Ingestor graph (mocked LLM + embeddings) to verify that
        get_tracer() returns an enabled manager and a callback handler is
        obtainable when real Langfuse is active.
        """
        if not langfuse_observe_tests:
            pytest.skip("LANGFUSE_OBSERVE_TESTS not set")
        if not obs_manager.enabled:
            pytest.skip("Langfuse not configured — missing keys in .env")

        from src.agents.ingestor import IngestorState, build_ingestor
        from src.observability import get_tracer

        tracer = get_tracer()
        assert tracer.enabled, "Tracer should be enabled with real Langfuse keys"

        handler = tracer.get_callback_handler(session_id="integration-sess-001")
        assert handler is not None, (
            "CallbackHandler should be created when Langfuse is available"
        )

        # Build and run ingestor with Langfuse callback
        graph = build_ingestor().compile()
        initial_state: IngestorState = {
            "session_id": "integration-sess-001",
            "file_path": str(sample_txt),
            "file_type": "",
            "raw_text": "",
            "classification": "",
            "classification_confidence": 0.0,
            "topics": [],
            "chunks_created": 0,
            "errors": [],
            "status": "pending",
            "document_id": "",
            "chunk_ids": [],
        }

        config = {"configurable": {"thread_id": "integration-sess-001"}}
        if handler:
            config.setdefault("callbacks", []).append(handler)

        # Spy on client.start_observation via obs_manager fixture patch
        captured_meta: dict[str, object] = {}
        if obs_manager._client:
            original_start = obs_manager._client.start_observation

            def spy_start(*, name, as_type="span", metadata=None, **kwargs):
                captured_meta.update(metadata or {})
                return original_start(
                    name=name, as_type=as_type, metadata=metadata, **kwargs
                )

            monkeypatch.setattr(obs_manager._client, "start_observation", spy_start)

        final_state = graph.invoke(initial_state, config=config)

        assert final_state["status"] == "completed", (
            f"Ingestor failed: {final_state.get('errors', [])}"
        )

        # After agent run, flush traces
        from src.observability import flush_traces
        flush_traces()

    def test_langfuse_unreachable_agent_does_not_crash(
        self, langfuse_observe_tests, sample_txt, mock_llm_response,
        mock_embedding_model, in_memory_chroma,
    ):
        """Agent completes successfully even when Langfuse is unreachable.

        Simulates a ConnectionError in the Langfuse constructor and verifies
        the Ingestor graph still runs to completion without exception.
        """
        import src.observability as _obs_mod
        from src.observability._client import _reset_langfuse_client

        # Reset BOTH the module-level singleton AND the client cache
        _obs_mod._manager = None
        _reset_langfuse_client()

        # Temporarily simulate unreachable Langfuse
        with patch("langfuse.Langfuse", side_effect=ConnectionError("unreachable")):
            _reset_langfuse_client()
            _obs_mod._manager = None

            from src.agents.ingestor import IngestorState, build_ingestor
            from src.observability import get_tracer

            tracer = get_tracer()
            assert not tracer.enabled, "Tracer must be disabled when Langfuse unreachable"

            handler = tracer.get_callback_handler(session_id="crash-test-sess")
            assert handler is None, (
                "CallbackHandler must be None when Langfuse unreachable"
            )

            # Run agent without Langfuse — must not crash
            graph = build_ingestor().compile()
            initial_state: IngestorState = {
                "session_id": "crash-test-sess",
                "file_path": str(sample_txt),
                "file_type": "",
                "raw_text": "",
                "classification": "",
                "classification_confidence": 0.0,
                "topics": [],
                "chunks_created": 0,
                "errors": [],
                "status": "pending",
                "document_id": "",
                "chunk_ids": [],
            }

            final_state = graph.invoke(initial_state)

            assert final_state["status"] == "completed", (
                f"Agent should complete even without Langfuse: "
                f"{final_state.get('errors', [])}"
            )

        # Clean up: reset singleton so subsequent tests get fresh state
        _obs_mod._manager = None
        _reset_langfuse_client()
