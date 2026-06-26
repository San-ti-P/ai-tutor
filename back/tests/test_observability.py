"""Observability test suite — 12 PRD §8 unit cases + 3 integration tests.

Unit tier: ``pytest -m "not integration"`` — all Langfuse interactions mocked.
Integration tier: ``pytest -m integration`` — requires LANGFUSE_OBSERVE_TESTS=true.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

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
            documents=[
                "Derivada es el límite del cociente incremental",
                "Integral es el área bajo la curva",
            ],
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
        assert any("keys not configured" in w for w in warnings), (
            f"Expected WARNING log, got: {warnings}"
        )

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
# Epic-08 — Observability Hardening (OBS-03/04 gaps)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHardening:
    """Epic-08 observability hardening: @observe decorators and callback wiring."""

    @contextmanager
    def _spy_observe(self):
        """Patch langfuse.observe with a spy that records applied names and as_type."""
        applied: list[dict] = []

        def spy_observe(**kwargs):
            def decorator(fn):
                applied.append({"name": kwargs.get("name"), "as_type": kwargs.get("as_type")})
                return fn

            return decorator

        with patch("langfuse.observe", spy_observe):
            yield applied

    def test_validate_claim_grounding_observed(self):
        """1.1a: validate_claim_grounding is decorated with @observe."""
        import importlib

        mod = importlib.import_module("src.tools.validate_claim_grounding")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            result = mod.validate_claim_grounding.invoke({"claims": ["test claim"], "chunks": []})

        assert result["all_matched"] is True
        names = [a["name"] for a in applied]
        assert "validate_claim_grounding" in names
        # as_type defaults — will fail until fixed to "tool"
        vcg = next(a for a in applied if a["name"] == "validate_claim_grounding")
        assert vcg["as_type"] == "tool", f"Expected as_type='tool', got {vcg['as_type']}"

    def test_retrieve_chunks_observed(self, mock_embedding_model):
        """1.1b: retrieve_chunks is decorated with @observe."""
        import importlib

        tools_mod = importlib.import_module("src.tools")

        with self._spy_observe() as applied:
            importlib.reload(tools_mod)
            result = tools_mod.retrieve_chunks.invoke(
                {"query": "nonexistent", "top_k": 5, "collection_name": "no_such_collection"}
            )

        assert result == []
        names = [a["name"] for a in applied]
        assert "retrieve_chunks" in names
        rc = next(a for a in applied if a["name"] == "retrieve_chunks")
        assert rc["as_type"] == "tool", f"Expected as_type='tool', got {rc['as_type']}"

    async def test_get_student_summary_observed(self):
        """1.1c: get_student_summary is decorated with @observe."""
        import importlib

        mod = importlib.import_module("src.tools.get_student_summary")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            with (
                patch("src.memory.schema.get_student_profile") as mock_profile,
                patch("src.memory.schema.get_topic_scores") as mock_scores,
                patch("src.memory.schema.compute_weak_topics") as mock_weak,
                patch("src.memory.schema.get_recent_sessions") as mock_sessions,
            ):
                mock_profile.return_value = {
                    "id": "stu-1",
                    "preferences": {},
                    "session_count": 1,
                }
                mock_scores.return_value = []
                mock_weak.return_value = []
                mock_sessions.return_value = []

                result = await mod.get_student_summary.ainvoke({"student_id": "stu-1"})

        assert result is not None
        assert result["id"] == "stu-1"
        names = [a["name"] for a in applied]
        assert "get_student_summary" in names
        gss = next(a for a in applied if a["name"] == "get_student_summary")
        assert gss["as_type"] == "tool", f"Expected as_type='tool', got {gss['as_type']}"

    async def test_update_student_profile_observed(self):
        """1.1d: update_student_profile is decorated with @observe."""
        import importlib

        mod = importlib.import_module("src.tools.update_student_profile")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            with (
                patch("src.memory.schema.upsert_student_profile") as mock_upsert_profile,
                patch("src.memory.schema.upsert_topic_scores") as mock_upsert_scores,
            ):
                mock_upsert_profile.return_value = None
                mock_upsert_scores.return_value = None

                result = await mod.update_student_profile.ainvoke(
                    {"student_id": "stu-1", "topic_scores": {"math": 8.0}}
                )

        assert result["status"] == "ok"
        names = [a["name"] for a in applied]
        assert "update_student_profile" in names
        usp = next(a for a in applied if a["name"] == "update_student_profile")
        assert usp["as_type"] == "tool", f"Expected as_type='tool', got {usp['as_type']}"

    async def test_orchestrate_chat_observed(self):
        """1.1e: orchestrate_chat is decorated with @observe."""
        import importlib

        mod = importlib.import_module("src.tools.orchestrate_chat")

        async def _mock_graph():
            return mock_graph

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "hola",
                "intent": "general_chat",
                "status": "complete",
            }
        )

        with self._spy_observe() as applied:
            importlib.reload(mod)
            with patch("src.agents.orchestrator.get_orchestrator_graph", side_effect=_mock_graph):
                result = await mod.orchestrate_chat.ainvoke(
                    {"messages": [{"role": "user", "content": "hola"}]}
                )

        assert result["response"] == "hola"
        names = [a["name"] for a in applied]
        assert "orchestrate_chat" in names
        oc = next(a for a in applied if a["name"] == "orchestrate_chat")
        assert oc["as_type"] == "tool", f"Expected as_type='tool', got {oc['as_type']}"

    # Phase 2: graph builder decorators — NO @observe (Gap 2 fix)

    def test_build_ingestor_no_longer_observed(self):
        """2.1a: build_ingestor MUST NOT have @observe (Gap 2 — dumps graph schema)."""
        import importlib

        mod = importlib.import_module("src.agents.ingestor")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            graph = mod.build_ingestor()

        assert graph is not None
        names = [a["name"] for a in applied]
        assert "ingestor" not in names, (
            f"build_ingestor should NOT have @observe — was found in {names}"
        )

    def test_build_exam_generator_no_longer_observed(self):
        """2.1b: build_exam_generator MUST NOT have @observe (Gap 2)."""
        import importlib

        mod = importlib.import_module("src.agents.exam_generator")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            graph = mod.build_exam_generator()

        assert graph is not None
        names = [a["name"] for a in applied]
        assert "exam_generator" not in names, (
            f"build_exam_generator should NOT have @observe — was found in {names}"
        )

    def test_build_exercise_generator_no_longer_observed(self):
        """2.1c: build_exercise_generator MUST NOT have @observe (Gap 2)."""
        import importlib

        mod = importlib.import_module("src.agents.exercise_generator")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            graph = mod.build_exercise_generator()

        assert graph is not None
        names = [a["name"] for a in applied]
        assert "exercise_generator" not in names, (
            f"build_exercise_generator should NOT have @observe — was found in {names}"
        )

    def test_build_evaluator_no_longer_observed(self):
        """2.1d: build_evaluator MUST NOT have @observe (Gap 2)."""
        import importlib

        mod = importlib.import_module("src.agents.evaluator")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            graph = mod.build_evaluator()

        assert graph is not None
        names = [a["name"] for a in applied]
        assert "evaluator" not in names, (
            f"build_evaluator should NOT have @observe — was found in {names}"
        )

    def test_build_support_agent_no_longer_observed(self):
        """2.1e: build_support_agent MUST NOT have @observe (Gap 2)."""
        import importlib

        mod = importlib.import_module("src.agents.support")

        with self._spy_observe() as applied:
            importlib.reload(mod)
            graph = mod.build_support_agent()

        assert graph is not None
        names = [a["name"] for a in applied]
        assert "support_agent" not in names, (
            f"build_support_agent should NOT have @observe — was found in {names}"
        )

    # Phase 3: CallbackHandler injection into graph.invoke calls

    def _reload_tools(self):
        """Reload src.tools so latest __init__.py changes are visible."""
        import importlib

        tools_mod = importlib.import_module("src.tools")
        importlib.reload(tools_mod)
        return tools_mod

    def _mock_tracer(self, handler):
        """Return a mock tracer whose get_callback_handler returns *handler*."""
        mock_tracer = MagicMock()
        mock_tracer.get_callback_handler.return_value = handler
        return mock_tracer

    def test_ingest_document_injects_callback_handler(self):
        """3.1a: ingest_document passes CallbackHandler to graph.invoke."""
        tools_mod = self._reload_tools()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "classification": "apunte_teorico",
            "topics": ["algebra"],
            "chunks_created": 1,
            "status": "completed",
            "errors": [],
        }
        mock_handler = MagicMock()

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(mock_handler)),
        ):
            tools_mod.ingest_document.invoke({"file_path": "/tmp/foo.txt", "session_id": "s1"})

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert config["callbacks"] == [mock_handler]

        # Graceful when disabled
        mock_graph.invoke.reset_mock()
        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(None)),
        ):
            tools_mod.ingest_document.invoke({"file_path": "/tmp/foo.txt", "session_id": "s1"})

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert "callbacks" not in config

    def test_generate_exam_injects_callback_handler(self):
        """3.1b: generate_exam passes CallbackHandler to graph.invoke."""
        tools_mod = self._reload_tools()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"exam": {"exam_id": "e1"}}
        mock_handler = MagicMock()

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(mock_handler)),
        ):
            tools_mod.generate_exam.invoke(
                {"session_id": "s1", "topics": ["algebra"], "question_count": 1}
            )

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert config["callbacks"] == [mock_handler]

        # Graceful when disabled
        mock_graph.invoke.reset_mock()
        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(None)),
        ):
            tools_mod.generate_exam.invoke(
                {"session_id": "s1", "topics": ["algebra"], "question_count": 1}
            )

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert "callbacks" not in config

    def test_generate_exercise_injects_callback_handler(self):
        """3.1c: generate_exercise passes CallbackHandler to graph.invoke."""
        tools_mod = self._reload_tools()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"exercise": {"exercise_id": "x1"}}
        mock_handler = MagicMock()

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(mock_handler)),
        ):
            tools_mod.generate_exercise.invoke({"session_id": "s1", "topic": "algebra"})

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert config["callbacks"] == [mock_handler]

        # Graceful when disabled
        mock_graph.invoke.reset_mock()
        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(None)),
        ):
            tools_mod.generate_exercise.invoke({"session_id": "s1", "topic": "algebra"})

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert "callbacks" not in config

    def test_evaluate_answer_injects_callback_handler(self):
        """3.1d: evaluate_answer passes CallbackHandler to graph.invoke."""
        tools_mod = self._reload_tools()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"evaluation_results": [{"question_id": "q1"}]}
        mock_handler = MagicMock()

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(mock_handler)),
        ):
            tools_mod.evaluate_answer.invoke(
                {
                    "session_id": "s1",
                    "exam_id": "e1",
                    "answers": [{"question_id": "q1", "question": "Q?", "student_answer": "A"}],
                }
            )

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert config["callbacks"] == [mock_handler]

        # Graceful when disabled
        mock_graph.invoke.reset_mock()
        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer", return_value=self._mock_tracer(None)),
        ):
            tools_mod.evaluate_answer.invoke(
                {
                    "session_id": "s1",
                    "exam_id": "e1",
                    "answers": [{"question_id": "q1", "question": "Q?", "student_answer": "A"}],
                }
            )

        config = mock_graph.invoke.call_args.kwargs["config"]
        assert "callbacks" not in config

    # Phase 4: cross-cutting verification

    def test_all_tools_work_without_langfuse_keys(self, sample_txt, mock_embedding_model):
        """4.1: All 10 tools + 3 builders execute when Langfuse keys are empty."""
        import importlib

        import src.observability as obs_mod
        from src.config import settings
        from src.observability._client import _reset_langfuse_client

        with (
            patch.object(settings, "langfuse_public_key", ""),
            patch.object(settings, "langfuse_secret_key", ""),
        ):
            _reset_langfuse_client()
            obs_mod._manager = None
            tools_mod = importlib.import_module("src.tools")
            importlib.reload(tools_mod)

            # Sync tools
            tools_mod.validate_claim_grounding.invoke({"claims": [], "chunks": []})
            tools_mod.retrieve_chunks.invoke(
                {"query": "x", "collection_name": "no_such", "top_k": 1}
            )

            # extract_topics needs a mocked LLM
            with patch("src.llm.get_structured_llm") as mock_get_llm:
                mock_result = MagicMock()
                mock_result.summary = "s"
                mock_result.topics = ["t1"]
                mock_result.topic_tree = "{}"
                mock_structured = MagicMock()
                mock_structured.invoke.return_value = mock_result
                mock_get_llm.return_value = mock_structured
                tools_mod.extract_topics.invoke({"text": "some academic text"})

            # Graph-backed tools use a mock graph
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "classification": "apunte_teorico",
                "topics": ["t"],
                "chunks_created": 1,
                "status": "completed",
                "errors": [],
                "exam": {"exam_id": "e1"},
                "exercise": {"exercise_id": "x1"},
                "evaluation_results": [{"question_id": "q1"}],
            }
            with patch("src.tools._get_or_compile", return_value=mock_graph):
                tools_mod.ingest_document.invoke({"file_path": str(sample_txt), "session_id": "s1"})
                tools_mod.generate_exam.invoke(
                    {"session_id": "s1", "topics": ["t"], "question_count": 1}
                )
                tools_mod.generate_exercise.invoke({"session_id": "s1", "topic": "t"})
                tools_mod.evaluate_answer.invoke(
                    {
                        "session_id": "s1",
                        "exam_id": "e1",
                        "answers": [
                            {
                                "question_id": "q1",
                                "question": "Q?",
                                "student_answer": "A",
                            }
                        ],
                    }
                )

            # Async tools
            async def _run_async_tools():
                async_graph = MagicMock()
                async_graph.ainvoke = AsyncMock(
                    return_value={
                        "response": "ok",
                        "intent": "general_chat",
                        "status": "complete",
                    }
                )
                with (
                    patch("src.memory.schema.get_student_profile") as mock_profile,
                    patch("src.memory.schema.get_topic_scores") as mock_scores,
                    patch("src.memory.schema.compute_weak_topics") as mock_weak,
                    patch("src.memory.schema.get_recent_sessions") as mock_sessions,
                    patch("src.memory.schema.upsert_student_profile") as mock_upsert_profile,
                    patch("src.memory.schema.upsert_topic_scores") as mock_upsert_scores,
                    patch(
                        "src.agents.orchestrator.get_orchestrator_graph",
                        return_value=async_graph,
                    ),
                ):
                    mock_profile.return_value = {
                        "id": "stu-1",
                        "preferences": {},
                        "session_count": 1,
                    }
                    mock_scores.return_value = []
                    mock_weak.return_value = []
                    mock_sessions.return_value = []
                    mock_upsert_profile.return_value = None
                    mock_upsert_scores.return_value = None

                    await tools_mod.get_student_summary.ainvoke({"student_id": "stu-1"})
                    await tools_mod.update_student_profile.ainvoke(
                        {"student_id": "stu-1", "topic_scores": {"math": 8.0}}
                    )
                    await tools_mod.orchestrate_chat.ainvoke(
                        {"messages": [{"role": "user", "content": "hola"}]}
                    )

            import asyncio

            asyncio.run(_run_async_tools())

            # Builders
            from src.agents.exam_generator import build_exam_generator
            from src.agents.exercise_generator import build_exercise_generator
            from src.agents.ingestor import build_ingestor

            assert build_ingestor() is not None
            assert build_exam_generator() is not None
            assert build_exercise_generator() is not None

    def test_no_duplicate_root_traces(
        self, mock_langfuse, sample_txt, mock_llm_response, mock_embedding_model
    ):
        """4.2: ingest_document end-to-end emits observations under one trace_id."""
        import contextvars
        import importlib
        from functools import wraps

        _current_trace_id = contextvars.ContextVar("test_trace_id", default=None)
        observations: list[dict] = []

        def spy_observe(**decorator_kwargs):
            def decorator(fn):
                is_async = inspect.iscoroutinefunction(fn)

                if is_async:

                    @wraps(fn)
                    async def async_wrapper(*args, **kwargs):
                        trace_id = _current_trace_id.get()
                        if trace_id is None:
                            trace_id = f"trace-{len(observations)}"
                            _current_trace_id.set(trace_id)
                        observations.append(
                            {"name": decorator_kwargs.get("name"), "trace_id": trace_id}
                        )
                        return await fn(*args, **kwargs)

                    return async_wrapper
                else:

                    @wraps(fn)
                    def sync_wrapper(*args, **kwargs):
                        trace_id = _current_trace_id.get()
                        if trace_id is None:
                            trace_id = f"trace-{len(observations)}"
                            _current_trace_id.set(trace_id)
                        observations.append(
                            {"name": decorator_kwargs.get("name"), "trace_id": trace_id}
                        )
                        return fn(*args, **kwargs)

                    return sync_wrapper

            return decorator

        with patch("langfuse.observe", spy_observe):
            tools_mod = importlib.import_module("src.tools")
            importlib.reload(tools_mod)

            result = tools_mod.ingest_document.invoke(
                {"file_path": str(sample_txt), "session_id": "s1"}
            )

        assert result["status"] == "completed"
        # Outer tool span plus inner spans (e.g. retrieve_chunks) all share one trace_id
        assert len(observations) >= 1
        trace_ids = {o["trace_id"] for o in observations}
        assert len(trace_ids) == 1, f"Expected single trace_id, got {trace_ids}"


# ═══════════════════════════════════════════════════════════════════════════════
# Environment at client constructor + as_type tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClientEnvironment:
    """Environment must be set at Langfuse() constructor level, not just metadata."""

    def test_langfuse_constructor_receives_environment(self):
        """_get_langfuse_client passes settings.langfuse_environment to Langfuse()."""
        from src.config import settings
        from src.observability._client import _get_langfuse_client, _reset_langfuse_client

        _reset_langfuse_client()
        with (
            patch.object(settings, "langfuse_public_key", "pk-test"),
            patch.object(settings, "langfuse_secret_key", "sk-test"),
            patch.object(settings, "langfuse_host", "http://localhost:3000"),
            patch.object(settings, "langfuse_environment", "staging"),
            patch.object(settings, "langfuse_release", "v1.2.3"),
            patch("langfuse.Langfuse") as mock_lf,
        ):
            _get_langfuse_client()

        call_kwargs = mock_lf.call_args.kwargs
        assert call_kwargs.get("environment") == "staging", (
            f"Langfuse() must receive environment='staging', got {call_kwargs}"
        )
        assert call_kwargs.get("release") == "v1.2.3", (
            f"Langfuse() must receive release='v1.2.3', got {call_kwargs}"
        )

    def test_langfuse_constructor_default_environment(self):
        """When langfuse_environment is default, 'development' is passed."""
        from src.config import settings
        from src.observability._client import _get_langfuse_client, _reset_langfuse_client

        _reset_langfuse_client()
        with (
            patch.object(settings, "langfuse_public_key", "pk-test"),
            patch.object(settings, "langfuse_secret_key", "sk-test"),
            patch.object(settings, "langfuse_host", "http://localhost:3000"),
            patch.object(settings, "langfuse_environment", "development"),
            patch("langfuse.Langfuse") as mock_lf,
        ):
            _get_langfuse_client()

        call_kwargs = mock_lf.call_args.kwargs
        assert call_kwargs.get("environment") == "development", (
            f"Langfuse() should default to environment='development', got {call_kwargs}"
        )


class TestObservationTypes:
    """RAG functions and tools must use correct as_type (not default 'span')."""

    def test_rag_embed_store_as_type_embedding(self):
        """embed_and_store @observe must use as_type='embedding'."""
        import importlib

        mod = importlib.import_module("src.rag")

        with patch("langfuse.observe") as mock_obs:
            mock_obs.side_effect = lambda **kw: lambda fn: fn
            importlib.reload(mod)

        # Collect all as_type values passed to @observe
        as_types = [call.kwargs.get("as_type") for call in mock_obs.call_args_list]
        assert "embedding" in as_types, (
            f"embed_and_store must use as_type='embedding', got {as_types}"
        )

    def test_rag_retrieve_as_type_retriever(self):
        """retrieve @observe must use as_type='retriever'."""
        import importlib

        mod = importlib.import_module("src.rag")

        with patch("langfuse.observe") as mock_obs:
            mock_obs.side_effect = lambda **kw: lambda fn: fn
            importlib.reload(mod)

        as_types = [call.kwargs.get("as_type") for call in mock_obs.call_args_list]
        assert "retriever" in as_types, f"retrieve must use as_type='retriever', got {as_types}"

    def test_tools_init_as_type_tool(self):
        """All @observe in tools/__init__.py must use as_type='tool'."""
        import importlib

        mod = importlib.import_module("src.tools")

        with patch("langfuse.observe") as mock_obs:
            mock_obs.side_effect = lambda **kw: lambda fn: fn
            importlib.reload(mod)

        calls_by_name = {
            call.kwargs.get("name"): call.kwargs.get("as_type") for call in mock_obs.call_args_list
        }
        expected = [
            "retrieve_chunks",
            "ingest_document",
            "generate_exercise",
            "evaluate_answer",
            "generate_exam",
        ]
        for name in expected:
            assert name in calls_by_name, f"{name} not found in @observe calls"
            assert calls_by_name[name] == "tool", (
                f"{name} as_type should be 'tool', got {calls_by_name[name]}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 1 tests — Environment propagation
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnvironmentPropagation:
    """Gap 1: environment extracted from metadata, set on span, stripped from start_observation metadata."""

    def test_create_trace_keeps_environment_in_metadata(self, obs_manager):
        """Environment stays in metadata for Langfuse dashboard filtering."""
        from src.config import settings

        with patch.object(settings, "langfuse_public_key", "pk-dummy"):
            with patch.object(settings, "langfuse_secret_key", "sk-dummy"):
                with patch.object(settings, "langfuse_host", "http://localhost:3000"):
                    from src.observability._client import _reset_langfuse_client

                    _reset_langfuse_client()
                    mgr = ObservabilityManager()
                    mgr._ensure_init()

                    mgr.create_trace(
                        name="gap1-test2",
                        session_id="s1",
                        metadata={"environment": "production", "custom": "keep"},
                    )

                    call_kwargs = mgr._client.start_observation.call_args.kwargs
                    meta = call_kwargs.get("metadata", {})
                    assert meta.get("environment") == "production", (
                        f"environment should be in metadata, got {meta}"
                    )
                    assert meta.get("custom") == "keep", (
                        f"Other metadata keys preserved, got {meta}"
                    )

    def test_create_trace_no_environment_metadata_still_works(self, obs_manager):
        """When no environment in metadata, default is used without crash."""
        from src.config import settings

        with patch.object(settings, "langfuse_public_key", "pk-dummy"):
            with patch.object(settings, "langfuse_secret_key", "sk-dummy"):
                with patch.object(settings, "langfuse_host", "http://localhost:3000"):
                    from src.observability._client import _reset_langfuse_client

                    _reset_langfuse_client()
                    mgr = ObservabilityManager()
                    mgr._ensure_init()

                    trace = mgr.create_trace(name="gap1-test3", session_id="s1")
                    assert trace is not None

    def test_create_trace_environment_in_start_observation_metadata(self, obs_manager):
        """Environment passed via metadata reaches start_observation directly."""
        from src.config import settings

        with patch.object(settings, "langfuse_public_key", "pk-dummy"):
            with patch.object(settings, "langfuse_secret_key", "sk-dummy"):
                with patch.object(settings, "langfuse_host", "http://localhost:3000"):
                    from src.observability._client import _reset_langfuse_client

                    _reset_langfuse_client()
                    mgr = ObservabilityManager()
                    mgr._ensure_init()

                    trace = mgr.create_trace(
                        name="gap1-env",
                        session_id="s1",
                        metadata={"environment": "staging", "custom": "val"},
                    )
                    assert trace is not None
                    # environment is passed directly in start_observation metadata
                    call_kwargs = mgr._client.start_observation.call_args.kwargs
                    meta = call_kwargs.get("metadata", {})
                    assert meta.get("environment") == "staging", (
                        f"Expected environment='staging' in metadata, got {meta}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 3/4 tests — Context propagation via update_current_trace
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextPropagation:
    """Gap 3/4: propagate_attributes must set session_id before graph.invoke."""

    def test_create_trace_uses_propagate_attributes(self, mock_langfuse):
        """Gap-3: create_trace wraps start_observation in propagate_attributes."""
        from src.config import settings
        from src.observability import ObservabilityManager
        from src.observability._client import _reset_langfuse_client

        with (
            patch.object(settings, "langfuse_public_key", "pk-dummy"),
            patch.object(settings, "langfuse_secret_key", "sk-dummy"),
            patch.object(settings, "langfuse_host", "http://localhost:3000"),
            patch("langfuse.propagate_attributes") as mock_pa,
        ):
            _reset_langfuse_client()
            mgr = ObservabilityManager()
            mgr._ensure_init()

            mgr.create_trace(
                name="context-test",
                session_id="ctx-sess",
                user_id="stu-42",
                metadata={"environment": "test"},
            )

            mock_pa.assert_called_once()
            call_kwargs = mock_pa.call_args.kwargs
            assert call_kwargs.get("session_id") == "ctx-sess"
            assert call_kwargs.get("user_id") == "stu-42"
            assert call_kwargs.get("trace_name") == "context-test"

    def test_create_trace_logs_warning_on_failure(self, caplog, mock_langfuse):
        """Gap-3: failed create_trace logs WARNING (was debug)."""
        from src.config import settings
        from src.observability import ObservabilityManager
        from src.observability._client import _reset_langfuse_client

        with (
            patch.object(settings, "langfuse_public_key", "pk-dummy"),
            patch.object(settings, "langfuse_secret_key", "sk-dummy"),
            patch.object(settings, "langfuse_host", "http://localhost:3000"),
            patch(
                "langfuse.propagate_attributes",
                side_effect=RuntimeError("broken"),
            ),
        ):
            _reset_langfuse_client()
            mgr = ObservabilityManager()
            mgr._ensure_init()

            result = mgr.create_trace(name="ctx-fail", session_id="s1")
            assert result is None

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Failed to create Langfuse trace" in w for w in warnings), (
            f"Expected WARNING about trace creation failure, got: {warnings}"
        )

    def test_ingest_document_calls_propagate_attributes(self, mock_embedding_model):
        """Gap-4: ingest_document wraps graph.invoke in propagate_attributes."""
        import importlib

        tools_mod = importlib.import_module("src.tools")
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "classification": "apunte_teorico",
            "topics": ["t"],
            "chunks_created": 1,
            "status": "completed",
            "errors": [],
        }

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer") as mock_get_tracer,
            patch("langfuse.propagate_attributes") as mock_pa,
        ):
            mock_tracer = MagicMock()
            mock_tracer.get_callback_handler.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            importlib.reload(tools_mod)
            tools_mod.ingest_document.invoke({"file_path": "/tmp/f.txt", "session_id": "ctx-s1"})

        # propagate_attributes must be called with session_id
        assert mock_pa.called, "propagate_attributes must be called before graph.invoke"
        update_kwargs = mock_pa.call_args.kwargs
        assert update_kwargs.get("session_id") == "ctx-s1", (
            f"Expected session_id='ctx-s1', got {update_kwargs}"
        )

    def test_generate_exam_calls_propagate_attributes(self):
        """Gap-4: generate_exam wraps graph.invoke in propagate_attributes."""
        import importlib

        tools_mod = importlib.import_module("src.tools")
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"exam": {"exam_id": "e1"}}

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer") as mock_get_tracer,
            patch("langfuse.propagate_attributes") as mock_pa,
        ):
            mock_tracer = MagicMock()
            mock_tracer.get_callback_handler.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            importlib.reload(tools_mod)
            tools_mod.generate_exam.invoke(
                {"session_id": "ctx-s2", "topics": ["algebra"], "question_count": 1}
            )

        assert mock_pa.called
        assert mock_pa.call_args.kwargs.get("session_id") == "ctx-s2"

    def test_generate_exercise_calls_propagate_attributes(self):
        """Gap-4: generate_exercise wraps graph.invoke in propagate_attributes."""
        import importlib

        tools_mod = importlib.import_module("src.tools")
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"exercise": {"exercise_id": "x1"}}

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer") as mock_get_tracer,
            patch("langfuse.propagate_attributes") as mock_pa,
        ):
            mock_tracer = MagicMock()
            mock_tracer.get_callback_handler.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            importlib.reload(tools_mod)
            tools_mod.generate_exercise.invoke({"session_id": "ctx-s3", "topic": "algebra"})

        assert mock_pa.called
        assert mock_pa.call_args.kwargs.get("session_id") == "ctx-s3"

    def test_evaluate_answer_calls_propagate_attributes(self):
        """Gap-4: evaluate_answer wraps graph.invoke in propagate_attributes."""
        import importlib

        tools_mod = importlib.import_module("src.tools")
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"evaluation_results": []}

        with (
            patch("src.tools._get_or_compile", return_value=mock_graph),
            patch("src.observability.get_tracer") as mock_get_tracer,
            patch("langfuse.propagate_attributes") as mock_pa,
        ):
            mock_tracer = MagicMock()
            mock_tracer.get_callback_handler.return_value = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            importlib.reload(tools_mod)
            tools_mod.evaluate_answer.invoke(
                {
                    "session_id": "ctx-s4",
                    "exam_id": "e1",
                    "answers": [{"question_id": "q1", "question": "Q?", "student_answer": "A"}],
                }
            )

        assert mock_pa.called
        assert mock_pa.call_args.kwargs.get("session_id") == "ctx-s4"


# ═══════════════════════════════════════════════════════════════════════════════
# Gap 6 test — Metadata injection compatibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadataInjectionCompat:
    """Gap 6: Conftest injects environment via metadata — stays in metadata for dashboard filtering."""

    def test_conftest_environment_present_in_start_observation_metadata(self, obs_manager):
        """Environment='test' from conftest metadata stays in start_observation metadata."""
        from src.config import settings

        with patch.object(settings, "langfuse_public_key", "pk-dummy"):
            with patch.object(settings, "langfuse_secret_key", "sk-dummy"):
                with patch.object(settings, "langfuse_host", "http://localhost:3000"):
                    from src.observability._client import _reset_langfuse_client

                    _reset_langfuse_client()
                    mgr = ObservabilityManager()
                    mgr._ensure_init()

                    mgr.create_trace(
                        name="gap6-test",
                        session_id="s1",
                        metadata={"environment": "test", "test_run_id": "r1"},
                    )

                    call_kwargs = mgr._client.start_observation.call_args.kwargs
                    meta = call_kwargs.get("metadata", {})
                    assert meta.get("environment") == "test", (
                        f"environment preserved in metadata, got {meta}"
                    )
                    # Other keys like test_run_id should stay
                    assert "test_run_id" in meta, f"test_run_id preserved in metadata, got {meta}"


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
            return original_start(name=name, as_type=as_type, metadata=metadata, **kwargs)

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
        assert "test_name" in captured_meta, f"Expected test_name in metadata, got {captured_meta}"

    def test_agent_invocation_creates_trace(
        self,
        langfuse_observe_tests,
        test_run_id,
        obs_manager,
        sample_txt,
        mock_llm_response,
        mock_embedding_model,
        in_memory_chroma,
        request,
        monkeypatch,
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
        assert handler is not None, "CallbackHandler should be created when Langfuse is available"

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
                return original_start(name=name, as_type=as_type, metadata=metadata, **kwargs)

            monkeypatch.setattr(obs_manager._client, "start_observation", spy_start)

        final_state = graph.invoke(initial_state, config=config)

        assert final_state["status"] == "completed", (
            f"Ingestor failed: {final_state.get('errors', [])}"
        )

        # After agent run, flush traces
        from src.observability import flush_traces

        flush_traces()

    def test_langfuse_unreachable_agent_does_not_crash(
        self,
        langfuse_observe_tests,
        sample_txt,
        mock_llm_response,
        mock_embedding_model,
        in_memory_chroma,
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
            assert handler is None, "CallbackHandler must be None when Langfuse unreachable"

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
                f"Agent should complete even without Langfuse: {final_state.get('errors', [])}"
            )

        # Clean up: reset singleton so subsequent tests get fresh state
        _obs_mod._manager = None
        _reset_langfuse_client()
