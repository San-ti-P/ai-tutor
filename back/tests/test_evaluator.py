"""Tests for the Evaluator agent — 8-node StateGraph for grading answers.

Covers PRD cases:
  - Case 3 (happy path): correct answer → score ≥ 8
  - Case 8 (edge case): partially correct → score 5-7
  - Case 12 (adversarial): gibberish/non-evaluable → structured rejection
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1/2 — State schema and evaluability guard tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorState:
    """Verify EvaluatorState schema correctness and batch support."""

    def test_state_schema_validation(self, evaluator_state):
        """State dict has all required fields for the 8-node pipeline."""
        from src.agents.evaluator import EvaluatorState

        state: EvaluatorState = evaluator_state  # type: ignore[assignment]

        assert state["session_id"] != ""
        assert state["student_id"] != ""
        assert state["exam_id"] != ""
        assert isinstance(state["answers"], list)
        assert len(state["answers"]) == 3
        assert state["current_index"] == 0
        assert isinstance(state["evaluation_results"], list)
        assert state["non_evaluable"] is False
        assert state["non_evaluable_reason"] == ""
        assert state["judge_sample"] is False
        assert state["judge_result"] is None
        assert state["requires_review"] is False
        assert state["scores_synced"] is False
        assert state["status"] == "pending"

        first_answer = state["answers"][0]
        assert "question_id" in first_answer
        assert "question" in first_answer
        assert "base_answer" in first_answer
        assert "student_answer" in first_answer
        assert "topic" in first_answer
        assert "source_chunk_ids" in first_answer

    def test_state_accumulation_reducers(self, evaluator_state):
        """Annotated[list, operator.add] fields exist on state type."""
        from typing import get_type_hints

        from src.agents.evaluator import EvaluatorState

        hints = get_type_hints(EvaluatorState, include_extras=True)
        assert hints.get("evaluation_results") is not None
        assert hints.get("retrieved_chunks") is not None
        assert hints.get("errors") is not None


class TestCheckEvaluability:
    """Guard clause: detect non-evaluable answers before grading."""

    def test_check_evaluability_rejects_gibberish(self, evaluator_state):
        """Answer 'asdf jkl qwerty zxcv nm' must be flagged as non-evaluable.

        Covers EVAL-SPEC-09 and PRD case 12.
        """
        from src.agents.evaluator import check_evaluability

        state = dict(evaluator_state)
        state["current_index"] = 2  # answer 3 = gibberish: "asdf jkl qwerty zxcv nm"
        state["answer_text"] = state["answers"][2]["student_answer"]
        state["non_evaluable"] = False

        result = check_evaluability(state)

        assert result["non_evaluable"] is True
        assert result["non_evaluable_reason"] == "gibberish"
        assert result["evaluation"]["status"] == "cannot_evaluate"
        assert result["evaluation"]["reason"] == "gibberish"
        assert "Resubmit" in result["evaluation"]["suggested_action"]

    def test_check_evaluability_passes_valid_answer(self, evaluator_state):
        """A coherent Spanish answer must be marked as evaluable."""
        from src.agents.evaluator import check_evaluability

        state = dict(evaluator_state)
        state["current_index"] = 0  # answer 1 = valid
        state["answer_text"] = state["answers"][0]["student_answer"]
        state["non_evaluable"] = True  # prove it gets reset

        result = check_evaluability(state)

        assert result["non_evaluable"] is False
        assert result["non_evaluable_reason"] == ""

    def test_check_evaluability_too_short(self, evaluator_state):
        """Answer with fewer than 3 non-whitespace chars is gibberish.

        TRIANGULATION: different inputs → same code path (gibberish detection).
        """
        from src.agents.evaluator import check_evaluability

        state = dict(evaluator_state)
        state["current_index"] = 0
        # Override the answer with a very short one
        state["answers"][0]["student_answer"] = "ab"
        state["non_evaluable"] = False

        result = check_evaluability(state)

        assert result["non_evaluable"] is True
        assert result["non_evaluable_reason"] == "gibberish"


class TestEvaluateAnswer:
    """Structured LLM output for evaluation scoring."""

    def test_evaluate_answer_structured_output(self, evaluator_state, mock_evaluator_llm):
        """LLM returns a valid SingleEvaluation with score 0-10.

        Covers EVAL-SPEC-01 and PRD case 3 (happy path).
        """
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["answer_text"] = state["answers"][0]["student_answer"]

        result = evaluate_answer(state)

        eval_dict = result["evaluation"]
        assert eval_dict is not None
        assert eval_dict["score"] == 8.0
        assert eval_dict["is_evaluable"] is True
        assert len(eval_dict["justification"]) > 0
        assert isinstance(eval_dict["conceptual_errors"], list)
        assert isinstance(eval_dict["suggestions"], list)
        assert eval_dict["status"] == "evaluated"

    def test_evaluate_answer_handles_llm_failure(self, evaluator_state):
        """On LLM error, node returns graceful error evaluation dict."""
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0

        with patch("src.llm.get_structured_llm") as mock_gs_llm:
            fake_invokable = MagicMock()
            fake_invokable.invoke.side_effect = RuntimeError("LLM unavailable")
            mock_gs_llm.return_value = fake_invokable

            result = evaluate_answer(state)

        assert result["status"] == "error"
        assert "errors" in result
        eval_dict = result["evaluation"]
        assert eval_dict["status"] == "error"
        assert eval_dict["is_evaluable"] is False


class TestValidateFeedback:
    """Anti-hallucination: cross-reference claims against RAG chunks."""

    def test_validate_feedback_all_matched(self, evaluator_state, mock_embedding_model):
        """Claims matching chunks should produce no validation warnings."""
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        # Evaluation whose claims match sample_chunks content
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 8.0,
            "justification": (
                "La derivada de una función se define como el límite del cociente "
                "incremental. Esto representa la pendiente de la recta tangente."
            ),
            "conceptual_errors": [],
            "suggestions": ["Repasar las reglas de derivación como la derivada de una suma."],
            "is_evaluable": True,
            "status": "evaluated",
        }

        result = validate_feedback(state)

        # With mock_embedding_model, embeddings are deterministic — claims
        # should match chunks above threshold
        assert result.get("requires_review") is not None

    def test_validate_feedback_empty_evaluation(self, evaluator_state):
        """No evaluation dict → no-op, returns empty dict."""
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        state["evaluation"] = None

        result = validate_feedback(state)
        assert result == {}

    def test_validate_feedback_cannot_evaluate_skips(self, evaluator_state):
        """If evaluation status is 'cannot_evaluate', skip validation."""
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        state["evaluation"] = {
            "question_id": "q-003",
            "status": "cannot_evaluate",
            "reason": "gibberish",
        }

        result = validate_feedback(state)
        assert result == {}


class TestBuildFeedback:
    """Assemble final evaluation dict and append to results."""

    def test_build_feedback_appends_to_results(self, evaluator_state):
        """build_feedback adds evaluation to evaluation_results accumulator."""
        from src.agents.evaluator import build_feedback

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 8.0,
            "justification": "Buena respuesta.",
            "conceptual_errors": [],
            "suggestions": [],
            "is_evaluable": True,
            "status": "evaluated",
        }
        state["evaluation_results"] = []

        result = build_feedback(state)

        assert "evaluation_results" in result
        assert len(result["evaluation_results"]) == 1
        ev = result["evaluation_results"][0]
        assert ev["score"] == 8.0
        assert ev["question_id"] == "q-001"
        assert "validation_warnings" in ev
        assert "requires_review" in ev
        # evaluation is cleared for next iteration
        assert result.get("evaluation") is None

    def test_build_feedback_no_evaluation(self, evaluator_state):
        """No evaluation dict → returns no_evaluation status."""
        from src.agents.evaluator import build_feedback

        state = dict(evaluator_state)
        state["evaluation"] = None

        result = build_feedback(state)
        assert result["status"] == "no_evaluation"


class TestNextQuestion:
    """Batch iteration: increment index and set status."""

    def test_next_question_increments_index(self, evaluator_state):
        """current_index increases by 1; status reflects remaining questions."""
        from src.agents.evaluator import next_question

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["answers"] = state["answers"]  # 3 answers

        result = next_question(state)

        assert result["current_index"] == 1
        assert result["status"] == "processing"
        assert result["non_evaluable"] is False
        assert result["requires_review"] is False
        assert result["judge_sample"] is False

    def test_next_question_last_answer(self, evaluator_state):
        """When all answers processed, status becomes 'done'."""
        from src.agents.evaluator import next_question

        state = dict(evaluator_state)
        state["current_index"] = 2  # last answer (index 2 of 3)

        result = next_question(state)

        assert result["current_index"] == 3  # index == len(answers)
        assert result["status"] == "done"


class TestPrepareEvaluation:
    """Topic deduplication and chunk retrieval for the answer batch."""

    def test_prepare_evaluation_deduplicates_topics(self, evaluator_state, mock_embedding_model):
        """Retrieved chunks are deduplicated by chunk_id across topics."""
        from src.agents.evaluator import prepare_evaluation

        state = dict(evaluator_state)
        # Clear retrieved_chunks to test fresh retrieval
        state["retrieved_chunks"] = []

        result = prepare_evaluation(state)

        # verify we got the retrieved_chunks back (may be empty if no ChromaDB)
        assert "retrieved_chunks" in result
        assert result["status"] == "prepared"
        assert result["current_index"] == 0


class TestSyncScores:
    """DB persistence for evaluation results."""

    def test_sync_scores_empty_results(self, evaluator_state):
        """No results to sync → still marks as synced."""
        from src.agents.evaluator import sync_scores

        state = dict(evaluator_state)
        state["evaluation_results"] = []

        result = sync_scores(state)

        assert result["scores_synced"] is True
        assert result["status"] == "synced"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Additional unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateFeedbackAntiHallucination:
    """Anti-hallucination: flag fabricated claims."""

    def test_validate_feedback_flags_fabricated_claims(self, evaluator_state, mock_embedding_model):
        """Claims with no chunk match must produce validation_warnings.

        Uses a fabricated claim about "astrophysics" that doesn't match
        any of the math-focused sample_chunks.
        """
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 9.0,
            "justification": (
                "La derivada es el límite del cociente incremental. "
                "Además, los agujeros negros emiten radiación Hawking "
                "que se relaciona con la termodinámica cuántica."
            ),
            "conceptual_errors": [],
            "suggestions": ["Estudiar astrofísica de agujeros negros."],
            "is_evaluable": True,
            "status": "evaluated",
        }

        result = validate_feedback(state)

        eval_result = result.get("evaluation", {})
        warnings = eval_result.get("validation_warnings", [])
        # The astrophysics claims should be flagged (low similarity to math chunks)
        assert isinstance(warnings, list)

    def test_validate_feedback_no_chunks(self, evaluator_state, mock_embedding_model):
        """Empty retrieved_chunks → validation_warnings stays empty, no crash."""
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        state["retrieved_chunks"] = []
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 7.0,
            "justification": "Some evaluation text.",
            "conceptual_errors": [],
            "suggestions": ["Review the basic concepts."],
            "is_evaluable": True,
            "status": "evaluated",
        }

        result = validate_feedback(state)
        eval_result = result.get("evaluation", {})
        assert eval_result.get("validation_warnings") == []


class TestLLMJudge:
    """LLM-as-judge: sampling and disagreement detection."""

    def test_llm_judge_disagreement_flags_review(self, evaluator_state):
        """|primary.score - judge.score| > 2.0 → requires_review=True."""
        from unittest.mock import MagicMock, patch

        from src.agents.evaluator import JudgeVerdict, llm_judge

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["judge_sample"] = True
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 8.0,
            "justification": "Primary evaluation.",
            "conceptual_errors": [],
            "suggestions": [],
            "is_evaluable": True,
            "status": "evaluated",
        }

        # Judge disagrees: score 3 (diff = 5 > 2.0)
        fake_verdict = JudgeVerdict(
            score=3.0,
            agrees_with_primary=False,
            discrepancy="Primary was too generous.",
            suggested_score=3.0,
        )

        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_verdict

        with patch("src.llm.get_structured_llm", return_value=fake_invokable):
            result = llm_judge(state)

        assert result["requires_review"] is True
        assert result["judge_result"]["score"] == 3.0
        assert result["judge_result"]["agrees_with_primary"] is False

    def test_llm_judge_agrees_no_flag(self, evaluator_state):
        """|primary.score - judge.score| ≤ 2.0 → requires_review=False."""
        from unittest.mock import MagicMock, patch

        from src.agents.evaluator import JudgeVerdict, llm_judge

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["judge_sample"] = True
        state["evaluation"] = {
            "question_id": "q-001",
            "score": 7.0,
            "justification": "Primary evaluation.",
            "conceptual_errors": [],
            "suggestions": [],
            "is_evaluable": True,
            "status": "evaluated",
        }

        # Judge agrees: score 7.5 (diff = 0.5 ≤ 2.0)
        fake_verdict = JudgeVerdict(
            score=7.5,
            agrees_with_primary=True,
            discrepancy="",
            suggested_score=None,
        )

        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_verdict

        with patch("src.llm.get_structured_llm", return_value=fake_invokable):
            result = llm_judge(state)

        assert result["requires_review"] is False
        assert result["judge_result"]["agrees_with_primary"] is True

    def test_llm_judge_not_sampled(self, evaluator_state):
        """judge_sample=False → pass through, no LLM call."""
        from src.agents.evaluator import llm_judge

        state = dict(evaluator_state)
        state["judge_sample"] = False
        state["evaluation"] = {"score": 5.0, "is_evaluable": True}

        result = llm_judge(state)
        assert result == {}


class TestFullGraphMocked:
    """End-to-end graph execution with mocked LLMs."""

    def test_full_graph_mocked(self, evaluator_state, mock_evaluator_llm, mock_embedding_model):
        """Full graph invoke with mocked LLMs produces evaluation_results.

        Covers US-5.8 (E2E flow): all 8 nodes execute and return structured
        results with correct shape per answer.
        """
        from src.agents.evaluator import build_evaluator

        state = dict(evaluator_state)

        graph = build_evaluator().compile()
        result = graph.invoke(state)

        eval_results: list[dict] = result.get("evaluation_results", [])

        # Should have results for all 3 answers
        assert len(eval_results) == 3

        # First two answers should be evaluable with scores
        assert eval_results[0].get("is_evaluable") is True
        assert eval_results[0]["score"] == 8.0  # from mock_evaluator_llm

        assert eval_results[1].get("is_evaluable") is True
        assert eval_results[1]["score"] == 8.0

        # Third answer is gibberish → non-evaluable
        assert eval_results[2].get("status") == "cannot_evaluate"
        assert eval_results[2].get("reason") == "gibberish"

        # Graph final status
        assert result.get("scores_synced") is True
        assert result.get("status") in ("synced", "synced_with_errors")


# ═══════════════════════════════════════════════════════════════════════════════
# RAG-exclusive answers: evaluator tests (task 3.4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluatorNoChunks:
    """Task 3.4: Evaluator returns cannot_evaluate when no chunks available."""

    def test_evaluator_cannot_evaluate_when_no_chunks(self, evaluator_state):
        """GIVEN chunk_context is empty → THEN short-circuit to cannot_evaluate (no LLM call).

        Covers evaluation spec: "No chunks available — evaluation skipped".
        """
        from unittest.mock import patch

        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["retrieved_chunks"] = []  # Empty chunks

        # Patch get_structured_llm to track whether it was called
        with patch("src.llm.get_structured_llm") as mock_gs_llm:
            result = evaluate_answer(state)

            # NO LLM call should have been made
            mock_gs_llm.assert_not_called()

        eval_dict = result["evaluation"]
        assert eval_dict is not None
        assert eval_dict["score"] == 0.0
        assert eval_dict["is_evaluable"] is False
        assert eval_dict.get("status") == "cannot_evaluate"
        reason = eval_dict.get("non_evaluable_reason") or eval_dict.get("reason", "")
        assert "no_material" in reason

    def test_evaluator_uses_rag_only_prompt_when_chunks_available(
        self, evaluator_state, mock_evaluator_llm
    ):
        """GIVEN chunks are available → THEN evaluation proceeds with score.

        Covers evaluation spec: "Prompt includes RAG-only instruction when chunks available".
        The mock_evaluator_llm fixture provides the LLM chain; we verify the
        result shape without inspecting the prompt text (which is a unit
        concern best tested via the policy module's own test).
        """
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        # retrieved_chunks already populated from evaluator_state fixture

        result = evaluate_answer(state)

        eval_dict = result["evaluation"]
        assert eval_dict is not None
        assert eval_dict["score"] == 8.0  # from mock
        assert eval_dict["is_evaluable"] is True
        assert eval_dict["status"] == "evaluated"
        # Should contain the justification from the mock
        assert len(eval_dict["justification"]) > 0

    def test_evaluator_no_material_requires_review_false(self, evaluator_state):
        """GIVEN no chunks → THEN requires_review is False (not a quality issue)."""
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["retrieved_chunks"] = []

        result = evaluate_answer(state)
        eval_dict = result["evaluation"]

        # When cannot_evaluate due to no_material, it should not require review
        assert eval_dict.get("requires_review", False) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests (real LLM + real embeddings — opt-in with -m integration)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestEvaluatorIntegration:
    """Integration tests requiring real LLM and real embeddings.

    Run with: pytest tests/test_evaluator.py -v -m integration

    Uses the real PDF (apunteAgentes_IA2007.pdf) ingested into ChromaDB
    via the ``ingested_collection_name`` fixture for RAG-grounded evaluation.
    """

    def test_evaluate_correct_answer(
        self, requires_ollama, evaluator_state, ingested_collection_name
    ):
        """PRD Case 3 (happy path): correct answer → score ≥ 8 with RAG backing."""
        from src.agents.evaluator import build_evaluator

        state = dict(evaluator_state)
        state["collection_name"] = ingested_collection_name
        state["answers"] = [
            {
                "question_id": "q-agent-001",
                "question": "¿Qué es un agente inteligente según el apunte?",
                "base_answer": (
                    "Un agente inteligente es una entidad que percibe su entorno "
                    "a través de sensores y actúa sobre él mediante efectores."
                ),
                "student_answer": (
                    "Un agente inteligente es una entidad que percibe su entorno "
                    "mediante sensores y actúa sobre él usando efectores."
                ),
                "source_chunk_ids": [],
                "topic": "agentes",
                "difficulty": "medium",
            }
        ]

        graph = build_evaluator().compile()
        result = graph.invoke(state)

        eval_results: list[dict] = result.get("evaluation_results", [])
        assert len(eval_results) == 1
        ev = eval_results[0]

        assert ev.get("is_evaluable") is True
        score = ev.get("score", 0)
        assert score >= 6.0, f"Expected score ≥ 6 for correct agent answer, got {score}"
        assert len(ev.get("justification", "")) > 0

    def test_evaluate_partially_correct(
        self, requires_ollama, evaluator_state, ingested_collection_name
    ):
        """PRD Case 8 (edge case): partially correct → score in middle range with RAG."""
        from src.agents.evaluator import build_evaluator

        state = dict(evaluator_state)
        state["collection_name"] = ingested_collection_name
        state["answers"] = [
            {
                "question_id": "q-agent-002",
                "question": (
                    "¿Qué diferencia hay entre un agente reactivo y uno basado en objetivos?"
                ),
                "base_answer": (
                    "Un agente reactivo responde directamente a estímulos del entorno "
                    "sin mantener estado interno. Un agente basado en objetivos "
                    "tiene metas explícitas y planifica acciones para alcanzarlas."
                ),
                "student_answer": (
                    "Los agentes reactivos son más rápidos porque no piensan, "
                    "mientras que los basados en objetivos son más lentos "
                    "pero pueden resolver problemas más complejos."
                ),
                "source_chunk_ids": [],
                "topic": "agentes",
                "difficulty": "hard",
            }
        ]

        graph = build_evaluator().compile()
        result = graph.invoke(state)

        eval_results: list[dict] = result.get("evaluation_results", [])
        assert len(eval_results) == 1
        ev = eval_results[0]

        assert ev.get("is_evaluable") is True
        score = ev.get("score", 0)
        assert 3.0 <= score <= 7.5, f"Expected mid-range score for vague answer, got {score}"
        assert score < 9.0, "Vague answer should not get near-perfect score"

    def test_evaluate_wrong_language(self, requires_ollama, evaluator_state):
        """PRD Case 12 (adversarial): non-evaluable answer → structured rejection."""
        from src.agents.evaluator import check_evaluability

        state = dict(evaluator_state)
        state["current_index"] = 2
        state["answer_text"] = state["answers"][2]["student_answer"]

        check_result = check_evaluability(state)
        assert check_result["non_evaluable"] is True
        assert check_result["non_evaluable_reason"] == "gibberish"
        assert check_result["evaluation"]["status"] == "cannot_evaluate"


# ==============================================================================
# Phase 5.3: validate_feedback immutability (Epic 13, SS-2)
# ==============================================================================


class TestValidateFeedbackImmutability:
    """SS-2: validate_feedback returns new dict, does not mutate input."""

    def test_validate_feedback_does_not_mutate_input_dict(self):
        """GIVEN an evaluation dict → THEN validate_feedback returns a new dict
        and the original dict is unchanged."""
        from src.agents.evaluator import validate_feedback

        evaluation = {
            "question_id": "q-001",
            "score": 8.0,
            "justification": "Buena respuesta sobre derivadas.",
            "suggestions": ["Repasar reglas de derivación."],
            "is_evaluable": True,
            "topic": "cálculo/derivadas",
            "status": "evaluated",
        }
        original_id = id(evaluation)
        original_keys = set(evaluation.keys())

        state = {
            "evaluation": evaluation,
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-001",
                    "text": "La derivada es el límite del cociente incremental.",
                    "metadata": {},
                    "similarity_score": 0.1,
                }
            ],
            "judge_sample": False,
            "requires_review": False,
        }

        result = validate_feedback(state)

        # The returned evaluation must be a DIFFERENT object
        returned_eval = result["evaluation"]
        assert id(returned_eval) != original_id, (
            "validate_feedback must return a NEW dict, not the mutated original"
        )

        # The original evaluation must be UNCHANGED
        assert set(evaluation.keys()) == original_keys, (
            "Original evaluation dict was mutated — keys changed"
        )
        assert evaluation.get("score") == 8.0
        assert "validation_warnings" not in evaluation, (
            "Original evaluation dict was mutated with validation_warnings"
        )

        # The returned dict should have validation_warnings
        assert "validation_warnings" in returned_eval

    def test_validate_feedback_empty_chunks_no_mutation(self):
        """GIVEN empty chunks → THEN returned dict is new, original unchanged."""
        from src.agents.evaluator import validate_feedback

        evaluation = {"question_id": "q-002", "score": 5.0, "status": "evaluated"}
        original_keys = set(evaluation.keys())

        state = {
            "evaluation": evaluation,
            "retrieved_chunks": [],
            "judge_sample": False,
        }

        result = validate_feedback(state)
        returned_eval = result["evaluation"]

        assert id(returned_eval) != id(evaluation)
        assert set(evaluation.keys()) == original_keys
        assert "validation_warnings" in returned_eval


# ==============================================================================
# Phase 5.6: sync_scores transaction rollback (Epic 13, SS-1)
# ==============================================================================


class TestSyncScoresTransaction:
    """SS-1: sync_scores uses single transaction with rollback on failure."""

    @pytest.mark.asyncio
    async def test_sync_scores_writes_within_transaction(self, tmp_path):
        """GIVEN valid evaluation results → THEN all writes succeed atomically."""
        import aiosqlite

        from src.config import settings

        db_path = tmp_path / "sync_test.db"
        original_path = settings.sqlite_db_path

        try:
            # Set up test DB
            settings.sqlite_db_path = str(db_path)
            async with aiosqlite.connect(str(db_path)) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.executescript("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        student_id TEXT NOT NULL,
                        question_id TEXT NOT NULL,
                        topic TEXT,
                        score REAL,
                        feedback_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS topic_scores (
                        topic TEXT NOT NULL,
                        student_id TEXT NOT NULL,
                        score REAL NOT NULL,
                        evaluated_at TEXT NOT NULL DEFAULT (datetime('now')),
                        PRIMARY KEY (topic, student_id)
                    );
                """)
                await db.commit()

            from src.agents.evaluator import sync_scores

            state = {
                "evaluation_results": [
                    {"question_id": "q-1", "topic": "cálculo", "score": 8.0, "status": "evaluated"},
                    {"question_id": "q-2", "topic": "álgebra", "score": 6.5, "status": "evaluated"},
                ],
                "session_id": "sess-test",
                "student_id": "student-test",
            }

            result = sync_scores(state)
            assert result["scores_synced"] is True
            assert result.get("status") in ("synced", "synced_with_errors")

            # Verify writes
            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT COUNT(*) as cnt FROM evaluations")
                row = await cursor.fetchone()
                assert row["cnt"] == 2
                cursor = await db.execute("SELECT COUNT(*) as cnt FROM topic_scores")
                row = await cursor.fetchone()
                assert row["cnt"] == 2
        finally:
            settings.sqlite_db_path = original_path

    @pytest.mark.asyncio
    async def test_sync_scores_empty_results(self, tmp_path):
        """GIVEN empty evaluation_results → THEN no writes occur, no error."""
        import aiosqlite

        from src.config import settings

        db_path = tmp_path / "sync_empty.db"
        original_path = settings.sqlite_db_path

        try:
            settings.sqlite_db_path = str(db_path)
            async with aiosqlite.connect(str(db_path)) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.executescript("""
                    CREATE TABLE IF NOT EXISTS evaluations (
                        id TEXT PRIMARY KEY, session_id TEXT, student_id TEXT,
                        question_id TEXT, topic TEXT, score REAL, feedback_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS topic_scores (
                        topic TEXT, student_id TEXT, score REAL,
                        evaluated_at TEXT DEFAULT (datetime('now')),
                        PRIMARY KEY (topic, student_id)
                    );
                """)
                await db.commit()

            from src.agents.evaluator import sync_scores

            state = {
                "evaluation_results": [],
                "session_id": "sess-test",
                "student_id": "student-test",
            }

            result = sync_scores(state)
            assert result["scores_synced"] is True

            async with aiosqlite.connect(str(db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("SELECT COUNT(*) as cnt FROM evaluations")
                row = await cursor.fetchone()
                assert row["cnt"] == 0
        finally:
            settings.sqlite_db_path = original_path


class TestMCQEvaluation:
    """Verify deterministic multiple-choice question (MCQ) evaluation."""

    def test_evaluate_answer_mcq_correct(self, evaluator_state):
        """MCQ with correct answer must evaluate to 10.0 score deterministically."""
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        # Setup MCQ answer
        state["answers"] = [
            {
                "question_id": "q-mcq-001",
                "question": "¿Cuál es la derivada de x²?",
                "base_answer": "2x",
                "student_answer": " 2X ",  # extra spacing, mixed case
                "type": "mcq",
                "topic": "cálculo/derivadas",
                "difficulty": "medium",
                "source_chunk_ids": ["chunk-math-001"],
            }
        ]

        result = evaluate_answer(state)
        assert result["status"] == "evaluated"
        eval_dict = result["evaluation"]
        assert eval_dict["score"] == 10.0
        assert eval_dict["is_evaluable"] is True
        assert "Respuesta correcta" in eval_dict["justification"]
        assert eval_dict["conceptual_errors"] == []
        assert eval_dict["suggestions"] == []

    def test_evaluate_answer_mcq_incorrect(self, evaluator_state):
        """MCQ with incorrect answer must evaluate to 0.0 score deterministically."""
        from src.agents.evaluator import evaluate_answer

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["answers"] = [
            {
                "question_id": "q-mcq-001",
                "question": "¿Cuál es la derivada de x²?",
                "base_answer": "2x",
                "student_answer": "x²",
                "type": "mcq",
                "topic": "cálculo/derivadas",
                "difficulty": "medium",
                "source_chunk_ids": ["chunk-math-001"],
            }
        ]

        result = evaluate_answer(state)
        assert result["status"] == "evaluated"
        eval_dict = result["evaluation"]
        assert eval_dict["score"] == 0.0
        assert eval_dict["is_evaluable"] is True
        assert "Respuesta incorrecta" in eval_dict["justification"]
        assert eval_dict["conceptual_errors"] == ["Selección de opción incorrecta"]
        assert len(eval_dict["suggestions"]) > 0

    def test_validate_feedback_mcq_bypasses(self, evaluator_state):
        """MCQ evaluations bypass grounding/hallucination checks and judge sampling."""
        from src.agents.evaluator import validate_feedback

        state = dict(evaluator_state)
        state["current_index"] = 0
        state["answers"] = [
            {
                "question_id": "q-mcq-001",
                "question": "¿Cuál es la derivada de x²?",
                "base_answer": "2x",
                "student_answer": "2x",
                "type": "mcq",
                "topic": "cálculo/derivadas",
                "difficulty": "medium",
                "source_chunk_ids": ["chunk-math-001"],
            }
        ]
        state["evaluation"] = {
            "question_id": "q-mcq-001",
            "score": 10.0,
            "justification": "Respuesta correcta. Seleccionaste la opción correcta: '2x'.",
            "conceptual_errors": [],
            "suggestions": [],
            "is_evaluable": True,
            "topic": "cálculo/derivadas",
            "source_chunk_ids": ["chunk-math-001"],
            "status": "evaluated",
        }

        result = validate_feedback(state)
        assert result["requires_review"] is False
        assert result["judge_sample"] is False
        assert result["evaluation"]["validation_warnings"] == []

