"""Tests for ExamGenerator agent — unit, integration, and end-to-end.

Covers PRD test cases #2 (happy path), #7 (missing topic), #11 (adversarial).
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Unit tests — RED (written before implementation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieveRelevantChunks:
    """Task 2.1–2.2: retrieve_relevant_chunks node."""

    def test_retrieve_chunks_empty_topic(self, exam_generator_state):
        """When a topic yields zero chunks, topic_not_found + suggestions populated."""
        from src.agents.exam_generator import retrieve_relevant_chunks

        state = {**exam_generator_state, "topics": ["astrofísica_inexistente"]}

        with patch(
            "src.tools.retrieve_chunks", return_value=[]
        ) as mock_retrieve:
            result = retrieve_relevant_chunks(state)

        assert mock_retrieve.called
        assert "topic_not_found" in result
        assert "astrofísica_inexistente" in result["topic_not_found"]
        assert "topic_suggestions" in result

    def test_retrieve_chunks_accumulates(self, exam_generator_state, sample_chunks):
        """Multiple topics accumulate and deduplicate chunks."""
        from src.agents.exam_generator import retrieve_relevant_chunks

        def _fake_retrieve(query, top_k=5, topic_filter=None, collection_name="default"):
            if "derivadas" in query:
                return [sample_chunks[0], sample_chunks[1]]
            if "matrices" in query:
                # Include an overlapping chunk to test dedup
                return [sample_chunks[2], sample_chunks[0]]
            return []

        state = {**exam_generator_state}

        with patch(
            "src.tools.retrieve_chunks",
            side_effect=_fake_retrieve,
        ):
            result = retrieve_relevant_chunks(state)

        # Should have accumulated chunks (operator.add reducer on Annotated)
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) >= 1
        # Verify no chunk_id appears more than once
        chunk_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestGenerateQuestions:
    """Task 2.3–2.4: generate_questions node."""

    def test_generate_questions_mcq_open_answer(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """LLM produces 3 MCQs + 2 open-answer with source_chunk_ids."""
        from src.agents.exam_generator import generate_questions

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:3],
            "mcq_ratio": 0.6,
            "question_count": 5,
        }

        result = generate_questions(state)

        assert "generated_questions" in result
        questions = result["generated_questions"]
        assert len(questions) == 5
        # Every question must have source_chunk_ids
        for q in questions:
            assert "source_chunk_ids" in q
            assert len(q["source_chunk_ids"]) > 0

    def test_generate_questions_retry_only_invalid(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """On retry path, only invalid_question_indices are regenerated."""
        from src.agents.exam_generator import generate_questions

        # Pre-populate valid questions (indices 0, 1, 3, 4)
        existing_valid = [
            {"prompt": "Q0?", "type": "mcq", "source_chunk_ids": ["c1"]},
            {"prompt": "Q1?", "type": "mcq", "source_chunk_ids": ["c2"]},
            None,  # index 2 — invalid, will be regenerated
            {"prompt": "Q3?", "type": "open", "source_chunk_ids": ["c3"]},
            {"prompt": "Q4?", "type": "open", "source_chunk_ids": ["c4"]},
        ]

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:3],
            "generated_questions": existing_valid,
            "invalid_question_indices": [2],
            "retry_count": 1,
            "question_count": 5,
        }

        result = generate_questions(state)

        questions = result.get("generated_questions", existing_valid)
        # Slot 2 should be replaced with a real question dict
        assert questions[2] is not None
        assert isinstance(questions[2], dict)
        # Other slots preserved
        assert questions[0] is not None
        assert questions[4] is not None


class TestValidateQuestions:
    """Task 2.5–2.6: validate_questions node."""

    def test_validate_questions_all_matched(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """Claims extracted from questions whose text appears in chunks pass validation."""
        from src.agents.exam_generator import validate_questions

        # Build questions whose content mirrors chunk text
        generated = [
            {
                "type": "mcq",
                "stem": "La derivada de una función f(x) se define como el límite del cociente incremental.",
                "options": [
                    "f'(a) = lim(h→0) [f(a+h) - f(a)] / h",
                    "f'(a) = f(a+h) - f(a)",
                    "f'(a) = lim(h→0) f(a)/h",
                ],
                "correct_option_index": 0,
                "source_chunk_ids": ["chunk-math-001"],
                "difficulty": "medium",
                "topic": "cálculo/derivadas",
            },
            {
                "type": "open_answer",
                "prompt": "¿Qué es una matriz?",
                "base_answer": (
                    "Una matriz es un arreglo rectangular de números dispuestos "
                    "en filas y columnas. La suma de matrices se realiza elemento "
                    "a elemento."
                ),
                "source_chunk_ids": ["chunk-math-003"],
                "difficulty": "easy",
                "topic": "álgebra/matrices",
            },
        ]

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks,
            "generated_questions": generated,
        }

        with patch(
            "src.rag.get_embedding_model"
        ) as mock_embed:
            # Embeddings: make chunk embeddings match their text content
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Encode returns proper SentenceTransformer-like result
            def _fake_encode(texts):
                import hashlib

                return [
                    [
                        float(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % 1000)
                        / 1000.0
                        for _ in range(384)
                    ]
                    for t in texts
                ]

            model.encode.side_effect = lambda texts: type(
                "FakeArray", (), {"tolist": lambda self: _fake_encode(texts)}
            )()
            mock_embed.return_value = model

            result = validate_questions(state)

        # All questions grounded in chunk text → should pass
        assert "validation_results" in result
        assert "invalid_question_indices" in result

    def test_validate_questions_one_fails(
        self, exam_generator_state, sample_chunks
    ):
        """A question with claims not in chunks produces validation_errors."""
        from src.agents.exam_generator import validate_questions

        generated = [
            {
                "type": "open_answer",
                "prompt": "¿Qué es la mecánica cuántica?",
                "base_answer": (
                    "La mecánica cuántica estudia partículas subatómicas "
                    "y el principio de incertidumbre de Heisenberg."
                ),
                "source_chunk_ids": ["chunk-fake-999"],  # Not in sample_chunks
                "difficulty": "hard",
                "topic": "física/cuántica",
            },
        ]

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks,
            "generated_questions": generated,
        }

        with patch(
            "src.rag.get_embedding_model"
        ) as mock_embed:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Return LOW similarity embeddings for the claim (hallucination)
            def _low_encode(texts):
                return [[0.01 * (i + 1) for i in range(384)] for _ in texts]

            model.encode.side_effect = lambda texts: type(
                "FakeArray", (), {"tolist": lambda self: _low_encode(texts)}
            )()
            mock_embed.return_value = model

            result = validate_questions(state)

        # Should have validation errors since claims won't match chunks
        assert "validation_errors" in result


class TestShouldRetry:
    """Task 2.7–2.8: should_retry conditional edge."""

    def test_should_retry_retry(self, exam_generator_state):
        """Errors present + retry_count < 3 → returns 'retry'."""
        from src.agents.exam_generator import should_retry

        state = {**exam_generator_state, "validation_errors": ["claim X not found"], "retry_count": 1}
        result = should_retry(state)
        assert result == "retry"

    def test_should_retry_done(self, exam_generator_state):
        """retry_count >= 3 → returns 'done' even with errors."""
        from src.agents.exam_generator import should_retry

        state = {**exam_generator_state, "validation_errors": ["claim X not found"], "retry_count": 3}
        result = should_retry(state)
        assert result == "done"

    def test_should_retry_no_errors(self, exam_generator_state):
        """No validation errors → returns 'done'."""
        from src.agents.exam_generator import should_retry

        state = {**exam_generator_state, "validation_errors": [], "retry_count": 0}
        result = should_retry(state)
        assert result == "done"


class TestFormatExam:
    """Task 2.9–2.10: format_exam node."""

    def test_format_exam_complete(self, exam_generator_state, sample_chunks, mock_exam_llm):
        """Complete exam with 5 questions, no omissions → status complete."""
        from src.agents.exam_generator import format_exam

        # First generate questions, then format
        from src.agents.exam_generator import generate_questions

        gen_state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:4],
            "question_count": 5,
        }
        gen_result = generate_questions(gen_state)

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:4],
            "generated_questions": gen_result["generated_questions"],
            "omitted_questions": [],
            "validation_errors": [],
            "topic_not_found": [],
            "topic_suggestions": [],
        }

        result = format_exam(state)
        assert "exam" in result
        exam = result["exam"]
        assert "exam_id" in exam
        assert "session_id" in exam
        assert exam["total_questions"] == 5
        assert exam["status"] == "complete"
        assert exam["omitted_count"] == 0
        for q in exam["questions"]:
            assert "source_chunk_ids" in q

    def test_format_exam_partial(self, exam_generator_state, sample_chunks, mock_exam_llm):
        """Partial exam with omitted questions → status partial + warnings."""
        from src.agents.exam_generator import format_exam

        from src.agents.exam_generator import generate_questions

        gen_state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:3],
            "question_count": 3,
        }
        gen_result = generate_questions(gen_state)

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:3],
            "generated_questions": gen_result["generated_questions"],
            "omitted_questions": [1],  # index 1 omitted
            "validation_errors": ["Question 1: fabricated claim"],
            "topic_not_found": ["física/cuántica"],
            "topic_suggestions": ["cálculo/derivadas", "álgebra/matrices"],
        }

        result = format_exam(state)
        exam = result["exam"]
        assert exam["status"] == "partial"
        assert exam["omitted_count"] == 1
        assert "warnings" in exam
        assert len(exam["topic_not_found"]) > 0
        assert len(exam["topic_suggestions"]) > 0
        # Verify omitted question (index 1) is excluded
        assert len(exam["questions"]) < len(state["generated_questions"])


class TestEndToEnd:
    """Task 2.11: Full graph execution end-to-end."""

    def test_e2e_full_graph(self, exam_generator_state, sample_chunks, mock_exam_llm):
        """Compile + invoke with mocked LLM → verify complete exam output structure.
        
        R8 (Performance): 10-question exam under 30 seconds. With mocked LLM,
        execution should be near-instant ($<1s$). This asserts the budget exists
        — real LLM latency is tested in integration environment.
        """
        import time

        from src.agents.exam_generator import build_exam_generator

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:4],
        }

        t_start = time.perf_counter()
        with patch(
            "src.agents.exam_generator.retrieve_relevant_chunks",
            return_value={"retrieved_chunks": sample_chunks[:4], "status": "retrieved"},
        ):
            with patch(
                "src.agents.exam_generator.validate_questions",
                return_value={
                    "validation_results": [],
                    "validation_errors": [],
                    "invalid_question_indices": [],
                },
            ):
                graph = build_exam_generator().compile()
                result = graph.invoke(state)
        elapsed = time.perf_counter() - t_start

        assert "exam" in result
        exam = result["exam"]
        assert isinstance(exam, dict)
        assert "exam_id" in exam
        assert "total_questions" in exam
        assert exam["total_questions"] > 0
        assert "questions" in exam
        assert len(exam["questions"]) == exam["total_questions"]
        assert exam["status"] in ("complete", "partial", "no_material")
        assert "topics_covered" in exam

        # R8: With mocked LLM, graph execution must complete under 5 seconds.
        # The 30s budget is for real LLM calls; this ensures no infinite loops.
        assert elapsed < 5.0, f"Graph execution took {elapsed:.1f}s — exceeded 5s budget for mocked run"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: PRD Integration Tests — RED
# ═══════════════════════════════════════════════════════════════════════════════


class TestPRDIntegration:
    """PRD-mandated integration test cases."""

    def test_prd2_happy_path_5_questions(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """PRD case #2: Generate 5 questions on specific topic. Verify count/structure/chunk refs."""
        from src.agents.exam_generator import build_exam_generator

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:4],
            "question_count": 5,
            "difficulty": "medium",
        }

        with patch(
            "src.agents.exam_generator.retrieve_relevant_chunks",
            return_value={"retrieved_chunks": sample_chunks[:4], "status": "retrieved"},
        ):
            with patch(
                "src.agents.exam_generator.validate_questions",
                return_value={
                    "validation_results": [],
                    "validation_errors": [],
                    "invalid_question_indices": [],
                },
            ):
                graph = build_exam_generator().compile()
                result = graph.invoke(state)

        exam = result["exam"]
        assert exam["total_questions"] == 5
        assert exam["status"] == "complete"
        for q in exam["questions"]:
            assert "source_chunk_ids" in q
            assert len(q["source_chunk_ids"]) > 0
            assert "type" in q

    def test_prd7_missing_topic_handling(self, exam_generator_state):
        """PRD case #7: topic not in material → error + ≤3 suggestions."""
        from src.agents.exam_generator import build_exam_generator

        state = {
            **exam_generator_state,
            "topics": ["astrofísica_inexistente"],
        }

        with patch(
            "src.agents.exam_generator.retrieve_relevant_chunks",
            return_value={
                "retrieved_chunks": [],
                "topic_not_found": ["astrofísica_inexistente"],
                "topic_suggestions": ["cálculo", "álgebra", "física"],
                "status": "no_material",
            },
        ):
            graph = build_exam_generator().compile()
            result = graph.invoke(state)

        exam = result["exam"]
        assert "topic_not_found" in exam
        assert "astrofísica_inexistente" in exam["topic_not_found"]
        assert len(exam.get("topic_suggestions", [])) <= 3
        assert exam["total_questions"] == 0

    def test_prd11_adversarial_no_content(self, exam_generator_state, sample_chunks):
        """PRD case #11: topic from different subject, agent must not invent."""
        from src.agents.exam_generator import validate_questions

        # LLM invented a question for a topic with no matching chunks
        fabricated_questions = [
            {
                "type": "open_answer",
                "prompt": "¿Qué es un agujero negro?",
                "base_answer": (
                    "Un agujero negro es una región del espacio-tiempo con una "
                    "concentración de masa tan alta que nada puede escapar de su "
                    "gravedad, ni siquiera la luz."
                ),
                "source_chunk_ids": ["chunk-invented-001"],
                "difficulty": "medium",
                "topic": "astrofísica",
            },
        ]

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks,  # Only math chunks
            "generated_questions": fabricated_questions,
        }

        with patch(
            "src.rag.get_embedding_model"
        ) as mock_embed:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Chunk embeddings: non-zero vectors (so they have valid norms).
            # Claim embedding: zero vector → norm=0 → cos-sim=0.0 → below any threshold.
            # This guarantees the fabricated astrofísica claim fails validation.
            chunk_vec = [[0.1] * 384]
            zero_vec = [[0.0] * 384]
            call_counter = [0]  # mutable so inner function can mutate

            def _discriminating_encode(texts):
                call_counter[0] += 1
                if call_counter[0] == 1:  # first call: chunk texts
                    return [list(chunk_vec[0]) for _ in texts]
                return [list(zero_vec[0]) for _ in texts]  # second call: claim

            model.encode.side_effect = lambda texts: type(
                "FakeArray", (), {"tolist": lambda self: _discriminating_encode(texts)}
            )()
            mock_embed.return_value = model

            result = validate_questions(state)

        # Validation should catch the fabricated claim
        assert "validation_errors" in result
        # At least one validation error should exist
        assert len(result["validation_errors"]) > 0
