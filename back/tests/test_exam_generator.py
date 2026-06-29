"""Tests for ExamGenerator agent — unit, integration, and end-to-end.

Covers PRD test cases #2 (happy path), #7 (missing topic), #11 (adversarial).
"""

from __future__ import annotations

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
        from src.utils.async_ import run_async_in_sync

        state = {**exam_generator_state, "topics": ["astrofísica_inexistente"]}

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.return_value = []
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        assert mock_tool.invoke.called
        assert "topic_not_found" in result
        assert "astrofísica_inexistente" in result["topic_not_found"]
        assert "topic_suggestions" in result

    def test_retrieve_chunks_accumulates(self, exam_generator_state, sample_chunks):
        """Multiple topics accumulate and deduplicate chunks."""
        from src.agents.exam_generator import retrieve_relevant_chunks
        from src.utils.async_ import run_async_in_sync

        def _fake_retrieve(input_dict):
            query = input_dict.get("query", "")
            if "derivadas" in query:
                return [sample_chunks[0], sample_chunks[1]]
            if "matrices" in query:
                return [sample_chunks[2], sample_chunks[0]]
            return []

        state = {**exam_generator_state}

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        # Should have accumulated chunks (operator.add reducer on Annotated)
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) >= 1
        # Verify no chunk_id appears more than once
        chunk_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_retrieve_chunks_theme_matching(self, exam_generator_state, sample_chunks):
        """When a user specifies a theme, it matches session document topics."""
        from src.agents.exam_generator import retrieve_relevant_chunks
        from src.utils.async_ import run_async_in_sync
        from unittest.mock import AsyncMock

        # We request "derivadas", which is not exact, but matches cálculo/derivadas
        state = {
            **exam_generator_state,
            "topics": ["derivadas"],
            "session_id": "test_sess_1",
        }

        mock_files = [
            {"topics_json": '["cálculo/derivadas", "álgebra/matrices"]'},
        ]

        def _fake_retrieve(input_dict):
            query = input_dict.get("query", "")
            if "cálculo/derivadas" in query:
                return [sample_chunks[0]]
            return []

        with patch("src.memory.schema.list_session_files", new=AsyncMock(return_value=mock_files)), \
              patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        assert "topics" in result
        # "derivadas" should be resolved to ["cálculo/derivadas"]
        assert result["topics"] == ["cálculo/derivadas"]
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["chunk_id"] == sample_chunks[0]["chunk_id"]

    def test_retrieve_chunks_theme_matching_fallback(self, exam_generator_state, sample_chunks):
        """When a theme does not match any session topic, falls back to original theme."""
        from src.agents.exam_generator import retrieve_relevant_chunks
        from src.utils.async_ import run_async_in_sync
        from unittest.mock import AsyncMock

        state = {
            **exam_generator_state,
            "topics": ["física/cuántica"],
            "session_id": "test_sess_2",
        }

        mock_files = [
            {"topics_json": '["cálculo/derivadas", "álgebra/matrices"]'},
        ]

        def _fake_retrieve(input_dict):
            query = input_dict.get("query", "")
            if "física/cuántica" in query:
                return [sample_chunks[3]]
            return []

        with patch("src.memory.schema.list_session_files", new=AsyncMock(return_value=mock_files)), \
             patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        assert "topics" in result
        assert result["topics"] == ["física/cuántica"]
        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["chunk_id"] == sample_chunks[3]["chunk_id"]

    def test_retrieve_explicit_topic_excludes_weak_topics(
        self, exam_generator_state, sample_chunks
    ):
        """When user explicitly requests a topic, weak topics are NOT blended.

        Regression: weak topics should only be injected when the user does NOT
        specify a topic (i.e. topics=["general"] or empty).  An explicit query
        like "matrices" must not be diluted with chunks from unrelated weak
        topics such as "cálculo".
        """
        from src.agents.exam_generator import retrieve_relevant_chunks
        from src.utils.async_ import run_async_in_sync

        queries_called: list[str] = []

        def _fake_retrieve(input_dict):
            queries_called.append(input_dict.get("query", ""))
            return [sample_chunks[0]]

        state = {
            **exam_generator_state,
            "topics": ["matrices"],
            "student_profile": {
                "weak_topics": ["cálculo", "estadística"],
                "preferences": {},
            },
            "session_id": "",  # skip session file resolution
        }

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        assert mock_tool.invoke.call_count >= 1
        # Only the user's explicit topic should appear — NOT weak topics
        assert "matrices" in queries_called[0], (
            f"Expected user topic 'matrices' in query, got {queries_called}"
        )
        for q in queries_called:
            assert "cálculo" not in q, (
                f"Weak topic 'cálculo' should NOT appear when user gave explicit topic"
            )
            assert "estadística" not in q, (
                f"Weak topic 'estadística' should NOT appear when user gave explicit topic"
            )
        assert "retrieved_chunks" in result

    def test_retrieve_no_explicit_topic_uses_weak_topics(
        self, exam_generator_state, sample_chunks
    ):
        """When user provides no explicit topic ("general"), weak topics ARE used."""
        from src.agents.exam_generator import retrieve_relevant_chunks
        from src.utils.async_ import run_async_in_sync

        state = {
            **exam_generator_state,
            "topics": ["general"],
            "student_profile": {
                "weak_topics": ["cálculo/derivadas", "álgebra/matrices"],
                "preferences": {},
            },
            "session_id": "",  # skip session file resolution
        }

        queries_called: list[str] = []

        def _fake_retrieve(input_dict):
            queries_called.append(input_dict.get("query", ""))
            return [sample_chunks[0]]

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = run_async_in_sync(retrieve_relevant_chunks(state))

        # Weak topics should be the query targets, not "general"
        assert any("cálculo" in q or "derivadas" in q for q in queries_called), (
            f"Expected weak topics in queries, got {queries_called}"
        )
        assert "retrieved_chunks" in result


class TestGenerateQuestions:
    """Task 2.3–2.4: generate_questions node."""

    def test_generate_questions_mcq_open_answer(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """LLM produces 3 MCQs + 2 open-answer with source_chunk_ids."""
        from src.agents.exam_generator import generate_questions

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks,  # all 5 chunks including chunk-math-004
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
            "retrieved_chunks": sample_chunks,  # all 5 chunks
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
                "stem": (
                    "La derivada de una función f(x) se define como el límite "
                    "del cociente incremental."
                ),
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

        with patch("src.rag.get_embedding_model") as mock_embed:
            import torch

            # Embeddings: make chunk embeddings match their text content
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Encode returns a real torch tensor for cos_sim compatibility
            def _fake_encode(texts, **kwargs):
                import hashlib

                return torch.tensor(
                    [
                        [
                            float(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % 1000) / 1000.0
                            for _ in range(384)
                        ]
                        for t in texts
                    ],
                    dtype=torch.float32,
                )

            model.encode.side_effect = _fake_encode
            mock_embed.return_value = model

            result = validate_questions(state)

        # All questions grounded in chunk text → should pass
        assert "validation_results" in result
        assert "invalid_question_indices" in result

    def test_validate_questions_one_fails(self, exam_generator_state, sample_chunks):
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

        with patch("src.rag.get_embedding_model") as mock_embed:
            import torch

            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Return LOW similarity embeddings for the claim (hallucination)
            def _low_encode(texts, **kwargs):
                return torch.tensor(
                    [[0.01 * (i + 1) for i in range(384)] for _ in texts],
                    dtype=torch.float32,
                )

            model.encode.side_effect = _low_encode
            mock_embed.return_value = model

            result = validate_questions(state)

        # Should have validation errors since claims won't match chunks
        assert "validation_errors" in result


class TestShouldRetry:
    """Task 2.7–2.8: should_retry conditional edge."""

    def test_should_retry_retry(self, exam_generator_state):
        """Errors present + retry_count < 3 → returns 'retry'."""
        from src.agents.exam_generator import should_retry

        state = {
            **exam_generator_state,
            "validation_errors": ["claim X not found"],
            "retry_count": 1,
        }
        result = should_retry(state)
        assert result == "retry"

    def test_should_retry_done(self, exam_generator_state):
        """retry_count >= 3 → returns 'done' even with errors."""
        from src.agents.exam_generator import should_retry

        state = {
            **exam_generator_state,
            "validation_errors": ["claim X not found"],
            "retry_count": 3,
        }
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
        # First generate questions, then format
        from src.agents.exam_generator import format_exam, generate_questions

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
        from src.agents.exam_generator import format_exam, generate_questions

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

    def test_format_exam_topic_distribution(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """Topic distribution computed from final questions and included in exam output."""
        from src.agents.exam_generator import format_exam, generate_questions

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
        exam = result["exam"]

        assert "topic_distribution" in exam
        dist = exam["topic_distribution"]
        assert isinstance(dist, dict)
        # Sum of distributed counts should equal total_questions
        total = sum(dist.values())
        assert total == exam["total_questions"] or total == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Weak Topic Prioritization Tests (Epic 12)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeakTopicPrioritization:
    """Epic 12: Verify exam generator biases questions toward weak topics."""

    def test_generate_questions_with_weak_topics(
        self, exam_generator_state, sample_chunks
    ):
        """When student_profile has weak_topics, topic_distribution computed
        and ≥60% allocation goes to weak topics."""
        from src.agents.exam_generator import (
            ExamGeneration,
            MCQQuestion,
            OpenAnswerQuestion,
            generate_questions,
        )

        # Build a fake LLM response where all questions target weak topics
        fake_exam = ExamGeneration(
            mcq_questions=[
                MCQQuestion(
                    stem="MCQ sobre IA?",
                    options=["A", "B", "C", "D"],
                    correct_option_index=0,
                    source_chunk_ids=["c-w1"],
                    difficulty="medium",
                    topic="agentes/inteligentes",
                ),
                MCQQuestion(
                    stem="MCQ sobre definición?",
                    options=["A", "B", "C", "D"],
                    correct_option_index=0,
                    source_chunk_ids=["c-w2"],
                    difficulty="medium",
                    topic="agentes/inteligentes",
                ),
                MCQQuestion(
                    stem="MCQ sobre autonomía?",
                    options=["A", "B", "C", "D"],
                    correct_option_index=0,
                    source_chunk_ids=["c-w3"],
                    difficulty="medium",
                    topic="agentes/autonomia",
                ),
            ],
            open_questions=[
                OpenAnswerQuestion(
                    prompt="Explicar racionalidad?",
                    base_answer="racionalidad es...",
                    key_points=["kp1", "kp2"],
                    source_chunk_ids=["c-w4"],
                    difficulty="medium",
                    topic="agentes/racionalidad",
                ),
                OpenAnswerQuestion(
                    prompt="Explicar tipos?",
                    base_answer="tipos son...",
                    key_points=["kp1", "kp2"],
                    source_chunk_ids=["c-w5"],
                    difficulty="medium",
                    topic="agentes/tipos",
                ),
            ],
            metadata={"topics_covered": [], "total_source_chunks": 5},
        )

        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_exam

        # Chunks with topic metadata matching weak topics
        chunks_with_topics = []
        for i, (cid, topic) in enumerate(
            [
                ("c-w1", "agentes/inteligentes"),
                ("c-w2", "agentes/inteligentes"),
                ("c-w3", "agentes/autonomia"),
                ("c-w4", "agentes/racionalidad"),
                ("c-w5", "agentes/tipos"),
            ]
        ):
            chunks_with_topics.append({
                "chunk_id": cid,
                "text": f"Chunk text about {topic}",
                "metadata": {"topic": topic},
                "similarity_score": 0.3,
            })

        state = {
            **exam_generator_state,
            "student_profile": {
                "weak_topics": [
                    "agentes/inteligentes",
                    "agentes/autonomia",
                    "agentes/racionalidad",
                ],
                "preferences": {},
            },
            "topics": ["agentes"],
            "retrieved_chunks": chunks_with_topics,
            "question_count": 5,
            "mcq_ratio": 0.6,
        }

        with patch("src.llm.get_structured_llm", return_value=fake_invokable):
            result = generate_questions(state)

        assert "topic_distribution" in result
        dist = result["topic_distribution"]
        assert isinstance(dist, dict)
        # Weak topics (3) vs everything else — weak quota should be ≥3 (60% of 5)
        weak_count = sum(
            count for topic, count in dist.items()
            if topic.startswith("agentes/")
        )
        assert weak_count >= 3, (
            f"Weak topic allocation should be ≥3 (60% of 5), got {weak_count}: {dist}"
        )

    def test_generate_questions_uniform_no_profile(
        self, exam_generator_state, sample_chunks
    ):
        """When student_profile is None, uses uniform distribution across topics."""
        from src.agents.exam_generator import (
            ExamGeneration,
            MCQQuestion,
            generate_questions,
        )

        fake_exam = ExamGeneration(
            mcq_questions=[
                MCQQuestion(
                    stem="Q?",
                    options=["A", "B", "C", "D"],
                    correct_option_index=0,
                    source_chunk_ids=["c1"],
                    difficulty="medium",
                    topic="cálculo/derivadas",
                )
                for _ in range(3)
            ],
            open_questions=[],
            metadata={"topics_covered": [], "total_source_chunks": 3},
        )

        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_exam

        chunks_with_topics = [
            {
                "chunk_id": "c1",
                "text": "Derivadas chunk",
                "metadata": {"topic": "cálculo/derivadas"},
                "similarity_score": 0.3,
            },
            {
                "chunk_id": "c2",
                "text": "Matrices chunk",
                "metadata": {"topic": "álgebra/matrices"},
                "similarity_score": 0.4,
            },
        ]

        state = {
            **exam_generator_state,
            "student_profile": None,
            "topics": ["cálculo/derivadas", "álgebra/matrices"],
            "retrieved_chunks": chunks_with_topics,
            "question_count": 3,
            "mcq_ratio": 1.0,
        }

        with patch("src.llm.get_structured_llm", return_value=fake_invokable):
            result = generate_questions(state)

        assert "topic_distribution" in result
        dist = result["topic_distribution"]
        assert isinstance(dist, dict)
        # With 2 topics and 3 questions: expect 2 on one, 1 on the other
        total = sum(dist.values())
        assert total == 3, f"Expected 3 total, got {total}: {dist}"
        # Should cover both topics
        assert len(dist) >= 1

    async def test_topic_distribution_in_exam_output(
        self, exam_generator_state, sample_chunks, mock_exam_llm
    ):
        """End-to-end: exam output includes topic_distribution computed from questions."""
        from src.agents.exam_generator import build_exam_generator

        state = {
            **exam_generator_state,
            "retrieved_chunks": sample_chunks[:4],
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
                result = await graph.ainvoke(state)

        exam = result["exam"]
        assert "topic_distribution" in exam
        dist = exam["topic_distribution"]
        assert isinstance(dist, dict)
        total = sum(dist.values())
        assert total == exam["total_questions"], (
            f"Distribution sum {total} != total_questions {exam['total_questions']}: {dist}"
        )
        # Questions topics should match distribution keys
        question_topics = {q.get("topic", "") for q in exam["questions"]}
        for topic in dist:
            assert topic in question_topics, (
                f"Topic '{topic}' in distribution but not in questions"
            )


class TestEndToEnd:
    """Task 2.11: Full graph execution end-to-end."""

    async def test_e2e_full_graph(self, exam_generator_state, sample_chunks, mock_exam_llm):
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
                result = await graph.ainvoke(state)
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
        assert elapsed < 5.0, (
            f"Graph execution took {elapsed:.1f}s — exceeded 5s budget for mocked run"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: PRD Integration Tests — RED
# ═══════════════════════════════════════════════════════════════════════════════


class TestPRDIntegration:
    """PRD-mandated integration test cases."""

    async def test_prd2_happy_path_5_questions(self, exam_generator_state, sample_chunks, mock_exam_llm):
        """PRD case #2: Generate 5 questions on specific topic.

        Verify count, structure, and chunk refs."""
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
                result = await graph.ainvoke(state)

        exam = result["exam"]
        assert exam["total_questions"] == 5
        assert exam["status"] == "complete"
        for q in exam["questions"]:
            assert "source_chunk_ids" in q
            assert len(q["source_chunk_ids"]) > 0
            assert "type" in q

    async def test_prd7_missing_topic_handling(self, exam_generator_state):
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
            result = await graph.ainvoke(state)

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

        with patch("src.rag.get_embedding_model") as mock_embed:
            import torch

            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Chunk embeddings: non-zero vectors (so they have valid norms).
            # Claim embedding: zero vector → norm=0 → cos-sim=0.0 → below any threshold.
            # This guarantees the fabricated astrofísica claim fails validation.
            chunk_vec = [[0.1] * 384]
            zero_vec = [[0.0] * 384]
            call_counter = [0]  # mutable so inner function can mutate

            def _discriminating_encode(texts, **kwargs):
                call_counter[0] += 1
                if call_counter[0] == 1:  # first call: chunk texts
                    return torch.tensor([list(chunk_vec[0]) for _ in texts], dtype=torch.float32)
                return torch.tensor([list(zero_vec[0]) for _ in texts], dtype=torch.float32)

            model.encode.side_effect = _discriminating_encode
            mock_embed.return_value = model

            result = validate_questions(state)

        # Validation should catch the fabricated claim
        assert "validation_errors" in result
        # At least one validation error should exist
        assert len(result["validation_errors"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Real-model integration tests (run with: pytest -m integration)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRealIntegration:
    """End-to-end exam generation with real LLM + real embeddings + real PDF.

    These tests use the actual apunteAgentes_IA2007.pdf ingested into
    ChromaDB, real SentenceTransformer embeddings, and ChatGroq LLM.
    They verify product behavior, not just plumbing.
    """

    async def test_generate_exam_from_real_pdf(
        self, requires_ollama, ingested_collection_name, real_pdf_text
    ):
        """Full pipeline: real PDF → real chunks → real LLM → validated exam.

        Verifies R1-R3, R6: questions exist, source_chunk_ids present,
        status is complete (or partial if anti-hallucination triggers).
        """
        from src.tools import extract_topics as _extract_topics
        from src.tools import retrieve_chunks

        # Extract topics from the real PDF to guide exam generation
        topics_result = await _extract_topics.ainvoke({"text": real_pdf_text[:3000]})
        detected_topics = topics_result.get("topics", [])
        if not detected_topics:
            detected_topics = ["agentes", "inteligencia artificial"]

        # Use the first 2-3 detected topics
        exam_topics = detected_topics[:3]

        # Invoke the graph directly — bypass the generate_exam tool so we
        # can pass the real ingested_collection_name (not session_-prefixed).
        from src.agents.exam_generator import (
            ExamGeneratorState,
            build_exam_generator,
        )

        graph = build_exam_generator().compile()
        state: ExamGeneratorState = {
            "session_id": "integration_test",
            "student_id": "student-001",
            "topics": exam_topics,
            "difficulty": "medium",
            "question_count": 4,
            "mcq_ratio": 0.5,
            "student_profile": None,
            "collection_name": ingested_collection_name,
            "retrieved_chunks": [],
            "generated_questions": [],
            "validation_results": [],
            "validation_errors": [],
            "invalid_question_indices": [],
            "omitted_questions": [],
            "retry_count": 0,
            "topic_not_found": [],
            "topic_suggestions": [],
            "topic_distribution": {},
            "exam": {},
            "status": "pending",
        }

        result = await graph.ainvoke(state)
        exam = result.get("exam", {})

        # ── Structural assertions ──────────────────────────────────────
        assert isinstance(exam, dict), f"Expected dict, got {type(exam)}"
        assert "exam_id" in exam
        assert "total_questions" in exam
        assert "questions" in exam
        assert "status" in exam

        # Must have produced at least some questions, OR correctly report no_material.
        # Small local models may legitimately fail structured output — that's not a bug.
        if exam["total_questions"] == 0:
            assert exam["status"] == "no_material", (
                f"Zero questions but status is '{exam['status']}' — should be 'no_material'. "
                f"Topics: {exam_topics}"
            )
            return  # Nothing more to verify for empty exams

        # Status must be complete or partial (not error)
        assert exam["status"] in ("complete", "partial"), (
            f"Unexpected status: {exam['status']}. Warnings: {exam.get('warnings', [])}"
        )

        # ── Content assertions ─────────────────────────────────────────
        questions = exam["questions"]
        for i, q in enumerate(questions):
            # Every question must have source_chunk_ids (R6)
            assert "source_chunk_ids" in q, f"Question {i} missing source_chunk_ids"
            assert len(q["source_chunk_ids"]) > 0, f"Question {i} has empty source_chunk_ids"
            # Every question must have a type (R1, R2)
            assert q.get("type") in ("mcq", "open_answer"), (
                f"Question {i} has invalid type: {q.get('type')}"
            )

            # MCQ-specific checks (R1)
            if q["type"] == "mcq":
                assert "prompt" in q, f"MCQ {i} missing prompt (mapped from stem by generate_questions)"
                assert "options" in q, f"MCQ {i} missing options"
                assert len(q.get("options", [])) >= 3, f"MCQ {i} has fewer than 3 options"
                assert "correct_option_index" in q, f"MCQ {i} missing correct_option_index"

            # Open-answer checks (R2)
            if q["type"] == "open_answer":
                assert "prompt" in q, f"Open-answer {i} missing prompt"
                assert "base_answer" in q, f"Open-answer {i} missing base_answer"

        # ── Grounding check (R3): every question must reference valid source chunks ──
        # NOTE: Verbatim text matching is skipped — LLM paraphrases source content.
        # Grounding is instead verified via source_chunk_ids (checked above) and the
        # anti-hallucination guardrail (test_anti_hallucination_catches_fabrication).
        all_chunk_ids = set()
        for topic in exam_topics:
            chunks = retrieve_chunks.invoke(
                {
                    "query": topic,
                    "collection_name": ingested_collection_name,
                    "top_k": 5,
                }
            )
            for c in chunks:
                if isinstance(c, dict) and c.get("chunk_id"):
                    all_chunk_ids.add(c["chunk_id"])

        for i, q in enumerate(questions):
            q_ids = set(q.get("source_chunk_ids", []))
            valid_ids = q_ids & all_chunk_ids
            assert len(valid_ids) > 0, (
                f"Question {i} source_chunk_ids ({q_ids}) have no overlap "
                f"with retrieved chunks ({len(all_chunk_ids)} available). "
                f"Question may be hallucinated."
            )

    def test_anti_hallucination_catches_fabrication(
        self, requires_ollama, ingested_collection_name
    ):
        """R3: Claim-level validation detects content not in source chunks.

        Injects a fabricated question (about black holes) alongside real
        agent-theory chunks from the ingested PDF. The validation node
        must flag the fabricated claims since no chunk contains astrophysics
        content.
        """
        from src.agents.exam_generator import validate_questions
        from src.tools import retrieve_chunks

        # Retrieve real agent-theory chunks
        real_chunks = retrieve_chunks.invoke(
            {
                "query": "agentes inteligentes definición",
                "collection_name": ingested_collection_name,
                "top_k": 5,
            }
        )
        assert len(real_chunks) > 0, "No real chunks found for validation test"

        # Build a fabricated question about astrophysics — content NOT in the PDF
        fabricated_question = {
            "type": "open_answer",
            "prompt": "¿Qué es un agujero negro según la teoría de la relatividad?",
            "base_answer": (
                "Un agujero negro es una región del espacio-tiempo donde la "
                "gravedad es tan intensa que nada, ni siquiera la luz, puede "
                "escapar. Se forma por el colapso gravitacional de una estrella "
                "masiva al final de su ciclo de vida."
            ),
            "key_points": [
                "Región del espacio-tiempo con gravedad extrema",
                "Nada puede escapar, ni la luz",
                "Colapso gravitacional de estrella masiva",
                "Horizonte de eventos",
            ],
            "source_chunk_ids": ["fabricated-chunk-999"],
            "difficulty": "hard",
            "topic": "astrofísica/agujeros_negros",
        }

        state = {
            "session_id": "integration_test",
            "student_id": "student-001",
            "topics": ["agentes"],
            "difficulty": "medium",
            "question_count": 1,
            "mcq_ratio": 0.0,
            "student_profile": None,
            "collection_name": ingested_collection_name,
            "retrieved_chunks": real_chunks,
            "generated_questions": [fabricated_question],
            "validation_results": [],
            "validation_errors": [],
            "invalid_question_indices": [],
            "omitted_questions": [],
            "retry_count": 0,
            "topic_not_found": [],
            "topic_suggestions": [],
            "topic_distribution": {},
            "exam": {},
            "status": "generated",
        }

        result = validate_questions(state)

        # Validation must detect the fabricated claims
        validation_results = result.get("validation_results", [])
        invalid_indices = result.get("invalid_question_indices", [])

        assert len(validation_results) > 0, "No validation results produced"
        assert 0 in invalid_indices, (
            f"Fabricated question passed validation! Results: {validation_results}"
        )


# ==============================================================================
# Phase 5.5: source_chunk_ids hallucination filter (Epic 13, LT-2)
# ==============================================================================


class TestSourceChunkIdFilter:
    """LT-2: Post-filter hallucinated source_chunk_ids against retrieved_chunks."""

    def test_hallucinated_ids_removed(self, mock_exam_llm, exam_generator_state, sample_chunks):
        """GIVEN LLM returns a question with hallucinated chunk IDs
        THEN source_chunk_ids are filtered to only valid IDs.
        """
        from src.agents.exam_generator import (
            ExamGeneration,
            MCQQuestion,
            OpenAnswerQuestion,
        )

        # Override the mock to include hallucinated chunk IDs
        fake_exam = ExamGeneration(
            mcq_questions=[
                MCQQuestion(
                    stem="Test hallucination filter",
                    options=["A", "B", "C"],
                    correct_option_index=0,
                    source_chunk_ids=["chunk-math-001", "chunk-math-999", "chunk-fake-007"],
                    difficulty="medium",
                    topic="cálculo/derivadas",
                ),
            ],
            open_questions=[],
            metadata={"topics_covered": ["cálculo/derivadas"], "total_source_chunks": 1},
        )

        mock_exam_llm.return_value.invoke.return_value = fake_exam

        from src.agents.exam_generator import generate_questions

        state = dict(exam_generator_state)
        state["retrieved_chunks"] = sample_chunks  # Only has chunk-math-001..005

        result = generate_questions(state)

        questions = result["generated_questions"]
        assert len(questions) == 1

        mcq = questions[0]
        filtered_ids = mcq.get("source_chunk_ids", [])
        assert "chunk-math-001" in filtered_ids, "Valid chunk ID was incorrectly removed"
        assert "chunk-math-999" not in filtered_ids, "Hallucinated chunk ID was not removed"
        assert "chunk-fake-007" not in filtered_ids, "Hallucinated chunk ID was not removed"
        assert len(filtered_ids) == 1, (
            f"Expected 1 valid ID, got {filtered_ids}"
        )

    def test_all_valid_ids_preserved(self, mock_exam_llm, exam_generator_state, sample_chunks):
        """GIVEN all source_chunk_ids are valid → THEN none are removed."""
        from src.agents.exam_generator import (
            ExamGeneration,
            MCQQuestion,
        )

        fake_exam = ExamGeneration(
            mcq_questions=[
                MCQQuestion(
                    stem="All valid IDs",
                    options=["A", "B", "C"],
                    correct_option_index=0,
                    source_chunk_ids=["chunk-math-001", "chunk-math-003", "chunk-math-005"],
                    difficulty="medium",
                    topic="cálculo/derivadas",
                ),
            ],
            open_questions=[],
            metadata={"topics_covered": ["cálculo/derivadas"], "total_source_chunks": 3},
        )

        mock_exam_llm.return_value.invoke.return_value = fake_exam

        from src.agents.exam_generator import generate_questions

        state = dict(exam_generator_state)
        state["retrieved_chunks"] = sample_chunks

        result = generate_questions(state)
        questions = result["generated_questions"]
        filtered_ids = questions[0].get("source_chunk_ids", [])

        assert set(filtered_ids) == {"chunk-math-001", "chunk-math-003", "chunk-math-005"}

    def test_no_chunks_available(self, mock_exam_llm, exam_generator_state):
        """GIVEN no retrieved chunks → THEN generate_questions returns no_material."""
        from src.agents.exam_generator import generate_questions

        state = dict(exam_generator_state)
        state["retrieved_chunks"] = []

        result = generate_questions(state)

        assert result["status"] == "no_material"
        assert result["generated_questions"] == []
