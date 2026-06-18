"""Tests for ExerciseGenerator agent — unit, integration, and end-to-end.

Covers PRD test cases EX-01 (happy path), EX-02 (missing topic), EX-NFR-01 (adversarial).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Unit tests — RED (written before implementation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieveRelevantChunks:
    """Task 2.1: retrieve_relevant_chunks node."""

    def test_retrieve_empty_topic_suggestions(self, exercise_generator_state):
        """When topic yields zero chunks, topic_not_found + suggestions populated."""
        from src.agents.exercise_generator import retrieve_relevant_chunks

        state = {**exercise_generator_state, "topic": "astrofísica_inexistente"}

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.return_value = []
            result = retrieve_relevant_chunks(state)

        assert mock_tool.invoke.called
        assert "topic_not_found" in result
        assert "astrofísica_inexistente" in result["topic_not_found"]
        assert "topic_suggestions" in result

    def test_retrieve_chunks_accumulate_deduplicate(self, exercise_generator_state, sample_chunks):
        """Chunks accumulate and deduplicate by chunk_id."""
        from src.agents.exercise_generator import retrieve_relevant_chunks

        def _fake_retrieve(input_dict):
            query = input_dict.get("query", "")
            if "derivadas" in query:
                return [sample_chunks[0], sample_chunks[1]]
            return []

        state = {**exercise_generator_state}

        with patch("src.tools.retrieve_chunks") as mock_tool:
            mock_tool.invoke.side_effect = _fake_retrieve
            result = retrieve_relevant_chunks(state)

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) >= 1
        # Verify no chunk_id appears more than once
        chunk_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestGenerateExercise:
    """Task 2.2: generate_exercise node."""

    def test_generate_exercise_happy_path(
        self, exercise_generator_state, sample_chunks, mock_exercise_llm
    ):
        """LLM produces a PracticalExercise with 3-step model solution."""
        from src.agents.exercise_generator import generate_exercise

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
            "topic": "cálculo/derivadas",
            "difficulty": "medium",
        }

        result = generate_exercise(state)

        assert "generated_exercise" in result
        exercise = result["generated_exercise"]
        assert isinstance(exercise, dict)
        assert "statement" in exercise
        assert "given_data" in exercise
        assert "question" in exercise
        assert "source_chunk_ids" in exercise
        assert len(exercise["source_chunk_ids"]) > 0
        # Must have model_solution with steps
        solution = exercise.get("model_solution", {})
        assert "steps" in solution
        assert len(solution["steps"]) >= 2
        assert "final_answer" in solution
        assert result.get("status") == "generated"

    def test_generate_exercise_retry_merges_with_errors(
        self, exercise_generator_state, sample_chunks, mock_exercise_llm
    ):
        """On retry, regenerated exercise replaces previous, retry_count increments."""
        from src.agents.exercise_generator import generate_exercise

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
            "retry_count": 1,
            "validation_errors": ["exercise claim 'agujero negro' not found in source chunks"],
            "generated_exercise": {
                "statement": "Fabricated: black hole dynamics",
                "source_chunk_ids": [],
            },
        }

        result = generate_exercise(state)

        assert "generated_exercise" in result
        exercise = result["generated_exercise"]
        # Should have been replaced with the mocked valid exercise
        assert "statement" in exercise
        assert "model_solution" in exercise
        # retry_count should increment
        assert result["retry_count"] > state["retry_count"]


class TestValidateExercise:
    """Task 2.3: validate_exercise node."""

    def test_validate_all_claims_matched(self, exercise_generator_state, sample_chunks):
        """Claims extracted from exercise grounded in chunks pass validation.

        Builds an exercise whose content mirrors chunk text literally so that
        MD5-based fake embeddings match exactly.
        """
        from src.agents.exercise_generator import validate_exercise

        # Build exercise from chunk text directly
        chunk_001 = sample_chunks[0]["text"]
        chunk_002 = sample_chunks[1]["text"]

        # Extract usable sentences from chunk text
        statement_sentence = chunk_001.split(". ")[0] + "."
        given_sentence = "f'(a) = lim(h→0) [f(a+h) - f(a)] / h."
        question_sentence = "¿Qué representa la derivada de una función en un punto?"

        exercise = {
            "statement": statement_sentence,
            "given_data": given_sentence,
            "question": question_sentence,
            "source_chunk_ids": ["chunk-math-001", "chunk-math-002"],
            "difficulty": "medium",
            "topic": "cálculo/derivadas",
            "model_solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "description": chunk_001[:90] + ".",
                        "result": "La pendiente de la recta tangente a la curva en ese punto.",
                        "source_chunk_ids": ["chunk-math-001"],
                    },
                    {
                        "step_number": 2,
                        "description": chunk_002[:90] + ".",
                        "result": "Las reglas de derivación permiten calcular derivadas rápidamente.",
                        "source_chunk_ids": ["chunk-math-002"],
                    },
                ],
                "final_answer": "La derivada representa la pendiente de la recta tangente.",
                "key_concepts": ["límite del cociente incremental"],
                "source_chunk_ids": ["chunk-math-001", "chunk-math-002"],
            },
        }

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
            "generated_exercise": exercise,
        }

        with patch("src.rag.get_embedding_model") as mock_embed:
            import hashlib

            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            def _fake_encode(texts):
                return [
                    [
                        float(int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % 1000) / 1000.0
                        for _ in range(384)
                    ]
                    for t in texts
                ]

            model.encode.side_effect = lambda texts: type(
                "FakeArray", (), {"tolist": lambda self: _fake_encode(texts)}
            )()
            mock_embed.return_value = model

            result = validate_exercise(state)

        # Claims grounded in chunks → should pass
        assert "validation_passed" in result
        assert result.get("validation_passed", False) is True
        assert len(result.get("validation_errors", [])) == 0

    def test_validate_one_claim_fails(self, exercise_generator_state, sample_chunks):
        """A fabricated exercise with claims not in chunks produces validation_errors."""
        from src.agents.exercise_generator import validate_exercise

        fabricated_exercise = {
            "statement": "La relatividad general describe los agujeros negros.",
            "given_data": "Masa solar M = 10^30 kg",
            "question": "Calculá el radio de Schwarzschild del Sol.",
            "source_chunk_ids": ["chunk-fake-999"],
            "difficulty": "hard",
            "topic": "astrofísica",
            "model_solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Aplicar fórmula de Schwarzschild R = 2GM/c²",
                        "result": "R = 2.95 km",
                        "source_chunk_ids": ["chunk-fake-999"],
                    },
                ],
                "final_answer": "El radio de Schwarzschild del Sol es 2.95 km.",
                "key_concepts": ["radio de Schwarzschild"],
                "source_chunk_ids": ["chunk-fake-999"],
            },
        }

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks,  # Only math chunks
            "generated_exercise": fabricated_exercise,
        }

        with patch("src.rag.get_embedding_model") as mock_embed:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            # Chunk embeddings: non-zero. Claim embeddings: zero vector → cos-sim=0
            chunk_vec = [[0.1] * 384]
            zero_vec = [[0.0] * 384]
            call_counter = [0]

            def _discriminating_encode(texts):
                call_counter[0] += 1
                if call_counter[0] == 1:  # first call: chunk texts
                    return [list(chunk_vec[0]) for _ in texts]
                return [list(zero_vec[0]) for _ in texts]  # second call: claims

            model.encode.side_effect = lambda texts: type(
                "FakeArray",
                (),
                {"tolist": lambda self: _discriminating_encode(texts)},
            )()
            mock_embed.return_value = model

            result = validate_exercise(state)

        # Validation should catch fabricated claims
        assert "validation_errors" in result
        assert len(result["validation_errors"]) > 0
        assert result.get("validation_passed", True) is False


class TestShouldRetry:
    """Task 2.4: should_retry conditional edge."""

    def test_should_retry_retry(self, exercise_generator_state):
        """Errors present + retry_count < 3 → returns 'retry'."""
        from src.agents.exercise_generator import should_retry

        state = {
            **exercise_generator_state,
            "validation_errors": ["claim X not found in source chunks"],
            "retry_count": 1,
        }
        result = should_retry(state)
        assert result == "retry"

    def test_should_retry_done(self, exercise_generator_state):
        """retry_count >= 3 → returns 'done' even with errors."""
        from src.agents.exercise_generator import should_retry

        state = {
            **exercise_generator_state,
            "validation_errors": ["claim X not found"],
            "retry_count": 3,
        }
        result = should_retry(state)
        assert result == "done"

    def test_should_retry_no_errors(self, exercise_generator_state):
        """No validation errors → returns 'done'."""
        from src.agents.exercise_generator import should_retry

        state = {**exercise_generator_state, "validation_errors": [], "retry_count": 0}
        result = should_retry(state)
        assert result == "done"

    def test_should_retry_terminal_no_material(self, exercise_generator_state):
        """Status no_material → returns 'done' even with errors."""
        from src.agents.exercise_generator import should_retry

        state = {
            **exercise_generator_state,
            "validation_errors": ["claim X not found"],
            "retry_count": 0,
            "status": "no_material",
        }
        result = should_retry(state)
        assert result == "done"

    def test_should_retry_terminal_error(self, exercise_generator_state):
        """Status error → returns 'done'."""
        from src.agents.exercise_generator import should_retry

        state = {
            **exercise_generator_state,
            "validation_errors": ["retrieval failed"],
            "retry_count": 0,
            "status": "error",
        }
        result = should_retry(state)
        assert result == "done"


class TestFormatExercise:
    """Task 2.5: format_exercise node."""

    def test_format_exercise_complete(
        self, exercise_generator_state, sample_chunks, mock_exercise_llm
    ):
        """Complete exercise with all valid → status complete, exercise_id UUID4, generated_at ISO."""
        from src.agents.exercise_generator import format_exercise, generate_exercise

        gen_state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
        }
        gen_result = generate_exercise(gen_state)

        state = {
            **exercise_generator_state,
            "generated_exercise": gen_result["generated_exercise"],
            "validation_passed": True,
            "validation_errors": [],
            "topic_not_found": [],
            "topic_suggestions": [],
            "retry_count": 1,
        }

        result = format_exercise(state)
        assert "exercise" in result
        exercise = result["exercise"]
        assert "exercise_id" in exercise
        assert "session_id" in exercise
        assert "generated_at" in exercise
        assert "statement" in exercise
        assert exercise["status"] == "complete"

    def test_format_exercise_partial(
        self, exercise_generator_state, sample_chunks, mock_exercise_llm
    ):
        """Partial exercise with validation errors → status partial + warnings."""
        from src.agents.exercise_generator import format_exercise, generate_exercise

        gen_state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
        }
        gen_result = generate_exercise(gen_state)

        state = {
            **exercise_generator_state,
            "generated_exercise": gen_result["generated_exercise"],
            "validation_passed": False,
            "validation_errors": ["exercise claim 'fabricated' not found in source chunks"],
            "topic_not_found": ["astrofísica"],
            "topic_suggestions": ["cálculo/derivadas", "álgebra/matrices"],
            "retry_count": 3,
        }

        result = format_exercise(state)
        exercise = result["exercise"]
        assert exercise.get("status") == "partial"
        assert "warnings" in exercise
        assert len(exercise["warnings"]) > 0
        assert "topic_not_found" in exercise
        assert len(exercise["topic_not_found"]) > 0
        assert "topic_suggestions" in exercise
        assert len(exercise["topic_suggestions"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: PRD Integration Tests — RED
# ═══════════════════════════════════════════════════════════════════════════════


class TestPRDIntegration:
    """PRD-mandated integration test cases for ExerciseGenerator."""

    def test_ex01_happy_path(self, exercise_generator_state, sample_chunks, mock_exercise_llm):
        """PRD EX-01: Generate complex exercise with ≥2 solution steps."""
        from src.agents.exercise_generator import build_exercise_generator

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
        }

        with patch(
            "src.agents.exercise_generator.retrieve_relevant_chunks",
            return_value={
                "retrieved_chunks": sample_chunks[:2],
                "status": "retrieved",
            },
        ):
            with patch(
                "src.agents.exercise_generator.validate_exercise",
                return_value={
                    "validation_passed": True,
                    "validation_errors": [],
                },
            ):
                graph = build_exercise_generator().compile()
                result = graph.invoke(state)

        exercise = result["exercise"]
        assert exercise["status"] == "complete"
        # Verify model solution has ≥2 steps
        solution = exercise.get("model_solution", {})
        assert "steps" in solution
        assert len(solution["steps"]) >= 2
        assert "final_answer" in solution
        # Verify exercise has content fields
        for field in ("statement", "given_data", "question"):
            assert field in exercise
            assert len(exercise[field]) > 0
        # Verify source grounding
        assert "source_chunk_ids" in exercise
        assert len(exercise["source_chunk_ids"]) > 0

    def test_ex02_missing_topic(self, exercise_generator_state):
        """PRD EX-02: topic not in material → no_material + ≤3 suggestions."""
        from src.agents.exercise_generator import build_exercise_generator

        state = {
            **exercise_generator_state,
            "topic": "astrofísica_inexistente",
        }

        with patch(
            "src.agents.exercise_generator.retrieve_relevant_chunks",
            return_value={
                "retrieved_chunks": [],
                "topic_not_found": ["astrofísica_inexistente"],
                "topic_suggestions": ["cálculo", "álgebra", "física"],
                "status": "no_material",
            },
        ):
            graph = build_exercise_generator().compile()
            result = graph.invoke(state)

        exercise = result["exercise"]
        assert "topic_not_found" in exercise
        assert "astrofísica_inexistente" in exercise["topic_not_found"]
        assert len(exercise.get("topic_suggestions", [])) <= 3
        assert exercise["status"] == "no_material"

    def test_exnfr_adversarial(self, exercise_generator_state, sample_chunks):
        """PRD EX-NFR-01: fabricated astrophysics exercise caught by validation, status partial."""
        from src.agents.exercise_generator import build_exercise_generator

        fabricated_ex = {
            "statement": "La relatividad general describe los agujeros negros.",
            "given_data": "M = 10^30 kg",
            "question": "Calculá el radio de Schwarzschild.",
            "difficulty": "hard",
            "topic": "astrofísica",
            "source_chunk_ids": ["chunk-invented-001"],
            "model_solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Fórmula R = 2GM/c²",
                        "result": "R ≈ 3 km",
                        "source_chunk_ids": ["chunk-invented-001"],
                    },
                ],
                "final_answer": "El radio de Schwarzschild es ~3 km.",
                "key_concepts": ["agujero negro"],
                "source_chunk_ids": ["chunk-invented-001"],
            },
        }

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks,
            "generated_exercise": fabricated_ex,
            "status": "generated",
        }

        with patch("src.rag.get_embedding_model") as mock_embed:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 384

            chunk_vec = [[0.1] * 384]
            zero_vec = [[0.0] * 384]
            call_counter = [0]

            def _discriminating_encode(texts):
                call_counter[0] += 1
                if call_counter[0] == 1:
                    return [list(chunk_vec[0]) for _ in texts]
                return [list(zero_vec[0]) for _ in texts]

            model.encode.side_effect = lambda texts: type(
                "FakeArray",
                (),
                {"tolist": lambda self: _discriminating_encode(texts)},
            )()
            mock_embed.return_value = model

            graph = build_exercise_generator().compile()
            result = graph.invoke(state)

        exercise = result["exercise"]
        # Should be partial because validation caught fabrication
        assert exercise["status"] in ("partial", "error", "no_material")
        # Warnings must exist
        assert "warnings" in exercise


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: End-to-End Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Task 4.3: Full graph execution end-to-end."""

    def test_e2e_full_graph(self, exercise_generator_state, sample_chunks, mock_exercise_llm):
        """Compile + invoke with mocked LLM → verify exercise output structure.

        With mocked LLM, execution should be near-instant (<5s). This asserts
        the budget exists — real LLM latency is tested in integration environment.
        """
        import time

        from src.agents.exercise_generator import build_exercise_generator

        state = {
            **exercise_generator_state,
            "retrieved_chunks": sample_chunks[:2],
        }

        t_start = time.perf_counter()
        with patch(
            "src.agents.exercise_generator.retrieve_relevant_chunks",
            return_value={"retrieved_chunks": sample_chunks[:2], "status": "retrieved"},
        ):
            with patch(
                "src.agents.exercise_generator.validate_exercise",
                return_value={
                    "validation_passed": True,
                    "validation_errors": [],
                },
            ):
                graph = build_exercise_generator().compile()
                result = graph.invoke(state)
        elapsed = time.perf_counter() - t_start

        assert "exercise" in result
        exercise = result["exercise"]
        assert isinstance(exercise, dict)
        assert "exercise_id" in exercise
        assert "session_id" in exercise
        assert "generated_at" in exercise
        assert "topic" in exercise
        assert "difficulty" in exercise
        assert "exercise_type" in exercise
        assert "statement" in exercise
        assert "given_data" in exercise
        assert "question" in exercise
        assert "model_solution" in exercise
        solution = exercise["model_solution"]
        assert "steps" in solution
        assert len(solution["steps"]) >= 2
        assert "final_answer" in solution
        assert "key_concepts" in solution
        assert "status" in exercise
        assert exercise["status"] in ("complete", "partial", "no_material")

        # E2E with mocks must complete under 5 seconds
        assert elapsed < 5.0, (
            f"Graph execution took {elapsed:.1f}s — exceeded 5s budget for mocked run"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Real-model integration tests (run with: pytest -m integration)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRealIntegration:
    """End-to-end exercise generation with real LLM + real embeddings + real PDF.

    These tests use the actual apunteAgentes_IA2007.pdf ingested into
    ChromaDB, real SentenceTransformer embeddings, and ChatGroq LLM.
    """

    def test_generate_exercise_from_real_pdf(
        self, requires_ollama, ingested_collection_name, real_pdf_text
    ):
        """Full pipeline: real PDF → real chunks → real LLM → validated exercise.

        Verifies exercise structure, model_solution steps ≥ 2, source_chunk_ids
        non-empty, and status is complete (or partial if anti-hallucination triggers).
        """
        from src.agents.exercise_generator import (
            ExerciseGeneratorState,
            build_exercise_generator,
        )

        # Use a topic likely present in the PDF
        topic = "agentes inteligentes"

        graph = build_exercise_generator().compile()
        state: ExerciseGeneratorState = {
            "session_id": "integration_test",
            "student_id": "student-001",
            "topic": topic,
            "difficulty": "medium",
            "exercise_type": "problem_solving",
            "collection_name": ingested_collection_name,
            "student_profile": None,
            "retrieved_chunks": [],
            "generated_exercise": {},
            "validation_passed": False,
            "validation_errors": [],
            "retry_count": 0,
            "topic_not_found": [],
            "topic_suggestions": [],
            "exercise": {},
            "status": "pending",
        }

        result = graph.invoke(state)
        exercise = result.get("exercise", {})

        # ── Structural assertions ──────────────────────────────────────
        assert isinstance(exercise, dict), f"Expected dict, got {type(exercise)}"
        assert "exercise_id" in exercise
        assert "topic" in exercise
        assert "status" in exercise

        # Must have produced an exercise OR correctly report no_material
        if exercise["status"] == "no_material":
            return  # Nothing more to verify for empty exercises

        # Status must be complete or partial (not error)
        assert exercise["status"] in ("complete", "partial"), (
            f"Unexpected status: {exercise['status']}. Warnings: {exercise.get('warnings', [])}"
        )

        # ── Content assertions ─────────────────────────────────────────
        assert "statement" in exercise, "Exercise missing statement"
        assert len(exercise["statement"]) > 0
        assert "given_data" in exercise, "Exercise missing given_data"
        assert "question" in exercise, "Exercise missing question"
        assert "model_solution" in exercise, "Exercise missing model_solution"

        solution = exercise["model_solution"]
        assert "steps" in solution, "Model solution missing steps"
        assert len(solution["steps"]) >= 2, (
            f"Expected ≥2 solution steps, got {len(solution['steps'])}"
        )
        assert "final_answer" in solution, "Model solution missing final_answer"
        assert "key_concepts" in solution, "Model solution missing key_concepts"

        # Source grounding
        assert "source_chunk_ids" in exercise
        assert len(exercise["source_chunk_ids"]) > 0, (
            "Exercise has empty source_chunk_ids — not grounded"
        )

    def test_anti_hallucination_catches_fabrication(self, requires_ollama, ingested_collection_name):
        """EX-NFR-01: Claim-level validation detects fabricated content.

        Injects a fabricated exercise (about black holes) alongside real
        agent-theory chunks from the ingested PDF. The validate_exercise node
        must flag the fabricated claims since no chunk contains astrophysics.
        """
        from src.agents.exercise_generator import validate_exercise
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

        # Build a fabricated exercise about astrophysics
        fabricated_exercise = {
            "statement": (
                "Un agujero negro es una región del espacio-tiempo con una "
                "concentración de masa tan alta que nada puede escapar."
            ),
            "given_data": "Masa solar M = 2 × 10^30 kg, G = 6.67 × 10^-11 N·m²/kg²",
            "question": "Calculá el radio de Schwarzschild del Sol.",
            "source_chunk_ids": ["fabricated-chunk-999"],
            "difficulty": "hard",
            "topic": "astrofísica/agujeros_negros",
            "model_solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "description": "Aplicar la fórmula de Schwarzschild: R = 2GM/c²",
                        "result": "R = 2.95 km",
                        "source_chunk_ids": ["fabricated-chunk-999"],
                    },
                    {
                        "step_number": 2,
                        "description": "Verificar unidades en el sistema internacional",
                        "result": "Unidades consistentes",
                        "source_chunk_ids": ["fabricated-chunk-999"],
                    },
                ],
                "final_answer": "El radio de Schwarzschild del Sol es aproximadamente 2.95 km.",
                "key_concepts": [
                    "radio de Schwarzschild",
                    "relatividad general",
                    "agujero negro",
                ],
                "source_chunk_ids": ["fabricated-chunk-999"],
            },
        }

        state = {
            "session_id": "integration_test",
            "student_id": "student-001",
            "topic": "agentes",
            "difficulty": "medium",
            "exercise_type": "problem_solving",
            "collection_name": ingested_collection_name,
            "student_profile": None,
            "retrieved_chunks": real_chunks,
            "generated_exercise": fabricated_exercise,
            "validation_passed": False,
            "validation_errors": [],
            "retry_count": 0,
            "topic_not_found": [],
            "topic_suggestions": [],
            "exercise": {},
            "status": "generated",
        }

        result = validate_exercise(state)

        # Validation must detect the fabricated claims
        validation_passed = result.get("validation_passed", True)
        validation_errors = result.get("validation_errors", [])

        assert not validation_passed, (
            f"Fabricated exercise passed validation! Errors: {validation_errors}"
        )
        assert len(validation_errors) > 0, "No validation errors produced for fabricated exercise"
