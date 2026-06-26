"""Shared test fixtures for the Ingestor + RAG + ExamGenerator test suite."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Load .env before any test runs
from dotenv import load_dotenv as _load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    _load_dotenv(_env_path)


def pytest_configure(config: pytest.Config) -> None:
    """Set Langfuse environment BEFORE any test creates the SDK global singleton."""
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "test")


_PROVIDER_MODULE_MAP: dict[str, str] = {
    "ollama": "langchain_ollama.ChatOllama",
    "groq": "langchain_groq.ChatGroq",
    "opencode-go": "langchain_openai.ChatOpenAI",
    "opencode-go-anthropic": "langchain_anthropic.ChatAnthropic",
    "openai": "langchain_openai.ChatOpenAI",
}


def _llm_provider_module() -> str:
    """Return the import path for the current LLM provider's chat class.

    Based on ``settings.llm_provider`` so mock patches target the right class.
    Supports ollama, groq, opencode-go, and openai providers.
    """
    from src.config import settings

    return _PROVIDER_MODULE_MAP.get(settings.llm_provider, "langchain_ollama.ChatOllama")


@contextmanager
def patch_llm(fake_return: Any):
    """Context manager that patches get_structured_llm for deterministic testing.

    Patches ``src.llm.get_structured_llm`` to return a callable that
    ignores the schema and returns *fake_return* directly. This works
    across all providers (Ollama, Groq, etc.) because it bypasses the
    entire LLM instantiation and chain assembly.
    """
    from src.llm import get_structured_llm as _original_get_structured_llm

    fake_invokable = MagicMock()
    fake_invokable.invoke.return_value = fake_return

    with patch("src.llm.get_structured_llm", return_value=fake_invokable) as mock_gs_llm:
        yield mock_gs_llm


@pytest.fixture
def temp_dir():
    """Temporary directory that cleans up after test."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_pdf(temp_dir) -> Path:
    """Generate a valid minimal PDF for testing parse_document.

    Uses reportlab to create a PDF with extractable text. Skips the test
    if reportlab is not available (the fallback minimal PDF has no text
    that pdfminer can extract).
    """
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed — required for PDF test fixtures")

    pdf_path = temp_dir / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Álgebra lineal: espacio vectorial")
    c.drawString(100, 730, "Capítulo 1: Vectores y matrices")
    c.drawString(100, 710, "1.1 Definición de vector")
    c.drawString(100, 690, "Un vector es un elemento de un espacio vectorial.")
    c.save()
    return pdf_path


@pytest.fixture
def sample_txt(temp_dir) -> Path:
    """Create a sample TXT file with academic content."""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text(
        "Álgebra lineal: espacio vectorial\n\n"
        "Capítulo 1: Vectores y matrices\n\n"
        "1.1 Definición de vector\n"
        "Un vector es un elemento de un espacio vectorial. "
        "Los vectores pueden sumarse y multiplicarse por escalares.\n\n"
        "1.2 Matrices\n"
        "Una matriz es un arreglo rectangular de números."
    )
    return txt_path


@pytest.fixture
def non_academic_txt(temp_dir) -> Path:
    """Create a non-academic text file for rejection testing."""
    txt_path = temp_dir / "random.txt"
    txt_path.write_text(
        "Breaking News: Local sports team wins championship. "
        "The weather today is sunny with a chance of rain. "
        "Celebrity gossip and entertainment news updates."
    )
    return txt_path


@pytest.fixture
def in_memory_chroma(temp_dir):
    """Ephemeral ChromaDB client for tests — no persistence."""
    import chromadb

    # Use temp directory for ephemeral ChromaDB
    client = chromadb.PersistentClient(path=str(temp_dir / "chroma_test"))
    yield client
    # Cleanup happens via temp_dir fixture


@pytest.fixture
def mock_llm_response():
    """Mock LLM that returns a valid academic classification."""
    mock_result = MagicMock()
    mock_result.classification = "apunte_teorico"
    mock_result.confidence = 0.95
    mock_result.topics = ["álgebra", "vectores", "matrices", "espacio vectorial"]

    with patch_llm(mock_result) as mock:
        yield mock


@pytest.fixture
def ingestor_state(sample_txt):
    """Base IngestorState for agent tests."""
    from src.agents.ingestor import IngestorState

    return IngestorState(
        session_id=str(uuid.uuid4()),
        file_path=str(sample_txt),
        file_type="",
        raw_text="",
        classification="",
        topics=[],
        chunks_created=0,
        errors=[],
        status="pending",
        classification_confidence=0.0,
        document_id="",
        chunk_ids=[],
    )


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model that returns deterministic embeddings as tensors."""
    import torch

    with patch("src.rag.get_embedding_model") as mock_get:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Make encode return real torch tensors for cos_sim/batch compatibility
        def _fake_encode(texts, **kwargs):
            return torch.tensor(
                [
                    [0.1 * (i + 1 + (hash(t) % 10) * 0.01) for i in range(384)]
                    for t in texts
                ],
                dtype=torch.float32,
            )

        mock_model.encode.side_effect = _fake_encode
        mock_get.return_value = mock_model
        yield mock_get


# ── ExamGenerator fixtures ───────────────────────────────────────────────────


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Return a list of realistic mock chunk dicts for exam generation testing.

    Each chunk has chunk_id, text, metadata, and similarity_score.
    Content covers álgebra and cálculo topics for validation testing.
    """
    return [
        {
            "chunk_id": "chunk-math-001",
            "text": (
                "La derivada de una función f(x) en un punto a se define como el límite "
                "del cociente incremental: f'(a) = lim(h→0) [f(a+h) - f(a)] / h. "
                "Esto representa la pendiente de la recta tangente a la curva en ese punto."
            ),
            "metadata": {"topic": "cálculo/derivadas", "source": "apunte_calculo.pdf"},
            "similarity_score": 0.12,
        },
        {
            "chunk_id": "chunk-math-002",
            "text": (
                "Reglas de derivación: derivada de una suma [f+g]' = f' + g', "
                "derivada de producto [f·g]' = f'·g + f·g', "
                "derivada de cociente [f/g]' = (f'·g - f·g') / g²."
            ),
            "metadata": {"topic": "cálculo/derivadas", "source": "apunte_calculo.pdf"},
            "similarity_score": 0.08,
        },
        {
            "chunk_id": "chunk-math-003",
            "text": (
                "Una matriz es un arreglo rectangular de números dispuestos en filas "
                "y columnas. La suma de matrices se realiza elemento a elemento y solo "
                "es posible cuando ambas matrices tienen las mismas dimensiones."
            ),
            "metadata": {"topic": "álgebra/matrices", "source": "apunte_algebra.pdf"},
            "similarity_score": 0.15,
        },
        {
            "chunk_id": "chunk-math-004",
            "text": (
                "El rango de una matriz es el número máximo de columnas linealmente "
                "independientes. Se puede calcular mediante eliminación gaussiana. "
                "Una matriz cuadrada es invertible si y solo si su rango es máximo."
            ),
            "metadata": {"topic": "álgebra/matrices", "source": "apunte_algebra.pdf"},
            "similarity_score": 0.10,
        },
        {
            "chunk_id": "chunk-math-005",
            "text": (
                "La integral definida de una función entre a y b representa el área "
                "bajo la curva. El Teorema Fundamental del Cálculo establece que la "
                "integración y la derivación son operaciones inversas."
            ),
            "metadata": {"topic": "cálculo/integrales", "source": "apunte_calculo.pdf"},
            "similarity_score": 0.14,
        },
    ]


@pytest.fixture
def sample_student_profile() -> dict:
    """Return a sample student profile with weak topics and preferences."""
    return {
        "weak_topics": ["cálculo/derivadas", "álgebra/matrices"],
        "preferences": {"difficulty": "medium", "mcq_ratio": 0.6},
    }


@pytest.fixture
def exam_generator_state(sample_chunks) -> dict:
    """Return the base state dict for ExamGenerator graph invocation."""
    return {
        "session_id": str(uuid.uuid4()),
        "student_id": "student-001",
        "topics": ["cálculo/derivadas", "álgebra/matrices"],
        "difficulty": "medium",
        "question_count": 5,
        "mcq_ratio": 0.6,
        "student_profile": None,
        "collection_name": "",
        "retrieved_chunks": [],
        "generated_questions": [],
        "validation_results": [],
        "validation_errors": [],
        "invalid_question_indices": [],
        "omitted_questions": [],
        "retry_count": 0,
        "topic_not_found": [],
        "topic_suggestions": [],
        "exam": {},
        "status": "pending",
    }


@pytest.fixture
def mock_exam_llm():
    """Patch ChatGroq to return a valid ExamGeneration structured output.

    Mocks the entire ChatGroq with_structured_output → invoke chain so
    generate_questions can run without a real LLM. The mock returns
    3 MCQs + 2 open-answer questions with source_chunk_ids.
    """
    from src.agents.exam_generator import (
        ExamGeneration,
        MCQQuestion,
        OpenAnswerQuestion,
    )

    fake_exam = ExamGeneration(
        mcq_questions=[
            MCQQuestion(
                stem="¿Cuál es la definición de derivada?",
                options=[
                    "El límite del cociente incremental",
                    "La pendiente de una recta cualquiera",
                    "El producto de dos funciones",
                    "La integral de una función",
                ],
                correct_option_index=0,
                source_chunk_ids=["chunk-math-001"],
                difficulty="medium",
                topic="cálculo/derivadas",
            ),
            MCQQuestion(
                stem="¿Cuál es la derivada de una suma de funciones?",
                options=[
                    "f' · g'",
                    "f' + g'",
                    "f' / g'",
                    "f' - g'",
                ],
                correct_option_index=1,
                source_chunk_ids=["chunk-math-002"],
                difficulty="medium",
                topic="cálculo/derivadas",
            ),
            MCQQuestion(
                stem="¿Qué es una matriz?",
                options=[
                    "Un arreglo rectangular de números",
                    "Una función continua",
                    "Un vector unitario",
                    "Una derivada parcial",
                ],
                correct_option_index=0,
                source_chunk_ids=["chunk-math-003"],
                difficulty="easy",
                topic="álgebra/matrices",
            ),
        ],
        open_questions=[
            OpenAnswerQuestion(
                prompt="Explica el concepto de derivada y su interpretación geométrica.",
                base_answer=(
                    "La derivada es el límite del cociente incremental. "
                    "Geométricamente representa la pendiente de la recta tangente."
                ),
                key_points=[
                    "Límite del cociente incremental",
                    "Pendiente de la recta tangente",
                    "Tasa de cambio instantánea",
                ],
                source_chunk_ids=["chunk-math-001"],
                difficulty="medium",
                topic="cálculo/derivadas",
            ),
            OpenAnswerQuestion(
                prompt="Describe cómo se calcula el rango de una matriz.",
                base_answer=(
                    "El rango se calcula mediante eliminación gaussiana. "
                    "Es el número de columnas linealmente independientes."
                ),
                key_points=[
                    "Columnas linealmente independientes",
                    "Eliminación gaussiana",
                    "Matriz invertible si rango máximo",
                ],
                source_chunk_ids=["chunk-math-004"],
                difficulty="hard",
                topic="álgebra/matrices",
            ),
        ],
        metadata={
            "topics_covered": ["cálculo/derivadas", "álgebra/matrices"],
            "total_source_chunks": 4,
        },
    )

    with patch("src.llm.get_structured_llm") as mock_gs_llm:
        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_exam
        mock_gs_llm.return_value = fake_invokable
        yield mock_gs_llm


@pytest.fixture
def requires_ollama():
    """Skip integration tests when Ollama is not reachable or model not pulled."""
    from langchain_ollama import ChatOllama

    from src.config import settings

    try:
        llm = ChatOllama(model=settings.ollama_model_name, base_url=settings.ollama_base_url)
        llm.invoke("ping")
    except Exception as exc:
        pytest.skip(
            f"Ollama not reachable or model "
            f"'{settings.ollama_model_name}' not available: {exc}"
        )


@pytest.fixture
def real_pdf_path() -> Path:
    """Return path to the real academic PDF for integration tests.

    Looks for apunteAgentes_IA2007.pdf in tests/fixtures/.
    Skips the test if the file is not found.
    """
    pdf = Path(__file__).resolve().parent / "fixtures" / "apunteAgentes_IA2007.pdf"
    if not pdf.exists():
        pytest.skip(f"Real PDF not found at {pdf}")
    return pdf


@pytest.fixture
def real_pdf_text(real_pdf_path: Path) -> str:
    """Parse the real academic PDF with markitdown and return raw text."""
    import markitdown

    md = markitdown.MarkItDown()
    result = md.convert(str(real_pdf_path))
    text = result.text_content
    if not text or not text.strip():
        pytest.skip("Real PDF parsed but produced no extractable text")
    return text


# ── ExerciseGenerator fixtures ──────────────────────────────────────────────


@pytest.fixture
def exercise_generator_state(sample_chunks) -> dict:
    """Return the base state dict for ExerciseGenerator graph invocation."""
    return {
        "session_id": str(uuid.uuid4()),
        "student_id": "student-001",
        "topic": "cálculo/derivadas",
        "difficulty": "medium",
        "exercise_type": "problem_solving",
        "collection_name": "",
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


@pytest.fixture
def mock_exercise_llm():
    """Patch ChatGroq to return a valid ExerciseGeneration structured output.

    Mocks the entire ChatGroq with_structured_output → invoke chain so
    generate_exercise can run without a real LLM. The mock returns
    one PracticalExercise with a 3-step ModelSolution.
    """
    from src.agents.exercise_generator import (
        ExerciseGeneration,
        ExerciseStep,
        ModelSolution,
        PracticalExercise,
    )

    fake_exercise = ExerciseGeneration(
        exercises=[
            PracticalExercise(
                statement=(
                    "Un estudiante quiere calcular la derivada de la función "
                    "f(x) = 3x² + 2x - 5 en el punto x = 2."
                ),
                given_data="f(x) = 3x² + 2x - 5, x₀ = 2",
                question="Calculá f'(2) usando la definición de derivada.",
                difficulty="medium",
                topic="cálculo/derivadas",
                source_chunk_ids=["chunk-math-001"],
                model_solution=ModelSolution(
                    steps=[
                        ExerciseStep(
                            step_number=1,
                            description="Escribir la definición de derivada como límite",
                            result="f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
                            source_chunk_ids=["chunk-math-001"],
                        ),
                        ExerciseStep(
                            step_number=2,
                            description="Evaluar f(2+h) y f(2)",
                            result="f(2+h) = 3(2+h)² + 2(2+h) - 5",
                            source_chunk_ids=["chunk-math-001", "chunk-math-002"],
                        ),
                        ExerciseStep(
                            step_number=3,
                            description="Calcular el límite y obtener f'(2)",
                            result="f'(2) = 14",
                            source_chunk_ids=["chunk-math-001", "chunk-math-002"],
                        ),
                    ],
                    final_answer="La derivada de f(x) en x=2 es 14.",
                    key_concepts=[
                        "definición de derivada como límite",
                        "cálculo de límite",
                        "evaluación de funciones",
                    ],
                    source_chunk_ids=["chunk-math-001", "chunk-math-002"],
                ),
            ),
        ],
        metadata={
            "topics_covered": ["cálculo/derivadas"],
            "total_source_chunks": 2,
        },
    )

    with patch("src.llm.get_structured_llm") as mock_gs_llm:
        fake_invokable = MagicMock()
        fake_invokable.invoke.return_value = fake_exercise
        mock_gs_llm.return_value = fake_invokable
        yield mock_gs_llm


# ── Orchestrator fixtures ───────────────────────────────────────────────────


@pytest.fixture
def orchestrator_state() -> dict:
    """Base OrchestratorState dict for agent tests."""
    return {
        "session_id": "test-session-001",
        "user_message": "Generame un examen de derivadas",
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
    }


# ── Evaluator fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def evaluator_state(sample_chunks) -> dict:
    """Return base EvaluatorState dict with a 3-answer batch for graph tests."""
    import uuid

    return {
        "session_id": str(uuid.uuid4()),
        "student_id": "student-001",
        "exam_id": "exam-test-001",
        "trace_id": str(uuid.uuid4()),
        "answers": [
            {
                "question_id": "q-001",
                "question": "¿Cuál es la definición de derivada?",
                "base_answer": (
                    "La derivada es el límite del cociente incremental: "
                    "f'(a) = lim(h→0) [f(a+h) - f(a)] / h."
                ),
                "student_answer": (
                    "La derivada es el límite del cociente incremental, "
                    "representa la pendiente de la tangente."
                ),
                "source_chunk_ids": ["chunk-math-001"],
                "topic": "cálculo/derivadas",
                "difficulty": "medium",
            },
            {
                "question_id": "q-002",
                "question": "¿Cómo se calcula el rango de una matriz?",
                "base_answer": (
                    "El rango se calcula mediante eliminación gaussiana. "
                    "Es el número de columnas linealmente independientes."
                ),
                "student_answer": (
                    "Se calcula con eliminación gaussiana y es el número "
                    "de columnas independientes."
                ),
                "source_chunk_ids": ["chunk-math-004"],
                "topic": "álgebra/matrices",
                "difficulty": "hard",
            },
            {
                "question_id": "q-003",
                "question": "¿Qué establece el Teorema Fundamental del Cálculo?",
                "base_answer": (
                    "Establece que la integración y la derivación son "
                    "operaciones inversas."
                ),
                "student_answer": "asdf jkl qwerty zxcv nm",
                "source_chunk_ids": ["chunk-math-005"],
                "topic": "cálculo/integrales",
                "difficulty": "easy",
            },
        ],
        "current_index": 0,
        "answer_text": "",
        "ocr_extracted_text": None,
        "ocr_confidence": 0.0,
        "retrieved_chunks": sample_chunks,
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


@pytest.fixture
def mock_evaluator_llm():
    """Patch LLM to return a valid SingleEvaluation (score=8).

    Mocks the entire with_structured_output → invoke chain so the
    evaluate_answer node can run without a real LLM.
    """
    from src.agents.evaluator import SingleEvaluation

    fake_eval = SingleEvaluation(
        score=8.0,
        justification=(
            "El estudiante demuestra comprensión del concepto de derivada "
            "como límite del cociente incremental y su interpretación "
            "geométrica como pendiente de la recta tangente."
        ),
        conceptual_errors=[],
        suggestions=["Profundizar en la interpretación física de la derivada."],
        is_evaluable=True,
    )

    with patch_llm(fake_eval) as mock_llm:
        yield mock_llm


@pytest.fixture
def mock_judge_llm():
    """Patch LLM to return a JudgeVerdict that agrees with primary.

    Judge score is within ±1 of primary, so requires_review stays False.
    """
    from src.agents.evaluator import JudgeVerdict

    fake_verdict = JudgeVerdict(
        score=7.5,
        agrees_with_primary=True,
        discrepancy="",
        suggested_score=None,
    )

    with patch_llm(fake_verdict) as mock_llm:
        yield mock_llm


@pytest.fixture
def ingested_collection_name(real_pdf_path: Path, temp_dir: Path) -> str:
    """Ingest the real PDF into an ephemeral ChromaDB collection.

    Uses real markitdown parsing, real SentenceTransformer embeddings,
    and real ChromaDB storage. Yields the collection name for retrieval.

    This is expensive — only use in integration tests.
    """
    import uuid

    import markitdown

    from src.rag import chunk_text, embed_and_store

    md = markitdown.MarkItDown()
    result = md.convert(str(real_pdf_path))
    text = result.text_content
    if not text or not text.strip():
        pytest.skip("Real PDF produced no extractable text for ingestion")

    chunks = chunk_text(text)
    if not chunks:
        pytest.skip("Real PDF produced zero chunks after splitting")

    session_id = str(uuid.uuid4())
    collection_name = f"integration_{session_id}"

    metadatas = [
        {
            "source_file": real_pdf_path.name,
            "chunk_index": i,
            "classification": "apunte_teorico",
        }
        for i in range(len(chunks))
    ]

    embed_and_store(
        [c.page_content for c in chunks],
        metadatas,
        collection_name,
    )

    return collection_name


# ── Observability fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_run_id() -> str:
    """Unique identifier for grouping all traces from one test run.

    Used by the Langfuse metadata injection to tag every trace with the
    same ``test_run_id``, enabling filtering in the Langfuse dashboard.
    """
    return str(uuid.uuid4())


@pytest.fixture(scope="session")
def langfuse_observe_tests() -> bool:
    """Return True when ``LANGFUSE_OBSERVE_TESTS=true`` in environment.

    When True, ``mock_langfuse`` becomes a no-op and the real Langfuse
    client is used.  Integration tests can inspect this fixture to skip
    when tracing is not requested.
    """
    return os.environ.get("LANGFUSE_OBSERVE_TESTS", "").lower() == "true"


@pytest.fixture(autouse=True)
def _inject_test_metadata_for_integration(
    request: pytest.FixtureRequest,
    langfuse_observe_tests: bool,
    test_run_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse: inject test metadata into create_trace for integration tests.

    Only activates when ``LANGFUSE_OBSERVE_TESTS=true`` AND the current
    test carries the ``@pytest.mark.integration`` marker.

    Monkeypatches ``ObservabilityManager.create_trace`` at the *class*
    level so every instance (including the ``get_tracer()`` singleton)
    benefits from the injection.  The patch wraps the method that was
    active at fixture-setup time, so other fixtures (like ``obs_manager``)
    can add their own metadata layers without conflict.
    """
    if not langfuse_observe_tests:
        return

    marker = request.node.get_closest_marker("integration")
    if marker is None:
        return

    # Skip if the test already uses obs_manager — obs_manager does its own patch
    if "obs_manager" in getattr(request.node, "fixturenames", set()):
        return

    from src.observability import ObservabilityManager

    _current = ObservabilityManager.create_trace

    def _patched_create_trace(
        self: ObservabilityManager,
        *,
        name: str,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Any:
        meta = {**(metadata or {})}
        meta.setdefault("environment", "test")
        meta.setdefault("test_run_id", test_run_id)
        meta.setdefault("source", "pytest-integration")
        meta.setdefault("test_name", request.node.name)
        return _current(self, name=name, session_id=session_id, user_id=user_id, metadata=meta)  # type: ignore[arg-type]

    monkeypatch.setattr(ObservabilityManager, "create_trace", _patched_create_trace)


@pytest.fixture
def mock_langfuse(langfuse_observe_tests: bool) -> Any:
    """Patch langfuse.Langfuse for unit tests; no-op when real tracing requested.

    When ``langfuse_observe_tests`` is True this fixture yields ``None``
    and performs no patching — the real Langfuse client (configured via
    ``.env`` keys) is allowed through.

    All Langfuse client interactions become no-op mocks so unit tests
    never touch the network.  The mock client returns a MagicMock from
    ``.trace()`` that supports chained ``.generation()`` / ``.span()`` /
    ``.score()`` / ``.end()`` methods.

    Also temporarily injects dummy keys into settings so the singleton
    client path is exercised even when the test ``.env`` is empty.
    """
    if langfuse_observe_tests:
        yield None
        return

    from src.config import settings as _settings

    with (
        patch.object(_settings, "langfuse_public_key", "pk-test-dummy", create=False),
        patch.object(_settings, "langfuse_secret_key", "sk-test-dummy", create=False),
        patch.object(_settings, "langfuse_host", "http://localhost:3000", create=False),
        patch("langfuse.Langfuse") as mock_client_cls,
        patch.dict("os.environ", {"LANGFUSE_TRACING_ENVIRONMENT": "test"}, clear=False),
    ):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_span = MagicMock()

        # v2 API (legacy, for backward compat in tests)
        mock_client.trace.return_value = mock_trace
        # v4 API (current)
        mock_client.start_observation.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation
        mock_trace.span.return_value = mock_span

        mock_client_cls.return_value = mock_client
        yield mock_client_cls


@pytest.fixture
def mock_observe():
    """Patch langfuse.observe to a transparent pass-through.

    The decorated function executes normally — no tracing intent is
    altered, but the real Langfuse decorator is never invoked.
    """
    with patch("langfuse.observe", lambda **kw: (lambda fn: fn)):
        yield


@pytest.fixture
def obs_manager(
    mock_langfuse: Any,
    langfuse_observe_tests: bool,
    test_run_id: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Return an ObservabilityManager wired to the active Langfuse client.

    When ``langfuse_observe_tests`` is False (default): the client is
    mocked via ``mock_langfuse`` and no metadata injection occurs.

    When ``langfuse_observe_tests`` is True: the real Langfuse client
    (configured via ``.env``) is used AND ``create_trace`` is monkeypatched
    at the *class* level to inject the four test metadata tags:
    ``environment``, ``test_run_id``, ``source``, ``test_name``.

    The class-level patch means even ``get_tracer().create_trace()``
    (used by production code paths) receives the injected metadata.
    """
    from src.observability import ObservabilityManager
    from src.observability._client import _reset_langfuse_client

    _reset_langfuse_client()
    mgr = ObservabilityManager()
    mgr._ensure_init()

    if langfuse_observe_tests and mgr.enabled:
        # Inject test metadata into ALL create_trace calls (class-level patch)
        _original = ObservabilityManager.create_trace

        def _patched_create_trace(
            self: ObservabilityManager,
            *,
            name: str,
            session_id: str,
            user_id: str | None = None,
            metadata: dict[str, object] | None = None,
        ) -> Any:
            meta = {**(metadata or {})}
            meta.setdefault("environment", "test")
            meta.setdefault("test_run_id", test_run_id)
            meta.setdefault("source", "pytest-integration")
            meta.setdefault("test_name", request.node.name)
            return _original(self, name=name, session_id=session_id, user_id=user_id, metadata=meta)  # type: ignore[arg-type]

        monkeypatch.setattr(ObservabilityManager, "create_trace", _patched_create_trace)

    yield mgr
