"""Shared test fixtures for the Ingestor + RAG + ExamGenerator test suite."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load .env before any test runs — ChatGroq reads GROQ_API_KEY from os.environ
from dotenv import load_dotenv as _load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    _load_dotenv(_env_path)


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
    with patch("langchain_groq.ChatGroq") as mock:
        # Create a mock classification result
        mock_result = MagicMock()
        mock_result.classification = "apunte_teorico"
        mock_result.confidence = 0.95
        mock_result.topics = ["álgebra", "vectores", "matrices", "espacio vectorial"]

        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_result

        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured
        mock.return_value = mock_instance
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
    """Mock embedding model that returns deterministic embeddings."""
    with patch("src.rag.get_embedding_model") as mock_get:
        mock_model = MagicMock()
        # Return deterministic 384-dim embeddings scaled by hash of input
        mock_model.encode.return_value.tolist.return_value = [
            [0.1 * (i + 1 + (hash(chunk) % 10) * 0.01) for i in range(384)] for chunk in []
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Make encode return proper numpy-like list for each call
        def _fake_encode(texts):
            return [[0.1 * (i + 1 + (hash(t) % 10) * 0.01) for i in range(384)] for t in texts]

        mock_model.encode.side_effect = lambda texts: type(
            "FakeArray", (), {"tolist": lambda self: _fake_encode(texts)}
        )()
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

    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_exam
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured
        mock_groq.return_value = mock_instance
        yield mock_groq


# ── Real-model integration fixtures (opt-in via `-m integration`) ─────────────


@pytest.fixture
def requires_groq():
    """Skip integration tests when GROQ_API_KEY is not configured."""
    from src.config import settings

    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY not set — real LLM integration test skipped")


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

    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = fake_exercise
        mock_instance = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured
        mock_groq.return_value = mock_instance
        yield mock_groq


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
