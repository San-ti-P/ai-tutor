"""Shared test fixtures for the Ingestor + RAG test suite."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
        ocr_confidence=0.0,
        needs_ocr_confirmation=False,
        errors=[],
        status="pending",
        classification_confidence=0.0,
        ocr_expressions=[],
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
            [0.1 * (i + 1 + (hash(chunk) % 10) * 0.01) for i in range(384)]
            for chunk in []
        ]
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Make encode return proper numpy-like list for each call
        def _fake_encode(texts):
            return [
                [0.1 * (i + 1 + (hash(t) % 10) * 0.01) for i in range(384)]
                for t in texts
            ]

        mock_model.encode.side_effect = lambda texts: type(
            "FakeArray", (), {"tolist": lambda self: _fake_encode(texts)}
        )()
        mock_get.return_value = mock_model
        yield mock_get
