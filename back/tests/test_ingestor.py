"""Integration tests for the Ingestor agent — PRD test cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.ingestor import (
    classify_document,
    parse_document,
)

# ──────────────────────────────────────────────────────────────────────────────
# PRD Case #1 — Happy path: well-formatted PDF
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestorHappyPath:
    def test_parse_pdf_document(self, sample_pdf, ingestor_state):
        """PRD Case #1: Ingest well-formatted PDF."""
        state = dict(ingestor_state)
        state["file_path"] = str(sample_pdf)

        result = parse_document(state)
        assert result["status"] == "parsed"
        assert result["file_type"] == "pdf"
        assert len(result["raw_text"]) > 0

    def test_parse_txt_document(self, sample_txt, ingestor_state):
        """TXT parsing succeeds and extracts raw text."""
        state = dict(ingestor_state)
        state["file_path"] = str(sample_txt)

        result = parse_document(state)
        assert result["status"] == "parsed"
        assert result["file_type"] == "text"
        assert "Álgebra lineal" in result["raw_text"]

    def test_classify_academic_document(self, sample_txt, ingestor_state, mock_llm_response):
        """Classification returns academic label with topics."""
        state = dict(ingestor_state)
        state["file_path"] = str(sample_txt)

        parsed = parse_document(state)
        state.update(parsed)

        result = classify_document(state)
        assert result["classification"] == "apunte_teorico"
        assert len(result["topics"]) >= 3
        assert result["classification_confidence"] >= 0.9


# ──────────────────────────────────────────────────────────────────────────────
# PRD Case #5 — Incremental ingestion
# ──────────────────────────────────────────────────────────────────────────────


class TestIncrementalIngestion:
    def test_incremental_ingestion(self, sample_txt, temp_dir):
        """PRD Case #5: Two files added, chunks accumulate."""
        with patch("src.rag.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            def _fake_encode(texts):
                return [[0.1 * (i + 1) for i in range(384)] for _ in texts]

            mock_model.encode.side_effect = lambda texts: _FakeEncodeResult(_fake_encode(texts))
            mock_get_model.return_value = mock_model

            with patch("src.rag.get_chroma_client") as mock_get_client:
                import chromadb

                client = chromadb.PersistentClient(path=str(temp_dir / "chroma_incremental"))
                mock_get_client.return_value = client

                from src.rag import chunk_text, embed_and_store

                # Ingest first file
                collection = "test_incremental"
                text1 = sample_txt.read_text()
                chunks1 = chunk_text(text1)
                ids1 = embed_and_store(
                    [c.page_content for c in chunks1],
                    [{"source": "file1"} for _ in chunks1],
                    collection,
                )
                count1 = len(ids1)

                # Second ingestion
                text2 = "Ecuaciones diferenciales: capítulo adicional."
                chunks2 = chunk_text(text2)
                ids2 = embed_and_store(
                    [c.page_content for c in chunks2],
                    [{"source": "file2"} for _ in chunks2],
                    collection,
                )
                count2 = len(ids2)

                # Both should have contributed chunks
                assert count1 > 0
                assert count2 > 0
                # Total chunks = sum of both
                coll = client.get_collection(collection)
                assert coll.count() >= count1 + count2


# ──────────────────────────────────────────────────────────────────────────────
# PRD Case #10 — Non-academic rejection
# ──────────────────────────────────────────────────────────────────────────────


class TestNonAcademicRejection:
    def test_reject_non_academic_content(self, non_academic_txt, ingestor_state):
        """PRD Case #10: Non-academic text is rejected."""
        # Create a mock that returns non-academic classification
        with patch("src.llm.get_structured_llm") as mock_gs_llm:
            mock_result = MagicMock()
            mock_result.classification = "no_academico"
            mock_result.confidence = 0.88
            mock_result.topics = []

            fake_invokable = MagicMock()
            fake_invokable.invoke.return_value = mock_result
            mock_gs_llm.return_value = fake_invokable

            state = dict(ingestor_state)
            state["file_path"] = str(non_academic_txt)

            parsed = parse_document(state)
            state.update(parsed)

            result = classify_document(state)
            assert result["status"] == "rejected_non_academic"
            assert result["classification"] == "no_academico"
            assert len(result["errors"]) > 0
            assert "non-academic" in result["errors"][0].lower()

    def test_empty_text_rejected(self, temp_dir, ingestor_state):
        """Empty file is rejected at classification."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("   \n  \n  ")  # whitespace only

        state = dict(ingestor_state)
        state["file_path"] = str(empty_file)

        parsed = parse_document(state)
        state.update(parsed)

        result = classify_document(state)
        assert result["status"] == "rejected"
        assert len(result["errors"]) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Image rejection (scope restriction: OCR deferred)
# ──────────────────────────────────────────────────────────────────────────────


class TestImageRejection:
    def test_image_file_rejected(self, temp_dir, ingestor_state):
        """Image files are rejected with a descriptive message (OCR deferred)."""
        img_file = temp_dir / "notes.png"
        img_file.write_text("fake image content")

        state = dict(ingestor_state)
        state["file_path"] = str(img_file)

        result = parse_document(state)
        assert result["status"] == "rejected"
        assert len(result["errors"]) > 0
        assert "not yet supported" in result["errors"][0].lower()


# ──────────────────────────────────────────────────────────────────────────────
# extract_topics tool
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractTopics:
    def test_extract_topics_from_text(self):
        """extract_topics returns structured topics from text input."""
        from src.tools import extract_topics

        with patch("src.llm.get_structured_llm") as mock_gs_llm:
            mock_result = MagicMock()
            mock_result.summary = "Resumen sobre álgebra lineal."
            mock_result.topics = ["álgebra", "vectores", "matrices"]
            mock_result.topic_tree = {"álgebra": {"vectores": {}, "matrices": {}}}

            fake_invokable = MagicMock()
            fake_invokable.invoke.return_value = mock_result
            mock_gs_llm.return_value = fake_invokable

            result = extract_topics.invoke({"text": "Álgebra lineal: vectores y matrices."})
            assert result["summary"] == "Resumen sobre álgebra lineal."
            assert len(result["topics"]) == 3
            assert "topic_tree" in result

    def test_extract_topics_from_file(self, sample_txt):
        """extract_topics parses a TXT file and extracts topics."""
        from src.tools import extract_topics

        with patch("src.llm.get_structured_llm") as mock_gs_llm:
            mock_result = MagicMock()
            mock_result.summary = "Material sobre álgebra lineal."
            mock_result.topics = ["álgebra", "vectores"]
            mock_result.topic_tree = {"álgebra": {"vectores": {}}}

            fake_invokable = MagicMock()
            fake_invokable.invoke.return_value = mock_result
            mock_gs_llm.return_value = fake_invokable

            result = extract_topics.invoke({"file_path": str(sample_txt)})
            assert "summary" in result
            assert len(result["topics"]) > 0

    def test_extract_topics_no_input(self):
        """extract_topics errors when neither text nor file_path given."""
        from src.tools import extract_topics

        result = extract_topics.invoke({})
        assert "error" in result
        assert "either" in result["error"].lower()

    def test_extract_topics_nonexistent_file(self):
        """extract_topics errors gracefully on missing file."""
        from src.tools import extract_topics

        result = extract_topics.invoke({"file_path": "/nonexistent/file.pdf"})
        assert "error" in result
        assert "not found" in result["error"].lower()


# ──────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ──────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_parse_nonexistent_file(self, ingestor_state):
        """parse_document handles missing file gracefully."""
        state = dict(ingestor_state)
        state["file_path"] = "/nonexistent/file.pdf"

        result = parse_document(state)
        assert result["status"] == "error"
        assert len(result["errors"]) > 0
        assert "not found" in result["errors"][0].lower()

    def test_parse_unsupported_format(self, temp_dir, ingestor_state):
        """parse_document rejects unsupported formats."""
        bad_file = temp_dir / "test.exe"
        bad_file.write_text("not a document")

        state = dict(ingestor_state)
        state["file_path"] = str(bad_file)

        result = parse_document(state)
        assert result["status"] == "rejected"
        assert len(result["errors"]) > 0
        assert "unsupported" in result["errors"][0].lower()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _FakeEncodeResult:
    """Minimal fake to mimic sentence-transformers encode().tolist() chain."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def tolist(self):
        return self._embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Real-model integration tests (run with: pytest -m integration)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRealPDFIngestion:
    """Ingest the real academic PDF with real markitdown + ChromaDB."""

    def test_parse_real_pdf(self, real_pdf_path, real_pdf_text):
        """Real academic PDF parses and produces extractable text."""
        assert len(real_pdf_text) > 500, (
            f"Real PDF produced only {len(real_pdf_text)} chars — too little"
        )
        # Should contain recognizable agent-related terms
        lower = real_pdf_text.lower()
        agent_terms = ["agente", "inteligente", "ambiente", "racional"]
        found = [t for t in agent_terms if t in lower]
        assert len(found) >= 2, (
            f"Real PDF text doesn't look like agent theory. "
            f"Found terms: {found}. Preview: {real_pdf_text[:300]}"
        )

    def test_ingest_real_pdf(self, ingested_collection_name):
        """Real PDF ingestion produces ChromaDB collection with chunks."""
        from chromadb import PersistentClient

        from src.config import settings

        client = PersistentClient(path=settings.chroma_persist_directory)
        collection = client.get_collection(name=ingested_collection_name)
        count = collection.count()
        assert count > 0, "Ingested collection is empty"

    def test_classify_real_pdf(self, requires_ollama, real_pdf_text):
        """Real LLM classifies the PDF as academic material."""
        from src.agents.ingestor import classify_document

        state = {
            "raw_text": real_pdf_text,
            "file_path": "apunteAgentes_IA2007.pdf",
            "session_id": "integration_test",
            "file_type": "pdf",
            "classification": "",
            "classification_confidence": 0.0,
            "topics": [],
            "chunks_created": 0,
            "errors": [],
            "status": "pending",
            "document_id": "",
            "chunk_ids": [],
        }

        result = classify_document(state)
        assert result["classification"] == "apunte_teorico", (
            f"Expected 'apunte_teorico', got '{result.get('classification')}'. "
            f"Confidence: {result.get('classification_confidence')}"
        )
        assert len(result.get("topics", [])) >= 3, (
            f"Expected ≥3 topics, got {len(result.get('topics', []))}"
        )

    def test_retrieve_from_real_pdf(self, ingested_collection_name):
        """Semantic retrieval finds relevant chunks from the real PDF."""
        from src.tools import retrieve_chunks

        # Query using terms we know are in the PDF
        results = retrieve_chunks.invoke(
            {
                "query": "agentes inteligentes",
                "collection_name": ingested_collection_name,
                "top_k": 3,
            }
        )

        assert len(results) > 0, "No chunks retrieved for known topic"
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert len(r["text"]) > 0
