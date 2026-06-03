"""Integration tests for the Ingestor agent — PRD test cases."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.ingestor import (
    check_ocr_confidence,
    classify_document,
    parse_document,
    run_ocr_if_needed,
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

    def test_classify_academic_document(
        self, sample_txt, ingestor_state, mock_llm_response
    ):
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
    def test_incremental_ingestion(
        self, sample_txt, temp_dir
    ):
        """PRD Case #5: Two files added, chunks accumulate."""
        with patch("src.rag.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            def _fake_encode(texts):
                return [[0.1 * (i + 1) for i in range(384)] for _ in texts]

            mock_model.encode.side_effect = lambda texts: _FakeEncodeResult(
                _fake_encode(texts)
            )
            mock_get_model.return_value = mock_model

            with patch("src.rag.get_chroma_client") as mock_get_client:
                import chromadb

                client = chromadb.PersistentClient(
                    path=str(temp_dir / "chroma_incremental")
                )
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
# PRD Case #6 — Math PDF OCR skipped for text files
# ──────────────────────────────────────────────────────────────────────────────

class TestOcrHandling:
    def test_ocr_skipped_for_text_files(self, sample_txt, ingestor_state):
        """PRD Case #6: OCR is skipped for non-image files."""
        state = dict(ingestor_state)
        state["file_path"] = str(sample_txt)
        state["file_type"] = "text"

        result = run_ocr_if_needed(state)
        assert result["status"] == "ocr_skipped"
        assert result["ocr_confidence"] == 1.0
        assert result["needs_ocr_confirmation"] is False
        assert result["ocr_expressions"] == []


# ──────────────────────────────────────────────────────────────────────────────
# PRD Case #9 — Low OCR confidence triggers confirmation
# ──────────────────────────────────────────────────────────────────────────────

class TestOcrConfidenceGating:
    def test_low_ocr_confidence_triggers_confirmation(self):
        """PRD Case #9: Confidence < 0.85 triggers confirmation."""
        result = check_ocr_confidence({"ocr_confidence": 0.4})
        assert result == "request_confirmation"

    def test_high_ocr_confidence_proceeds(self):
        """Confidence >= 0.85 proceeds to chunking."""
        result = check_ocr_confidence({"ocr_confidence": 0.9})
        assert result == "proceed"

    def test_exact_threshold_boundary(self):
        """Confidence exactly at threshold proceeds."""
        result = check_ocr_confidence({"ocr_confidence": 0.85})
        assert result == "proceed"


# ──────────────────────────────────────────────────────────────────────────────
# PRD Case #10 — Non-academic rejection
# ──────────────────────────────────────────────────────────────────────────────

class TestNonAcademicRejection:
    def test_reject_non_academic_content(
        self, non_academic_txt, ingestor_state
    ):
        """PRD Case #10: Non-academic text is rejected."""
        # Create a mock that returns non-academic classification
        with patch("langchain_groq.ChatGroq") as mock_llm:
            mock_result = MagicMock()
            mock_result.classification = "no_academico"
            mock_result.confidence = 0.88
            mock_result.topics = []

            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_result

            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_structured
            mock_llm.return_value = mock_instance

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
