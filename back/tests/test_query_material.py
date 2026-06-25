"""Tests for the query_material tool."""

from __future__ import annotations

from unittest.mock import patch


class TestQueryMaterial:
    """Unit tests for query_material RAG answer tool."""

    def test_query_material_no_chunks(self):
        """Empty retrieval returns a friendly no-material message."""
        from src.tools.query_material import query_material

        with patch("src.tools.query_material._rag_retrieve", return_value=[]):
            result = query_material.invoke(
                {"query": "¿Qué dice el apunte?", "session_id": "sess-123"}
            )

        assert result["chunks_found"] == 0
        assert result["sources"] == []
        assert "No encontré material" in result["answer"]

    def test_query_material_answers_from_chunks(self):
        """Retrieved chunks are passed to the LLM and an answer is returned."""
        from src.tools.query_material import query_material

        chunks = [
            {
                "chunk_id": "c1",
                "text": "La derivada de x^2 es 2x.",
                "metadata": {},
                "similarity_score": 0.1,
            }
        ]

        with patch("src.tools.query_material._rag_retrieve", return_value=chunks):
            with patch(
                "src.tools.query_material.get_llm",
                return_value=_FakeLLM("La derivada es 2x."),
            ):
                result = query_material.invoke(
                    {"query": "¿Cuál es la derivada de x^2?", "session_id": "sess-123"}
                )

        assert result["chunks_found"] == 1
        assert result["answer"] == "La derivada es 2x."
        assert result["sources"] == ["La derivada de x^2 es 2x."]


class _FakeLLM:
    """Minimal fake LLM returning a fixed string response."""

    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _prompt: str, **_kwargs):
        class _Response:
            content = self._content

        return _Response()
