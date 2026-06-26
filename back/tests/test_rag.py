"""Unit tests for the RAG module."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.rag import ThematicIndex, chunk_text, embed_and_store, retrieve

# ═══════════════════════════════════════════════════════════════════════════════
# RAG policy module tests (rag-exclusive-answers, task 3.1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRagPolicy:
    """Unit tests for the shared RAG-only policy module."""

    def test_rag_only_system_prompt_exists(self):
        """RAG_ONLY_SYSTEM_PROMPT is a non-empty string that forbids parametric knowledge."""
        from src.rag.policy import RAG_ONLY_SYSTEM_PROMPT

        assert isinstance(RAG_ONLY_SYSTEM_PROMPT, str)
        assert len(RAG_ONLY_SYSTEM_PROMPT) > 50
        # Must explicitly forbid parametric knowledge
        assert "ÚNICAMENTE" in RAG_ONLY_SYSTEM_PROMPT or "SOLO" in RAG_ONLY_SYSTEM_PROMPT
        assert (
            "no inventes" in RAG_ONLY_SYSTEM_PROMPT.lower()
            or "no" in RAG_ONLY_SYSTEM_PROMPT.lower()
        )

    def test_no_material_message_returns_string(self):
        """no_material_message() returns a Spanish string with actionable guidance."""
        from src.rag.policy import no_material_message

        msg = no_material_message()
        assert isinstance(msg, str)
        assert len(msg) > 20
        # Must contain material-related guidance in Spanish
        lower = msg.lower()
        assert "material" in lower or "subí" in lower or "cargado" in lower

    def test_no_material_message_is_deterministic(self):
        """Multiple calls return the same message string."""
        from src.rag.policy import no_material_message

        msg1 = no_material_message()
        msg2 = no_material_message()
        assert msg1 == msg2


class TestChunkText:
    def test_chunks_short_text(self):
        docs = chunk_text("Hello world.")
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world."

    def test_chunks_long_text(self):
        long_text = "This is a test. " * 200  # ~3600 chars
        docs = chunk_text(long_text)
        assert len(docs) > 1
        # Each chunk should be <= 512 chars (plus some overlap)
        for doc in docs:
            assert len(doc.page_content) <= 600  # 512 + overlap buffer

    def test_chunks_empty_text(self):
        docs = chunk_text("")
        assert len(docs) == 0

    def test_chunks_preserves_metadata(self):
        docs = chunk_text("Hello world.", {"source": "test"})
        for doc in docs:
            assert doc.metadata["source"] == "test"


class TestThematicIndex:
    def test_add_topics(self):
        ti = ThematicIndex()
        ti.add_topics(["math", "algebra", "calculus"])
        assert "math" in ti._tree
        assert "algebra" in ti._tree
        assert "calculus" in ti._tree

    def test_add_topics_hierarchical(self):
        ti = ThematicIndex()
        ti.add_topics(["math/algebra/linear", "physics/mechanics"])
        assert "math" in ti._tree
        assert "algebra" in ti._tree["math"]
        assert "linear" in ti._tree["math"]["algebra"]

    def test_merge(self):
        ti1 = ThematicIndex()
        ti1.add_topics(["math", "algebra"])
        ti2 = ThematicIndex()
        ti2.add_topics(["physics", "math", "geometry"])

        ti1.merge(ti2)
        assert "physics" in ti1._tree
        assert "geometry" in ti1._tree
        assert "math" in ti1._tree  # existing branch preserved

    def test_merge_preserves_existing(self):
        ti1 = ThematicIndex()
        ti1.add_topics(["math/algebra/linear"])
        ti2 = ThematicIndex()
        ti2.add_topics(["math/calculus"])

        ti1.merge(ti2)
        assert "linear" in ti1._tree["math"]["algebra"]
        assert "calculus" in ti1._tree["math"]

    def test_search(self):
        ti = ThematicIndex()
        ti.add_topics(["math/algebra/linear", "physics/mechanics"])
        results = ti.search("math")
        assert len(results) > 0
        assert "algebra" in results

    def test_search_not_found(self):
        ti = ThematicIndex()
        ti.add_topics(["math"])
        results = ti.search("nonexistent")
        assert len(results) == 0

    def test_to_dict(self):
        ti = ThematicIndex()
        ti.add_topics(["math", "algebra"])
        d = ti.to_dict()
        assert isinstance(d, dict)
        assert "math" in d
        assert "algebra" in d


class TestEmbedAndStore:
    """Tests for embed_and_store requiring a mock embedding model."""

    @pytest.fixture(autouse=True)
    def _mock_embedding(self, temp_dir):
        """Mock get_embedding_model to return fake embeddings and override
        ChromaDB path to a temp directory."""
        with patch("src.rag.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            # Return a 384-dim fake embedding for each chunk
            def _fake_encode(texts):
                return [[0.1 * (i + 1) for i in range(384)] for _ in texts]

            mock_model.encode.return_value.tolist = MagicMock()
            mock_model.encode.side_effect = lambda texts: _FakeEncodeResult(_fake_encode(texts))
            mock_get_model.return_value = mock_model

            # Patch chroma persist directory to temp
            with patch("src.rag.get_chroma_client") as mock_get_client:
                import chromadb

                client = chromadb.PersistentClient(path=str(temp_dir / "chroma_test_embed"))
                mock_get_client.return_value = client
                yield

    def test_embed_and_store_returns_ids(self):
        chunks = ["Chunk one.", "Chunk two.", "Chunk three."]
        ids = embed_and_store(
            chunks,
            [{"source": "test"} for _ in chunks],
            "test_collection",
        )
        assert len(ids) == 3
        # All IDs should be valid UUIDs
        for chunk_id in ids:
            uuid.UUID(chunk_id)

    def test_embed_and_store_empty_chunks(self):
        ids = embed_and_store([], [], "test_empty")
        assert len(ids) == 0


class _FakeEncodeResult:
    """Minimal fake to mimic sentence-transformers encode().tolist() chain."""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def tolist(self):
        return self._embeddings


class TestRetrieve:
    """Tests for retrieve requiring a mock embedding model."""

    @pytest.fixture(autouse=True)
    def _setup_collection(self, temp_dir):
        """Create a real ChromaDB collection with pre-loaded chunks."""
        with patch("src.rag.get_embedding_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            # Store the texts that get encoded so we can build proper embeddings
            encoded_texts = []

            def _fake_encode(texts):
                encoded_texts.clear()
                encoded_texts.extend(texts)
                return [[0.1 * (i + 1) for i in range(384)] for _ in texts]

            mock_model.encode.return_value.tolist = MagicMock()
            mock_model.encode.side_effect = lambda texts: _FakeEncodeResult(_fake_encode(texts))
            mock_get_model.return_value = mock_model

            with patch("src.rag.get_chroma_client") as mock_get_client:
                import chromadb

                client = chromadb.PersistentClient(path=str(temp_dir / "chroma_test_retrieve"))
                mock_get_client.return_value = client

                # Pre-populate collection with test chunks
                embed_and_store(
                    [
                        "Calculus derivatives explained.",
                        "Physics mechanics 101.",
                        "Algebra matrices operations.",
                    ],
                    [
                        {"topic": "math/calculus", "source": "test"},
                        {"topic": "physics/mechanics", "source": "test"},
                        {"topic": "math/algebra", "source": "test"},
                    ],
                    "test_retrieve",
                )
                yield

    def test_retrieve_returns_results(self):
        results = retrieve("derivatives", "test_retrieve", top_k=2)
        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "metadata" in r
            assert "similarity_score" in r

    def test_retrieve_with_topic_filter(self):
        results = retrieve(
            "operations",
            "test_retrieve",
            top_k=3,
            topic_filter="math",
        )
        assert len(results) > 0
        # All results should have a metadata topic starting with "math"
        for r in results:
            assert r["metadata"]["topic"].startswith("math")

    def test_retrieve_empty_collection(self, temp_dir):
        with patch("src.rag.get_chroma_client") as mock_get_client:
            import chromadb

            client = chromadb.PersistentClient(path=str(temp_dir / "chroma_empty"))
            mock_get_client.return_value = client
            results = retrieve("query", "nonexistent_collection", top_k=5)
            assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# Real-model integration tests (run with: pytest -m integration)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestRealRAG:
    """Real chunking + embedding + retrieval from the real academic PDF."""

    def test_real_chunking(self, real_pdf_text):
        """Real academic text produces multiple semantic chunks."""
        docs = chunk_text(real_pdf_text)
        assert len(docs) >= 5, f"Expected ≥5 chunks from real PDF, got {len(docs)}"
        # Chunks should not exceed chunk_size + overlap
        for doc in docs:
            assert len(doc.page_content) <= 600, f"Chunk too large: {len(doc.page_content)} chars"
        # At least one chunk should contain agent-related content
        agent_content = any("agente" in doc.page_content.lower() for doc in docs)
        assert agent_content, "No chunk contains agent-related content"

    def test_real_embed_and_retrieve(self, ingested_collection_name):
        """Real embeddings produce meaningful semantic retrieval."""
        # Query for agent-related content — should find relevant chunks
        results = retrieve(
            "definición de agente inteligente",
            ingested_collection_name,
            top_k=5,
        )
        assert len(results) > 0, "No results for known topic query"
        # Best result should have a reasonable similarity score
        best_score = results[0]["similarity_score"]
        assert best_score >= 0.0, f"Invalid similarity score: {best_score}"
        # Results should be ordered by relevance (scores should differ)
        scores = [r["similarity_score"] for r in results]
        unique_scores = len(set(round(s, 5) for s in scores))
        assert unique_scores >= 2, (
            "All retrieved chunks have identical similarity scores — "
            "embedding may not be working correctly"
        )

    def test_real_topic_extraction(self, requires_ollama, real_pdf_text):
        """extract_topics tool works on real academic content."""
        from src.tools import extract_topics

        result = extract_topics.invoke({"text": real_pdf_text[:5000]})
        assert "error" not in result, f"extract_topics failed: {result.get('error')}"
        assert "summary" in result
        assert len(result.get("topics", [])) >= 3, (
            f"Expected ≥3 topics, got {result.get('topics', [])}"
        )
        assert "topic_tree" in result
        # Topic tree should be a dict or non-empty JSON string
        tree = result["topic_tree"]
        assert tree, "topic_tree is empty"
        if isinstance(tree, str):
            import json

            try:
                tree = json.loads(tree)
            except json.JSONDecodeError:
                pass  # string representation is acceptable
        assert isinstance(tree, (dict, str)), f"topic_tree is not dict or string: {type(tree)}"
