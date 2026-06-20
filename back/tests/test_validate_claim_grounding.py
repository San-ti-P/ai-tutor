"""Unit tests for src/tools/validate_claim_grounding.py — anti-hallucination @tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import settings

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_claims() -> list[str]:
    """Claims that should match sample chunks."""
    return [
        "La derivada de una función se define como el límite del cociente incremental.",
        "La suma de matrices se realiza elemento a elemento.",
    ]


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Source chunks with text and chunk_id."""
    return [
        {
            "chunk_id": "chunk-math-001",
            "text": (
                "La derivada de una función f(x) en un punto a se define como "
                "el límite del cociente incremental: f'(a) = lim(h->0) [f(a+h)-f(a)]/h."
            ),
        },
        {
            "chunk_id": "chunk-math-003",
            "text": (
                "Una matriz es un arreglo rectangular de números. La suma de "
                "matrices se realiza elemento a elemento."
            ),
        },
    ]


# ── Test helpers ─────────────────────────────────────────────────────────────


def _make_matching_model(dim: int = 384):
    """Model where chunks and claims get identical embeddings → cos_sim=1.0."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        # All vectors are [1/sqrt(dim), ...] → cos_sim between any pair = 1.0
        return torch.full((len(texts), dim), 1.0 / (dim**0.5), dtype=torch.float32)

    model.encode.side_effect = _encode
    return model


def _make_diverging_model(chunk_value: float, claim_value: float, dim: int = 384):
    """Model where chunks and claims get different embeddings.

    Normalised dot product ≈ chunk_value * claim_value.
    """
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    call_counter = [0]

    def _encode(texts, **kwargs):
        call_counter[0] += 1
        if call_counter[0] == 1:  # First call: chunks
            return torch.full((len(texts), dim), chunk_value, dtype=torch.float32)
        # Second call: claims
        return torch.full((len(texts), dim), claim_value, dtype=torch.float32)

    model.encode.side_effect = _encode
    return model


def _make_zero_model(dim: int = 384):
    """Model returning all-zero vectors → cos_sim=0.0."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim

    def _encode(texts, **kwargs):
        return torch.zeros((len(texts), dim), dtype=torch.float32)

    model.encode.side_effect = _encode
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestValidateClaimGroundingFlagOnly:
    """flag_only mode: returns all_matched + claim_results, never retry."""

    def test_all_claims_matched(self, sample_claims, sample_chunks):
        """All claims above threshold → all_matched=True, no missing."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_matching_model()

            result = validate_claim_grounding.invoke({
                "claims": sample_claims,
                "chunks": sample_chunks,
                "mode": "flag_only",
                "threshold": 0.1,
            })

        assert result["all_matched"] is True
        assert len(result["claim_results"]) == 2
        assert all(r["matched"] for r in result["claim_results"])
        assert len(result["missing_claims"]) == 0
        assert "should_retry" not in result

    def test_some_claims_below_threshold(self, sample_claims, sample_chunks):
        """Claims below threshold → all_matched=False, missing populated."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            # Chunks: unit vector, Claims: zero → cos_sim=0.0 < any threshold
            mock_embed.return_value = _make_diverging_model(
                chunk_value=1.0, claim_value=0.0
            )

            result = validate_claim_grounding.invoke({
                "claims": sample_claims,
                "chunks": sample_chunks,
                "mode": "flag_only",
                "threshold": 0.5,
            })

        assert result["all_matched"] is False
        assert len(result["claim_results"]) == 2
        assert all(not r["matched"] for r in result["claim_results"])
        assert len(result["missing_claims"]) == 2
        assert "should_retry" not in result

    def test_default_threshold_from_settings(self, sample_claims, sample_chunks):
        """When threshold not provided, uses settings.anti_hallucination_threshold."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        original = settings.anti_hallucination_threshold
        settings.anti_hallucination_threshold = 0.99

        try:
            with patch("src.rag.get_embedding_model") as mock_embed:
                # cos_sim=1.0 < 0.99 → False (hmm, 1.0 > 0.99, still matched)
                # cos_sim=0.0 < 0.99 → True (unmatched)
                mock_embed.return_value = _make_diverging_model(
                    chunk_value=1.0, claim_value=0.0
                )

                result = validate_claim_grounding.invoke({
                    "claims": sample_claims,
                    "chunks": sample_chunks,
                    "mode": "flag_only",
                })

            assert result["all_matched"] is False
        finally:
            settings.anti_hallucination_threshold = original

    def test_empty_claims_returns_empty(self, sample_chunks):
        """Empty claims list → all_matched=True, empty results."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        result = validate_claim_grounding.invoke({
            "claims": [],
            "chunks": sample_chunks,
            "mode": "flag_only",
        })

        assert result["all_matched"] is True
        assert result["claim_results"] == []
        assert result["missing_claims"] == []

    def test_empty_chunks_returns_empty(self, sample_claims):
        """Empty chunks list → all_matched=True, empty results."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        result = validate_claim_grounding.invoke({
            "claims": sample_claims,
            "chunks": [],
            "mode": "flag_only",
        })

        assert result["all_matched"] is True
        assert result["claim_results"] == []
        assert result["missing_claims"] == []


class TestValidateClaimGroundingRetryTrigger:
    """retry_trigger mode: includes should_retry boolean."""

    def test_all_matched_no_retry(self, sample_claims, sample_chunks):
        """All claims above threshold → should_retry=False."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_matching_model()

            result = validate_claim_grounding.invoke({
                "claims": sample_claims,
                "chunks": sample_chunks,
                "mode": "retry_trigger",
                "threshold": 0.1,
            })

        assert result["all_matched"] is True
        assert result["should_retry"] is False

    def test_some_unmatched_triggers_retry(self, sample_claims, sample_chunks):
        """Claims below threshold → should_retry=True."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_diverging_model(
                chunk_value=1.0, claim_value=0.0
            )

            result = validate_claim_grounding.invoke({
                "claims": sample_claims,
                "chunks": sample_chunks,
                "mode": "retry_trigger",
                "threshold": 0.9,
            })

        assert result["all_matched"] is False
        assert result["should_retry"] is True

    def test_best_chunk_id_present(self, sample_claims, sample_chunks):
        """Each claim_result includes best_chunk_id from the source chunks."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_matching_model()

            result = validate_claim_grounding.invoke({
                "claims": sample_claims,
                "chunks": sample_chunks,
                "mode": "retry_trigger",
                "threshold": 0.1,
            })

        for cr in result["claim_results"]:
            assert "best_chunk_id" in cr
            assert cr["best_chunk_id"] in {
                "chunk-math-001",
                "chunk-math-003",
            }


class TestValidateClaimGroundingEdgeCases:
    """Edge-case handling."""

    def test_zero_norm_embeddings_handled(self):
        """Zero vectors → similarity = 0.0 → below any positive threshold."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        claim = "some claim"
        chunk = {"chunk_id": "c1", "text": "source text"}

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_zero_model()

            result = validate_claim_grounding.invoke({
                "claims": [claim],
                "chunks": [chunk],
                "mode": "flag_only",
                "threshold": 0.5,
            })

        # Zero-norm → similarity = 0.0 → below any positive threshold → unmatched
        assert result["all_matched"] is False
        assert result["claim_results"][0]["similarity_score"] == 0.0

    def test_single_claim_single_chunk(self):
        """Minimal input: 1 claim, 1 chunk."""
        from src.tools.validate_claim_grounding import validate_claim_grounding

        with patch("src.rag.get_embedding_model") as mock_embed:
            mock_embed.return_value = _make_matching_model()

            result = validate_claim_grounding.invoke({
                "claims": ["Test claim"],
                "chunks": [{"chunk_id": "c1", "text": "Test chunk"}],
                "mode": "flag_only",
                "threshold": 0.5,
            })

        assert result["all_matched"] is True
        assert len(result["claim_results"]) == 1
        assert result["claim_results"][0]["matched"] is True
