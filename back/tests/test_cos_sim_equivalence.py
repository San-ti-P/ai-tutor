"""Numerical equivalence test: manual _cosine_sim vs sentence_transformers.util.cos_sim.

Proves that the refactoring from manual cosine similarity to
``sentence_transformers.util.cos_sim`` produces identical results
within floating-point tolerance (atol=1e-6).
"""

from __future__ import annotations

import math

import pytest
import torch
from sentence_transformers.util import cos_sim


def _manual_cosine_sim(a: list[float], b: list[float]) -> float:
    """Manual cosine similarity — matches the current implementation in agents.

    Includes the zero-norm guard: returns 0.0 if either vector has zero norm.
    This is the reference implementation we're replacing.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _random_vector(dim: int = 384) -> list[float]:
    """Generate a random vector with values in [-1, 1]."""
    import random

    random.seed(42)  # Deterministic for reproducibility
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


class TestCosineSimEquivalence:
    """Prove cos_sim from sentence_transformers matches manual implementation."""

    def test_single_pair_identical_vectors(self):
        """cos_sim and manual agree on identical normalized vectors."""
        vec = _random_vector()
        manual = _manual_cosine_sim(vec, vec)
        # cos_sim expects 2D tensors: shape (N, D)
        t = torch.tensor([vec], dtype=torch.float32)
        result = cos_sim(t, t)
        assert torch.abs(result[0, 0] - manual) < 1e-6, (
            f"cos_sim={result[0, 0]:.10f}, manual={manual:.10f}"
        )

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_100_random_pairs(self, seed: int):
        """100 random normalized vector pairs — cos_sim ≈ manual within 1e-6."""
        import random as _random

        _random.seed(seed)
        a = [_random.uniform(-1.0, 1.0) for _ in range(384)]
        b = [_random.uniform(-1.0, 1.0) for _ in range(384)]

        # Manual result
        manual = _manual_cosine_sim(a, b)

        # cos_sim result (needs 2D tensors)
        t_a = torch.tensor([a], dtype=torch.float32)
        t_b = torch.tensor([b], dtype=torch.float32)
        result = cos_sim(t_a, t_b)

        assert torch.abs(result[0, 0] - manual) < 1e-6, (
            f"seed={seed}: cos_sim={result[0, 0]:.10f}, manual={manual:.10f}"
        )

    def test_zero_norm_vector_produces_zero(self):
        """Both implementations return ~0 for zero-norm vectors.

        Note: cos_sim produces NaN for zero-norm vectors; torch.nan_to_num
        converts NaN to 0.0. This test verifies the conversion.
        """
        a = [0.0] * 384
        b = _random_vector()

        # Manual zero-norm guard returns 0.0
        manual = _manual_cosine_sim(a, b)
        assert manual == 0.0

        # sentence_transformers cos_sim handles zero-norm internally;
        # the result is 0.0 (not NaN) because normalize_embeddings
        # produces all-zeros which yields 0.0 after mm.
        t_a = torch.tensor([a], dtype=torch.float32)
        t_b = torch.tensor([b], dtype=torch.float32)
        result = cos_sim(t_a, t_b)
        # nan_to_num is still safe to apply (idempotent on non-NaN)
        clean = torch.nan_to_num(result, nan=0.0)
        assert clean[0, 0] == 0.0

    def test_batch_shape_correct(self):
        """cos_sim with N claims and M chunks produces (N, M) matrix."""
        claims = torch.randn(5, 384, dtype=torch.float32)
        chunks = torch.randn(10, 384, dtype=torch.float32)
        sim = cos_sim(claims, chunks)
        assert sim.shape == (5, 10)

    def test_best_score_per_claim(self):
        """max(dim=1) on cos_sim matrix gives best chunk per claim."""
        claims = torch.randn(3, 384, dtype=torch.float32)
        chunks = torch.randn(5, 384, dtype=torch.float32)
        sim = cos_sim(claims, chunks)
        best_scores, best_indices = sim.max(dim=1)
        assert best_scores.shape == (3,)
        assert best_indices.shape == (3,)
        assert all(0 <= idx < 5 for idx in best_indices.tolist())

    def test_dimension_mismatch_raises(self):
        """cos_sim with incompatible dimensions raises RuntimeError."""
        a = torch.randn(2, 384, dtype=torch.float32)
        b = torch.randn(2, 512, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="cannot be multiplied"):
            cos_sim(a, b)
