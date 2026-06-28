"""Anti-hallucination claim validation tool — reusable @tool for all agents.

Provides ``validate_claim_grounding``, a LangChain ``@tool`` that embeds
claims and source chunks, computes cosine similarity via
``sentence_transformers.util.cos_sim``, and returns per-claim grounding
results.

Supports two modes:
- ``flag_only``: returns warnings for ungrounded claims (Evaluator)
- ``retry_trigger``: includes ``should_retry`` boolean for retry loops
  (ExamGenerator, ExerciseGenerator)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from langfuse import observe

from src.config import settings


@tool
@observe(name="validate_claim_grounding", as_type="tool")
def validate_claim_grounding(
    claims: list[str],
    chunks: list[dict],
    mode: Literal["flag_only", "retry_trigger"] = "flag_only",
    threshold: float | None = None,
) -> dict:
    """Validate claims against source chunks via embedding cosine similarity.

    Batch-encodes all claims and chunk texts, computes the cosine similarity
    matrix with ``sentence_transformers.util.cos_sim``, and returns per-claim
    grounding results. Zero-norm vectors are handled via ``torch.nan_to_num``.

    Args:
        claims: List of claim strings to validate (e.g. sentence-level claims
            extracted from generated content).
        chunks: List of chunk dicts from ChromaDB. Each dict must have
            ``chunk_id`` (str) and ``text`` (str) keys.
        mode: ``"flag_only"`` returns warnings without retry advice (used by
            Evaluator, which only flags ungrounded feedback). ``"retry_trigger"``
            adds a ``should_retry`` boolean indicating whether regeneration
            is recommended (used by ExamGenerator and ExerciseGenerator retry
            loops).
        threshold: Cosine similarity threshold. Claims with best-match score
            below this value are flagged as ungrounded. Defaults to
            ``settings.anti_hallucination_threshold``.

    Returns:
        A dict with:
            - ``all_matched`` (bool): True if every claim has similarity >= threshold.
            - ``claim_results`` (list[dict]): Per-claim dict with ``claim_text``,
              ``best_chunk_id``, ``similarity_score``, ``matched``.
            - ``missing_claims`` (list[dict]): Claims below threshold, each with
              ``claim`` and ``best_score``.
            - ``should_retry`` (bool): Only present in ``retry_trigger`` mode.
              True when ``all_matched`` is False (recommends regeneration).
    """
    if threshold is None:
        threshold = settings.anti_hallucination_threshold

    # ── Fast path: empty inputs ──
    if not claims or not chunks:
        result: dict = {
            "all_matched": True,
            "claim_results": [],
            "missing_claims": [],
        }
        if mode == "retry_trigger":
            result["should_retry"] = False
        return result

    import torch
    from sentence_transformers.util import cos_sim

    from src.rag import get_embedding_model

    model = get_embedding_model()

    # ── Batch encode claims and chunks ──
    # E5 models require asymmetric prefixes: claims are queries, chunks are passages.
    chunk_texts = [c.get("text", "") for c in chunks]
    claim_embeddings = model.encode(
        [f"query: {cl}" for cl in claims], convert_to_tensor=True
    )
    chunk_embeddings = model.encode(
        [f"passage: {ct}" for ct in chunk_texts], convert_to_tensor=True
    )

    # ── Pre-flight dimension assertion ──
    assert claim_embeddings.shape[1] == chunk_embeddings.shape[1], (
        f"Embedding dimension mismatch: claims={claim_embeddings.shape[1]}, "
        f"chunks={chunk_embeddings.shape[1]}"
    )

    # ── Single matrix cosine similarity ──
    sim_matrix = cos_sim(claim_embeddings, chunk_embeddings)
    best_scores, best_indices = sim_matrix.max(dim=1)
    best_scores = torch.nan_to_num(best_scores, nan=0.0)

    # ── Per-claim results ──
    claim_results: list[dict] = []
    missing_claims: list[dict] = []
    all_matched = True

    for ci, claim in enumerate(claims):
        score = best_scores[ci].item()
        chunk_idx = int(best_indices[ci].item())
        best_chunk_id = chunks[chunk_idx].get("chunk_id", "")

        matched = score >= threshold
        if not matched:
            all_matched = False
            missing_claims.append({"claim": claim[:200], "best_score": round(score, 4)})

        claim_results.append(
            {
                "claim_text": claim[:200],
                "best_chunk_id": best_chunk_id,
                "similarity_score": round(score, 4),
                "matched": matched,
            }
        )

    result = {
        "all_matched": all_matched,
        "claim_results": claim_results,
        "missing_claims": missing_claims,
    }

    if mode == "retry_trigger":
        result["should_retry"] = not all_matched

    return result
