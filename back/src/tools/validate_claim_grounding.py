"""Anti-hallucination claim validation tool — reusable @tool for all agents.

Provides ``validate_claim_grounding``, a LangChain ``@tool`` that validates
claims against source chunks using a two-tier approach:

1. **Embedding cosine similarity** (fast first pass): flags claims whose
   best-match similarity is below the configured threshold.
2. **LLM semantic validation** (fallback for rejected claims): the LLM
   re-checks whether a claim is factually supported by ANY chunk, even
   if the wording differs (paraphrasing, synonyms, short terms).

Supports two modes:
- ``flag_only``: returns warnings for ungrounded claims (Evaluator)
- ``retry_trigger``: includes ``should_retry`` boolean for retry loops
  (ExamGenerator, ExerciseGenerator)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.tools import tool
from langfuse import observe
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)


class ClaimVerdict(BaseModel):
    """LLM verdict for a single claim grounding check."""

    claim_index: int = Field(description="0-based index of the claim in the input list")
    grounded: bool = Field(description="Whether the claim is factually supported by any chunk")
    reasoning: str = Field(description="Brief explanation in Spanish (1-2 sentences)")


class LLMGroundingCheck(BaseModel):
    """Batch LLM grounding verdicts for multiple claims."""

    verdicts: list[ClaimVerdict] = Field(description="One verdict per claim checked")


def _llm_grounding_check(
    claims_to_check: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    context: str | None = None,
) -> list[ClaimVerdict]:
    """Use LLM to semantically validate claims against source chunks.

    Only called for claims that failed the embedding similarity check.
    The LLM sees the claim text and ALL available chunks, and determines
    whether the claim is factually supported — even if paraphrased.

    When *context* is provided (full question/exercise text), it is included
    before the claims so the LLM understands what the question is asking and
    can properly evaluate whether answer-fragment claims are factually supported.
    """
    from src.llm import get_structured_llm

    if not claims_to_check or not all_chunks:
        return []

    # Build chunk context — truncate each to avoid token overflow
    chunk_context_parts: list[str] = []
    for i, c in enumerate(all_chunks):
        cid = c.get("chunk_id", f"chunk-{i}")
        text = c.get("text", "")
        chunk_context_parts.append(f"[{cid}] {text[:400]}")
    chunk_context = "\n\n".join(chunk_context_parts)[:8000]

    # Build claims list
    claims_text = "\n".join(
        f"  [{ci['index']}] {ci['claim'][:200]}" for ci in claims_to_check
    )

    # Build context header — included BEFORE claims so the LLM can evaluate
    # whether answer-fragment claims are factually supported by chunks
    context_block = ""
    if context:
        context_block = f"CONTEXTO DE LA PREGUNTA/EJERCICIO:\n{context[:2000]}\n\n"

    prompt = f"""Revisa si las siguientes afirmaciones estan respaldadas por los fragmentos
del material de estudio proporcionados. Para cada afirmacion, determina si
el contenido FACTUAL esta presente en ALGUN fragmento, incluso si la redaccion
es diferente (parafraseo, sinonimos, terminos tecnicos equivalentes).

FRAGMENTOS DEL MATERIAL:
{chunk_context}

{context_block}AFIRMACIONES A VERIFICAR:
{claims_text}

Para cada afirmacion, indica:
- grounded: true si el contenido factual esta en los fragmentos, false si no
- reasoning: explicacion breve en espanol citando el fragmento relevante

Responde SOLO con JSON valido."""

    try:
        structured_llm = get_structured_llm(LLMGroundingCheck, temperature=settings.validate_grounding_temperature)
        result: LLMGroundingCheck = structured_llm.invoke(prompt)
        return result.verdicts
    except Exception as exc:
        logger.warning("LLM grounding check failed: %s", exc)
        return []


@tool
@observe(name="validate_claim_grounding", as_type="tool")
def validate_claim_grounding(
    claims: list[str],
    chunks: list[dict[str, Any]],
    mode: Literal["flag_only", "retry_trigger"] = "flag_only",
    threshold: float | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Validate claims against source chunks via embedding + LLM fallback.

    Two-tier validation:
    1. Embedding cosine similarity (fast, cheap) — flags claims below threshold.
    2. LLM semantic check (for rejected claims) — re-evaluates whether the
       claim is factually grounded even if paraphrased.

    Args:
        claims: List of claim strings to validate.
        chunks: List of chunk dicts from ChromaDB. Each dict must have
            ``chunk_id`` (str) and ``text`` (str) keys.
        mode: ``"flag_only"`` returns warnings without retry advice.
            ``"retry_trigger"`` adds ``should_retry`` boolean.
        threshold: Cosine similarity threshold. Defaults to
            ``settings.anti_hallucination_threshold``.
        context: Optional full question/exercise text. When provided, it is
            passed to the LLM fallback so it can properly evaluate whether
            answer-fragment claims are factually supported by chunks.

    Returns:
        A dict with:
            - ``all_matched`` (bool): True if every claim is grounded.
            - ``claim_results`` (list[dict]): Per-claim dict with ``claim_text``,
              ``best_chunk_id``, ``similarity_score``, ``matched``, and
              ``llm_verified`` (bool, only for initially rejected claims).
            - ``missing_claims`` (list[dict]): Claims still ungrounded after LLM check.
            - ``should_retry`` (bool): Only present in ``retry_trigger`` mode.
    """
    if threshold is None:
        threshold = settings.anti_hallucination_threshold

    # -- Fast path: empty inputs --
    if not claims or not chunks:
        result: dict[str, Any] = {
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

    # -- Batch encode claims and chunks --
    chunk_texts = [c.get("text", "") for c in chunks]
    claim_embeddings = model.encode(
        [f"query: {cl}" for cl in claims], convert_to_tensor=True
    )
    chunk_embeddings = model.encode(
        [f"passage: {ct}" for ct in chunk_texts], convert_to_tensor=True
    )

    # -- Pre-flight dimension assertion --
    assert claim_embeddings.shape[1] == chunk_embeddings.shape[1], (
        f"Embedding dimension mismatch: claims={claim_embeddings.shape[1]}, "
        f"chunks={chunk_embeddings.shape[1]}"
    )

    # -- Single matrix cosine similarity --
    sim_matrix = cos_sim(claim_embeddings, chunk_embeddings)
    best_scores, best_indices = sim_matrix.max(dim=1)
    best_scores = torch.nan_to_num(best_scores, nan=0.0)

    # -- Per-claim results (embedding pass) --
    claim_results: list[dict[str, Any]] = []
    rejected_claims: list[dict[str, Any]] = []  # for LLM fallback

    for ci, claim in enumerate(claims):
        score = best_scores[ci].item()
        chunk_idx = int(best_indices[ci].item())
        best_chunk_id = chunks[chunk_idx].get("chunk_id", "")

        matched = score >= threshold
        result_entry = {
            "claim_text": claim[:200],
            "best_chunk_id": best_chunk_id,
            "similarity_score": round(score, 4),
            "matched": matched,
        }

        if not matched:
            rejected_claims.append({
                "index": ci,
                "claim": claim[:200],
                "best_chunk_id": best_chunk_id,
                "similarity_score": round(score, 4),
            })

        claim_results.append(result_entry)

    # -- LLM fallback for rejected claims --
    # Only trigger LLM for claims with embedding scores above a floor (0.25).
    # Claims with very low embedding scores (< 0.25) are almost certainly
    # fabricated — skip the expensive LLM call.
    LLM_FALLBACK_FLOOR = 0.25
    llm_candidates = [
        rc for rc in rejected_claims if rc["similarity_score"] >= LLM_FALLBACK_FLOOR
    ]

    if llm_candidates:
        logger.info(
            "Embedding rejected %d/%d claims — %d qualify for LLM fallback (score >= %.2f)",
            len(rejected_claims),
            len(claims),
            len(llm_candidates),
            LLM_FALLBACK_FLOOR,
        )
        verdicts = _llm_grounding_check(llm_candidates, chunks, context=context)

        # Merge LLM verdicts back into claim_results
        for v in verdicts:
            idx = v.claim_index
            if 0 <= idx < len(claim_results):
                claim_results[idx]["llm_verified"] = True
                if v.grounded:
                    claim_results[idx]["matched"] = True
                    claim_results[idx]["llm_grounded"] = True
                    claim_results[idx]["llm_reasoning"] = v.reasoning
    elif rejected_claims:
        logger.info(
            "Embedding rejected %d/%d claims — all below LLM floor (%.2f), skipping fallback",
            len(rejected_claims),
            len(claims),
            LLM_FALLBACK_FLOOR,
        )

    # -- Recompute all_matched and missing_claims after LLM fallback --
    all_matched = True
    missing_claims: list[dict[str, Any]] = []
    for cr in claim_results:
        if not cr["matched"]:
            all_matched = False
            missing_claims.append({
                "claim": cr["claim_text"],
                "best_score": cr["similarity_score"],
                "llm_verified": cr.get("llm_verified", False),
            })

    result = {
        "all_matched": all_matched,
        "claim_results": claim_results,
        "missing_claims": missing_claims,
    }

    if mode == "retry_trigger":
        result["should_retry"] = not all_matched

    return result
