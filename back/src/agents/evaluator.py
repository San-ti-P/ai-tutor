"""Evaluator Agent — Chain-of-Thought StateGraph for grading student answers.

8-node batch evaluation pipeline:
  prepare → check_evaluability → evaluate → validate_feedback
    → [optional llm_judge] → build_feedback → next_question
    → (loop or sync_scores → END)

Anti-hallucination via claim-level embedding cross-reference against
retrieved RAG chunks (threshold 0.55). LLM-as-judge sampling at
configurable rate (default 10%) for quality assurance.
"""

from __future__ import annotations

import logging
import math
import operator
import random
from typing import Annotated

try:
    from langfuse import observe
except ImportError:
    def observe(name: str | None = None):  # noqa: D103
        def decorator(fn):  # noqa: D103
            return fn
        return decorator

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── Pydantic structured-output models ────────────────────────────────────────


class SingleEvaluation(BaseModel):
    """LLM structured output for a single answer evaluation."""

    score: float = Field(ge=0, le=10, description="Numerical score from 0 to 10")
    justification: str = Field(
        description="Detailed justification referencing specific concepts"
    )
    conceptual_errors: list[str] = Field(
        default_factory=list,
        description="List of conceptual mistakes found in the answer",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable review suggestions for the student",
    )
    is_evaluable: bool = Field(
        default=True,
        description="Whether the answer is coherent enough to be evaluated",
    )


class JudgeVerdict(BaseModel):
    """LLM-as-judge second-pass evaluation output."""

    score: float = Field(ge=0, le=10, description="Independent judge score 0-10")
    agrees_with_primary: bool = Field(
        description="Whether judge agrees with the primary evaluation"
    )
    discrepancy: str = Field(
        default="",
        description="Explanation of any disagreement between judge and primary",
    )
    suggested_score: float | None = Field(
        default=None,
        description="Judge's suggested score if disagreement found",
    )


# ── Graph state schema ───────────────────────────────────────────────────────


class EvaluatorState(TypedDict):
    """State for the 8-node batch evaluation StateGraph.

    Fields with ``Annotated[list, operator.add]`` are reducers that
    accumulate across graph iterations (new values are appended, not
    overwritten).
    """

    # ── request metadata ──
    session_id: str
    student_id: str
    exam_id: str
    trace_id: str

    # ── batch answers ──
    answers: list[dict]
    # Each answer dict: question_id, question, base_answer, student_answer,
    # answer_image (optional), source_chunk_ids, topic, difficulty
    current_index: int
    answer_text: str

    # ── OCR passthrough ──
    ocr_extracted_text: str | None
    ocr_confidence: float

    # ── RAG chunks (accumulated across topics) ──
    retrieved_chunks: Annotated[list[dict], operator.add]
    collection_name: str  # Override for session_{session_id}; used by prepare_evaluation

    # ── per-question evaluation state ──
    evaluation: dict | None
    evaluation_results: Annotated[list[dict], operator.add]

    # ── non-evaluable guard ──
    non_evaluable: bool
    non_evaluable_reason: str

    # ── LLM-as-judge ──
    judge_sample: bool
    judge_result: dict | None
    requires_review: bool

    # ── finalisation ──
    scores_synced: bool
    errors: Annotated[list[str], operator.add]
    status: str


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: cosine similarity
# ═══════════════════════════════════════════════════════════════════════════════


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Node: prepare_evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_evaluation(state: EvaluatorState) -> dict:
    """Deduplicate topics from answers, retrieve RAG chunks per topic.

    Queries ChromaDB for each unique topic across the answer batch and
    accumulates retrieved chunks via the ``retrieved_chunks`` reducer.
    No LLM call — pure retrieval.
    """
    from src.tools import retrieve_chunks as _retrieve_chunks

    answers: list[dict] = state.get("answers", [])
    session_id: str = state.get("session_id", "")
    collection_name = state.get("collection_name") or f"session_{session_id}"

    # Deduplicate topics preserving order
    seen_topics: set[str] = set()
    unique_topics: list[str] = []
    for ans in answers:
        topic = ans.get("topic", "")
        if topic and topic not in seen_topics:
            seen_topics.add(topic)
            unique_topics.append(topic)

    all_chunks: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for topic in unique_topics:
        try:
            chunks = _retrieve_chunks.invoke(
                {
                    "query": topic,
                    "top_k": 5,
                    "collection_name": collection_name,
                }
            )
            for chunk in chunks:
                cid = chunk.get("chunk_id", "")
                if cid and cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_chunks.append(chunk)
        except Exception as exc:
            logger.warning("Retrieval failed for topic '%s': %s", topic, exc)

    return {
        "retrieved_chunks": all_chunks,
        "current_index": 0,
        "status": "prepared",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: check_evaluability
# ═══════════════════════════════════════════════════════════════════════════════


def _has_vowel(word: str) -> bool:
    """Return True if word contains at least one vowel character."""
    vowels = set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
    return any(ch in vowels for ch in word)


def _is_gibberish(text: str) -> bool:
    """Heuristic: detect keyboard-smashing gibberish via vowel ratio.

    If more than 40% of whitespace-separated tokens (length ≥ 2) contain
    no vowels, the text is likely keyboard smashing.
    """
    if not text or not text.strip():
        return True

    tokens = [t for t in text.split() if len(t) >= 2]
    if not tokens:
        return True

    vowelless_count = sum(1 for t in tokens if not _has_vowel(t))
    return vowelless_count / len(tokens) > 0.4


def _guess_language_mismatch(text: str, question: str) -> bool:
    """Heuristic: detect likely language mismatch via character-set analysis.

    Returns True if answer has significant non-Latin-1 content while
    question context is Spanish/Latin-script — indicating possible
    wrong-language answer.
    """
    if not text or not text.strip():
        return False

    # Count characters by Unicode block
    latin_count = 0
    cjk_count = 0
    for ch in text:
        cp = ord(ch)
        if cp < 0x300:  # Basic Latin, Latin-1 Supplement, Latin Extended
            latin_count += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF:
            cjk_count += 1

    total = latin_count + cjk_count
    if total == 0:
        return False

    # If >50% of non-space chars are CJK, flag as language mismatch
    return cjk_count > total * 0.5


def check_evaluability(state: EvaluatorState) -> dict:
    """Guard clause: detect non-evaluable answers before grading.

    Rules (fast, deterministic):
    1. Length < 3 non-whitespace chars → gibberish
    2. Character-set heuristic: CJK-heavy answer when question is Latin → wrong language

    Returns ``non_evaluable=True`` with a reason string on match, or
    ``non_evaluable=False`` to proceed to the evaluation node.
    """
    answers: list[dict] = state.get("answers", [])
    idx: int = state.get("current_index", 0)

    if idx >= len(answers):
        return {"non_evaluable": True, "non_evaluable_reason": "index_out_of_range"}

    current_answer = answers[idx]
    student_answer = current_answer.get("student_answer", "").strip()
    question = current_answer.get("question", "")

    # Rule 1: too short → gibberish
    non_ws = "".join(ch for ch in student_answer if not ch.isspace())
    if len(non_ws) < 3:
        return {
            "non_evaluable": True,
            "non_evaluable_reason": "gibberish",
            "evaluation": {
                "question_id": current_answer.get("question_id", ""),
                "status": "cannot_evaluate",
                "reason": "gibberish",
                "suggested_action": "Resubmit a coherent answer",
            },
        }

    # Rule 2: keyboard-smashing gibberish (vowel-ratio heuristic)
    if _is_gibberish(student_answer):
        return {
            "non_evaluable": True,
            "non_evaluable_reason": "gibberish",
            "evaluation": {
                "question_id": current_answer.get("question_id", ""),
                "status": "cannot_evaluate",
                "reason": "gibberish",
                "suggested_action": "Resubmit a coherent answer",
            },
        }

    # Rule 3: character-set heuristic for wrong language
    if _guess_language_mismatch(student_answer, question):
        return {
            "non_evaluable": True,
            "non_evaluable_reason": "language_mismatch",
            "evaluation": {
                "question_id": current_answer.get("question_id", ""),
                "status": "cannot_evaluate",
                "reason": "language_mismatch",
                "suggested_action": "Provide answer in the same language as the question",
            },
        }

    return {"non_evaluable": False, "non_evaluable_reason": ""}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: evaluate_answer
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_answer(state: EvaluatorState) -> dict:
    """Grade current answer via structured LLM call with RAG context.

    Builds a Chain-of-Thought prompt including:
    - The question text and base answer
    - The student's answer
    - Top relevant RAG chunks as reference material

    Uses ``with_structured_output(SingleEvaluation)`` with temperature=0
    for deterministic grading. Catches ``is_evaluable=False`` returned
    by the LLM itself as an additional guard.

    Decorated with ``@observe()`` for Langfuse tracing.
    """
    from src.config import settings

    answers: list[dict] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    chunks: list[dict] = state.get("retrieved_chunks", [])

    if idx >= len(answers):
        return {"errors": ["current_index out of range"], "status": "error"}

    current = answers[idx]
    question = current.get("question", "")
    base_answer = current.get("base_answer", "")
    student_answer = current.get("student_answer", "")
    topic = current.get("topic", "")

    try:
        # Build chunk context (truncated to avoid token overflow)
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:6000]

        # Adapt prompt to presence/absence of RAG chunks
        if chunk_context:
            reference_section = f"""MATERIAL DE REFERENCIA (chunks del apunte):
{chunk_context}
"""
            reference_rule = (
                "6. Cada afirmación en la justificación DEBE estar "
                "respaldada por el material de referencia."
            )
        else:
            reference_section = ""
            reference_rule = (
                "6. Evaluá basándote en tu conocimiento del tema y la respuesta esperada. "
                "Sé conservador con el puntaje ante incertidumbre."
            )

        prompt = f"""Evaluá la siguiente respuesta de un estudiante universitario.

PREGUNTA:
{question}

RESPUESTA ESPERADA (base answer):
{base_answer}

RESPUESTA DEL ESTUDIANTE:
{student_answer}

{reference_section}
INSTRUCCIONES:
1. Asigná un puntaje de 0 a 10 basado en corrección conceptual, completitud y claridad.
2. Justificá el puntaje mencionando conceptos específicos.
3. Identificá errores conceptuales concretos (lista vacía si no hay).
4. Proporcioná sugerencias de estudio accionables basadas en los errores detectados.
5. Si la respuesta es incoherente, en otro idioma, o no se puede evaluar,
   establecé is_evaluable=false.
{reference_rule}"""

        llm_cls, llm_kwargs = settings.llm_kwargs
        llm = llm_cls(**llm_kwargs)
        structured_llm = llm.with_structured_output(SingleEvaluation)

        result: SingleEvaluation = structured_llm.invoke(prompt)

        evaluation_dict = {
            "question_id": current.get("question_id", ""),
            "score": result.score,
            "justification": result.justification,
            "conceptual_errors": result.conceptual_errors,
            "suggestions": result.suggestions,
            "is_evaluable": result.is_evaluable,
            "topic": topic,
            "source_chunk_ids": current.get("source_chunk_ids", []),
            "status": "evaluated",
        }

        if not result.is_evaluable:
            evaluation_dict["status"] = "cannot_evaluate"
            evaluation_dict["reason"] = "LLM determined answer not evaluable"

        return {
            "evaluation": evaluation_dict,
            "status": "evaluated",
        }

    except Exception as exc:
        logger.exception("evaluate_answer LLM call failed")
        return {
            "evaluation": {
                "question_id": current.get("question_id", ""),
                "score": 0.0,
                "justification": "",
                "conceptual_errors": [],
                "suggestions": [],
                "is_evaluable": False,
                "topic": topic,
                "source_chunk_ids": [],
                "status": "error",
                "error": str(exc),
            },
            "errors": [f"Evaluation error: {exc}"],
            "status": "error",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: validate_feedback
# ═══════════════════════════════════════════════════════════════════════════════


def _split_claims(text: str) -> list[str]:
    """Split text into sentence-level claims for embedding comparison."""
    import re

    if not text or not text.strip():
        return []

    # Split on sentence boundaries: . ! ? followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Also try semicolon and newline splits for more granular claims
    result: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Further split long sentences on semicolons
        parts = [p.strip() for p in s.split(";") if p.strip()]
        result.extend(parts)
    return [c for c in result if len(c) > 10]  # Skip very short fragments


def validate_feedback(state: EvaluatorState) -> dict:
    """Anti-hallucination: cross-reference evaluation claims against RAG chunks.

    Algorithm:
    1. Extract sentence-level claims from ``justification`` + ``suggestions``
    2. Embed claims with SentenceTransformer
    3. Compute cosine similarity against each chunk in ``retrieved_chunks``
    4. Flag claims whose best-match similarity is below ``anti_hallucination_threshold``
    5. If any claim flagged → ``requires_review=True``
    6. Randomly sample for LLM-as-judge (default 10% rate)

    Does NOT retry — only flags. NO retry.
    """
    from src.config import settings
    from src.rag import get_embedding_model

    evaluation: dict | None = state.get("evaluation")
    chunks: list[dict] = state.get("retrieved_chunks", [])

    if not evaluation or evaluation.get("status") == "cannot_evaluate":
        return {}

    justification = evaluation.get("justification", "")
    suggestions_list: list[str] = evaluation.get("suggestions", [])

    # Combine all evaluator-generated text
    all_text = justification + " " + " ".join(suggestions_list)
    claims = _split_claims(all_text)

    # Determine judge sampling (default 10%)
    sample_rate = settings.judge_sample_rate
    sampled = random.random() < sample_rate

    if not claims or not chunks:
        evaluation["validation_warnings"] = []
        return {
            "evaluation": evaluation,
            "judge_sample": sampled,
        }

    try:
        model = get_embedding_model()

        claim_embeddings = model.encode(claims).tolist()
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = model.encode(chunk_texts).tolist()

        threshold = settings.anti_hallucination_threshold
        validation_warnings: list[dict] = []

        for i, claim in enumerate(claims):
            best_sim = 0.0
            best_chunk_id = ""
            for j, chunk_vec in enumerate(chunk_embeddings):
                sim = _cosine_sim(claim_embeddings[i], chunk_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_chunk_id = chunks[j].get("chunk_id", "")

            if best_sim < threshold:
                validation_warnings.append(
                    {
                        "claim": claim,
                        "best_match_chunk": best_chunk_id,
                        "similarity": round(best_sim, 4),
                    }
                )

        evaluation["validation_warnings"] = validation_warnings
        has_warnings = len(validation_warnings) > 0

        return {
            "evaluation": evaluation,
            "requires_review": has_warnings,
            "judge_sample": sampled,
        }

    except Exception as exc:
        logger.exception("validate_feedback embedding failed")
        evaluation["validation_warnings"] = []
        return {
            "evaluation": evaluation,
            "errors": [f"Validation error: {exc}"],
            "judge_sample": sampled,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: llm_judge (Phase 3 — stub)
# ═══════════════════════════════════════════════════════════════════════════════


def llm_judge(state: EvaluatorState) -> dict:
    """LLM-as-judge: second-pass evaluation on configurable sample rate.

    Guard:
    1. Check ``judge_sample`` flag — only run if set by validate_feedback
       or explicitly sampled.
    2. Compute ``random.random() < sample_rate`` (default 0.10).

    On sample: independent LLM call with same context, separate
    ``JudgeVerdict`` structured output. Disagreement threshold:
    ``|primary.score - judge.score| > 2.0`` → ``requires_review=True``.

    No retry — disagreement is a flag, not a correction.
    """
    from src.config import settings

    evaluation: dict | None = state.get("evaluation")
    if not evaluation or not evaluation.get("is_evaluable"):
        return {}

    if not state.get("judge_sample", False):
        return {}

    answers: list[dict] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    chunks: list[dict] = state.get("retrieved_chunks", [])

    if idx >= len(answers):
        return {}

    current = answers[idx]

    try:
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:6000]

        prompt = f"""Actuá como juez de segunda instancia. Re-evaluá la siguiente respuesta.

PREGUNTA:
{current.get('question', '')}

RESPUESTA ESPERADA:
{current.get('base_answer', '')}

RESPUESTA DEL ESTUDIANTE:
{current.get('student_answer', '')}

MATERIAL DE REFERENCIA:
{chunk_context}

EVALUACIÓN PRIMARIA (puntaje: {evaluation.get('score', 0)}):
{evaluation.get('justification', '')}

INSTRUCCIONES:
1. Asigná tu propio puntaje independiente de 0 a 10.
2. Indicá si estás de acuerdo con la evaluación primaria.
3. Si hay discrepancia significativa, explicala.
4. Si sugerís un puntaje diferente, incluilo en suggested_score."""

        llm_cls, llm_kwargs = settings.llm_kwargs
        llm = llm_cls(**llm_kwargs)
        structured_llm = llm.with_structured_output(JudgeVerdict)

        judge: JudgeVerdict = structured_llm.invoke(prompt)

        judge_dict: dict = {
            "score": judge.score,
            "agrees_with_primary": judge.agrees_with_primary,
            "discrepancy": judge.discrepancy,
            "suggested_score": judge.suggested_score,
        }

        primary_score = evaluation.get("score", 0.0)
        disagreement = abs(primary_score - judge.score)

        return {
            "judge_result": judge_dict,
            "requires_review": disagreement > 2.0,
        }

    except Exception as exc:
        logger.exception("llm_judge LLM call failed")
        return {
            "errors": [f"Judge error: {exc}"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: build_feedback
# ═══════════════════════════════════════════════════════════════════════════════


def build_feedback(state: EvaluatorState) -> dict:
    """Assemble final evaluation dict and append to ``evaluation_results``.

    Merges the evaluation dict with validation warnings and judge verdict
    into a single result object, then appends it to the accumulator.
    """
    evaluation: dict | None = state.get("evaluation")
    answers: list[dict] = state.get("answers", [])
    idx: int = state.get("current_index", 0)

    if evaluation is None:
        return {"status": "no_evaluation"}

    # Ensure question_id and topic are present
    if idx < len(answers):
        evaluation.setdefault("question_id", answers[idx].get("question_id", ""))
        evaluation.setdefault("topic", answers[idx].get("topic", ""))
        evaluation.setdefault("source_chunk_ids", answers[idx].get("source_chunk_ids", []))

    evaluation.setdefault("validation_warnings", [])
    evaluation.setdefault("requires_review", state.get("requires_review", False))

    if state.get("judge_result"):
        evaluation["judge_verdict"] = state["judge_result"]

    return {
        "evaluation_results": [evaluation],
        "evaluation": None,  # Clear for next iteration
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: next_question
# ═══════════════════════════════════════════════════════════════════════════════


def next_question(state: EvaluatorState) -> dict:
    """Increment current_index. Route back or to sync_scores."""
    idx: int = state.get("current_index", 0)
    answers: list[dict] = state.get("answers", [])
    new_idx = idx + 1

    return {
        "current_index": new_idx,
        "non_evaluable": False,
        "non_evaluable_reason": "",
        "requires_review": False,
        "judge_sample": False,
        "judge_result": None,
        "status": "done" if new_idx >= len(answers) else "processing",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: sync_scores
# ═══════════════════════════════════════════════════════════════════════════════


def sync_scores(state: EvaluatorState) -> dict:
    """Persist evaluation results to DB and mark sync complete.

    Writes each result to the ``evaluations`` table and the
    ``topic_scores`` table via the memory module. Also returns
    scores dict for optional Support Agent routing.
    """
    import asyncio
    import uuid as _uuid

    from src.memory.schema import save_evaluation

    results: list[dict] = state.get("evaluation_results", [])
    session_id: str = state.get("session_id", "")
    student_id: str = state.get("student_id", "")

    errors: list[str] = []

    for result in results:
        try:
            eval_record = {
                "id": str(_uuid.uuid4()),
                "session_id": session_id,
                "student_id": student_id,
                "question_id": result.get("question_id", ""),
                "topic": result.get("topic", ""),
                "score": result.get("score", 0.0),
                "feedback_json": "",
            }

            # Run async DB write synchronously (acceptable for graph node)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context — create task
                    asyncio.ensure_future(save_evaluation(eval_record))
                else:
                    loop.run_until_complete(save_evaluation(eval_record))
            except RuntimeError:
                # No event loop — create one
                asyncio.run(save_evaluation(eval_record))

        except Exception as exc:
            logger.warning("Failed to save evaluation for %s: %s", result.get("question_id"), exc)
            errors.append(f"DB write failed for {result.get('question_id')}: {exc}")

    if errors:
        return {"scores_synced": True, "errors": errors, "status": "synced_with_errors"}

    return {"scores_synced": True, "status": "synced"}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional routing helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _route_after_check(state: EvaluatorState) -> str:
    """Route after check_evaluability: non-evaluable skips to build_feedback."""
    if state.get("non_evaluable"):
        return "build_feedback"
    return "evaluate_answer"


def _route_after_validate(state: EvaluatorState) -> str:
    """Route after validate_feedback: sample judge or skip to build_feedback."""
    if state.get("judge_sample"):
        return "llm_judge"
    return "build_feedback"


def _route_after_next(state: EvaluatorState) -> str:
    """Route after next_question: loop back or proceed to sync."""
    answers: list[dict] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    if idx < len(answers):
        return "check_evaluability"
    return "sync_scores"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════


@observe(name="evaluator")
def build_evaluator() -> StateGraph:
    """Build the 8-node evaluator StateGraph with conditional routing.

    Topology:
      START → prepare → check_evaluability
        ├── non-eval → build_feedback → next_question
        └── evaluable → evaluate → validate_feedback
              ├── judge_sample → llm_judge → build_feedback → next_question
              └── no_judge → build_feedback → next_question
      next_question
        ├── more answers → check_evaluability (loop)
        └── done → sync_scores → END
    """
    builder = StateGraph(EvaluatorState)

    builder.add_node("prepare_evaluation", prepare_evaluation)
    builder.add_node("check_evaluability", check_evaluability)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("validate_feedback", validate_feedback)
    builder.add_node("llm_judge", llm_judge)
    builder.add_node("build_feedback", build_feedback)
    builder.add_node("next_question", next_question)
    builder.add_node("sync_scores", sync_scores)

    # Entry
    builder.add_edge(START, "prepare_evaluation")
    builder.add_edge("prepare_evaluation", "check_evaluability")

    # check_evaluability → conditional branch
    builder.add_conditional_edges(
        "check_evaluability",
        _route_after_check,
        {"build_feedback": "build_feedback", "evaluate_answer": "evaluate_answer"},
    )

    # evaluate → validate
    builder.add_edge("evaluate_answer", "validate_feedback")

    # validate → conditional: judge or skip
    builder.add_conditional_edges(
        "validate_feedback",
        _route_after_validate,
        {"llm_judge": "llm_judge", "build_feedback": "build_feedback"},
    )

    # judge → build_feedback
    builder.add_edge("llm_judge", "build_feedback")

    # build_feedback → next_question
    builder.add_edge("build_feedback", "next_question")

    # next_question → loop or end
    builder.add_conditional_edges(
        "next_question",
        _route_after_next,
        {"check_evaluability": "check_evaluability", "sync_scores": "sync_scores"},
    )

    # sync → END
    builder.add_edge("sync_scores", END)

    return builder
