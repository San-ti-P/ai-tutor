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

import json
import logging
import operator
import random
from typing import Annotated, Any

from src.config import settings

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── Pydantic structured-output models ────────────────────────────────────────


class SingleEvaluation(BaseModel):
    """LLM structured output for a single answer evaluation."""

    score: float = Field(ge=0, le=10, description="Numerical score from 0 to 10")
    justification: str = Field(description="Detailed justification referencing specific concepts")
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
    answers: list[dict[str, Any]]
    # Each answer dict: question_id, question, base_answer, student_answer,
    # answer_image (optional), source_chunk_ids, topic, difficulty
    current_index: int
    answer_text: str

    # ── OCR passthrough ──
    ocr_extracted_text: str | None
    ocr_confidence: float

    # ── RAG chunks (accumulated across topics) ──
    retrieved_chunks: Annotated[list[dict[str, Any]], operator.add]
    collection_name: str  # Override for session_{session_id}; used by prepare_evaluation

    # ── per-question evaluation state ──
    evaluation: dict[str, Any] | None
    evaluation_results: Annotated[list[dict[str, Any]], operator.add]

    # ── non-evaluable guard ──
    non_evaluable: bool
    non_evaluable_reason: str

    # ── LLM-as-judge ──
    judge_sample: bool
    judge_result: dict[str, Any] | None
    requires_review: bool

    # ── finalisation ──
    scores_synced: bool
    errors: Annotated[list[str], operator.add]
    status: str


def _deep_merge_tree_ev(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Deep-merge nested dict, mutating target."""
    for key, value in source.items():
        if key not in target:
            target[key] = {}
        if isinstance(value, dict) and isinstance(target[key], dict):
            _deep_merge_tree_ev(target[key], value)


# ═══════════════════════════════════════════════════════════════════════════════
# Node: prepare_evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_evaluation(state: EvaluatorState) -> dict[str, Any]:
    """Deduplicate topics from answers, retrieve RAG chunks per topic.

    Queries ChromaDB for each unique topic across the answer batch and
    accumulates retrieved chunks via the ``retrieved_chunks`` reducer.
    No LLM call — pure retrieval.
    """
    from src.tools import retrieve_chunks as _retrieve_chunks
    from src.config import settings

    answers: list[dict[str, Any]] = state.get("answers", [])
    session_id: str = state.get("session_id", "")
    collection_name = state.get("collection_name") or f"session_{session_id}"

    # Load topic descriptions from session files
    topic_descriptions: dict[str, str] | None = None
    topic_tree: dict[str, Any] | None = None
    if session_id:
        from src.memory.schema import list_session_files
        from src.utils.async_ import run_async_in_sync
        import json as _json

        try:
            session_files = run_async_in_sync(list_session_files(session_id))
            _all_descs: dict[str, str] = {}
            _merged_tree: dict[str, Any] = {}
            for sf in session_files:
                descs_json = sf.get("topic_descriptions_json")
                if descs_json:
                    try:
                        sf_descs = _json.loads(descs_json)
                        if isinstance(sf_descs, dict):
                            for k, v in sf_descs.items():
                                if isinstance(v, str) and v.strip():
                                    _all_descs[k] = v
                    except Exception:
                        pass
                tree_json = sf.get("topic_tree_json")
                if tree_json:
                    try:
                        sf_tree = _json.loads(tree_json)
                        if isinstance(sf_tree, dict) and sf_tree:
                            _deep_merge_tree_ev(_merged_tree, sf_tree)
                    except Exception:
                        pass
            if _all_descs:
                topic_descriptions = _all_descs
            if _merged_tree:
                topic_tree = _merged_tree
        except Exception:
            pass

    # Deduplicate topics preserving order
    seen_topics: set[str] = set()
    unique_topics: list[str] = []
    for ans in answers:
        topic = ans.get("topic", "")
        if topic and topic not in seen_topics:
            seen_topics.add(topic)
            unique_topics.append(topic)

    all_chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for topic in unique_topics:
        try:
            chunks = _retrieve_chunks.invoke(
                {
                    "query": topic,
                    "top_k": settings.retrieval_top_k,
                    "collection_name": collection_name,
                    "topic_descriptions": topic_descriptions,
                    "topic_tree": topic_tree,
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


def check_evaluability(state: EvaluatorState) -> dict[str, Any]:
    """Guard clause: detect non-evaluable answers before grading.

    Rules (fast, deterministic):
    1. Length < 3 non-whitespace chars → gibberish
    2. Character-set heuristic: CJK-heavy answer when question is Latin → wrong language

    Returns ``non_evaluable=True`` with a reason string on match, or
    ``non_evaluable=False`` to proceed to the evaluation node.
    """
    answers: list[dict[str, Any]] = state.get("answers", [])
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


def evaluate_answer(state: EvaluatorState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Grade current answer via structured LLM call with RAG context.

    Builds a Chain-of-Thought prompt including:
    - The question text and base answer
    - The student's answer
    - Top relevant RAG chunks as reference material

    Uses ``with_structured_output(SingleEvaluation)`` with temperature=0
    for deterministic grading. Catches ``is_evaluable=False`` returned
    by the LLM itself as an additional guard.

    When ``chunk_context`` is empty, short-circuits to ``cannot_evaluate``
    with ``non_evaluable_reason="no_material"`` — no LLM call is made.

    Decorated with ``@observe()`` for Langfuse tracing.
    """
    from src.llm import get_structured_llm
    from src.rag.policy import EVALUATOR_SYSTEM_PROMPT, RAG_ONLY_SYSTEM_PROMPT, no_material_message

    answers: list[dict[str, Any]] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])

    logger.info(
        "[evaluate_answer] START | session=%s | answers=%d | idx=%d",
        state["session_id"],
        len(answers),
        idx,
    )

    if idx >= len(answers):
        return {"errors": ["current_index out of range"], "status": "error"}

    current = answers[idx]
    qtype = current.get("type", "open")
    question = current.get("question", "")
    base_answer = current.get("base_answer", "") or ""
    student_answer = current.get("student_answer", "") or ""
    topic = current.get("topic", "")

    # MCQ evaluation logic (deterministic, bypass LLM call)
    if qtype == "mcq":
        clean_student = student_answer.strip()
        clean_base = base_answer.strip()
        is_correct = (
            clean_student.lower() == clean_base.lower() if clean_student and clean_base else False
        )
        score = 10.0 if is_correct else 0.0

        if is_correct:
            justification = f"Respuesta correcta. Seleccionaste la opción correcta: '{clean_base}'."
            conceptual_errors = []
            suggestions = []
        else:
            if clean_student:
                justification = f"Respuesta incorrecta. Seleccionaste '{clean_student}' pero la opción correcta era '{clean_base}'."
            else:
                justification = f"Respuesta incorrecta. No seleccionaste ninguna opción. La opción correcta era '{clean_base}'."
            conceptual_errors = ["Selección de opción incorrecta"]
            suggestions = ["Repasar el material de lectura y reintentar el examen."]

        evaluation_dict = {
            "question_id": current.get("question_id", ""),
            "score": score,
            "justification": justification,
            "conceptual_errors": conceptual_errors,
            "suggestions": suggestions,
            "is_evaluable": True,
            "topic": topic,
            "source_chunk_ids": current.get("source_chunk_ids", []),
            "status": "evaluated",
        }
        return {
            "evaluation": evaluation_dict,
            "status": "evaluated",
        }

    # Build numbered fragments (no IDs so they don't appear in LLM output)
    fragment_texts = [c.get("text", "") for c in chunks]
    chunk_context = "\n\n".join(
        f"--- Fragmento {i + 1} ---\n{text}" for i, text in enumerate(fragment_texts)
    )[:6000]

    # Guard: no chunks → short-circuit to cannot_evaluate (no LLM call)
    if not chunk_context:
        return {
            "evaluation": {
                "question_id": current.get("question_id", ""),
                "score": 0.0,
                "justification": no_material_message(),
                "conceptual_errors": [],
                "suggestions": [],
                "is_evaluable": False,
                "non_evaluable_reason": "no_material",
                "requires_review": False,
                "topic": topic,
                "source_chunk_ids": current.get("source_chunk_ids", []),
                "source_chunks": [],
                "status": "cannot_evaluate",
            },
            "status": "cannot_evaluate",
        }

    try:
        reference_section = f"""MATERIAL DE REFERENCIA (chunks del apunte):
{chunk_context}
"""

        prompt = f"""{EVALUATOR_SYSTEM_PROMPT}

Evaluá la siguiente respuesta de un estudiante universitario. Respondé SIEMPRE en español.

PREGUNTA:
{question}

RESPUESTA ESPERADA (base answer):
{base_answer}

RESPUESTA DEL ESTUDIANTE:
{student_answer}

{reference_section}
INSTRUCCIONES:
1. Asigná un puntaje de 0 a 10 basado en corrección conceptual, completitud y claridad.
2. Justificá el puntaje mencionando conceptos específicos, en español.
3. Identificá errores conceptuales concretos (lista vacía si no hay).
4. Proporcioná sugerencias de estudio accionables basadas en los errores detectados.
5. Si la respuesta es incoherente, en otro idioma, o no se puede evaluar,
   establecé is_evaluable=false.
6. Usa el material de referencia como contexto para verificar conceptos, pero la
   fuente principal para el puntaje es la comparacion con la RESPUESTA ESPERADA."""

        structured_llm = get_structured_llm(
            SingleEvaluation, temperature=settings.evaluator_temperature
        )

        invoke_kwargs = {"config": config} if config is not None else {}
        result: SingleEvaluation = structured_llm.invoke(prompt, **invoke_kwargs)

        evaluation_dict = {
            "question_id": current.get("question_id", ""),
            "score": result.score,
            "justification": result.justification,
            "conceptual_errors": result.conceptual_errors,
            "suggestions": result.suggestions,
            "is_evaluable": result.is_evaluable,
            "topic": topic,
            "source_chunk_ids": current.get("source_chunk_ids", []),
            "source_chunks": fragment_texts,
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


def validate_feedback(state: EvaluatorState) -> dict[str, Any]:
    """Anti-hallucination: cross-reference evaluation claims against RAG chunks.

    Algorithm:
    1. Extract sentence-level claims from ``justification`` + ``suggestions``
       via ``src.utils.text.split_into_claims``
    2. Delegate to ``validate_claim_grounding`` (flag_only mode) for
       batch embedding and cosine similarity
    3. Flag claims whose best-match similarity is below threshold
    4. If any claim flagged → ``requires_review=True``
    5. Randomly sample for LLM-as-judge (default 10% rate)

    Does NOT retry — only flags. NO retry.
    """
    from src.config import settings
    from src.tools.validate_claim_grounding import validate_claim_grounding
    from src.utils.text import split_into_claims

    evaluation: dict[str, Any] | None = state.get("evaluation")
    chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])

    answers: list[dict[str, Any]] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    if idx < len(answers):
        qtype = answers[idx].get("type", "open")
        if qtype == "mcq":
            return {
                "evaluation": {**evaluation, "validation_warnings": []} if evaluation else {},
                "requires_review": False,
                "judge_sample": False,
            }

    if not evaluation or evaluation.get("status") == "cannot_evaluate":
        return {}

    justification = evaluation.get("justification", "")
    suggestions_list: list[str] = evaluation.get("suggestions", [])

    # Combine all evaluator-generated text
    all_text = justification + " " + " ".join(suggestions_list)
    claims = split_into_claims(all_text)

    # Determine judge sampling (default 10%)
    sample_rate = settings.judge_sample_rate
    sampled = random.random() < sample_rate

    if not claims or not chunks:
        return {
            "evaluation": {**evaluation, "validation_warnings": []},
            "judge_sample": sampled,
        }

    try:
        # Delegate to anti-hallucination tool (flag_only — no retry)
        result = validate_claim_grounding.invoke(
            {
                "claims": claims,
                "chunks": chunks,
                "mode": "flag_only",
            }
        )

        validation_warnings: list[dict[str, Any]] = []
        for cr in result.get("claim_results", []):
            if not cr["matched"]:
                validation_warnings.append(
                    {
                        "claim": cr["claim_text"],
                        "best_match_chunk": cr["best_chunk_id"],
                        "similarity": cr["similarity_score"],
                    }
                )

        has_warnings = len(validation_warnings) > 0
        updated_evaluation = {**evaluation, "validation_warnings": validation_warnings}

        return {
            "evaluation": updated_evaluation,
            "requires_review": has_warnings,
            "judge_sample": sampled,
        }

    except Exception as exc:
        logger.exception("validate_feedback embedding failed")
        return {
            "evaluation": {**evaluation, "validation_warnings": []},
            "errors": [f"Validation error: {exc}"],
            "judge_sample": sampled,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: llm_judge (Phase 3 — stub)
# ═══════════════════════════════════════════════════════════════════════════════


def llm_judge(state: EvaluatorState, config: RunnableConfig | None = None) -> dict[str, Any]:
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
    from src.llm import get_structured_llm

    evaluation: dict[str, Any] | None = state.get("evaluation")
    if not evaluation or not evaluation.get("is_evaluable"):
        return {}

    if not state.get("judge_sample", False):
        return {}

    answers: list[dict[str, Any]] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])

    if idx >= len(answers):
        return {}

    current = answers[idx]

    try:
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:6000]

        prompt = f"""Actuá como juez de segunda instancia. Re-evaluá la siguiente respuesta.
Respondé SIEMPRE en español.

PREGUNTA:
{current.get("question", "")}

RESPUESTA ESPERADA:
{current.get("base_answer", "")}

RESPUESTA DEL ESTUDIANTE:
{current.get("student_answer", "")}

MATERIAL DE REFERENCIA:
{chunk_context}

EVALUACIÓN PRIMARIA (puntaje: {evaluation.get("score", 0)}):
{evaluation.get("justification", "")}

INSTRUCCIONES:
1. Asigná tu propio puntaje independiente de 0 a 10.
2. Indicá si estás de acuerdo con la evaluación primaria.
3. Si hay discrepancia significativa, explicala en español.
4. Si sugerís un puntaje diferente, incluilo en suggested_score."""

        structured_llm = get_structured_llm(
            JudgeVerdict, temperature=settings.evaluator_temperature
        )

        invoke_kwargs = {"config": config} if config is not None else {}
        judge: JudgeVerdict = structured_llm.invoke(prompt, **invoke_kwargs)

        judge_dict: dict[str, Any] = {
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


def build_feedback(state: EvaluatorState) -> dict[str, Any]:
    """Assemble final evaluation dict and append to ``evaluation_results``.

    Merges the evaluation dict with validation warnings and judge verdict
    into a single result object, then appends it to the accumulator.
    """
    evaluation: dict[str, Any] | None = state.get("evaluation")
    answers: list[dict[str, Any]] = state.get("answers", [])
    idx: int = state.get("current_index", 0)

    if evaluation is None:
        return {"status": "no_evaluation"}

    # Ensure question_id, topic, and answer context are present
    if idx < len(answers):
        evaluation.setdefault("question_id", answers[idx].get("question_id", ""))
        evaluation.setdefault("topic", answers[idx].get("topic", ""))
        evaluation.setdefault("source_chunk_ids", answers[idx].get("source_chunk_ids", []))
        evaluation.setdefault("question_text", answers[idx].get("question", ""))
        evaluation.setdefault("student_answer", answers[idx].get("student_answer", ""))

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


def next_question(state: EvaluatorState) -> dict[str, Any]:
    """Increment current_index. Route back or to sync_scores."""
    idx: int = state.get("current_index", 0)
    answers: list[dict[str, Any]] = state.get("answers", [])
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


def sync_scores(state: EvaluatorState) -> dict[str, Any]:
    """Persist evaluation results to DB."""
    import uuid as _uuid

    from src.memory.schema import save_evaluation, upsert_topic_scores
    from src.utils.async_ import run_async_in_sync

    results: list[dict[str, Any]] = state.get("evaluation_results", [])
    session_id: str = state.get("session_id", "")
    student_id: str = state.get("student_id", "")
    exam_id: str = state.get("exam_id", "")

    errors: list[str] = []
    topic_score_pairs: list[dict[str, Any]] = []

    for result in results:
        try:
            feedback = {
                "justification": result.get("justification", ""),
                "conceptual_errors": result.get("conceptual_errors", []),
                "suggestions": result.get("suggestions", []),
                "is_evaluable": result.get("is_evaluable", True),
                "non_evaluable_reason": result.get("non_evaluable_reason", ""),
                "requires_review": result.get("requires_review", False),
                "judge_score": (
                    result.get("judge_verdict", {}).get("score")
                    if isinstance(result.get("judge_verdict"), dict)
                    else None
                ),
                "question_text": result.get("question_text", ""),
                "student_answer": result.get("student_answer", ""),
                "source_chunks": result.get("source_chunks", []),
            }
            eval_record = {
                "id": str(_uuid.uuid4()),
                "session_id": session_id,
                "student_id": student_id,
                "question_id": result.get("question_id", ""),
                "exam_id": exam_id,
                "topic": result.get("topic", ""),
                "score": result.get("score", 0.0),
                "feedback_json": json.dumps(feedback, ensure_ascii=False),
            }
            run_async_in_sync(save_evaluation(eval_record))

            topic = result.get("topic", "")
            score = result.get("score", 0.0)
            if topic and result.get("status") != "cannot_evaluate":
                topic_score_pairs.append({"topic": topic, "score": score})
        except Exception as exc:
            logger.exception("sync_scores failed for question %s", result.get("question_id", ""))
            errors.append(f"Save error: {exc}")

    if student_id and topic_score_pairs:
        try:
            run_async_in_sync(upsert_topic_scores(student_id, session_id, topic_score_pairs))
        except Exception as exc:
            logger.exception("sync_scores topic_scores upsert failed")
            errors.append(f"Topic scores error: {exc}")

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
    answers: list[dict[str, Any]] = state.get("answers", [])
    idx: int = state.get("current_index", 0)
    if idx < len(answers):
        return "check_evaluability"
    return "sync_scores"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_evaluator() -> StateGraph[EvaluatorState, Any, Any]:
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
