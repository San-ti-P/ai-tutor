"""ExamGenerator Agent — deterministic StateGraph for generating personalized exams.

Produces source-grounded exams (MCQ + open-answer) from ingested material.
Uses pure StateGraph topology (not ReAct): batch retrieval → single structured
LLM call → claim-level embedding validation → 3-retry loop → format.
"""

from __future__ import annotations

import operator
from datetime import UTC
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ── Pydantic structured-output models ────────────────────────────────────────


class MCQQuestion(BaseModel):
    """Multiple-choice question with stem, options, and source grounding."""

    stem: str = Field(description="Question stem (the question text)")
    options: list[str] = Field(description="3-5 answer options, exactly one correct")
    correct_option_index: int = Field(description="0-based index of the correct option")
    source_chunk_ids: list[str] = Field(description="ChromaDB chunk IDs supporting this question")
    difficulty: str = Field(description="'easy' | 'medium' | 'hard'")
    topic: str = Field(description="Primary topic this question covers")


class OpenAnswerQuestion(BaseModel):
    """Open-ended question with base answer for evaluator grading."""

    prompt: str = Field(description="Open-ended question prompt")
    base_answer: str = Field(description="Expected answer — used by Evaluator (Epic 5) for grading")
    key_points: list[str] = Field(description="3-5 key points the answer should include")
    source_chunk_ids: list[str] = Field(description="ChromaDB chunk IDs supporting this question")
    difficulty: str = Field(description="'easy' | 'medium' | 'hard'")
    topic: str = Field(description="Primary topic this question covers")


class ExamGeneration(BaseModel):
    """Batch exam generation — all questions in a single structured LLM call."""

    mcq_questions: list[MCQQuestion] = Field(default_factory=list)
    open_questions: list[OpenAnswerQuestion] = Field(default_factory=list)
    metadata: dict = Field(
        default_factory=dict,
        description="{topics_covered: [...], total_source_chunks: N}",
    )


class ClaimCheck(BaseModel):
    """Single atomic claim extracted from a question, with its best chunk match."""

    claim_text: str
    source: str  # "stem", "option", "base_answer", "key_point"
    question_index: int
    best_chunk_id: str | None = None
    similarity_score: float = 0.0
    matched: bool = False


class ValidationResult(BaseModel):
    """Per-question validation result: all claims, pass/fail, matched chunks."""

    question_index: int
    valid: bool
    claims_checked: list[ClaimCheck] = Field(default_factory=list)
    missing_claims: list[str] = Field(default_factory=list)
    matched_chunk_ids: list[str] = Field(default_factory=list)


# ── State schema ─────────────────────────────────────────────────────────────


class ExamGeneratorState(TypedDict):
    session_id: str
    student_id: str
    topics: list[str]
    difficulty: str
    question_count: int
    mcq_ratio: float
    student_profile: dict | None
    collection_name: str
    retrieved_chunks: Annotated[list[dict], operator.add]
    generated_questions: list[dict]
    validation_results: list[dict]
    validation_errors: Annotated[list[str], operator.add]
    invalid_question_indices: list[int]
    omitted_questions: Annotated[list[int], operator.add]
    retry_count: int
    topic_not_found: list[str]
    topic_suggestions: list[str]
    exam: dict
    status: str


# ── Node implementations ─────────────────────────────────────────────────────


def retrieve_relevant_chunks(state: ExamGeneratorState) -> dict:
    """Retrieve top-K relevant chunks from ChromaDB for the requested topics.

    Iterates state["topics"], calling retrieve_chunks per topic. Handles empty
    results by querying ThematicIndex for close-topic suggestions. Deduplicates
    by chunk_id. Weak topics from student_profile get 2× retrieval weight.
    """
    import logging

    from src.tools import retrieve_chunks as _retrieve_chunks

    logger = logging.getLogger(__name__)

    try:
        topics: list[str] = state.get("topics", [])
        session_id: str = state.get("session_id", "")
        collection_name = state.get("collection_name") or f"session_{session_id}"
        student_profile = state.get("student_profile")

        # Determine topics with optional weak-topic boosting
        weak_topics: list[str] = []
        if student_profile and isinstance(student_profile, dict):
            weak_topics = student_profile.get("weak_topics", [])

        # Build weighted topic list: weak topics appear twice
        weighted_topics: list[str] = list(topics)
        for wt in weak_topics:
            if wt not in topics:
                weighted_topics.append(wt)
            weighted_topics.append(wt)  # 2× weight

        # If no topics at all, use weak topics
        if not weighted_topics and weak_topics:
            weighted_topics = list(set(weak_topics))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_topics: list[str] = []
        for t in weighted_topics:
            if t not in seen:
                seen.add(t)
                unique_topics.append(t)

        # Retrieve chunks for each topic
        all_chunks: list[dict] = []
        topic_not_found: list[str] = []
        seen_chunk_ids: set[str] = set()

        for topic in unique_topics:
            chunks = _retrieve_chunks.invoke(
                {
                    "query": topic,
                    "top_k": 5,
                    "collection_name": collection_name,
                }
            )
            if not chunks:
                topic_not_found.append(topic)
            else:
                for chunk in chunks:
                    cid = chunk.get("chunk_id", "")
                    if cid and cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        all_chunks.append(chunk)

        # For missing topics, query ThematicIndex for suggestions
        topic_suggestions: list[str] = []
        if topic_not_found:
            try:
                from src.rag import ThematicIndex as _ThematicIndex

                # Build a temporary index from chunk metadata
                ti = _ThematicIndex()
                for chunk in all_chunks:
                    meta_topic = (
                        chunk.get("metadata", {}).get("topic", "")
                        if isinstance(chunk.get("metadata"), dict)
                        else ""
                    )
                    if meta_topic:
                        ti.add_topics([meta_topic])

                for missing in topic_not_found:
                    # Try searching with each segment of the missing topic path
                    parts = missing.split("/")
                    for i in range(len(parts)):
                        prefix = "/".join(parts[: i + 1])
                        if i == 0:
                            suggestions = list(ti.to_dict().keys())[:3]
                        else:
                            suggestions = ti.search(prefix)
                        if suggestions:
                            topic_suggestions.extend(suggestions[:3])
                            break

                topic_suggestions = list(dict.fromkeys(topic_suggestions))[:3]
            except Exception:
                logger.debug("ThematicIndex suggestion lookup failed", exc_info=True)

        return {
            "retrieved_chunks": all_chunks,
            "topic_not_found": topic_not_found,
            "topic_suggestions": topic_suggestions,
            "collection_name": collection_name,
            "status": "retrieved" if all_chunks else "no_material",
        }

    except Exception as exc:
        logger.exception("retrieve_relevant_chunks failed")
        return {
            "validation_errors": [f"Retrieval error: {exc}"],
            "status": "error",
        }


def generate_questions(state: ExamGeneratorState) -> dict:
    """Generate exam questions via a single structured LLM call.

    Builds a prompt from retrieved chunks, user preferences, student profile,
    and requested topics/difficulty. Uses ChatGroq with structured_output to
    get a deterministic ExamGeneration. On retry, only invalid slots are
    regenerated — valid questions from previous passes are preserved.
    """
    import logging

    from src.config import settings

    logger = logging.getLogger(__name__)

    try:
        chunks: list[dict] = state.get("retrieved_chunks", [])
        # Guard: no chunks → cannot generate
        if not chunks:
            return {
                "generated_questions": [],
                "status": "no_material",
            }

        question_count: int = state.get("question_count", 5)
        mcq_ratio: float = state.get("mcq_ratio", 0.5)
        difficulty: str = state.get("difficulty", "medium")
        student_profile = state.get("student_profile")
        retry_count: int = state.get("retry_count", 0)
        invalid_indices: list[int] = state.get("invalid_question_indices", [])
        existing_questions: list = state.get("generated_questions", [])
        validation_errors: list[str] = state.get("validation_errors", [])

        # On retry: only generate replacements for invalid slots
        target_count = (
            len(invalid_indices) if retry_count > 0 and invalid_indices else question_count
        )

        if target_count <= 0:
            return {"generated_questions": existing_questions, "status": "generated"}

        # Build chunk context
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:8000]  # Truncate to avoid token overflow

        # Build preferences section
        prefs_lines: list[str] = []
        mcq_n = max(1, round(target_count * mcq_ratio))
        open_n = target_count - mcq_n
        prefs_lines.append(f"Total questions: {target_count}")
        prefs_lines.append(f"MCQ questions: {mcq_n}")
        prefs_lines.append(f"Open-answer questions: {open_n}")
        prefs_lines.append(f"Difficulty: {difficulty}")

        if student_profile and isinstance(student_profile, dict):
            weak = student_profile.get("weak_topics", [])
            if weak:
                prefs_lines.append(f"Prioritize weak topics: {', '.join(weak)}")

        # Retry instructions
        retry_instructions = ""
        if retry_count > 0 and invalid_indices:
            retry_instructions = (
                f"\nIMPORTANTE: Solo generá preguntas para los índices {invalid_indices}. "
                f"Las preguntas anteriores fallaron validación porque sus afirmaciones "
                f"no coincidían con los chunks fuente. Asegurate de que CADA afirmación "
                f"provenga directamente de un chunk etiquetado con [CHUNK:...].\n"
                f"Errores anteriores: {'; '.join(validation_errors[-5:])}"
            )

        header = (
            "Generá un examen académico basado EXCLUSIVAMENTE en "
            "los siguientes chunks de material de estudio."
        )
        req_open = (
            "- Para open-answer: prompts que requieran explicación "
            "(no sí/no), incluir base_answer y 3-5 key_points."
        )
        prompt = f"""{header}

{chunk_context}

PREFERENCIAS:
{chr(10).join(prefs_lines)}{retry_instructions}

REQUISITOS:
- Cada pregunta DEBE basarse en hechos textuales de los chunks provistos.
- Para MCQs: 3-5 opciones, exactamente una correcta, distractores plausibles.
{req_open}
- Incluí source_chunk_ids (los IDs entre corchetes [CHUNK:xxx]) por cada pregunta.
- Cada pregunta debe tener los campos: topic y difficulty.
"""

        llm_cls, llm_kwargs = settings.llm_kwargs
        llm = llm_cls(**llm_kwargs)
        structured_llm = llm.with_structured_output(ExamGeneration)
        result: ExamGeneration = structured_llm.invoke(prompt)

        # Convert Pydantic models to dicts
        new_questions: list[dict] = []
        for mcq in result.mcq_questions:
            q = mcq.model_dump()
            q["type"] = "mcq"
            new_questions.append(q)
        for oa in result.open_questions:
            q = oa.model_dump()
            q["type"] = "open_answer"
            new_questions.append(q)

        # On retry: merge new questions into existing, replacing invalid slots
        if retry_count > 0 and invalid_indices and existing_questions:
            merged = list(existing_questions)
            for idx, new_q in zip(invalid_indices, new_questions):
                if idx < len(merged):
                    merged[idx] = new_q
                else:
                    merged.append(new_q)
            final_questions = merged
        else:
            final_questions = new_questions

        # Increment retry_count on retry path (conditional edge cannot mutate state)
        next_retry = retry_count + 1 if retry_count > 0 else retry_count

        return {
            "generated_questions": final_questions,
            "retry_count": next_retry,
            "status": "generated",
        }

    except Exception as exc:
        logger.exception("generate_questions failed")
        return {
            "validation_errors": [f"Generation error: {exc}"],
            "status": "error",
        }


def validate_questions(state: ExamGeneratorState) -> dict:
    """Validate every question by checking claims against source chunks.

    Extracts atomic claims from each question (stem + options for MCQ,
    base_answer + key_points for open-answer). Embeds claims and compares
    cosine similarity against chunk embeddings. Claims below the
    anti_hallucination_threshold are flagged as errors.
    """
    import logging
    import math
    import re

    from src.config import settings
    from src.rag import get_embedding_model

    logger = logging.getLogger(__name__)

    try:
        chunks: list[dict] = state.get("retrieved_chunks", [])
        questions: list[dict] = state.get("generated_questions", [])
        threshold: float = settings.anti_hallucination_threshold
        model = get_embedding_model()

        if not questions:
            return {
                "validation_results": [],
                "validation_errors": [],
                "invalid_question_indices": [],
            }

        # Pre-embed all chunk texts (batch)
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = model.encode(chunk_texts).tolist() if chunk_texts else []

        validation_results: list[dict] = []
        all_errors: list[str] = []
        invalid_indices: list[int] = []

        # Regex to split on sentence boundaries
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")

        for qi, question in enumerate(questions):
            qtype = question.get("type", "")
            claims: list[str] = []

            if qtype == "mcq":
                stem = question.get("stem", "")
                claims.extend(
                    c.strip() for c in sentence_split_re.split(stem) if len(c.strip()) >= 20
                )
                for opt in question.get("options", []):
                    opt_clean = opt.strip()
                    if len(opt_clean) >= 20:
                        claims.append(opt_clean)
            else:  # open_answer
                base = question.get("base_answer", "")
                claims.extend(
                    c.strip() for c in sentence_split_re.split(base) if len(c.strip()) >= 20
                )
                for kp in question.get("key_points", []):
                    kp_clean = kp.strip()
                    if len(kp_clean) >= 10:
                        claims.append(kp_clean)

            # If no claims extracted, try using the full stem/base_answer
            if not claims:
                if qtype == "mcq":
                    claims = [question.get("stem", "")]
                else:
                    claims = [question.get("base_answer", "")]
                claims = [c for c in claims if len(c.strip()) >= 10]

            claim_checks: list[dict] = []
            missing: list[str] = []
            matched_chunk_ids: list[str] = []
            all_matched = True

            for claim in claims:
                claim_embedding = model.encode([claim]).tolist()[0]

                best_score = 0.0
                best_chunk_id = None

                for ci, chunk_emb in enumerate(chunk_embeddings):
                    # Cosine similarity: dot product of normalized vectors
                    dot = sum(a * b for a, b in zip(claim_embedding, chunk_emb))
                    norm_a = math.sqrt(sum(a * a for a in claim_embedding))
                    norm_b = math.sqrt(sum(b * b for b in chunk_emb))
                    if norm_a > 0 and norm_b > 0:
                        sim = dot / (norm_a * norm_b)
                    else:
                        sim = 0.0
                    if sim > best_score:
                        best_score = sim
                        best_chunk_id = chunks[ci].get("chunk_id", "")

                matched = best_score >= threshold
                if not matched:
                    all_matched = False
                    missing.append(claim[:80])

                if best_chunk_id and best_chunk_id not in matched_chunk_ids:
                    matched_chunk_ids.append(best_chunk_id)

                claim_checks.append(
                    {
                        "claim_text": claim[:200],
                        "best_chunk_id": best_chunk_id,
                        "similarity_score": round(best_score, 4),
                        "matched": matched,
                    }
                )

            vr = {
                "question_index": qi,
                "valid": all_matched,
                "claims_checked": claim_checks,
                "missing_claims": missing,
                "matched_chunk_ids": matched_chunk_ids,
            }
            validation_results.append(vr)

            if not all_matched:
                invalid_indices.append(qi)
                all_errors.append(
                    f"Question {qi}: {len(missing)} claim(s) not found in source chunks"
                )

        return {
            "validation_results": validation_results,
            "validation_errors": all_errors,
            "invalid_question_indices": invalid_indices,
        }

    except Exception as exc:
        logger.exception("validate_questions failed")
        return {
            "validation_errors": [f"Validation error: {exc}"],
        }


def should_retry(state: ExamGeneratorState) -> str:
    """Return 'retry' if validation errors exist AND retry_count < 3, else 'done'.

    Does NOT retry on retrieval errors (status='error' or 'no_material') —
    those are terminal for this invocation.
    """
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)
    status = state.get("status", "")
    if status in ("error", "no_material"):
        return "done"
    if errors and retry_count < 3:
        return "retry"
    return "done"


def format_exam(state: ExamGeneratorState) -> dict:
    """Package validated questions into final exam dict with metadata.

    Removes omitted questions, computes topic coverage, adds warnings,
    and sets status (complete | partial | no_material).
    """
    import uuid as _uuid
    from datetime import datetime

    try:
        questions: list[dict] = state.get("generated_questions", [])
        omitted_indices: list[int] = state.get("omitted_questions", [])
        validation_errors: list[str] = state.get("validation_errors", [])
        topic_not_found: list[str] = state.get("topic_not_found", [])
        topic_suggestions: list[str] = state.get("topic_suggestions", [])

        # Filter out omitted questions
        omitted_set = set(omitted_indices)
        final_questions = [q for i, q in enumerate(questions) if i not in omitted_set]

        # Compute topics covered
        topics_covered = list(
            dict.fromkeys(q.get("topic", "") for q in final_questions if q.get("topic"))
        )

        # Determine status
        if not final_questions:
            exam_status = "no_material"
        elif topic_not_found and not topics_covered:
            exam_status = "no_material"
        elif omitted_indices or validation_errors:
            exam_status = "partial"
        else:
            exam_status = "complete"

        exam = {
            "exam_id": str(_uuid.uuid4()),
            "session_id": state.get("session_id", ""),
            "student_id": state.get("student_id", ""),
            "generated_at": datetime.now(UTC).isoformat(),
            "total_questions": len(final_questions),
            "questions": final_questions,
            "topics_covered": topics_covered,
            "source_chunks_total": len(state.get("retrieved_chunks", [])),
            "omitted_count": len(omitted_indices),
            "topic_not_found": topic_not_found,
            "topic_suggestions": topic_suggestions,
            "status": exam_status,
            "warnings": validation_errors,
        }

        return {"exam": exam, "status": exam_status}
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("format_exam failed")
        return {
            "exam": {"error": str(exc), "status": "error"},
            "status": "error",
            "validation_errors": [f"Format error: {exc}"],
        }


def build_exam_generator() -> StateGraph:
    """Build and return the ExamGenerator LangGraph."""
    builder = StateGraph(ExamGeneratorState)

    builder.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("validate_questions", validate_questions)
    builder.add_node("format_exam", format_exam)

    builder.add_edge(START, "retrieve_relevant_chunks")
    builder.add_edge("retrieve_relevant_chunks", "generate_questions")
    builder.add_edge("generate_questions", "validate_questions")
    builder.add_conditional_edges(
        "validate_questions",
        should_retry,
        {"retry": "generate_questions", "done": "format_exam"},
    )
    builder.add_edge("format_exam", END)

    return builder
