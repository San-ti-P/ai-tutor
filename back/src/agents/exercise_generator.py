"""ExerciseGenerator Agent — deterministic StateGraph for generating practice exercises.

Mirrors ExamGenerator topology: START → retrieve → generate → validate →
[conditional: retry → generate | done → format] → END.
Produces complex practical exercises with multi-step model solutions grounded
in source chunks. Anti-hallucination claim-level embedding validation with
3-retry loop.
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ── Pydantic structured-output models ────────────────────────────────────────


class ExerciseStep(BaseModel):
    """A single step in the multi-step model solution."""

    step_number: int = Field(description="Sequential step number (1-based)")
    description: str = Field(description="What to do in this step")
    result: str = Field(description="Intermediate result of this step")
    source_chunk_ids: list[str] = Field(description="Chunk IDs supporting this step")


class ModelSolution(BaseModel):
    """Complete model solution with ordered steps and final answer."""

    steps: list[ExerciseStep] = Field(description="Ordered solution steps (3-6 steps)")
    final_answer: str = Field(description="Final concise answer to the problem")
    key_concepts: list[str] = Field(description="3-5 key concepts the exercise teaches")
    source_chunk_ids: list[str] = Field(description="All chunk IDs referenced across steps")


class PracticalExercise(BaseModel):
    """A single practical exercise with statement, data, question, and model solution."""

    statement: str = Field(description="Problem statement / context")
    given_data: str = Field(description="Provided values, formulas, or data")
    question: str = Field(description="What the student must solve")
    difficulty: str = Field(description="'easy' | 'medium' | 'hard'")
    topic: str = Field(description="Primary topic path (e.g. 'cálculo/derivadas')")
    source_chunk_ids: list[str] = Field(description="Chunk IDs grounding the exercise")
    model_solution: ModelSolution


class ExerciseGeneration(BaseModel):
    """Structured output for exercise generation — single exercise per invocation."""

    exercises: list[PracticalExercise] = Field(default_factory=list)
    metadata: dict = Field(
        default_factory=dict,
        description="{topics_covered, total_source_chunks}",
    )


# ── State schema ─────────────────────────────────────────────────────────────


class ExerciseGeneratorState(TypedDict):
    session_id: str
    student_id: str
    topic: str
    difficulty: str
    exercise_type: str
    collection_name: str
    student_profile: dict | None
    retrieved_chunks: Annotated[list[dict], operator.add]
    generated_exercise: dict
    validation_passed: bool
    validation_errors: Annotated[list[str], operator.add]
    retry_count: int
    topic_not_found: list[str]
    topic_suggestions: list[str]
    exercise: dict
    status: str


def retrieve_relevant_chunks(state: ExerciseGeneratorState) -> dict:
    """Retrieve top-K relevant chunks from ChromaDB for the requested topic.

    Single-topic retrieval (unlike ExamGenerator's multi-topic loop).
    On empty results, queries ThematicIndex for close-topic suggestions.
    Deduplicates by chunk_id.
    """
    import logging

    from src.tools import retrieve_chunks as _retrieve_chunks

    logger = logging.getLogger(__name__)

    try:
        topic: str = state.get("topic", "")
        session_id: str = state.get("session_id", "")
        collection_name = state.get("collection_name") or f"session_{session_id}"

        if not topic:
            return {
                "retrieved_chunks": [],
                "topic_not_found": [""],
                "topic_suggestions": [],
                "collection_name": collection_name,
                "status": "no_material",
            }

        chunks = _retrieve_chunks.invoke(
            {
                "query": topic,
                "top_k": 5,
                "collection_name": collection_name,
            }
        )

        all_chunks: list[dict] = []
        seen_chunk_ids: set[str] = set()
        topic_not_found: list[str] = []

        if not chunks:
            topic_not_found.append(topic)
        else:
            for chunk in chunks:
                cid = chunk.get("chunk_id", "")
                if cid and cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_chunks.append(chunk)

        # For missing topic, query ThematicIndex for suggestions
        topic_suggestions: list[str] = []
        if topic_not_found:
            try:
                from src.rag import ThematicIndex as _ThematicIndex

                ti = _ThematicIndex()
                # Build index from chunk metadata (empty if no chunks)
                parts = topic.split("/")
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


def generate_exercise(state: ExerciseGeneratorState) -> dict:
    """Generate a practice exercise via a single structured LLM call.

    Builds a prompt from retrieved chunks, topic, difficulty, and type.
    Uses ChatGroq with structured_output to get a deterministic
    ExerciseGeneration. On retry, injects previous validation_errors
    as context to guide regeneration.
    """
    import logging

    from src.config import settings

    logger = logging.getLogger(__name__)

    try:
        chunks: list[dict] = state.get("retrieved_chunks", [])
        # Guard: no chunks → cannot generate
        if not chunks:
            return {
                "generated_exercise": {},
                "status": "no_material",
            }

        topic: str = state.get("topic", "")
        difficulty: str = state.get("difficulty", "medium")
        exercise_type: str = state.get("exercise_type", "problem_solving")
        retry_count: int = state.get("retry_count", 0)
        validation_errors: list[str] = state.get("validation_errors", [])

        # Build chunk context
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:8000]  # Truncate to avoid token overflow

        # Build preferences section
        prefs_lines: list[str] = []
        prefs_lines.append(f"Tema: {topic}")
        prefs_lines.append(f"Dificultad: {difficulty}")
        prefs_lines.append(f"Tipo de ejercicio: {exercise_type}")

        # Retry instructions
        retry_instructions = ""
        if retry_count > 0 and validation_errors:
            retry_instructions = (
                f"\nIMPORTANTE: El ejercicio anterior falló validación. "
                f"Errores: {'; '.join(validation_errors[-5:])}. "
                f"Asegurate de que CADA afirmación provenga directamente de "
                f"un chunk etiquetado con [CHUNK:...].\n"
            )

        prompt = f"""Generá un ejercicio práctico académico basado EXCLUSIVAMENTE en los siguientes chunks de material de estudio.

{chunk_context}

PREFERENCIAS:
{chr(10).join(prefs_lines)}{retry_instructions}

REQUISITOS:
- El ejercicio debe incluir: enunciado (statement), datos proporcionados (given_data), y una pregunta que requiera aplicación multi-paso.
- Proporcioná una solución modelo (model_solution) con 3-6 pasos detallados.
- Cada paso debe incluir: step_number, description, result, y source_chunk_ids.
- Incluí final_answer y 3-5 key_concepts.
- Cada hecho DEBE provenir de los chunks fuente. Incluí source_chunk_ids (los IDs entre corchetes [CHUNK:xxx]).
- El ejercicio debe tener campos: topic, difficulty.
"""

        llm_cls, llm_kwargs = settings.llm_kwargs
        llm = llm_cls(**llm_kwargs)
        structured_llm = llm.with_structured_output(ExerciseGeneration)
        result: ExerciseGeneration = structured_llm.invoke(prompt)

        # Extract first PracticalExercise
        exercises = result.exercises
        if not exercises:
            return {
                "generated_exercise": {},
                "status": "generated",
            }

        ex = exercises[0]
        exercise_dict = {
            "statement": ex.statement,
            "given_data": ex.given_data,
            "question": ex.question,
            "difficulty": ex.difficulty,
            "topic": ex.topic,
            "source_chunk_ids": ex.source_chunk_ids,
            "model_solution": {
                "steps": [
                    {
                        "step_number": s.step_number,
                        "description": s.description,
                        "result": s.result,
                        "source_chunk_ids": s.source_chunk_ids,
                    }
                    for s in ex.model_solution.steps
                ],
                "final_answer": ex.model_solution.final_answer,
                "key_concepts": ex.model_solution.key_concepts,
                "source_chunk_ids": ex.model_solution.source_chunk_ids,
            },
        }

        # Increment retry_count
        next_retry = retry_count + 1 if retry_count > 0 else 1

        return {
            "generated_exercise": exercise_dict,
            "retry_count": next_retry,
            "status": "generated",
        }

    except Exception as exc:
        logger.exception("generate_exercise failed")
        return {
            "validation_errors": [f"Generation error: {exc}"],
            "status": "error",
        }


def validate_exercise(state: ExerciseGeneratorState) -> dict:
    """Validate exercise claims against source chunks via embedding similarity.

    Dual claim extraction from:
      (a) statement + given_data + question split on sentence boundaries >=20 chars
      (b) model_solution.steps[].description + result + final_answer

    Batch-encodes all claims and chunks, then uses
    ``sentence_transformers.util.cos_sim`` for a single matrix operation.
    Claims below anti_hallucination_threshold are flagged as errors.
    """
    import logging

    import torch
    from sentence_transformers.util import cos_sim

    from src.config import settings
    from src.rag import get_embedding_model
    from src.utils.text import split_into_claims

    logger = logging.getLogger(__name__)

    try:
        chunks: list[dict] = state.get("retrieved_chunks", [])
        exercise: dict = state.get("generated_exercise", {})
        threshold: float = settings.anti_hallucination_threshold
        model = get_embedding_model()

        if not exercise:
            return {
                "validation_passed": False,
                "validation_errors": ["No exercise to validate"],
            }

        # Pre-embed all chunk texts (batch, as tensor)
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = (
            model.encode(chunk_texts, convert_to_tensor=True)
            if chunk_texts
            else torch.empty(0)
        )

        # ── Extract claims from exercise text ──
        claims: list[str] = []
        for field in ("statement", "given_data", "question"):
            text = exercise.get(field, "")
            claims.extend(split_into_claims(text, min_length=20))

        # ── Extract claims from model solution steps ──
        solution = exercise.get("model_solution", {})
        for step in solution.get("steps", []):
            desc = step.get("description", "")
            result = step.get("result", "")
            if len(desc.strip()) >= 20:
                claims.append(desc.strip())
            if len(result.strip()) >= 10:
                claims.append(result.strip())

        final_answer = solution.get("final_answer", "")
        if len(final_answer.strip()) >= 10:
            claims.append(final_answer.strip())

        # Fallback: if no claims, use full statement + question
        if not claims:
            fallback = (
                f"{exercise.get('statement', '')} {exercise.get('question', '')}"
            ).strip()
            if len(fallback) >= 10:
                claims = [fallback]

        # ── Batch encode all claims ──
        if not claims:
            return {"validation_passed": True, "validation_errors": []}

        claim_embeddings = model.encode(claims, convert_to_tensor=True)

        if chunk_embeddings.shape[0] == 0:
            # No chunks — all claims unmatched
            return {
                "validation_passed": False,
                "validation_errors": [
                    f"Claim {ci} ('{c[:80]}') cannot be validated: no source chunks"
                    for ci, c in enumerate(claims)
                ],
            }

        assert claim_embeddings.shape[1] == chunk_embeddings.shape[1], (
            f"Embedding dimension mismatch: claims={claim_embeddings.shape[1]}, "
            f"chunks={chunk_embeddings.shape[1]}"
        )

        # ── Single matrix cosine similarity ──
        sim_matrix = cos_sim(claim_embeddings, chunk_embeddings)
        best_scores, _ = sim_matrix.max(dim=1)
        best_scores = torch.nan_to_num(best_scores, nan=0.0)

        # ── Flag claims below threshold ──
        all_errors: list[str] = []
        all_matched = True

        for ci, claim in enumerate(claims):
            score = best_scores[ci].item()
            if score < threshold:
                all_matched = False
                all_errors.append(
                    f"Claim {ci} ('{claim[:80]}') similarity {score:.4f} "
                    f"below threshold {threshold} — not grounded in source chunks"
                )

        return {
            "validation_passed": all_matched,
            "validation_errors": all_errors,
        }

    except Exception as exc:
        logger.exception("validate_exercise failed")
        return {
            "validation_errors": [f"Validation error: {exc}"],
            "validation_passed": False,
        }

        # Pre-embed all chunk texts (batch)
        chunk_texts = [c.get("text", "") for c in chunks]
        chunk_embeddings = model.encode(chunk_texts).tolist() if chunk_texts else []

        # Regex to split on sentence boundaries
        sentence_split_re = re.compile(r"(?<=[.!?])\s+")

        # Extract claims from exercise text
        claims: list[str] = []
        for field in ("statement", "given_data", "question"):
            text = exercise.get(field, "")
            claims.extend(c.strip() for c in sentence_split_re.split(text) if len(c.strip()) >= 20)

        # Extract claims from model solution steps
        solution = exercise.get("model_solution", {})
        for step in solution.get("steps", []):
            desc = step.get("description", "")
            result = step.get("result", "")
            if len(desc.strip()) >= 20:
                claims.append(desc.strip())
            if len(result.strip()) >= 10:
                claims.append(result.strip())

        final_answer = solution.get("final_answer", "")
        if len(final_answer.strip()) >= 10:
            claims.append(final_answer.strip())

        # If no claims extracted, use full statement and question as fallback
        if not claims:
            fallback = f"{exercise.get('statement', '')} {exercise.get('question', '')}".strip()
            if len(fallback) >= 10:
                claims = [fallback]

        all_errors: list[str] = []
        all_matched = True

        for ci, claim in enumerate(claims):
            claim_embedding = model.encode([claim]).tolist()[0]

            best_score = 0.0
            for chi, chunk_emb in enumerate(chunk_embeddings):
                sim = _cosine_sim(claim_embedding, chunk_emb)
                if sim > best_score:
                    best_score = sim

            if best_score < threshold:
                all_matched = False
                all_errors.append(
                    f"Claim {ci} ('{claim[:80]}') similarity {best_score:.4f} "
                    f"below threshold {threshold} — not grounded in source chunks"
                )

        return {
            "validation_passed": all_matched,
            "validation_errors": all_errors,
        }

    except Exception as exc:
        logger.exception("validate_exercise failed")
        return {
            "validation_errors": [f"Validation error: {exc}"],
            "validation_passed": False,
        }


def should_retry(state: ExerciseGeneratorState) -> str:
    """Return 'retry' if validation errors exist AND retry_count < 3, else 'done'.

    Does NOT retry on terminal statuses (error, no_material).
    """
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)
    status = state.get("status", "")
    if status in ("error", "no_material"):
        return "done"
    if errors and retry_count < 3:
        return "retry"
    return "done"


def format_exercise(state: ExerciseGeneratorState) -> dict:
    """Package validated exercise into final dict with metadata.

    Sets status (complete | partial | no_material), adds exercise_id (UUID4),
    generated_at (UTC ISO), warnings, topic_not_found, topic_suggestions.
    """
    import uuid as _uuid
    from datetime import UTC, datetime

    try:
        exercise: dict = state.get("generated_exercise", {})
        validation_passed: bool = state.get("validation_passed", False)
        validation_errors: list[str] = state.get("validation_errors", [])
        topic_not_found: list[str] = state.get("topic_not_found", [])
        topic_suggestions: list[str] = state.get("topic_suggestions", [])
        retry_count: int = state.get("retry_count", 0)
        status: str = state.get("status", "")

        # If retries exhausted and still invalid, mark as partial with warnings
        if retry_count >= 3 and not validation_passed and status != "no_material":
            exercise_status = "partial"
        elif status == "no_material":
            exercise_status = "no_material"
        elif status == "error":
            exercise_status = "error"
        elif not validation_passed and validation_errors:
            exercise_status = "partial"
        elif not exercise:
            exercise_status = "no_material"
        else:
            exercise_status = "complete"

        # Build final exercise dict
        final_exercise = {
            "exercise_id": str(_uuid.uuid4()),
            "session_id": state.get("session_id", ""),
            "student_id": state.get("student_id", ""),
            "generated_at": datetime.now(UTC).isoformat(),
            "topic": state.get("topic", ""),
            "difficulty": state.get("difficulty", ""),
            "exercise_type": state.get("exercise_type", ""),
            "statement": exercise.get("statement", ""),
            "given_data": exercise.get("given_data", ""),
            "question": exercise.get("question", ""),
            "model_solution": exercise.get("model_solution", {}),
            "source_chunk_ids": exercise.get("source_chunk_ids", []),
            "topics_covered": [state.get("topic", "")] if state.get("topic") else [],
            "source_chunks_total": len(state.get("retrieved_chunks", [])),
            "topic_not_found": topic_not_found,
            "topic_suggestions": topic_suggestions,
            "status": exercise_status,
            "warnings": validation_errors if exercise_status != "complete" else [],
        }

        return {"exercise": final_exercise, "status": exercise_status}
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("format_exercise failed")
        return {
            "exercise": {"error": str(exc), "status": "error"},
            "status": "error",
            "validation_errors": [f"Format error: {exc}"],
        }


def build_exercise_generator() -> StateGraph:
    """Build and return the ExerciseGenerator LangGraph."""
    builder = StateGraph(ExerciseGeneratorState)

    builder.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
    builder.add_node("generate_exercise", generate_exercise)
    builder.add_node("validate_exercise", validate_exercise)
    builder.add_node("format_exercise", format_exercise)

    builder.add_edge(START, "retrieve_relevant_chunks")
    builder.add_edge("retrieve_relevant_chunks", "generate_exercise")
    builder.add_edge("generate_exercise", "validate_exercise")
    builder.add_conditional_edges(
        "validate_exercise",
        should_retry,
        {"retry": "generate_exercise", "done": "format_exercise"},
    )
    builder.add_edge("format_exercise", END)

    return builder
