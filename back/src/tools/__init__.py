"""LangChain tool definitions consumed by all agents.

Tools are decorated with ``@tool`` for automatic schema generation.
They delegate to the RAG module or agent graphs for execution.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from langfuse import observe, propagate_attributes

from src.rag import retrieve as _rag_retrieve
from src.tools.get_student_summary import get_student_summary  # noqa: F401
from src.tools.orchestrate_chat import orchestrate_chat  # noqa: F401
from src.tools.query_material import query_material  # noqa: F401
from src.tools.update_student_profile import update_student_profile  # noqa: F401
from src.tools.validate_claim_grounding import validate_claim_grounding  # noqa: F401

logger = logging.getLogger(__name__)

# Cached compiled graphs — safe to reuse across invocations
_graph_cache: dict[str, object] = {}


def _get_or_compile(name: str, builder_path: str):
    """Return a compiled StateGraph, building on first call only."""
    if name not in _graph_cache:
        import importlib

        mod_name, fn_name = builder_path.rsplit(".", 1)
        builder = getattr(importlib.import_module(mod_name), fn_name)
        _graph_cache[name] = builder().compile()
    return _graph_cache[name]


@tool
@observe(name="retrieve_chunks", as_type="tool")
def retrieve_chunks(
    query: str,
    top_k: int = 5,
    topic_filter: str | None = None,
    collection_name: str = "default",
) -> list[dict]:
    """Retrieve top-K relevant chunks from ChromaDB for a given query.

    Args:
        query: The search query or topic.
        top_k: Number of chunks to retrieve (default 5).
        topic_filter: Optional topic prefix to filter results.
        collection_name: ChromaDB collection name (default "default").

    Returns:
        A list of chunk dicts with keys: chunk_id, text, metadata, similarity_score.
    """
    return _rag_retrieve(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        topic_filter=topic_filter,
    )


@tool
@observe(name="ingest_document", as_type="tool")
def ingest_document(
    file_path: str,
    session_id: str,
) -> dict:
    """Ingest a document: parse, classify, chunk, and embed into ChromaDB.

    This tool orchestrates the full ingestion pipeline. Called by the Ingestor
    agent graph. It returns a summary dict with file_path, classification,
    topics, chunks_created, status, and errors.

    The ChromaDB collection name is derived internally from the session ID
    (``session_{session_id}``) by the chunk_and_embed node.

    Note: OCR math extraction is deferred. Only PDF and TXT files are accepted.
    Image files (PNG/JPG) are rejected with a descriptive error.

    Args:
        file_path: Path to the uploaded file on disk.
        session_id: Current session ID for tracking.

    Returns:
        A dict with keys: file_path, classification, topics, chunks_created,
        status, errors.
    """
    from src.agents.ingestor import IngestorState

    graph = _get_or_compile("ingestor", "src.agents.ingestor.build_ingestor")
    initial_state: IngestorState = {
        "session_id": session_id,
        "file_path": file_path,
        "file_type": "",
        "raw_text": "",
        "classification": "",
        "classification_confidence": 0.0,
        "topics": [],
        "chunks_created": 0,
        "errors": [],
        "status": "pending",
        "document_id": "",
        "chunk_ids": [],
    }

    from src.observability import flush_traces, get_tracer

    tracer = get_tracer()
    config: dict = {}

    try:
        # Per Langfuse docs: CallbackHandler created INSIDE propagate_attributes
        # inherits trace-level session/user context.
        with propagate_attributes(session_id=session_id):
            handler = tracer.get_callback_handler(session_id=session_id, user_id=None)
            if handler:
                config["callbacks"] = [handler]
                config["metadata"] = {"langfuse_session_id": session_id}
            result = graph.invoke(initial_state, config=config)
    finally:
        flush_traces()

    return {
        "file_path": file_path,
        "classification": result.get("classification", ""),
        "topics": result.get("topics", []),
        "chunks_created": result.get("chunks_created", 0),
        "status": result.get("status", ""),
        "errors": result.get("errors", []),
    }


@tool
def extract_topics(
    text: str | None = None,
    file_path: str | None = None,
) -> dict:
    """Extract hierarchical topics from academic content.

    Analyzes the provided text (or reads text from a file) and returns
    a structured tree of topics. Use this tool to understand what topics
    a document covers before deciding which chunks to retrieve or what
    content to generate.

    At least one of ``text`` or ``file_path`` must be provided. When both
    are given, ``text`` takes precedence.

    Args:
        text: Raw text content to analyze for topics.
        file_path: Path to a file (PDF, TXT) to parse and analyze.

    Returns:
        A dict with:
            - summary: one-sentence summary of the content
            - topics: flat list of detected topic strings (3-15 items)
            - topic_tree: nested dict representing the hierarchical topic structure
    """
    from pathlib import Path

    from pydantic import BaseModel, Field

    class TopicExtraction(BaseModel):
        summary: str = Field(description="One-sentence summary of the content")
        topics: list[str] = Field(description="Flat list of detected topics (3-15 items)")
        topic_tree: str = Field(
            default="",
            description=(
                "Hierarchical topic structure as JSON string. "
                'Example: \'{"Math": {"Algebra": {}, "Calculus": {}}}\''
            ),
        )

    # Resolve input text
    if text is not None and text.strip():
        content = text
    elif file_path is not None:
        p = Path(file_path)
        if not p.exists():
            return {"error": f"File not found: {file_path}"}

        suffix = p.suffix.lower()
        if suffix == ".txt":
            content = p.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            try:
                from src.utils.text import parse_file_to_text

                content = parse_file_to_text(str(p))
            except Exception as exc:
                return {"error": f"Failed to parse PDF: {exc}"}
        else:
            return {"error": f"Unsupported file type: {suffix}. Use PDF or TXT."}

        if not content.strip():
            return {"error": "File is empty."}
    else:
        return {"error": "Either 'text' or 'file_path' must be provided."}

    # Truncate to avoid token limits
    content_preview = content[:5000]

    try:
        from src.llm import get_structured_llm

        structured_llm = get_structured_llm(TopicExtraction)

        prompt = (
            "Analizá el siguiente texto académico y extraé:\n"
            "1. Un resumen de una línea del contenido.\n"
            "2. Una lista plana de temas principales (3-15 temas).\n"
            "3. Un árbol de temas como texto, ejemplo: 'Matemáticas > Álgebra > Lineal'\n\n"
            f"Texto:\n{content_preview}"
        )

        result = structured_llm.invoke(prompt)
        return {
            "summary": result.summary,
            "topics": result.topics,
            "topic_tree": result.topic_tree,
        }
    except Exception as exc:
        logger.exception("extract_topics failed")
        return {"error": f"Topic extraction failed: {exc}"}


@tool
@observe(name="generate_exercise", as_type="tool")
def generate_exercise(
    session_id: str,
    topic: str,
    difficulty: str = "medium",
    exercise_type: str = "problem_solving",
    student_profile: dict | None = None,
) -> dict:
    """Generate a practical exercise with step-by-step model solution.

    Invokes the full ExerciseGenerator StateGraph: retrieves chunks from
    ChromaDB for the requested topic, generates a single complex exercise
    via structured LLM output (PracticalExercise with ModelSolution), validates
    against source chunks using claim-level embedding similarity, retries
    hallucinated content up to 3 times, and returns the final exercise dict.

    Args:
        session_id: The current session ID (determines ChromaDB collection).
        topic: Topic string for the exercise (e.g. 'cálculo/derivadas').
        difficulty: 'easy', 'medium', or 'hard' (default 'medium').
        exercise_type: 'problem_solving', 'proof', or 'calculation'
            (default 'problem_solving').
        student_profile: Optional dict with 'weak_topics' and 'preferences'.

    Returns:
        An exercise dict with keys: exercise_id, session_id, student_id,
        generated_at, topic, difficulty, exercise_type, statement, given_data,
        question, model_solution (steps, final_answer, key_concepts),
        source_chunk_ids, topics_covered, source_chunks_total, topic_not_found,
        topic_suggestions, status, warnings.
    """
    from src.agents.exercise_generator import ExerciseGeneratorState

    graph = _get_or_compile(
        "exercise_generator", "src.agents.exercise_generator.build_exercise_generator"
    )
    initial_state: ExerciseGeneratorState = {
        "session_id": session_id,
        "student_id": student_profile.get("student_id", "") if student_profile else "",
        "topic": topic,
        "difficulty": difficulty,
        "exercise_type": exercise_type,
        "collection_name": f"session_{session_id}",
        "student_profile": student_profile,
        "retrieved_chunks": [],
        "generated_exercise": {},
        "validation_passed": False,
        "validation_errors": [],
        "retry_count": 0,
        "topic_not_found": [],
        "topic_suggestions": [],
        "exercise": {},
        "status": "pending",
    }

    from src.observability import flush_traces, get_tracer

    tracer = get_tracer()
    config: dict = {}

    try:
        # Per Langfuse docs: CallbackHandler created INSIDE propagate_attributes
        # inherits trace-level session/user context.
        with propagate_attributes(session_id=session_id):
            handler = tracer.get_callback_handler(session_id=session_id, user_id=None)
            if handler:
                config["callbacks"] = [handler]
                config["metadata"] = {"langfuse_session_id": session_id}
            result = graph.invoke(initial_state, config=config)
    finally:
        flush_traces()

    return result.get("exercise", {})


@tool
@observe(name="evaluate_answer", as_type="tool")
def evaluate_answer(
    session_id: str,
    exam_id: str,
    answers: list[dict],
    student_id: str = "",
) -> list[dict]:
    """Evaluate a batch of student answers and return structured results.

    Invokes the full Evaluator StateGraph: retrieves chunks, checks
    evaluability, grades each answer via structured LLM output, validates
    feedback against source chunks (anti-hallucination), optionally runs
    LLM-as-judge on a sample, and persists scores to the evaluations table.

    Args:
        session_id: The current session ID (determines ChromaDB collection).
        exam_id: Identifier for the exam being evaluated.
        answers: List of answer dicts. Each dict must have: question_id,
            question, base_answer, student_answer. Optional: answer_image,
            source_chunk_ids, topic, difficulty.
        student_id: Student identifier for profile tracking.

    Returns:
        A list of evaluation result dicts, each with: question_id, score,
        justification, conceptual_errors, suggestions, is_evaluable,
        non_evaluable_reason, requires_review, validation_warnings, status.
    """
    import uuid

    from src.agents.evaluator import EvaluatorState

    graph = _get_or_compile("evaluator", "src.agents.evaluator.build_evaluator")
    initial_state: EvaluatorState = {
        "session_id": session_id,
        "student_id": student_id,
        "exam_id": exam_id,
        "trace_id": str(uuid.uuid4()),
        "answers": answers,
        "current_index": 0,
        "answer_text": "",
        "ocr_extracted_text": None,
        "ocr_confidence": 0.0,
        "retrieved_chunks": [],
        "evaluation": None,
        "evaluation_results": [],
        "non_evaluable": False,
        "non_evaluable_reason": "",
        "judge_sample": False,
        "judge_result": None,
        "requires_review": False,
        "scores_synced": False,
        "errors": [],
        "status": "pending",
    }

    from src.observability import flush_traces, get_tracer

    tracer = get_tracer()
    config: dict = {}

    try:
        # Per Langfuse docs: CallbackHandler created INSIDE propagate_attributes
        # inherits trace-level session/user context.
        with propagate_attributes(session_id=session_id):
            handler = tracer.get_callback_handler(session_id=session_id, user_id=None)
            if handler:
                config["callbacks"] = [handler]
                config["metadata"] = {"langfuse_session_id": session_id}
            result = graph.invoke(initial_state, config=config)
    finally:
        flush_traces()

    return result.get("evaluation_results", [])


@tool
@observe(name="generate_exam", as_type="tool")
def generate_exam(
    session_id: str,
    topics: list[str],
    difficulty: str = "medium",
    question_count: int = 5,
    mcq_ratio: float = 0.5,
    student_profile: dict | None = None,
) -> dict:
    """Generate a personalized exam with MCQs and open-answer questions.

    Invokes the full ExamGenerator StateGraph: retrieves chunks from ChromaDB,
    generates questions via structured LLM output, validates against source
    chunks, retries hallucinated questions up to 3 times, and returns the
    final exam dict.

    Args:
        session_id: The current session ID (determines ChromaDB collection).
        topics: List of topic strings to cover (e.g. ['cálculo/derivadas']).
        difficulty: 'easy', 'medium', or 'hard' (default 'medium').
        question_count: Total number of questions to generate (default 5).
        mcq_ratio: Fraction of questions that should be MCQ (default 0.5).
        student_profile: Optional dict with 'weak_topics' and 'preferences'.

    Returns:
        An exam dict with keys: exam_id, session_id, student_id,
        generated_at, total_questions, questions, topics_covered,
        source_chunks_total, omitted_count, topic_not_found,
        topic_suggestions, status, warnings.
    """
    from src.agents.exam_generator import ExamGeneratorState

    graph = _get_or_compile("exam_generator", "src.agents.exam_generator.build_exam_generator")
    initial_state: ExamGeneratorState = {
        "session_id": session_id,
        "student_id": student_profile.get("student_id", "") if student_profile else "",
        "topics": topics,
        "difficulty": difficulty,
        "question_count": question_count,
        "mcq_ratio": mcq_ratio,
        "student_profile": student_profile,
        "collection_name": f"session_{session_id}",
        "retrieved_chunks": [],
        "generated_questions": [],
        "validation_results": [],
        "validation_errors": [],
        "invalid_question_indices": [],
        "omitted_questions": [],
        "retry_count": 0,
        "topic_not_found": [],
        "topic_suggestions": [],
        "exam": {},
        "status": "pending",
    }

    from src.observability import flush_traces, get_tracer

    tracer = get_tracer()
    config: dict = {}

    try:
        # Per Langfuse docs: CallbackHandler created INSIDE propagate_attributes
        # inherits trace-level session/user context.
        with propagate_attributes(session_id=session_id):
            handler = tracer.get_callback_handler(session_id=session_id, user_id=None)
            if handler:
                config["callbacks"] = [handler]
                config["metadata"] = {"langfuse_session_id": session_id}
            result = graph.invoke(initial_state, config=config)
    finally:
        flush_traces()

    return result.get("exam", {})
