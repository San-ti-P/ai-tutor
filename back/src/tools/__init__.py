"""LangChain tool definitions consumed by all agents.

Tools are decorated with ``@tool`` for automatic schema generation.
They delegate to the RAG module or agent graphs for execution.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

from src.rag import retrieve as _rag_retrieve

# Cached compiled ingestor graph — safe to reuse across invocations
_ingestor_graph = None
# Cached compiled exam generator graph — same lazy singleton pattern
_exam_generator_graph = None


def _get_ingestor_graph():
    """Return the compiled Ingestor StateGraph, compiling on first call only."""
    global _ingestor_graph
    if _ingestor_graph is None:
        from src.agents.ingestor import build_ingestor

        _ingestor_graph = build_ingestor().compile()
    return _ingestor_graph


def _get_exam_generator_graph():
    """Return the compiled ExamGenerator StateGraph, compiling on first call only."""
    global _exam_generator_graph
    if _exam_generator_graph is None:
        from src.agents.exam_generator import build_exam_generator

        _exam_generator_graph = build_exam_generator().compile()
    return _exam_generator_graph


@tool
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

    graph = _get_ingestor_graph()
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

    result = graph.invoke(initial_state)
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

    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field

    class TopicExtraction(BaseModel):
        summary: str = Field(description="One-sentence summary of the content")
        topics: list[str] = Field(
            description="Flat list of detected topics (3-15 items)"
        )
        topic_tree: dict = Field(
            description=(
                "Hierarchical topic structure as nested dict. "
                "Keys are topic names; values are sub-topic dicts. "
                'Example: {"Math": {"Algebra": {"Linear": {}}, "Calculus": {}}}'
            )
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
                import markitdown

                md = markitdown.MarkItDown()
                content = md.convert(str(p)).text_content
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
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        structured_llm = llm.with_structured_output(TopicExtraction)

        prompt = (
            "Analizá el siguiente texto académico y extraé:\n"
            "1. Un resumen de una línea del contenido.\n"
            "2. Una lista plana de temas principales (3-15 temas).\n"
            "3. Un árbol jerárquico de temas (dict anidado).\n\n"
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

    graph = _get_exam_generator_graph()
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

    result = graph.invoke(initial_state)
    return result.get("exam", {})
