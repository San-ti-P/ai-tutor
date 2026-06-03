"""LangChain tool definitions consumed by all agents.

Tools are decorated with ``@tool`` for automatic schema generation.
They delegate to the RAG module or agent graphs for execution.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag import retrieve as _rag_retrieve


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
    """Ingest a document: parse, classify, OCR, chunk, and embed into ChromaDB.

    This tool orchestrates the full ingestion pipeline. Called by the Ingestor
    agent graph. It returns a summary dict with file_path, classification,
    topics, chunks_created, status, and errors.

    The ChromaDB collection name is derived internally from the session ID
    (``session_{session_id}``) by the chunk_and_embed node.

    Args:
        file_path: Path to the uploaded file on disk.
        session_id: Current session ID for tracking.

    Returns:
        A dict with keys: file_path, classification, topics, chunks_created,
        status, errors.
    """
    from src.agents.ingestor import IngestorState, build_ingestor

    graph = build_ingestor().compile()
    initial_state: IngestorState = {
        "session_id": session_id,
        "file_path": file_path,
        "file_type": "",
        "raw_text": "",
        "classification": "",
        "topics": [],
        "chunks_created": 0,
        "ocr_confidence": 0.0,
        "needs_ocr_confirmation": False,
        "errors": [],
        "status": "pending",
        "classification_confidence": 0.0,
        "ocr_expressions": [],
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
