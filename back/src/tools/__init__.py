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
    collection_name: str = "default",
) -> dict:
    """Ingest a document: parse, classify, OCR, chunk, and embed into ChromaDB.

    This tool is intended to be called by the Ingestor agent graph, not directly.
    It returns a summary dict with file_path, classification, topics, and
    chunks_created.

    Args:
        file_path: Path to the uploaded file (PDF, PNG, TXT).
        session_id: Current session ID for tracking.
        collection_name: ChromaDB collection name (default "default").

    Returns:
        A dict with keys: file_path, classification, topics, chunks_created.
    """
    # Wired in Phase 3 when the full graph invoke is in place.
    raise NotImplementedError
