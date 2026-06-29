"""Tool for answering questions from ingested session material."""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from langfuse import observe

from src.llm import get_llm
from src.rag import retrieve as _rag_retrieve
from src.rag.policy import RAG_ONLY_SYSTEM_PROMPT, no_material_message
from src.config import settings

logger = logging.getLogger(__name__)


@tool
@observe(name="query_material", as_type="tool")
def query_material(
    query: str,
    session_id: str,
    top_k: int = 5,
) -> dict:
    """Answer a question using the ingested material for this session.

    Retrieves relevant chunks from the session's ChromaDB collection and
    synthesizes a concise answer with the LLM.

    Args:
        query: The user's question about the uploaded material.
        session_id: Current session ID (determines ChromaDB collection).
        top_k: Number of chunks to retrieve (default 5).

    Returns:
        A dict with keys: answer, sources (list of chunk texts), chunks_found.
    """
    collection_name = f"session_{session_id}"
    chunks = _rag_retrieve(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
    )

    if not chunks:
        return {
            "answer": no_material_message(),
            "sources": [],
            "chunks_found": 0,
        }

    # Build context from chunks
    context = "\n\n---\n\n".join(f"Fragmento {i + 1}:\n{c['text']}" for i, c in enumerate(chunks))

    prompt = (
        f"{RAG_ONLY_SYSTEM_PROMPT}\n\n"
        f"Fragmentos del material:\n{context}\n\n"
        f"Pregunta: {query}\n\n"
        "Respondé de forma clara y concisa en español."
    )

    try:
        llm = get_llm(temperature=settings.query_material_temperature)
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.exception("query_material LLM failed")
        return {
            "answer": f"No pude generar una respuesta: {exc}",
            "sources": [c["text"] for c in chunks],
            "chunks_found": len(chunks),
        }

    return {
        "answer": answer,
        "sources": [c["text"] for c in chunks],
        "chunks_found": len(chunks),
    }
