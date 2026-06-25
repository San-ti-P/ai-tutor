"""Ingestor Agent — linear StateGraph for document ingestion and processing.

Pipeline (OCR deferred): parse → classify → chunk/embed → END.
Error handling is per-node try/except; errors accumulate in state["errors"].

Scope restriction (June 2026):
- Image files are rejected — OCR math pipeline deferred to post-MVP.
- Only PDF and TXT are accepted.
- OCR helper functions kept below for reference; not wired into the graph.
"""

from __future__ import annotations

import logging
import operator
import uuid
from typing import Annotated, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class IngestorState(TypedDict):
    session_id: str
    file_path: str
    file_type: str
    raw_text: str
    classification: str
    classification_confidence: float
    topics: list[str]
    chunks_created: int
    errors: Annotated[list[str], operator.add]
    status: str
    document_id: str
    chunk_ids: list[str]


# ── Node implementations ────────────────────────────────────────────────────


def parse_document(state: IngestorState) -> dict:
    """Parse uploaded file using markitdown and extract raw text.

    Accepted: PDF, TXT.
    Rejected: images (PNG/JPG — OCR deferred), unsupported formats.

    Uses ``src.utils.text.parse_file_to_text`` — the single source of truth
    for markitdown-based parsing.
    """
    try:
        from pathlib import Path

        from src.utils.text import parse_file_to_text

        file_path = Path(state["file_path"])
        if not file_path.exists():
            return {"errors": [f"File not found: {file_path}"], "status": "error"}

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            file_type = "pdf"
        elif suffix == ".txt":
            file_type = "text"
        elif suffix in (".png", ".jpg", ".jpeg"):
            return {
                "errors": [
                    "Image files (PNG/JPG) are not yet supported. "
                    "OCR math extraction is deferred. Please upload PDF or TXT."
                ],
                "status": "rejected",
            }
        else:
            return {
                "errors": [f"Unsupported file type: {suffix}"],
                "status": "rejected",
            }

        raw_text = parse_file_to_text(str(file_path))

        return {
            "raw_text": raw_text,
            "file_type": file_type,
            "status": "parsed",
        }
    except Exception as e:
        logger.exception("parse_document failed")
        return {"errors": [f"Parse error: {e}"], "status": "error"}


def classify_document(state: IngestorState, config: RunnableConfig = None) -> dict:
    """Classify document type and detect topics using LLM."""
    from pydantic import BaseModel, Field

    from src.config import settings

    class Classification(BaseModel):
        classification: Literal[
            "apunte_teorico", "examen_previo", "ejercicio_resuelto", "no_academico"
        ]
        confidence: float = Field(ge=0.0, le=1.0)
        topics: list[str]

    try:
        raw_text = state.get("raw_text", "")
        if not raw_text or not raw_text.strip():
            return {
                "errors": ["Empty document — no extractable text"],
                "status": "rejected",
            }

        from src.llm import get_structured_llm

        structured_llm = get_structured_llm(Classification)

        prompt = f"""Analizá el siguiente texto académico y clasificalo.
Clases posibles: apunte_teorico, examen_previo, ejercicio_resuelto, no_academico.
Extraé también los temas principales (3-8 temas).

Texto:
{raw_text[:3000]}
"""
        invoke_kwargs = {"config": config} if config is not None else {}
        result = structured_llm.invoke(prompt, **invoke_kwargs)

        if result.classification == "no_academico":
            return {
                "classification": result.classification,
                "classification_confidence": result.confidence,
                "topics": result.topics,
                "errors": ["Content rejected: non-academic material"],
                "status": "rejected_non_academic",
            }

        if result.confidence < settings.classification_confidence_threshold:
            return {
                "classification": result.classification,
                "classification_confidence": result.confidence,
                "topics": result.topics,
                "status": "classification_uncertain",
            }

        return {
            "classification": result.classification,
            "classification_confidence": result.confidence,
            "topics": result.topics,
            "status": "classified",
        }
    except Exception as e:
        logger.exception("classify_document failed")
        return {"errors": [f"Classification error: {e}"], "status": "error"}


# ── Node implementation: chunk_and_embed ────────────────────────────────────


def chunk_and_embed(state: IngestorState) -> dict:
    """Split text into semantic chunks and store in ChromaDB with embeddings."""
    from src.rag import chunk_text, embed_and_store

    try:
        document_id = state.get("document_id") or str(uuid.uuid4())

        from pathlib import Path

        # Build metadata for each chunk
        base_metadata: dict[str, object] = {
            "document_id": document_id,
            "session_id": state["session_id"],
            "classification": state.get("classification", "unknown"),
            "source_file": Path(state["file_path"]).name,
        }

        # Chunk the raw text
        chunks = chunk_text(state["raw_text"])
        if not chunks:
            return {
                "document_id": document_id,
                "chunk_ids": [],
                "chunks_created": 0,
                "status": "completed",
            }

        chunk_texts = [c.page_content for c in chunks]
        topics = state.get("topics", [])
        primary_topic = topics[0] if topics else ""
        metadatas: list[dict[str, object]] = [
            {**base_metadata, "topic": primary_topic, "topics": topics, "chunk_index": i}
            for i in range(len(chunk_texts))
        ]

        # ChromaDB rejects empty list metadata values — prune them
        for meta in metadatas:
            for key in list(meta.keys()):
                if isinstance(meta[key], list) and not meta[key]:
                    del meta[key]

        # Embed and store
        collection_name = f"session_{state['session_id']}"
        chunk_ids = embed_and_store(chunk_texts, metadatas, collection_name)

        return {
            "document_id": document_id,
            "chunk_ids": chunk_ids,
            "chunks_created": len(chunk_ids),
            "status": "completed",
        }
    except Exception as e:
        logger.exception("chunk_and_embed failed")
        return {"errors": [f"Chunk/embed error: {e}"], "status": "error"}


# ── Graph builder ───────────────────────────────────────────────────────────


def build_ingestor() -> StateGraph:
    """Build and return the Ingestor LangGraph.

    Simplified graph (OCR deferred):
        START → parse_document → classify_document → chunk_and_embed → END
    """
    builder = StateGraph(IngestorState)

    builder.add_node("parse_document", parse_document)
    builder.add_node("classify_document", classify_document)
    builder.add_node("chunk_and_embed", chunk_and_embed)

    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "classify_document")
    builder.add_edge("classify_document", "chunk_and_embed")
    builder.add_edge("chunk_and_embed", END)

    return builder
