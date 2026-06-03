"""Ingestor Agent — linear StateGraph for document ingestion and processing.

Pipeline: parse → classify → OCR → [confidence gate] → chunk/embed → END.
Error handling is per-node try/except; errors accumulate in state["errors"].
"""

from __future__ import annotations

import base64
import logging
import operator
import uuid
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import settings

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
    ocr_confidence: float
    needs_ocr_confirmation: bool
    errors: Annotated[list[str], operator.add]
    status: str
    ocr_expressions: list[dict]
    document_id: str
    chunk_ids: list[str]


# ── Node implementations ────────────────────────────────────────────────────


def parse_document(state: IngestorState) -> dict:
    """Parse uploaded file using markitdown and extract raw text."""
    try:
        from pathlib import Path

        import markitdown

        file_path = Path(state["file_path"])
        if not file_path.exists():
            return {"errors": [f"File not found: {file_path}"], "status": "error"}

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            file_type = "pdf"
        elif suffix in (".png", ".jpg", ".jpeg"):
            file_type = "image"
        elif suffix == ".txt":
            file_type = "text"
        else:
            return {
                "errors": [f"Unsupported file type: {suffix}"],
                "status": "rejected",
            }

        md = markitdown.MarkItDown()
        result = md.convert(str(file_path))
        raw_text = result.text_content

        return {
            "raw_text": raw_text,
            "file_type": file_type,
            "status": "parsed",
        }
    except Exception as e:
        logger.exception("parse_document failed")
        return {"errors": [f"Parse error: {e}"], "status": "error"}


def classify_document(state: IngestorState) -> dict:
    """Classify document type and detect topics using LLM."""
    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field

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

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        structured_llm = llm.with_structured_output(Classification)

        prompt = f"""Analizá el siguiente texto académico y clasificalo.
Clases posibles: apunte_teorico, examen_previo, ejercicio_resuelto, no_academico.
Extraé también los temas principales (3-8 temas).

Texto:
{raw_text[:3000]}
"""
        result = structured_llm.invoke(prompt)

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


# ── OCR backend helpers ──────────────────────────────────────────────────────


def _ocr_mathpix(image_path: str) -> tuple[list[str], float]:
    """Call Mathpix API to extract LaTeX from image. Returns (expressions, confidence)."""
    import requests

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "https://api.mathpix.com/v3/text",
        json={
            "src": f"data:image/png;base64,{image_data}",
            "formats": ["latex_styled"],
        },
        headers={
            "app_id": settings.mathpix_app_id,
            "app_key": settings.mathpix_app_key,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Mathpix API error: {resp.status_code}")

    data = resp.json()
    expressions = [data.get("text", "")]
    confidence = data.get("confidence", 0.0)
    return expressions, confidence


def _ocr_noop(image_path: str) -> tuple[list[str], float]:  # noqa: ARG001
    """No-op OCR stub for when Mathpix is unavailable. Returns empty with confidence 0."""
    logger.warning("OCR noop: returning empty (Mathpix disabled)")
    return [], 0.0


# ── Node implementations (continued) ────────────────────────────────────────


def run_ocr_if_needed(state: IngestorState) -> dict:
    """Run OCR math extraction if document contains images with formulas."""
    try:
        if state.get("file_type") != "image":
            return {
                "ocr_confidence": 1.0,
                "ocr_expressions": [],
                "needs_ocr_confirmation": False,
                "status": "ocr_skipped",
            }

        # Choose backend: Mathpix if keys present, else noop
        if settings.mathpix_app_id and settings.mathpix_app_key:
            expressions, confidence = _ocr_mathpix(state["file_path"])
        else:
            expressions, confidence = _ocr_noop(state["file_path"])

        needs_confirmation = confidence < settings.ocr_confidence_threshold
        return {
            "ocr_expressions": expressions,
            "ocr_confidence": confidence,
            "needs_ocr_confirmation": needs_confirmation,
            "status": "awaiting_ocr_confirmation" if needs_confirmation else "ocr_done",
        }
    except Exception as e:
        logger.exception("run_ocr_if_needed failed")
        return {"errors": [f"OCR error: {e}"], "status": "ocr_failed"}


def check_ocr_confidence(
    state: IngestorState,
) -> Literal["proceed", "request_confirmation"]:
    """Check if OCR confidence meets threshold; request user confirmation if not."""
    if state.get("ocr_confidence", 1.0) >= settings.ocr_confidence_threshold:
        return "proceed"
    return "request_confirmation"


def chunk_and_embed(state: IngestorState) -> dict:
    """Split text into semantic chunks and store in ChromaDB with embeddings."""
    from src.rag import chunk_text, embed_and_store

    try:
        document_id = state.get("document_id") or str(uuid.uuid4())

        # Build metadata for each chunk
        base_metadata: dict[str, object] = {
            "document_id": document_id,
            "session_id": state["session_id"],
            "classification": state.get("classification", "unknown"),
            "source_file": state["file_path"],
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
        topic_str = ", ".join(topics) if topics else ""
        metadatas: list[dict[str, object]] = [
            {**base_metadata, "topic": topic_str, "chunk_index": i}
            for i in range(len(chunk_texts))
        ]

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
    """Build and return the Ingestor LangGraph."""
    builder = StateGraph(IngestorState)

    builder.add_node("parse_document", parse_document)
    builder.add_node("classify_document", classify_document)
    builder.add_node("run_ocr_if_needed", run_ocr_if_needed)
    builder.add_node("chunk_and_embed", chunk_and_embed)

    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "classify_document")
    builder.add_edge("classify_document", "run_ocr_if_needed")
    builder.add_conditional_edges(
        "run_ocr_if_needed",
        check_ocr_confidence,
        {"proceed": "chunk_and_embed", "request_confirmation": END},
    )
    builder.add_edge("chunk_and_embed", END)

    return builder
