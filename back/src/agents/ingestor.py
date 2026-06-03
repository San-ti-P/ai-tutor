"""Ingestor Agent — ReAct loop for document ingestion and processing."""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

import operator


class IngestorState(TypedDict):
    session_id: str
    file_path: str
    file_type: str
    raw_text: str
    classification: str
    topics: list[str]
    chunks_created: int
    ocr_confidence: float
    needs_ocr_confirmation: bool
    errors: Annotated[list[str], operator.add]
    status: str


def parse_document(state: IngestorState) -> dict:
    """Parse uploaded file using markitdown and extract raw text."""
    raise NotImplementedError


def classify_document(state: IngestorState) -> dict:
    """Classify document type (apunte, examen, ejercicio, etc.) and detect topics."""
    raise NotImplementedError


def run_ocr_if_needed(state: IngestorState) -> dict:
    """Run OCR math extraction if document contains images with formulas."""
    raise NotImplementedError


def check_ocr_confidence(state: IngestorState) -> Literal["proceed", "request_confirmation"]:
    """Check if OCR confidence meets threshold; request user confirmation if not."""
    raise NotImplementedError


def chunk_and_embed(state: IngestorState) -> dict:
    """Split text into semantic chunks and store in ChromaDB with embeddings."""
    raise NotImplementedError


def handle_error(state: IngestorState) -> dict:
    """Handle and log errors during ingestion pipeline."""
    raise NotImplementedError


def build_ingestor() -> StateGraph:
    """Build and return the Ingestor LangGraph."""
    builder = StateGraph(IngestorState)

    builder.add_node("parse_document", parse_document)
    builder.add_node("classify_document", classify_document)
    builder.add_node("run_ocr_if_needed", run_ocr_if_needed)
    builder.add_node("chunk_and_embed", chunk_and_embed)
    builder.add_node("handle_error", handle_error)

    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "classify_document")
    builder.add_edge("classify_document", "run_ocr_if_needed")
    builder.add_conditional_edges(
        "run_ocr_if_needed",
        check_ocr_confidence,
        {"proceed": "chunk_and_embed", "request_confirmation": END},
    )
    builder.add_edge("chunk_and_embed", END)
    builder.add_edge("handle_error", END)

    return builder
