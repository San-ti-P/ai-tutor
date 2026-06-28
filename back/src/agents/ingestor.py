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
    topic_tree: str
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
    import time

    t0 = time.monotonic()
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

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[parse_document] COMPLETE | session=%s | type=%s | len=%d | %dms",
            state["session_id"],
            file_type,
            len(raw_text),
            int(elapsed),
        )
        return {
            "raw_text": raw_text,
            "file_type": file_type,
            "status": "parsed",
        }
    except Exception as e:
        logger.exception("parse_document failed")
        return {"errors": [f"Parse error: {e}"], "status": "error"}


async def classify_document(state: IngestorState, config: RunnableConfig = None) -> dict:
    """Classify document type and detect topics using LLM.

    Runs the full-document topic extraction pipeline BEFORE the classification
    prompt, then injects pipeline-detected topics as context for the classifier.
    Classification itself still uses ``raw_text[:3000]`` preview.
    """
    import time

    from pydantic import BaseModel, Field

    from src.config import settings
    from src.topic_extraction import extract_topics_pipeline

    t0 = time.monotonic()
    logger.info(
        "[classify_document] START | session=%s",
        state["session_id"],
    )

    class Classification(BaseModel):
        classification: Literal[
            "apunte_teorico", "examen_previo", "ejercicio_resuelto", "no_academico"
        ]
        confidence: float = Field(ge=0.0, le=1.0)
        topics: list[str] = Field(default_factory=list)

    try:
        raw_text = state.get("raw_text", "")
        if not raw_text or not raw_text.strip():
            return {
                "errors": ["Empty document — no extractable text"],
                "status": "rejected",
            }

        # ── Run pipeline for full-document topic extraction ──────────────
        pipeline_result = await extract_topics_pipeline(raw_text)
        pipeline_topics = pipeline_result.get("topics", [])
        topic_tree = pipeline_result.get("topic_tree", "{}")

        # ── Conciliate topics with other files in the same session (cross-file conciliation) ──
        session_id = state.get("session_id")
        if session_id:
            import json
            from src.memory.schema import list_session_files
            from src.topic_extraction import reconcile_topics

            try:
                existing_files = await list_session_files(session_id)
                existing_topics = []
                for f in existing_files:
                    if state.get("document_id") and f.get("id") == state.get("document_id"):
                        continue
                    if f.get("topics_json"):
                        try:
                            ext_topics = json.loads(f["topics_json"])
                            existing_topics.extend(ext_topics)
                        except Exception:
                            pass

                if existing_topics:
                    reconciled_topics, topic_map = reconcile_topics(
                        pipeline_topics, existing_topics
                    )
                    pipeline_topics = reconciled_topics

                    # Also update the topic_tree JSON representation to reflect reconciled topics
                    if topic_tree and topic_tree != "{}":
                        try:
                            tree_data = json.loads(topic_tree)

                            def update_tree_keys(d: dict, mapping: dict) -> dict:
                                new_dict = {}
                                for k, v in d.items():
                                    new_k = mapping.get(k, k)
                                    if isinstance(v, dict):
                                        new_dict[new_k] = update_tree_keys(v, mapping)
                                    else:
                                        new_dict[new_k] = v
                                return new_dict

                            tree_data = update_tree_keys(tree_data, topic_map)
                            topic_tree = json.dumps(tree_data, ensure_ascii=False)
                        except Exception as e:
                            logger.warning("Failed to update topic tree keys: %s", e)
            except Exception as e:
                logger.warning("Failed to retrieve existing session files for topic conciliation: %s", e)

        from src.llm import get_structured_llm

        structured_llm = get_structured_llm(Classification)

        topics_str = ", ".join(pipeline_topics[:8]) if pipeline_topics else "(ninguno detectado)"
        prompt = f"""Analizá el siguiente texto académico y clasificalo.
Clases posibles: apunte_teorico, examen_previo, ejercicio_resuelto, no_academico.

Temas detectados en el documento completo: {topics_str}

Texto (vista previa):
{raw_text[:3000]}
"""
        invoke_kwargs = {"config": config} if config is not None else {}
        result = structured_llm.invoke(prompt, **invoke_kwargs)

        if result.classification == "no_academico":
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "[classify_document] COMPLETE | session=%s | rejected=non_academic | %dms",
                state["session_id"],
                int(elapsed),
            )
            return {
                "classification": result.classification,
                "classification_confidence": result.confidence,
                "topics": pipeline_topics,
                "topic_tree": topic_tree,
                "errors": ["Content rejected: non-academic material"],
                "status": "rejected_non_academic",
            }

        if result.confidence < settings.classification_confidence_threshold:
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(
                "[classify_document] COMPLETE | session=%s | uncertain | confidence=%.2f | %dms",
                state["session_id"],
                result.confidence,
                int(elapsed),
            )
            return {
                "classification": result.classification,
                "classification_confidence": result.confidence,
                "topics": pipeline_topics,
                "topic_tree": topic_tree,
                "status": "classification_uncertain",
            }

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[classify_document] COMPLETE | session=%s | class=%s | topics=%d | %dms",
            state["session_id"],
            result.classification,
            len(pipeline_topics),
            int(elapsed),
        )
        return {
            "classification": result.classification,
            "classification_confidence": result.confidence,
            "topics": pipeline_topics,
            "topic_tree": topic_tree,
            "status": "classified",
        }
    except Exception as e:
        logger.exception("classify_document failed")
        return {"errors": [f"Classification error: {e}"], "status": "error"}


# ── Node implementation: chunk_and_embed ────────────────────────────────────


def chunk_and_embed(state: IngestorState) -> dict:
    """Split text into semantic chunks and store in ChromaDB with embeddings."""
    import time

    from src.rag import chunk_text, embed_and_store

    t0 = time.monotonic()
    logger.info(
        "[chunk_and_embed] START | session=%s",
        state["session_id"],
    )
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

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[chunk_and_embed] COMPLETE | session=%s | chunks=%d | %dms",
            state["session_id"],
            len(chunk_ids),
            int(elapsed),
        )
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


def _route_after_parse(state: IngestorState) -> str:
    """Short-circuit: rejected or error status → END (no LLM calls)."""
    if state.get("status") in ("rejected", "error"):
        return "end"
    return "classify_document"


def build_ingestor() -> StateGraph:
    """Build and return the Ingestor LangGraph.

    Topology:
        START → parse_document
          ├── status in (\"rejected\", \"error\") → END
          └── status ok → classify_document → chunk_and_embed → END
    """
    builder = StateGraph(IngestorState)

    builder.add_node("parse_document", parse_document)
    builder.add_node("classify_document", classify_document)
    builder.add_node("chunk_and_embed", chunk_and_embed)

    builder.add_edge(START, "parse_document")
    builder.add_conditional_edges(
        "parse_document",
        _route_after_parse,
        {"classify_document": "classify_document", "end": END},
    )
    builder.add_edge("classify_document", "chunk_and_embed")
    builder.add_edge("chunk_and_embed", END)

    return builder
