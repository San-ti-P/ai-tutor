"""RAG module — ChromaDB vector store, MiniLM embeddings, and thematic index.

Provides the persistence and retrieval backbone consumed by all agents:
- Lazy singletons for the ChromaDB PersistentClient and SentenceTransformer model
- Semantic chunking via RecursiveCharacterTextSplitter
- embed_and_store / retrieve for the full write and read paths
- ThematicIndex for hierarchical topic tracking across sessions

Uses ``paraphrase-multilingual-MiniLM-L12-v2`` (configurable via ``embedding_model_name``).
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import observe
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------

_chroma_client: Any | None = None
_embedding_model: SentenceTransformer | None = None
_embedding_lock: threading.Lock = threading.Lock()


def get_chroma_client() -> Any:
    """Return the singleton PersistentClient, creating it on first call.

    Uses settings.chroma_persist_directory for on-disk storage.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        logger.info(
            "ChromaDB client initialised at %s",
            settings.chroma_persist_directory,
        )
    return _chroma_client


def get_embedding_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer model, loading on first call.

    Uses the model name configured in settings.embedding_model_name.
    The model files are cached on disk after the first download.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model_name)
        dims = _embedding_model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded: %s (dim=%d)",
            settings.embedding_model_name,
            dims,
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Split *text* into semantic chunks using a RecursiveCharacterTextSplitter.

    Separators try semantic boundaries first (paragraphs, lines, sentences)
    and fall back to character-level splits. Every chunk carries the supplied
    *metadata* merged with its positional ``chunk_index``.

    Returns an empty list when *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", ". ", "! ", "? ", "\n", " ", ""],
    )

    chunks = splitter.split_text(text)
    base_meta = metadata or {}

    return [
        Document(
            page_content=chunk,
            metadata={**base_meta, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Write path — embed & store
# ---------------------------------------------------------------------------


@observe(name="rag_embed_store", as_type="embedding")
def embed_and_store(
    chunks: list[str],
    metadatas: list[dict[str, Any]] | None,
    collection_name: str,
) -> list[str]:
    """Embed *chunks* with the local model and persist them in ChromaDB.

    Args:
        chunks: Text content of each chunk.
        metadatas: Per-chunk metadata dicts.  If ``None``, empty dicts are used.
        collection_name: ChromaDB collection name (typically session-scoped).

    Returns:
        The generated ChromaDB document IDs, one per chunk.
    """
    if not chunks:
        return []

    model = get_embedding_model()
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    with _embedding_lock:
        embeddings = model.encode([f"passage: {c}" for c in chunks]).tolist()

    if metadatas is None:
        metadatas = [{} for _ in chunks]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(
        "Stored %d chunks in collection '%s'",
        len(chunks),
        collection_name,
    )
    return chunk_ids


# ---------------------------------------------------------------------------
# Read path — retrieve
# ---------------------------------------------------------------------------




@observe(name="rag_retrieve", as_type="retriever")
def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 5,
    topic_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic similarity search over ChromaDB with optional topic filtering.

    When *topic_filter* is given, only chunks whose ``metadata["topic"]``
    starts with the filter prefix are returned (prefix match, not substring).

    Returns an empty list when the collection does not exist or no chunks
    match the query.

    Each result dict contains:
        - ``chunk_id``: ChromaDB document ID
        - ``text``: chunk content
        - ``metadata``: original metadata dict
        - ``similarity_score``: cosine distance (lower = more similar)
    """
    model = get_embedding_model()
    client = get_chroma_client()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.debug(
            "Collection '%s' not found, returning empty results",
            collection_name,
        )
        return []

    collection_count = collection.count()
    if collection_count == 0:
        return []

    # Fetch more results when post-filtering so we still have top_k after
    # discarding chunks whose topic doesn't match the prefix.
    fetch_k = min(top_k * 3, collection_count) if topic_filter else top_k

    with _embedding_lock:
        query_embedding = model.encode([f"query: {query}"]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    ids_list: list[str] = results.get("ids", [[]])[0]  # pyright: ignore[reportAssignmentType]
    if not ids_list:
        return []

    docs_list: list[str] = results.get("documents", [[""]])[0]  # pyright: ignore[reportAssignmentType]
    metas_list: list[dict[str, Any]] = results.get("metadatas", [[{}]])[0]  # pyright: ignore[reportAssignmentType]
    dists_list: list[float] = results.get("distances", [[0.0]])[0]  # pyright: ignore[reportAssignmentType]

    output: list[dict[str, Any]] = []
    for i in range(len(ids_list)):
        meta = metas_list[i] if i < len(metas_list) else {}

        # Post-filter: prefix match on metadata["topic"]
        if topic_filter and not meta.get("topic", "").startswith(topic_filter):
            continue

        output.append(
            {
                "chunk_id": ids_list[i],
                "text": docs_list[i] if i < len(docs_list) else "",
                "metadata": meta,
                "similarity_score": dists_list[i] if i < len(dists_list) else 0.0,
            }
        )

        if len(output) >= top_k:
            break

    return output


# ---------------------------------------------------------------------------
# Topic-description-aware retrieval (TDR-07)
# ---------------------------------------------------------------------------


def retrieve_by_topic(
    topic: str,
    topic_descriptions: dict[str, str] | None,
    collection_name: str,
    top_k: int = 5,
    topic_filter: str | None = None,
    topic_tree: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks using topic description as query, falling back to label.

    When the topic has a description in *topic_descriptions*, that
    description is used as the embedding query instead of the bare label.
    This keeps the semantic search focused on the matched topic without
    contaminating it with descriptions from parent, children, or sibling
    topics.

    Args:
        topic: The topic label (e.g. "Agentes inteligentes").
        topic_descriptions: Mapping from topic → description.  When the topic
            has an entry, the description is used as the embedding query
            instead of the bare label.  ``None`` or missing key → label used.
        collection_name: ChromaDB collection name.
        top_k: Number of chunks to retrieve.
        topic_filter: Optional prefix filter on ``metadata["topic"]``.
        topic_tree: Unused.  Kept for backward compatibility.

    Returns:
        List of chunk dicts from ``retrieve()``.
    """
    descs = topic_descriptions or {}
    query = descs.get(topic, topic)

    # Ensure non-empty query
    if not query or not query.strip():
        query = topic
    return retrieve(
        query=query,
        collection_name=collection_name,
        top_k=top_k,
        topic_filter=topic_filter,
    )


class ThematicIndex:
    """Hierarchical topic tree with merge support for incremental ingestion.

    Topics are represented as slash-separated paths (e.g. ``"Cálculo/Derivadas"``)
    and stored in a nested ``dict``.  Merging two indexes performs a deep-merge
    that preserves existing branches.
    """

    def __init__(self) -> None:
        self._tree: dict[str, Any] = {}

    # -- mutation -----------------------------------------------------------

    def add_topics(self, topics: list[str]) -> None:
        """Insert *topics* into the tree.

        Each string may be a slash-separated path (e.g. ``"math/algebra/linear"``)
        that creates intermediate nodes as needed.  Duplicate paths are idempotent.
        """
        for topic in topics:
            parts = [p.strip() for p in topic.split("/") if p.strip()]
            if not parts:
                continue
            node = self._tree
            for part in parts:
                if part not in node:
                    node[part] = {}
                node = node[part]

    def merge(self, other: ThematicIndex) -> None:
        """Deep-merge *other*'s topic tree into this one.

        Existing branches are preserved; new branches are added recursively.
        """

        def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
            for key, value in source.items():
                if key not in target:
                    target[key] = {}
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _deep_merge(target[key], value)

        _deep_merge(self._tree, other._tree)

    # -- query --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the full tree as a nested dict."""
        return self._tree

    def search(self, topic: str) -> list[str]:
        """Walk the tree to *topic* and return the keys of its children.

        If *topic* is not found or has no children, returns an empty list.
        """
        parts = [p.strip() for p in topic.split("/") if p.strip()]
        node = self._tree
        for part in parts:
            if part not in node:
                return []
            node = node[part]
        return list(node.keys())


# ---------------------------------------------------------------------------
# LLM Topic Matching
# ---------------------------------------------------------------------------


async def match_user_topics_to_session(
    user_topics: list[str],
    session_topics: list[str],
    topic_descriptions: dict[str, str] | None = None,
    topic_tree: dict[str, Any] | None = None,
    weak_topics: list[str] | None = None,
) -> dict[str, str]:
    """Match user-requested topics to session topics via LLM fuzzy matching.

    Returns a dict mapping matched session topic names to a RAG query
    sentence synthesized by the LLM.  The sentence combines the user's
    query intent with the matched topic's academic content, producing a
    richer embedding query than the bare topic label or raw description.

    When *weak_topics* is provided, the LLM is told to prioritize matches
    with topics the student has struggled with, so the student practices
    their weak areas even when asking about related broader concepts.

    When *topic_tree* is provided, parent topics (those with children) are
    tagged in the catalogue so the LLM prefers them over individual leaves
    for broad user queries.

    The LLM call is small (< 300 tokens) — one per request, not per chunk.
    Uses ``get_llm()`` with temperature=0 for deterministic matching.
    """
    if not user_topics or not session_topics:
        return {}

    from src.llm import get_llm

    # Identify parent topics (keys in topic_tree that have non-empty children)
    parent_topics: set[str] = set()
    if topic_tree:

        def _collect_parents(node: dict[str, Any]) -> None:
            for key, children in node.items():
                if isinstance(children, dict) and children:
                    parent_topics.add(key)
                    _collect_parents(children)

        _collect_parents(topic_tree)

    descs = topic_descriptions or {}
    # Build a catalogue line per session topic
    topic_lines: list[str] = []
    for t in session_topics:
        desc = descs.get(t, "").strip()
        tag = " [PADRE]" if t in parent_topics else ""
        if desc:
            topic_lines.append(f"- {t}{tag}: {desc}")
        else:
            topic_lines.append(f"- {t}{tag}")

    catalogue = "\n".join(topic_lines)
    user_list = "\n".join(f"- {t}" for t in user_topics)

    # Weak topics hint
    weak_hint = ""
    if weak_topics:
        wt_list = "\n".join(f"- {t}" for t in weak_topics)
        weak_hint = (
            f"\nTemas débiles del estudiante (priorizá matchear con estos):\n{wt_list}\n\n"
        )

    prompt = (
        "Sos un asistente que empareja temas académicos y construye "
        "oraciones de búsqueda semántica. Tu tarea es:\n"
        "1. Asociar cada tema solicitado por un estudiante con el tema más "
        "relevante del catálogo de la sesión.\n"
        "2. Para cada match, generar UNA oración en español académico "
        "(máximo 30 palabras) que sirva como query de búsqueda vectorial. "
        "La oración debe combinar la intención del estudiante con la "
        "descripción del tema del catálogo.\n\n"
        "Reglas:\n"
        "1. Si un tema del estudiante coincide exactamente o es claramente "
        "equivalente a un tema del catálogo, usá ese tema.\n"
        "2. Si un tema del estudiante es una versión más general, preferí "
        "el PADRE sobre un hijo individual.\n"
        "3. Los temas marcados [PADRE] agrupan varios subtemas.\n"
        "4. Si un tema del estudiante NO tiene equivalente claro, "
        "devolvé \"<<NO_MATCH>>\".\n"
        "5. La oración de búsqueda debe ser una frase completa, no una "
        "lista de palabras sueltas. Usá vocabulario académico preciso.\n"
        "6. Respondé SOLO con el mapeo, una línea por tema del estudiante, "
        "en el formato exacto:\n"
        "   tema_estudiante → tema_catalogo → oración_de_búsqueda\n\n"
        f"Catálogo de temas disponibles:\n{catalogue}\n"
        f"{weak_hint}"
        f"Temas solicitados por el estudiante:\n{user_list}\n\n"
        "Mapeo:"
    )

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.exception("LLM topic matching failed")
        return {}

    # Parse LLM response: "tema_estudiante → tema_catalogo → oración_de_búsqueda"
    result: dict[str, str] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "→" not in line:
            continue
        parts = [p.strip() for p in line.split("→")]
        if len(parts) < 2:
            continue
        matched = parts[1]
        if matched == "<<NO_MATCH>>":
            continue
        # Only keep matches that actually exist in session_topics
        if matched not in session_topics:
            continue
        # Use LLM-produced query sentence if available, else fall back to
        # the topic's own description from the catalogue.
        if len(parts) >= 3 and parts[2].strip():
            result[matched] = parts[2].strip()
        else:
            result[matched] = descs.get(matched, matched)

    return result
