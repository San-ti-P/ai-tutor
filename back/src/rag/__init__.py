"""RAG module — ChromaDB vector store, sentence-transformers embeddings, and thematic index.

Provides the persistence and retrieval backbone consumed by all agents:
- Lazy singletons for the ChromaDB PersistentClient and SentenceTransformer model
- Semantic chunking via RecursiveCharacterTextSplitter
- embed_and_store / retrieve for the full write and read paths
- ThematicIndex for hierarchical topic tracking across sessions
"""

from __future__ import annotations

import logging
import uuid

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------

_chroma_client: chromadb.PersistentClient | None = None
_embedding_model: SentenceTransformer | None = None


def get_chroma_client() -> chromadb.PersistentClient:
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


def chunk_text(text: str, metadata: dict | None = None) -> list[Document]:
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
        separators=["\n\n", "\n", ". ", " ", ""],
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


def embed_and_store(
    chunks: list[str],
    metadatas: list[dict] | None,
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
    embeddings = model.encode(chunks).tolist()

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


def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 5,
    topic_filter: str | None = None,
) -> list[dict]:
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

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    ids_list: list[str] = results.get("ids", [[]])[0]  # pyright: ignore[reportAssignmentType]
    if not ids_list:
        return []

    docs_list: list[str] = results.get("documents", [[""]])[0]  # pyright: ignore[reportAssignmentType]
    metas_list: list[dict] = results.get("metadatas", [[{}]])[0]  # pyright: ignore[reportAssignmentType]
    dists_list: list[float] = results.get("distances", [[0.0]])[0]  # pyright: ignore[reportAssignmentType]

    output: list[dict] = []
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
# ThematicIndex
# ---------------------------------------------------------------------------


class ThematicIndex:
    """Hierarchical topic tree with merge support for incremental ingestion.

    Topics are represented as slash-separated paths (e.g. ``"Cálculo/Derivadas"``)
    and stored in a nested ``dict``.  Merging two indexes performs a deep-merge
    that preserves existing branches.
    """

    def __init__(self) -> None:
        self._tree: dict = {}

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

        def _deep_merge(target: dict, source: dict) -> None:
            for key, value in source.items():
                if key not in target:
                    target[key] = {}
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _deep_merge(target[key], value)

        _deep_merge(self._tree, other._tree)

    # -- query --------------------------------------------------------------

    def to_dict(self) -> dict:
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
