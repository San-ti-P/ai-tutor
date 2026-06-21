# Epic 11: Topic Extraction Refinement — Long-Document NLP + Sequential LLM Pipeline

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.3 thematic index, §5.1 step 6, RF-06)
**Source Epic:** [epic-02-ingestor.md](./epic-02-ingestor.md) (US-2.5 Thematic index extraction, US-2.10 extract_topics tool)
**Delivery window:** Entrega 2 → Entrega 3 (refines existing ingest pipeline)

## Context

The current topic extraction has a hard truncation problem. Both `classify_document` (ingestor line 123) and `extract_topics` (tools/__init__.py line 202) cap input at 3000–5000 characters before sending to the LLM. For a 50-page academic PDF producing ~150,000 characters of markdown, the LLM sees only 2–3% of the content. Topics from pages 3–50 are invisible to extraction.

This is not a minor limitation — it violates RF-06 (thematic index must cover ALL ingested material) and produces exams that miss topics from the majority of the document.

The fix is a sequential chunk-and-extract pipeline: preprocess the full markdown, split into LLM-sized segments, extract topics from each segment, then unify results across all segments. Preprocessing (stopword removal, stemming) is applied at the **unification** stage — not before feeding to the LLM — because LLMs perform worse on stemmed/unnatural text.

## Scope

**In scope**
- Spanish NLP preprocessing utilities: stopword removal, Snowball stemming (for unification only)
- Configurable text segmentation into LLM-friendly chunks (token-aware, not raw character split)
- Sequential topic extraction across chunks (LLM called once per chunk)
- Optional parallel mode behind a config flag (sequential by default; respects rate limits)
- Topic unification pipeline: stem overlap + similarity scoring to deduplicate/merge topics
- Final hierarchical topic tree construction from unified topic list
- Replace single-pass truncation in `classify_document` and `extract_topics` with the new pipeline
- Pluggable architecture: the pipeline is a reusable module callable by Ingestor and any agent
- Preserve existing `ThematicIndex.merge()` — unification feeds into it
- Configurable chunk size and overlap (tune for token budget vs. topic granularity)

**Out of scope**
- Changing the LLM model used for topic extraction
- Modifying the ChromaDB chunking/embedding pipeline (this is about topic extraction, not RAG chunks)
- OCR pipeline changes
- Changing the document classification logic itself (just the topic extraction within it)
- Real-time/streaming topic extraction (batch only)
- Multi-language NLP (Spanish only — hardcoded for this course)

## Functional Requirements

- **TXR-01** The system must preprocess Spanish text for unification: remove Spanish stopwords and apply Snowball Spanish stemming to topic strings only — never to text fed to the LLM.
- **TXR-02** Long documents must be split into configurable token-aware segments (default: 4000 chars with 200 overlap) ensuring no chunk cuts mid-topic where possible.
- **TXR-03** Topic extraction must cover the entire document by calling the LLM sequentially on each segment; no segment is silently skipped.
- **TXR-04** When `settings.topic_extraction_parallel` is true, segments are processed concurrently (respecting provider rate limits); when false, sequentially.
- **TXR-05** After all segments are processed, a unification step merges duplicate/similar topics using stem overlap similarity (Jaccard on stemmed keywords) and a configurable similarity threshold.
- **TXR-06** The unified topic list feeds into a final LLM call that produces the hierarchical topic tree (or, if unification is clean enough, builds it deterministically from the flat list).
- **TXR-07** `classify_document` node in the Ingestor must use this pipeline instead of `raw_text[:3000]`.
- **TXR-08** `extract_topics` tool must use this pipeline instead of `content[:5000]`.
- **TXR-09** The pipeline must be a single reusable module (`src/topic_extraction/`) callable by any agent — not duplicated in Ingestor and tools.
- **TXR-10** Existing behavior for short documents (<1 segment) must be preserved: no degradation for small files.

## Non-Functional Requirements

- **TXR-NFR-01** Ingest of a 50-page PDF must complete topic extraction in under 45 seconds (sequential mode; RNF-01 ceiling is 30s for full exam gen — topic extraction is a sub-step).
- **TXR-NFR-02** Topic unification must produce ≤20 unique topics for a typical 50-page academic document (avoids topic explosion from near-duplicates).
- **TXR-NFR-03** NLP preprocessing must not add >2 seconds to total pipeline time for a 50-page document.
- **TXR-NFR-04** Parallel mode must respect provider rate limits — Groq free tier is 30 requests/min; Ollama has no limit but can OOM.
- **TXR-NFR-05** The pipeline must degrade gracefully on LLM failure for individual segments: skip the failed segment, log a warning, continue processing remaining segments.
- **TXR-NFR-06** The `ThematicIndex` merge remains the final persistence mechanism — unification produces clean topics, ThematicIndex stores them.

## Technical Notes

### Why NLP Preprocessing Is for Unification Only

LLMs are autoregressive models trained on natural language. Feeding them stemmed text like `"agnt intelig pued percibir entorn"` degrades their ability to recognize concepts, reason about topic boundaries, and produce coherent output. Stemming destroys the syntactic signals that LLMs rely on.

NLP preprocessing belongs in the **unification step**: after all segments produce their topic lists, each topic string is stemmed, stopwords are removed, and the resulting keyword sets are compared via Jaccard similarity. This identifies `"Agentes inteligentes y su entorno"` and `"El entorno de los agentes inteligentes"` as the same topic (stem set: `{agnt, intelig, entorn}`) without polluting the LLM's input.

### Pipeline Architecture

```
                             ┌─────────────────────┐
                             │   Full Markdown Text │
                             └──────────┬──────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Segment (split)   │
                              │  token-aware,      │
                              │  overlap-preserving│
                              └─────────┬─────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
              ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
              │  Segment 1 │       │  Segment 2 │  ...  │  Segment N │
              │  → LLM     │       │  → LLM     │       │  → LLM     │
              └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
               topics_1            topics_2            topics_N
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Unification       │
                              │  1. Stem + stopword│
                              │  2. Jaccard pairs  │
                              │  3. Cluster/merge  │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  Tree Construction │
                              │  LLM or deterministic│
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  ThematicIndex     │
                              │  (persist/merge)   │
                              └────────────────────┘
```

### NLP Stack (Spanish)

```python
# Stopwords: NLTK's Spanish stopword list (313 words)
from nltk.corpus import stopwords
spanish_stopwords = set(stopwords.words('spanish'))

# Stemmer: Snowball Spanish stemmer (light stemming, good for academic text)
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

# Unification similarity: Jaccard on stemmed keyword sets
def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
```

Both NLTK resources are already commonly available. The project may need to add `nltk` to dependencies and run `nltk.download('stopwords')` once.

### Segment Configuration

```python
# config.py additions
topic_segment_size: int = 4000       # chars per LLM call (≈1000-1500 tokens)
topic_segment_overlap: int = 200     # overlap to avoid boundary cuts
topic_extraction_parallel: bool = False  # sequential by default
topic_similarity_threshold: float = 0.6  # Jaccard threshold for merging topics
max_topics_per_document: int = 30    # cap to prevent explosion
```

### Sequential vs Parallel

| Mode | When | Tradeoff |
|------|------|----------|
| Sequential (default) | Groq free tier, low-memory Ollama | Respects rate limits, predictable memory. 50-page doc ≈ 38 segments → ~38 LLM calls |
| Parallel (`topic_extraction_parallel=true`) | Local Ollama with GPU, paid Groq tier | 3-5× faster but may OOM or hit rate limits. Uses `asyncio.gather` with semaphore |

Implementation uses `asyncio.Semaphore(max_concurrency)` in parallel mode to cap simultaneous LLM calls.

### Error Handling Per Segment

```python
# Each segment extraction is wrapped:
try:
    topics = await extract_topics_from_segment(segment_text)
    all_topics.append(topics)
except Exception as e:
    logger.warning("Segment %d/%d failed: %s", i, total, e)
    failed_segments.append(i)
    # Continue with remaining segments — don't abort
```

## New Module Structure

```
back/src/
├── topic_extraction/          # NEW — single reusable module
│   ├── __init__.py            # Public API: extract_topics_pipeline(text) -> dict
│   ├── preprocess.py          # stopword removal, stemming (for unification only)
│   ├── segment.py             # token-aware text segmentation
│   ├── extract.py             # LLM call per segment (sequential or parallel)
│   ├── unify.py               # Jaccard clustering, dedup, merge
│   └── tree.py                # Hierarchical tree from unified flat list
├── agents/
│   └── ingestor.py            # classify_document uses extract_topics_pipeline
├── tools/
│   └── __init__.py            # extract_topics tool uses extract_topics_pipeline
└── config.py                  # + topic extraction settings (segment size, threshold, parallel)
```

### Integration Points

**1. Ingestor `classify_document` node** (`back/src/agents/ingestor.py:93`)

Before:
```python
prompt = f"""Analizá el siguiente texto académico...
Texto:
{raw_text[:3000]}
"""
```

After:
```python
from src.topic_extraction import extract_topics_pipeline

topic_result = await extract_topics_pipeline(raw_text)
# topic_result = {"topics": [...], "topic_tree": {...}, "summary": "..."}
# Use topic_result["topics"] in the classification prompt (classification itself still uses a preview)
prompt = f"""Analizá el siguiente texto académico...
Temas detectados en el documento completo: {', '.join(topic_result['topics'])}
Texto (vista previa):
{raw_text[:3000]}
"""
```

**2. `extract_topics` tool** (`back/src/tools/__init__.py:136`)

Before: `content_preview = content[:5000]` → single LLM call.

After: Delegates entirely to `extract_topics_pipeline(content)` — returns the same dict shape (`summary`, `topics`, `topic_tree`) so no caller changes needed.

### Interaction with ThematicIndex

The existing `ThematicIndex.add_topics()` and `ThematicIndex.merge()` (`back/src/rag/__init__.py:257-286`) remain the persistence layer. The unification step in this epic produces clean, deduplicated topics; `ThematicIndex` receives them as a flat list and builds the tree structure via slash-separated paths.

### Dependencies

- `nltk` — Spanish stopwords + Snowball stemmer. If not already a dependency, add to `pyproject.toml`.
- No other new dependencies. All LLM calls use existing `src/llm.py` factory.
- No new vector store or database.

## Test Coverage

- New unit tests for `src/topic_extraction/preprocess.py`: stopword removal, stemming correctness on sample Spanish academic text
- New unit tests for `src/topic_extraction/segment.py`: segment boundaries, overlap, short-text passthrough
- New unit tests for `src/topic_extraction/unify.py`: Jaccard on identical topics (expect merge), distinct topics (expect keep separate), threshold edge cases
- New integration test: full pipeline on real 50-page PDF, verify ≥15 topics extracted, ≤30 after unification
- Existing tests for `classify_document` and `extract_topics` must pass with the new pipeline (mock LLM for unit; real LLM for integration)
- Update `tests_documentation.md` when adding integration tests

## User Stories

### US-11.1: Spanish NLP preprocessing utilities
- **As a** developer building the topic unification step
- **I want** functions to remove Spanish stopwords and apply Snowball stemming to topic strings
- **So that** I can compute accurate similarity between topics extracted from different segments
- **Acceptance criteria:**
  - `remove_stopwords(text: str) -> str` removes all 313 NLTK Spanish stopwords
  - `stem_topic(topic: str) -> set[str]` returns a set of stemmed, non-stopword keywords
  - `jaccard_similarity(set_a, set_b) -> float` returns 0.0–1.0
  - Functions never mutate the original text fed to LLM — only topic strings for unification
  - Unit tests with real Spanish academic phrases (e.g., "Agentes inteligentes y su entorno" stems to `{agnt, intelig, entorn}`)
- **Dependencies:** `nltk` package + `nltk.download('stopwords')`
- **Maps to:** TXR-01, TXR-NFR-03

### US-11.2: Token-aware text segmentation
- **As a** topic extraction pipeline
- **I want** long markdown text split into LLM-friendly segments with configurable size and overlap
- **So that** no segment exceeds the LLM's context window and topic boundaries are preserved
- **Acceptance criteria:**
  - `segment_text(text, size=4000, overlap=200) -> list[str]` produces N segments
  - Overlap prevents cutting mid-sentence or mid-paragraph
  - Short documents (<1 segment) return a single-element list (passthrough)
  - Segment count for 150K-char document ≈ 38 segments (150000 / (4000-200))
  - Empty text returns empty list; whitespace-only returns empty list
- **Dependencies:** —
- **Maps to:** TXR-02, TXR-10

### US-11.3: Sequential topic extraction across segments
- **As a** student uploading a long PDF
- **I want** the system to analyze every page for topics, not just the first few
- **So that** the thematic index covers the entire document
- **Acceptance criteria:**
  - Each segment is sent to the LLM with a consistent prompt: "Extraé los temas principales de este fragmento..."
  - All segment results are collected into a flat list of topic strings
  - Failed segments are logged and skipped — remaining segments continue
  - A 50-page PDF produces topics from pages 1–50, not just pages 1–2
  - The `classify_document` node and `extract_topics` tool both use this pipeline
- **Dependencies:** US-11.2
- **Maps to:** TXR-03, TXR-07, TXR-08, TXR-NFR-01, TXR-NFR-05

### US-11.4: Optional parallel extraction mode
- **As a** developer running on local Ollama with GPU
- **I want** segments to be processed concurrently when `topic_extraction_parallel=true`
- **So that** topic extraction is 3-5× faster when rate limits and memory permit
- **Acceptance criteria:**
  - Parallel mode uses `asyncio.gather` with configurable `Semaphore(max_concurrency)`
  - Sequential mode is the default (`topic_extraction_parallel=false`)
  - Parallel mode produces identical topic results to sequential mode (same LLM calls, different order)
  - Semaphore prevents OOM on local Ollama (default max_concurrency=4)
  - Groq rate limit errors are caught and retried with backoff
- **Dependencies:** US-11.3
- **Maps to:** TXR-04, TXR-NFR-04

### US-11.5: Topic unification via NLP similarity
- **As a** system
- **I want** near-duplicate topics from different segments merged into single canonical topics
- **So that** the thematic index is clean — no "Agentes inteligentes", "Agentes inteligentes y entorno", "Entorno de agentes" as three separate entries
- **Acceptance criteria:**
  - Each topic string is stemmed and stopword-removed to produce a keyword set
  - All topic pairs are compared via Jaccard similarity
  - Topics with Jaccard ≥ `topic_similarity_threshold` (default 0.6) are merged
  - Merged topics take the longest/richest string variant as canonical name
  - A 50-page academic PDF produces ≤20 unique topics after unification
  - Deterministic: given the same segment topics, unification always produces the same result
- **Dependencies:** US-11.1, US-11.3
- **Maps to:** TXR-01, TXR-05, TXR-NFR-02

### US-11.6: Hierarchical tree construction
- **As a** downstream agent (ExamGenerator, ExerciseGenerator)
- **I want** unified topics structured as a hierarchical tree with slash-separated paths
- **So that** retrieval can filter by topic prefix (e.g., "Agentes/Tipos" → all chunks about agent types)
- **Acceptance criteria:**
  - If unification produces clean, distinct topics: use a final LLM call to organize them hierarchically
  - If the topic list is small (<5 topics): deterministic tree construction is acceptable
  - Output format matches existing `topic_tree` field: nested dict as JSON string
  - Example: `{"Agentes": {"Tipos": {}, "Entorno": {}}, "RAG": {"Chunking": {}}}`
  - Feeds into `ThematicIndex.add_topics()` for persistence
- **Dependencies:** US-11.5
- **Maps to:** TXR-06, TXR-09

### US-11.7: Reusable pipeline module
- **As a** developer maintaining the codebase
- **I want** the entire topic extraction pipeline in `src/topic_extraction/`
- **So that** Ingestor and `extract_topics` tool share one implementation — no duplication
- **Acceptance criteria:**
  - `src/topic_extraction/__init__.py` exposes `extract_topics_pipeline(text) -> dict`
  - Return dict shape: `{"summary", "topics", "topic_tree", "segment_count", "failed_segments"}`
  - `classify_document` imports and calls `extract_topics_pipeline`
  - `extract_topics` tool imports and calls `extract_topics_pipeline`
  - Old `content[:5000]` truncation is removed from the tool
  - All existing tests pass — import path changes only
- **Dependencies:** US-11.3, US-11.5, US-11.6
- **Maps to:** TXR-09, TXR-07, TXR-08

### US-11.8: Configuration and tuning
- **As a** developer tuning for different LLM providers
- **I want** segment size, overlap, similarity threshold, and parallel mode configurable via settings
- **So that** I can optimize for Groq's 30K TPM vs Ollama's unlimited local throughput
- **Acceptance criteria:**
  - `settings.topic_segment_size` (default 4000)
  - `settings.topic_segment_overlap` (default 200)
  - `settings.topic_extraction_parallel` (default false)
  - `settings.topic_similarity_threshold` (default 0.6)
  - `settings.max_topics_per_document` (default 30)
  - All settings documented in `.env.example`
  - Changing settings does not require code changes
- **Dependencies:** —
- **Maps to:** TXR-02, TXR-04, TXR-05

### US-11.9: Integration with existing ingest flow
- **As a** student uploading a 50-page PDF
- **I want** the ingest confirmation to show topics extracted from the entire document
- **So that** I can verify the system understood all chapters, not just the introduction
- **Acceptance criteria:**
  - Full ingest flow (parse → classify → chunk/embed) uses the new pipeline
  - UI confirmation shows the complete unified topic list and tree
  - ChromaDB chunks are tagged with their corresponding topic from the unified set
  - Incremental ingestion: second PDF adds its topics, unified against existing topics
  - `ThematicIndex.merge()` correctly merges unified trees
- **Dependencies:** US-11.7
- **Maps to:** RF-03, RF-06, §5.1 step 6, TXR-06

### US-11.10: Full regression and integration test pass
- **As a** developer merging this epic
- **I want** all existing tests to pass and new pipeline tests to validate correctness
- **So that** the refinement introduces zero regressions and measurably improves topic coverage
- **Acceptance criteria:**
  - All 57 existing unit tests pass (no semantic changes to non-topic code)
  - All 12 existing integration tests pass (real PDF topic extraction now covers full document)
  - New pipeline unit tests: preprocessing, segmentation, unification, tree construction
  - New integration test: 50-page PDF produces ≥15 topics, ≤30 after unification, tree is non-empty
  - `ruff` format/lint clean
  - `tests_documentation.md` updated with new integration tests
- **Dependencies:** US-11.1 through US-11.9
- **Maps to:** TXR-NFR-01, TXR-NFR-02, TXR-NFR-05

