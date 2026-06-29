# Epic 14 — Topic Descriptions for RAG Retrieval

**Status:** Draft
**Depends on:** Epic 02 (Ingestor/RAG), Epic 11 (Topic Extraction)

> **Goal**: Add a short description to each extracted topic so retrieval queries embed rich semantic content instead of bare topic labels, fixing vocabulary-mismatch misses.

---

## 1. Problem

Every retrieval entry point passes a raw topic label straight to embedding:

```
"Agentes inteligentes" → MiniLM encode → ChromaDB search
```

A topic label is not what the content says. The chunk about agents contains `"ciclo percepción-acción"`, `"PEAS"`, `"reactivo"`, `"entorno"` — none of which appear in the label `"Agentes inteligentes"`. Cosine distance is high despite semantic relevance. Result: valid topics return zero chunks.

This affects exam generation, exercise generation, and answer evaluation — all three pass topic strings directly to `retrieve_chunks(topic=...)`.

---

## 2. Solution

### Pre-compute descriptions at ingest time, use them at query time.

```
# At INGEST (once per topic):
"Agentes inteligentes" → "Una entidad que percibe su entorno mediante sensores
y actúa sobre él con actuadores, persiguiendo objetivos de forma racional."

# At QUERY (replaces bare label):
description → MiniLM encode → ChromaDB search
```

The description has rich vocabulary overlap with the actual chunk text. No extra LLM calls at query time — descriptions are already stored.

### Fallback

If a topic has no description (legacy data, failed extraction), fall back to the topic label. Zero regression on existing sessions.

---

## 3. What Changes

### 3.1 Topic Extraction — generate descriptions

**File**: `back/src/topic_extraction/extract.py`

The segment-level LLM prompt currently extracts topic strings:
```
"Extraé los temas principales de este fragmento..."
→ ["Agentes inteligentes", "Razonamiento lógico", ...]
```

Extend the prompt to also produce a 1-sentence description per topic:
```
"Para cada tema, agregá una breve descripción (máximo 15 palabras)
que explique qué conceptos cubre."
→ [
    {"topic": "Agentes inteligentes", "description": "Entidad que percibe su entorno mediante sensores y actúa con actuadores"},
    {"topic": "Razonamiento lógico", "description": "Inferencia basada en reglas formales y lógica proposicional"},
  ]
```

Return type changes from `list[str]` to `list[dict]`. The unification step (`unify.py`) merges duplicates by topic name and keeps the richer description.

### 3.2 Pipeline output — include descriptions

**File**: `back/src/topic_extraction/__init__.py`

`extract_topics_pipeline()` return dict gains one key:

```python
return {
    "summary": summary,
    "topics": unified_topic_names,           # unchanged: list[str]
    "topic_descriptions": topic_desc_map,    # NEW: dict[str, str]
    "topic_tree": topic_tree_str,
    "segment_count": segment_count,
    "failed_segments": failed_segments,
}
```

`topic_descriptions` is `{"Agentes inteligentes": "Entidad que...", ...}`.

### 3.3 Database — persist descriptions

**File**: `back/src/memory/schema.py`

Add column `topic_descriptions_json TEXT DEFAULT '{}'` to `ingested_documents`.

```sql
ALTER TABLE ingested_documents ADD COLUMN topic_descriptions_json TEXT DEFAULT '{}';
```

New migration in `init_db()`. `insert_ingested_document()` writes the JSON blob. `list_session_files()` returns it.

### 3.4 Retrieval — use description as query

**File**: `back/src/rag/__init__.py`

New function:

```python
def retrieve_by_topic(
    topic: str,
    topic_descriptions: dict[str, str] | None,
    collection_name: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve chunks for a topic, using its description as query if available."""
    query = (topic_descriptions or {}).get(topic, topic)
    return retrieve(query=query, collection_name=collection_name, top_k=top_k)
```

If `topic_descriptions` is `None` or doesn't have the topic → falls back to `topic` label.

### 3.5 Agent retrofit

All three agents already receive session file data (topics list). They just need to also read `topic_descriptions_json` and pass it to `retrieve_by_topic`.

| Agent | File | Change |
|-------|------|--------|
| ExamGenerator | `src/agents/exam_generator.py:237` | `retrieve_chunks(query=topic)` → `retrieve_by_topic(topic, descriptions, collection)` |
| ExerciseGenerator | `src/agents/exercise_generator.py:116` | Same pattern |
| Evaluator | `src/agents/evaluator.py:153` | Same pattern |
| `query_material` tool | `src/tools/query_material.py` | No change — this uses user query text, not topics |

The `retrieve_chunks` tool (`src/tools/__init__.py:43`) gains an optional `topic_descriptions` parameter so agents can pass it through.

### 3.6 Config

**File**: `back/src/config.py`

```python
retrieval_use_topic_descriptions: bool = True  # Master toggle; disable = fallback to labels
```

---

## 4. Implementation Tasks

### Phase 1 — Description Extraction (Effort: M)

| # | File | Change |
|---|------|--------|
| 1.1 | `src/topic_extraction/extract.py` | Extend prompt to request descriptions. Change return type to `list[dict]` with `topic` + `description`. |
| 1.2 | `src/topic_extraction/unify.py` | Merge duplicate topics; keep richer description on collision (`max(len)`) |
| 1.3 | `src/topic_extraction/__init__.py` | Build `topic_descriptions` dict from unified list. Return it in pipeline result. |
| 1.4 | `tests/test_topic_extraction.py` | Add assertions: every topic has non-empty description, descriptions are Spanish, < 20 words. |

### Phase 2 — Persistence (Effort: S)

| # | File | Change |
|---|------|--------|
| 2.1 | `src/memory/schema.py` | Add `topic_descriptions_json` column. Migration in `init_db()`. |
| 2.2 | `src/memory/schema.py` (`insert_ingested_document`) | Write `topic_descriptions_json`. |
| 2.3 | `src/memory/schema.py` (`list_session_files`) | Return `topic_descriptions_json` in result dict. |
| 2.4 | `src/api/router.py` | Parse and return `topic_descriptions` in ingest/file-list endpoints. |
| 2.5 | `src/api/schemas.py` | Add `topic_descriptions: dict | None` to response schemas. |
| 2.6 | `src/agents/ingestor.py` | Pass `topic_descriptions_json` to `insert_ingested_document`. |

### Phase 3 — Retrieval & Agent Wiring (Effort: S)

| # | File | Change |
|---|------|--------|
| 3.1 | `src/rag/__init__.py` | Add `retrieve_by_topic()` function. |
| 3.2 | `src/tools/__init__.py` (`retrieve_chunks`) | Add optional `topic_descriptions` parameter; delegate to `retrieve_by_topic` when present. |
| 3.3 | `src/agents/exam_generator.py` | Load `topic_descriptions_json` from session files, pass to `retrieve_chunks`. |
| 3.4 | `src/agents/exercise_generator.py` | Same as ExamGenerator. |
| 3.5 | `src/agents/evaluator.py` | Same pattern. |
| 3.6 | `src/config.py` | Add `retrieval_use_topic_descriptions` setting. |
| 3.7 | `tests/test_rag.py` | Add test: `retrieve_by_topic` with description returns different (better) results than label-only. |

### Phase 4 — Regression Pass (Effort: S)

| # | File | Change |
|---|------|--------|
| 4.1 | All test files | Fix any test that relied on old return shape from extract/unify. |
| 4.2 | `tests_documentation.md` | Update with new tests. |

---

## 5. Acceptance Criteria

- [ ] `extract_topics_pipeline()` returns `topic_descriptions` dict with one description per topic
- [ ] Every extracted topic has a non-empty, Spanish, <20-word description
- [ ] `topic_descriptions_json` is persisted to DB and returned via API
- [ ] `retrieve_by_topic("Agentes inteligentes", descriptions, ...)` uses description text as query
- [ ] Legacy sessions without descriptions fall back to topic label (no crash, no regression)
- [ ] Exam generation with descriptions retrieves chunks for topics that previously returned empty
- [ ] All existing tests pass
- [ ] `ruff` format/lint clean

---

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM generates bad/low-quality descriptions | LOW | Prompt constrains to 15 words, academic Spanish. If description fails, topic label fallback still works. |
| Description adds latency at ingest | LOW | One extra sentence per topic in the LLM response. For 20 topics = ~20 extra tokens. Negligible. |
| Existing tests break on new return shape | MEDIUM | Phase 4 is dedicated to regression fix. Tests that mock topic extraction return values will need updating. |
| Descriptions stored as JSON in SQLite — no query-time perf impact | NONE | Read once per session, passed in-memory to agents. |
