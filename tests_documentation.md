# Test Documentation — ai-tutor

## Integration Tests

### Definition

Integration tests are tests that require **real external resources** — real LLM calls, real embedding models, real ChromaDB, or real PDF files. They are **not** run by default because they are slow, expensive, and depend on external services (Groq/Ollama, HuggingFace model downloads).

Tests that mock LLMs, embeddings, or ChromaDB are **unit tests** — they run on every commit.

### How to Run

```bash
# Unit tests only (default — fast, no external deps):
pytest tests/ -v

# Integration tests (requires Ollama or Groq API key):
pytest tests/ -m integration -v

# Everything:
pytest tests/ -v -m ""
```

### Provider Configuration

| Provider | Env | Model Config | Rate Limits |
|----------|-----|-------------|-------------|
| Ollama (local) | `LLM_PROVIDER=ollama` | `ollama_model_name` (default: `gemma4:e4b-it-q8_0`) | None — runs on your hardware |
| Groq (cloud) | `LLM_PROVIDER=groq` + `GROQ_API_KEY` | `groq_model_name` (default: `llama-3.1-8b-instant`) | Free tier: 30K TPM |

Set in `back/.env`:
```env
LLM_PROVIDER=ollama
GROQ_API_KEY=gsk_...      # only needed for groq
```

---

## Test Inventory

### Integration Tests (16 tests — `-m integration`)

| # | Test | File | Real Resource | What It Proves |
|---|------|------|--------------|----------------|
| 1 | `test_parse_real_pdf` | `test_ingestor.py` | markitdown + PDF | Real academic PDF parses to Spanish text with agent-theory terms |
| 2 | `test_ingest_real_pdf` | `test_ingestor.py` | ChromaDB + SentenceTransformer | Full ingestion pipeline populates a collection |
| 3 | `test_classify_real_pdf` | `test_ingestor.py` | ChatOllama/ChatGroq | LLM correctly classifies real PDF as `apunte_teorico` |
| 4 | `test_retrieve_from_real_pdf` | `test_ingestor.py` | ChromaDB + SentenceTransformer | Semantic search finds relevant chunks for "agentes inteligentes" |
| 5 | `test_real_chunking` | `test_rag.py` | RecursiveCharacterTextSplitter | Real academic text produces 5+ semantic chunks, all ≤512 tokens |
| 6 | `test_real_embed_and_retrieve` | `test_rag.py` | SentenceTransformer + ChromaDB | Embeddings produce distinct similarity scores (not all identical) |
| 7 | `test_real_topic_extraction` | `test_rag.py` | ChatOllama/ChatGroq | LLM extracts ≥3 topics + summary from real PDF text |
| 8 | `test_generate_exam_from_real_pdf` | `test_exam_generator.py` | ChatOllama/ChatGroq + all above | Full pipeline: PDF → chunks → LLM exam. Verifies R1, R2, R3, R6 |
| 9 | `test_anti_hallucination_catches_fabrication` | `test_exam_generator.py` | SentenceTransformer + ChromaDB | Claim-level validation catches fabricated astrophysics claims against real agent-theory chunks. Verifies R3 |
| 10 | `test_evaluate_correct_answer` | `test_evaluator.py` | ChatOllama + SentenceTransformer + ChromaDB (real PDF) | PRD Case 3: correct answer scored ≥6 by real LLM with RAG backing. Verifies EVAL-SPEC-01, EVAL-SPEC-02 |
| 11 | `test_evaluate_partially_correct` | `test_evaluator.py` | ChatOllama + SentenceTransformer + ChromaDB (real PDF) | PRD Case 8: partially correct answer scored mid-range. Verifies EVAL-SPEC-03, EVAL-SPEC-04 |
| 12 | `test_evaluate_wrong_language` | `test_evaluator.py` | None (rule-based guard) | PRD Case 12: gibberish rejected with structured cannot_evaluate response. Verifies EVAL-SPEC-09, EVAL-SPEC-10 |
| 13 | `test_full_pipeline_real_pdf` | `test_topic_extraction.py` | Ollama Cloud LLM + markitdown + real PDF | Epic 11 NFR: full topic extraction pipeline on real academic PDF. Verifies ≥3 topics extracted, topic_tree non-empty, segment_count > 0. Skips gracefully if Ollama Cloud API key missing or network fails. |
| 14 | `test_full_lifecycle_create_upload_chat_profile` | `test_session_lifecycle.py` | SQLite + ChromaDB + mocked LLM | Epic 9: full session lifecycle — create session, insert file metadata, insert evaluations, per-session profile aggregation, orchestrator graph flow with session context. |
| 15 | `test_delete_session_cascades` | `test_session_lifecycle.py` | SQLite + ChromaDB | Epic 9: session delete cascades ingested_documents and drops ChromaDB collection. |
| 16 | `test_integration_exam_generation_flow` | `test_session_lifecycle.py` | SQLite + ChromaDB + mocked exam generator | Epic 9: exam generation tool integration through session lifecycle. |

### Unit Tests (105 tests — default)

| File | Tests | Mocks | Covers |
|------|-------|-------|--------|
| `test_topic_extraction.py` | 48 | `get_llm` (AsyncMock), NLTK | TXR-01 through TXR-10: Spanish NLP preprocessing, markdown segmentation, sequential LLM extraction, Jaccard unification, tree construction, full pipeline API. Edge cases: Jaccard boundary (0.59/0.60), accented Unicode, long topic strings (>200 chars). |
| `test_exam_generator.py` | 15 | ChatGroq, retrieve_chunks, embeddings | Graph topology, state transitions, retry logic, dedup, output structure, PRD cases (mocked) |
| `test_ingestor.py` | 12 | ChatGroq | Parse, classify, incremental ingestion, non-academic rejection, image rejection, error handling. TXR-07 integration: classify_document → pipeline delegation. |
| `test_rag.py` | 10 | Embedding model, ChromaDB client | Chunking, ThematicIndex (CRUD + merge), embed/store, retrieve (with topic filter) |
| `test_evaluator.py` | 22 | ChatOllama (provider-aware mock), SentenceTransformer, retrieve_chunks | 8-node graph: state schema, evaluability guard (gibberish, language mismatch, length), structured LLM evaluation, anti-hallucination claim validation, LLM-as-judge sampling + disagreement, batch loop, DB sync, full graph E2E mocked |
| `test_sessions_api.py` | 10 | In-memory SQLite, mocked ChromaDB | Session CRUD endpoints: create, list, get, delete with cascade, 404 handling |
| `test_files_persistence.py` | 8 | In-memory SQLite, mocked ingest_document | File metadata persistence: insert, list ordering, missing session handling |
| `test_orchestrator_profile.py` | 8 | `get_student_summary` (AsyncMock) | Profile bootstrap: resolve_student_id, load_profile success/fallback, identity mapping |
| `test_short_term_memory.py` | 5 | In-memory SQLite, AsyncSqliteSaver, MagicMock LLM | Short-term memory: messages_history state field, synthesize appends, classify uses last N, checkpoint auto-restore |
| `test_session_profile.py` | 5 | In-memory SQLite | Per-session profile aggregation: topic scores from evaluations, weak topics sorted/capped, 404 for missing session |
| `test_session_context.py` | 18 | `list_session_files`, `get_session_profile` (AsyncMock), MagicMock LLM | Session context for agent: tool exports, load_session_context node populates files+progress, synthesize_response enrichment with profile+session_context+messages_history, graph wiring |

---

## PRD Test Case Coverage

All 12 cases from `init_PRD.md` §8. Mapping shows which test file covers each case and whether it uses real models.

| PRD # | Category | Description | Test | Real Models? |
|-------|----------|-------------|------|-------------|
| 1 | Happy Path | Ingest PDF, verify chunks | `test_parse_real_pdf` + `test_ingest_real_pdf` | ✅ Yes |
| 2 | Happy Path | Generate 5-question exam on specific topic | `test_prd2_happy_path_5_questions` (mock) + `test_generate_exam_from_real_pdf` (real) | Both |
| 3 | Happy Path | Evaluate correct answer (score ≥8) | `test_evaluate_answer_structured_output` (mock) + `test_evaluate_correct_answer` (real) | Both |
| 4 | Happy Path | Second session prioritizes weak topics | ⏳ Epic 1 + Epic 6 (not yet implemented) | — |
| 5 | Happy Path | Incremental ingestion (second PDF) | `test_incremental_ingestion` | ❌ Mock |
| 6 | Edge Case | PDF with complex tables/equations | ⏸️ Deferred (OCR math postponed) | — |
| 7 | Edge Case | Topic not in material | `test_prd7_missing_topic_handling` (mock) | ❌ Mock |
| 8 | Edge Case | Partially correct answer (score 5-7) | `test_evaluate_partially_correct` (real) | ✅ Yes |
| 9 | Edge Case | Low-confidence OCR confirmation | ⏸️ Deferred (OCR math postponed) | — |
| 10 | Adversarial | Non-academic random text rejected | `test_reject_non_academic_content` | ❌ Mock |
| 11 | Adversarial | Topic from different subject — no invention | `test_prd11_adversarial_no_content` (mock) + `test_anti_hallucination_catches_fabrication` (real) | Both |
| 12 | Adversarial | Answer in different language | `test_check_evaluability_rejects_gibberish` (mock) + `test_evaluate_wrong_language` (real) | Both |

**Coverage status**: 7/12 with real models, 3/12 mock-only, 2/12 deferred (OCR), 0/12 not yet implemented.

---

## TXR Requirement Test Coverage (Epic 11)

All 10 TXR requirements from `openspec/changes/epic-11-topic-extraction-refinement/specs.md` have at least one test.

| TXR | Description | Tests | Real LLM? |
|-----|-------------|-------|-----------|
| TXR-01 | Spanish NLP: stopword removal + Snowball stemming | `TestRemoveStopwords` (4), `TestStemTopic` (6 incl. accented, ñ) | ❌ Unit |
| TXR-02 | Markdown-aware segmentation with fallback | `TestSegmentText` (7) | ❌ Unit |
| TXR-03 | Sequential LLM extraction per segment | `TestExtractTopicsFromSegment` (4) | ❌ Mock |
| TXR-05 | Jaccard unification: merge, separate, canonical, cap | `TestUnifyTopics` (12 incl. boundary, accented, long strings) | ❌ Unit |
| TXR-06 | Hierarchical tree construction | `TestBuildTopicTree` (5) | ❌ Mock |
| TXR-07 | `classify_document` → pipeline delegation | `test_ingestor.py` (6 updated) | ❌ Mock |
| TXR-08 | `extract_topics` tool → pipeline delegation | `test_ingestor.py` tool tests | ❌ Mock |
| TXR-09 | Pipeline API shape (`extract_topics_pipeline`) | `TestExtractTopicsPipeline` (5) | ❌ Mock |
| TXR-10 | Short text passthrough | `test_short_text_passthrough`, `test_single_segment_skips_unify` | ❌ Mock |
| NFR | Full pipeline on real PDF (≥3 topics, tree non-empty) | `test_full_pipeline_real_pdf` (integration) | ✅ Ollama Cloud |

---

## Adding New Tests

### When to add a unit test (mock, runs every commit)
- Testing a state transition or graph edge
- Testing error handling / edge cases in node logic
- Testing output structure / format
- Testing that a node correctly accumulates, deduplicates, or filters data

### When to add an integration test (`@pytest.mark.integration`)
- Testing that a real LLM produces valid structured output
- Testing that real embeddings produce meaningful similarity scores
- Testing that the full pipeline (PDF → chunks → exam) works end-to-end
- Testing anti-hallucination with real semantic similarity
- Any test that depends on an external service (Groq, Ollama, HuggingFace download)

### Rules
1. Every integration test **must** have a `requires_ollama` fixture (or equivalent) so it skips gracefully when the provider is unavailable.
2. Integration tests go in the existing test files under a `@pytest.mark.integration` class — do not create separate files.
3. Test fixtures (PDFs, sample data) go in `back/tests/fixtures/`.
4. When adding or modifying an integration test or a PRD-mapped test, **update this document**.

---

## Test Fixtures

| Fixture | File | Purpose |
|---------|------|---------|
| `real_pdf_path` | `conftest.py` | Path to `tests/fixtures/apunteAgentes_IA2007.pdf`. Skips if missing. |
| `real_pdf_text` | `conftest.py` | Parsed text from the real PDF via markitdown. Skips if empty. |
| `ingested_collection_name` | `conftest.py` | Ingests real PDF into ChromaDB, returns collection name. Expensive. |
| `requires_ollama` | `conftest.py` | Skips test if Ollama is not reachable or model not pulled. |
| `sample_chunks` | `conftest.py` | Mock chunk dicts for unit tests (math topics). |
| `mock_exam_llm` | `conftest.py` | Mocks LLM (provider-aware) to return a valid ExamGeneration. |
| `exam_generator_state` | `conftest.py` | Base ExamGeneratorState dict for graph invocation. |
| `evaluator_state` | `conftest.py` | Base EvaluatorState dict with 3-answer batch, sample_chunks, and collection_name. |
| `mock_evaluator_llm` | `conftest.py` | Mocks LLM (provider-aware) to return SingleEvaluation score=8, is_evaluable=True. |
| `mock_judge_llm` | `conftest.py` | Mocks LLM (provider-aware) to return JudgeVerdict agreeing with primary (score=7.5). |
| `patch_llm` | `conftest.py` | Context manager helper: patches current provider's chat class with structured-output mock chain. |
| `_llm_provider_module` | `conftest.py` | Returns `"langchain_ollama.ChatOllama"` or `"langchain_groq.ChatGroq"` based on settings. |
