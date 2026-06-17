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

### Integration Tests (9 tests — `-m integration`)

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

### Unit Tests (35 tests — default)

| File | Tests | Mocks | Covers |
|------|-------|-------|--------|
| `test_exam_generator.py` | 15 | ChatGroq, retrieve_chunks, embeddings | Graph topology, state transitions, retry logic, dedup, output structure, PRD cases (mocked) |
| `test_ingestor.py` | 12 | ChatGroq | Parse, classify, incremental ingestion, non-academic rejection, image rejection, error handling |
| `test_rag.py` | 10 | Embedding model, ChromaDB client | Chunking, ThematicIndex (CRUD + merge), embed/store, retrieve (with topic filter) |

---

## PRD Test Case Coverage

All 12 cases from `init_PRD.md` §8. Mapping shows which test file covers each case and whether it uses real models.

| PRD # | Category | Description | Test | Real Models? |
|-------|----------|-------------|------|-------------|
| 1 | Happy Path | Ingest PDF, verify chunks | `test_parse_real_pdf` + `test_ingest_real_pdf` | ✅ Yes |
| 2 | Happy Path | Generate 5-question exam on specific topic | `test_prd2_happy_path_5_questions` (mock) + `test_generate_exam_from_real_pdf` (real) | Both |
| 3 | Happy Path | Evaluate correct answer (score ≥8) | ⏳ Epic 5 (not yet implemented) | — |
| 4 | Happy Path | Second session prioritizes weak topics | ⏳ Epic 1 + Epic 6 (not yet implemented) | — |
| 5 | Happy Path | Incremental ingestion (second PDF) | `test_incremental_ingestion` | ❌ Mock |
| 6 | Edge Case | PDF with complex tables/equations | ⏸️ Deferred (OCR math postponed) | — |
| 7 | Edge Case | Topic not in material | `test_prd7_missing_topic_handling` (mock) | ❌ Mock |
| 8 | Edge Case | Partially correct answer (score 5-7) | ⏳ Epic 5 (not yet implemented) | — |
| 9 | Edge Case | Low-confidence OCR confirmation | ⏸️ Deferred (OCR math postponed) | — |
| 10 | Adversarial | Non-academic random text rejected | `test_reject_non_academic_content` | ❌ Mock |
| 11 | Adversarial | Topic from different subject — no invention | `test_prd11_adversarial_no_content` (mock) + `test_anti_hallucination_catches_fabrication` (real) | Both |
| 12 | Adversarial | Answer in different language | ⏳ Epic 5 (not yet implemented) | — |

**Coverage status**: 5/12 with real models, 3/12 mock-only, 2/12 deferred (OCR), 2/12 not yet implemented (Epic 5 Evaluator).

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
1. Every integration test **must** have a `requires_groq` fixture (or equivalent) so it skips gracefully when the provider is unavailable.
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
| `requires_groq` | `conftest.py` | Skips test if `GROQ_API_KEY` is not configured in settings. |
| `sample_chunks` | `conftest.py` | Mock chunk dicts for unit tests (math topics). |
| `mock_exam_llm` | `conftest.py` | Mocks ChatGroq to return a valid ExamGeneration. |
| `exam_generator_state` | `conftest.py` | Base ExamGeneratorState dict for graph invocation. |
