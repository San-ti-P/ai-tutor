# Epic 10: Code Refactoring — Deduplication, Library Replacement & Architecture Hygiene

**Status:** Draft
**Source:** Codebase audit (`exploration/agent-refactoring-audit` + `exploration/library-replacement-audit` in engram)
**Delivery window:** Post-MVP hardening — before Entrega 3 (complete + defense)

## Context

All 6 agents are implemented and functional. However, a thorough codebase audit revealed significant duplication, architecture boundary violations, and one critical performance opportunity. The same cosine similarity function is copy-pasted 3 times, the anti-hallucination validation pipeline lives duplicated in exam_generator, exercise_generator, and evaluator, the LLM factory pattern appears 7 times, and the `api/router.py` compiles agent graphs directly — bypassing the `src/tools/` layer entirely.

More critically: `sentence-transformers` (already installed) includes `util.cos_sim` — a PyTorch-accelerated matrix cosine similarity that can replace all 3 manual copies AND their nested claim×chunk loops with a single `torch.matmul`. This same library also supports batched encoding via `model.encode(..., convert_to_tensor=True)`, eliminating per-claim single-vector encodes.

These issues don't block functionality today, but every future bug fix to the anti-hallucination logic would require patching 3 separate files. The architecture violations make the codebase harder to test and reason about.

## Scope

**In scope**
- Replace manual `_cosine_sim()` (×3 copies) + nested validation loops with `sentence_transformers.util.cos_sim`
- Extract duplicated anti-hallucination pipeline into `src/tools/validate_claim_grounding.py`
- Extract duplicated claim/sentence splitting into shared utilities
- Extract `_get_llm()` factory into `src/llm.py` (single source of truth)
- Extract `run_async_in_sync()` helper into `src/utils/`
- Fix API router to use `src/tools/` layer instead of compiling agent graphs directly
- Deduplicate markitdown parsing between Ingestor and `extract_topics` tool
- Evaluate Support Agent template-based responses for LLM migration
- Update all affected tests (import path changes, mocks)
- Establish `src/utils/` module with clear conventions

**Out of scope**
- Adding new features to any agent
- Changing agent graph topology or behavior
- OCR pipeline changes
- Frontend changes
- Langfuse tracing changes
- New dependencies (all replacements use already-installed libraries)

## Functional Requirements

- **REF-01** Cosine similarity across all agents must use `sentence_transformers.util.cos_sim` — no manual Python loops.
- **REF-02** Batched embedding via `model.encode(..., convert_to_tensor=True)` must replace per-claim single-vector encodes in validation pipelines.
- **REF-03** Anti-hallucination claim validation must be a single reusable `@tool` function in `src/tools/`, callable by exam_generator, exercise_generator, and evaluator.
- **REF-04** Claim/sentence splitting must be a single function in `src/utils/text.py`, callable by all 3 agents.
- **REF-05** LLM instantiation must be a single factory function — no agent imports `settings.llm_kwargs` directly.
- **REF-06** The `asyncio.get_event_loop()` try/except pattern must be a single `run_async_in_sync()` utility.
- **REF-07** `api/router.py` must invoke tools from `src/tools/`, not compile agent graphs with `build_*().compile()`.
- **REF-08** Document parsing (markitdown) must not be duplicated between `ingestor.parse_document` and `src.tools.extract_topics`.
- **REF-09** No functional behavior change in any agent — all existing tests must pass with only import path adjustments.
- **REF-10** The new `src/utils/` module must have a documented convention for what belongs there (pure functions, no I/O, no agent-specific logic).

## Non-Functional Requirements

- **REF-NFR-01** Cosine similarity replacement must produce numerically equivalent results (within floating-point epsilon of current manual implementation).
- **REF-NFR-02** Refactored anti-hallucination tool must support both flag-only mode (evaluator) and retry-trigger mode (exam/exercise generators).
- **REF-NFR-03** Batch encoding change must not increase memory usage beyond 2× current peak.
- **REF-NFR-04** All existing tests (51 plumbing + 11 integration) must pass after refactoring without semantic changes.
- **REF-NFR-05** New `src/utils/` module must have 100% unit test coverage for extracted functions.
- **REF-NFR-06** No new dependencies added — all replacements use already-installed libraries.

## Technical Notes

### Cosine Similarity Replacement (P0)

**Current** (3 copies in exam_generator, exercise_generator, evaluator):
```python
def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)
```

**Target** (single call):
```python
from sentence_transformers.util import cos_sim

# Batch encode all claims + chunks at once
claim_embeddings = model.encode(all_claims, convert_to_tensor=True)
chunk_embeddings = model.encode(all_chunks, convert_to_tensor=True)

# Single GPU matrix multiplication: shape (N_claims, N_chunks)
sim_matrix = cos_sim(claim_embeddings, chunk_embeddings)
best_scores = sim_matrix.max(dim=1)
```

Impact: replaces O(N×M) Python iterations × O(dim) manual arithmetic with 1 PyTorch matmul. 50-500× faster for 384-dim vectors. Eliminates 3 copies of `_cosine_sim`, 3 nested Python loops, and 3 per-claim single-vector encodes.

### Duplication Inventory

| Pattern | Current copies | Target |
|---------|---------------|--------|
| `_cosine_sim` + nested validation loops | 3 (exam, exercise, evaluator) | 1 call to `cos_sim` |
| `sentence_split_re` regex + claim extraction | 3 (exam, exercise, evaluator) | 1 util: `split_sentences()` |
| Anti-hallucination pipeline | 3 (exam, exercise, evaluator) | 1 tool: `validate_claim_grounding` |
| `_get_llm()` / `settings.llm_kwargs` | 7 (all agents + tools) | 1 factory: `src/llm.py` |
| `asyncio.get_event_loop()` wrapper | 5 (support, evaluator) | 1 util: `run_async_in_sync()` |
| `should_retry` helper | 2 (exam, exercise) | 1 util or inline in tool |

### Architecture Boundary Fix

**Current** — `api/router.py` imports agent internals:
```python
from src.agents.ingestor import IngestorState, build_ingestor
graph = build_ingestor().compile()
graph.invoke(state)
```

**Target** — use the tools layer:
```python
from src.tools import ingest_document
result = ingest_document.invoke({...})
```

The tools layer already wraps all agents as LangChain `@tool` functions. The router should invoke those tools, not bypass them.

### New Module Structure

```
src/
├── agents/          # Unchanged — agent graphs, one file per agent
├── tools/           # + validate_claim_grounding.py (new)
├── rag/             # Unchanged — ChromaDB, embeddings, retrieval
├── llm.py           # NEW — get_llm() factory, single source of truth
├── utils/           # NEW
│   ├── __init__.py
│   ├── text.py      # split_sentences(), split_into_claims()
│   ├── async_.py    # run_async_in_sync()
│   └── math.py      # (reserved for future — cos_sim lives in sentence_transformers)
├── config.py        # Unchanged — keep settings, remove llm_kwargs callers
└── api/             # Fixed — router uses tools layer
```

### Effort Estimate

| Priority | Items | Effort |
|----------|-------|--------|
| P0 — Library replacement | 1 item (cosine_sim + batch encode) | 2 hours |
| P1 — Eliminate duplication | 5 items (claim splitting, anti-hallucination tool, should_retry, sentence split, markitdown) | 1 day |
| P2 — Fix boundaries | 3 items (API router, LLM factory, async util) | 1.5 days |
| P3 — Tool consistency | 1 item (Support Agent LLM evaluation) | 0.5 day |
| Testing updates | All import paths, mocks, new util tests | 0.5 day |
| **Total** | **10 items** | **3-4 days** |

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| `cos_sim` produces slightly different results than manual implementation | Medium | Add tolerance assertion in tests. Verify with representative vector pairs. Floating-point equivalence within epsilon expected. |
| Batch encoding changes memory profile | Low | `convert_to_tensor=True` uses same underlying model. Claims are typically small (5-15) — negligible memory impact. |
| Test breakage from import path changes | High | Mock paths will change — agent tests that mock `_cosine_sim` must be updated. Mechanically straightforward. |
| Anti-hallucination tool mode confusion (flag vs retry) | Medium | Explicit `mode` parameter: `"flag_only"` for evaluator, `"retry_trigger"` for generators. |
| API router refactoring breaks /chat endpoint | Low | Tool layer is a thin wrapper — same underlying graph. `/chat` tests already mock LLM; should pass with tool invocation. |

## Test Coverage

- All 51 plumbing tests must pass after import path adjustments
- All 11 real LLM integration tests must pass (semantic behavior unchanged)
- New unit tests for `src/utils/text.py` (split_sentences, split_into_claims)
- New unit tests for `src/utils/async_.py` (run_async_in_sync)
- New unit tests for `src/llm.py` (factory returns correct provider)
- New unit tests for `src/tools/validate_claim_grounding.py` (both modes)
- Numerical equivalence test: manual `_cosine_sim` vs `cos_sim` on 100 random vector pairs

## User Stories

### US-10.1: Replace manual cosine similarity with sentence_transformers
- **As a** developer maintaining the anti-hallucination pipeline
- **I want** cosine similarity to use `sentence_transformers.util.cos_sim` with batched encoding
- **So that** I edit one file, not three, and get PyTorch-accelerated computation for free
- **Acceptance criteria:**
  - `_cosine_sim` removed from exam_generator, exercise_generator, and evaluator
  - All 3 validation nodes call `model.encode(..., convert_to_tensor=True)` + `cos_sim()`
  - Numerical results match current implementation within 1e-6 tolerance
  - All existing tests pass without semantic changes
- **Dependencies:** —
- **Maps to:** REF-01, REF-02, REF-NFR-01

### US-10.2: Extract shared anti-hallucination tool
- **As a** developer fixing a hallucination detection bug
- **I want** a single `validate_claim_grounding` tool used by all 3 validation agents
- **So that** I fix the bug once and all agents benefit
- **Acceptance criteria:**
  - New `src/tools/validate_claim_grounding.py` with `@tool` decorator
  - Supports `mode="flag_only"` (evaluator — returns results, never retries) and `mode="retry_trigger"` (generators — tells caller to retry)
  - exam_generator, exercise_generator, and evaluator import and use the tool
  - No duplication of claim extraction, embedding, or cosine comparison across agents
- **Dependencies:** US-10.1 (cosine sim replacement)
- **Maps to:** REF-03, REF-NFR-02

### US-10.3: Extract sentence splitting utilities
- **As a** developer
- **I want** a single `split_sentences()` and `split_into_claims()` in `src/utils/text.py`
- **So that** the same regex isn't copy-pasted across 3 files
- **Acceptance criteria:**
  - `split_sentences(text: str) -> list[str]` handles `.`, `!`, `?` boundaries
  - `split_into_claims(text: str) -> list[str]` handles sentences + semicolons
  - All 3 agents import from `src.utils.text`
  - Existing test behavior unchanged
- **Dependencies:** US-10.1
- **Maps to:** REF-04

### US-10.4: Centralize LLM instantiation
- **As a** developer changing the LLM provider
- **I want** a single `get_llm()` factory in `src/llm.py`
- **So that** I change the model name in one place, not 7
- **Acceptance criteria:**
  - `src/llm.py` exposes `get_llm()` and `get_structured_llm(schema)` factory functions
  - All 6 agents + tools use these factories — zero direct `settings.llm_kwargs` calls
  - Supports both Ollama and Groq providers (respects `settings.llm_provider`)
  - All existing tests pass
- **Dependencies:** —
- **Maps to:** REF-05

### US-10.5: Extract async-in-sync utility
- **As a** developer calling async functions from sync LangGraph nodes
- **I want** a single `run_async_in_sync()` utility
- **So that** the event-loop try/except boilerplate isn't duplicated 5 times
- **Acceptance criteria:**
  - `src/utils/async_.py` provides `run_async_in_sync(coro)`
  - Handles running event loop, no running loop, and thread-pool fallback
  - All 5 call sites (support ×3, evaluator ×2) use the utility
  - Existing async test behavior unchanged
- **Dependencies:** —
- **Maps to:** REF-06

### US-10.6: Fix API router architecture boundary
- **As a** developer testing the system
- **I want** the API router to invoke tools from `src/tools/`
- **So that** the architecture layers are respected and agents aren't coupled to HTTP
- **Acceptance criteria:**
  - `router.py` POST `/ingest` calls `ingest_document.invoke(...)` — no direct `build_ingestor().compile()`
  - `router.py` POST `/chat` calls the orchestrator through the tool layer or its public API — no agent state imports
  - `/chat` endpoint returns identical response format to current behavior
  - Integration test `test_chat_endpoint_returns_real_response` passes
- **Dependencies:** —
- **Maps to:** REF-07

### US-10.7: Deduplicate markitdown parsing
- **As a** developer maintaining document processing
- **I want** markitdown parsing in one place
- **So that** Ingestor and `extract_topics` tool don't each have their own parsing logic
- **Acceptance criteria:**
  - Markitdown parsing extracted to shared utility or lives only in Ingestor
  - `extract_topics` tool calls the Ingestor's parsing or a shared helper — no duplicate `MarkItDown()` instantiation
  - All ingestion and extraction tests pass
- **Dependencies:** —
- **Maps to:** REF-08

### US-10.8: Evaluate Support Agent responses
- **As a** student asking a profile/personalized question
- **I want** the Support Agent to use the LLM for natural responses
- **So that** its replies are as rich and contextual as the other agents'
- **Acceptance criteria:**
  - Decision document: evaluate whether templates or LLM is appropriate for Support Agent
  - If templates are kept: document the rationale
  - If LLM is adopted: implement with same `get_llm()` factory, test with real Ollama
- **Dependencies:** US-10.4 (LLM factory)
- **Maps to:** REF-09

### US-10.9: Full regression test pass
- **As a** developer merging this epic
- **I want** all 62 existing tests to pass
- **So that** the refactoring introduces zero regressions
- **Acceptance criteria:**
  - 51 plumbing tests pass (no Ollama needed) — `-m "not integration"`
  - 11 real LLM integration tests pass (requires Ollama) — `-m integration`
  - New util/tool unit tests added for extracted functions
  - ruff format/lint clean
- **Dependencies:** US-10.1 through US-10.8
- **Maps to:** REF-09, REF-NFR-04, REF-NFR-05

### US-10.10: Module conventions documented
- **As a** developer adding code to `src/utils/`
- **I want** clear conventions for what belongs there
- **So that** the module doesn't become a dumping ground
- **Acceptance criteria:**
  - `src/utils/__init__.py` docstring explains: pure functions only, no I/O, no agent-specific logic, must have tests
  - Code review checklist updated (or AGENTS.md updated)
- **Dependencies:** —
- **Maps to:** REF-10
