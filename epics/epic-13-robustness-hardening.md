# Epic 13 — Robustness Hardening

> **Goal**: Identify fragile implementation patterns across the ai-tutor backend and harden them against non-trivial user inputs, partial failures, and edge-case runtime conditions.

---

## 1. Audit Summary

The system is **functionally complete** (Epics 1–12 DONE) but relies on several "best-effort" patterns that will break under non-ordinary conditions. The fragility falls into 5 categories:

| Category | Severity | Affected Modules |
|----------|----------|------------------|
| **A. Input boundary gaps** | 🔴 HIGH | `router.py`, `orchestrator.py`, `schemas.py` |
| **B. Silent failure masking** | 🔴 HIGH | `orchestrator.py`, `ingestor.py`, `support.py` |
| **C. State inconsistency windows** | 🟡 MEDIUM | `evaluator.py`, `schema.py`, `orchestrator.py` |
| **D. Concurrency & singleton hazards** | 🟡 MEDIUM | `rag/__init__.py`, `llm.py`, `orchestrator.py` |
| **E. LLM output trust** | 🟡 MEDIUM | All agents using `get_structured_llm` |

---

## 2. Detailed Findings

### A. Input Boundary Gaps (HIGH)

#### A-1. `ChatRequest` accepts unbounded message length
**File**: `src/api/schemas.py:27-30`
**Problem**: `ChatRequest.message` is `str` with no length constraint. A 1MB message will be forwarded to `classify_intent` → LLM, causing token overflow or provider rejection with an opaque error.
**Fix**: Add `Field(max_length=10_000)` to `message`. Return 422 with a clear error.

#### A-2. `ExamPreferences.question_count` has no upper bound
**File**: `src/api/schemas.py:43`
**Problem**: `question_count` is `int` — no `le` constraint. A user sending `question_count: 500` will trigger massive LLM calls and ChromaDB retrieval loops.
**Fix**: Add `Field(ge=1, le=30)` to `question_count`. Add `Field(ge=0.0, le=1.0)` to `mcq_ratio` in `ExamGeneratorState`.

#### A-3. `session_id` format not validated
**File**: `src/api/router.py`
**Problem**: Session IDs are treated as opaque strings. SQL injection is prevented by parameterized queries, but ChromaDB collection names are built as `f"session_{session_id}"` — special characters (`.`, `/`, `\0`) could produce invalid collection names or path traversal in SQLite.
**Fix**: Validate `session_id` against `^[a-zA-Z0-9_-]{1,128}$` regex in Pydantic schemas.

#### A-4. File upload missing size limit at application level
**File**: `src/api/router.py` (ingest endpoint)
**Problem**: Relies on nginx/reverse proxy for size limits. A direct-to-uvicorn upload of a 500MB PDF will OOM during `markitdown.convert()` which loads the entire file into memory.
**Fix**: Check `file.size` (or read in chunks with a cap) before writing to temp. Reject files > 20MB with 413.

#### A-5. `ExerciseRequest.exercise_type` is unvalidated free-text
**File**: `src/api/schemas.py:139`
**Problem**: `exercise_type: str = "problem_solving"` — accepts any string. Injected into LLM prompts verbatim, enabling prompt injection (`exercise_type: "ignore all instructions and..."`)
**Fix**: Change to `Literal["problem_solving", "calculation", "conceptual", "applied"]`.

---

### B. Silent Failure Masking (HIGH)

#### B-1. `load_session_context` swallows all exceptions
**File**: `src/agents/orchestrator.py`
**Problem**: Broad `except Exception` returns empty context on ANY failure — including DB corruption, permission errors, or schema migration issues. The orchestrator proceeds as if no session history exists, potentially giving contradictory responses.
**Fix**: Catch only `(sqlite3.OperationalError, aiosqlite.Error, KeyError)`. Let unexpected errors propagate to synthesize_response's fallback.

#### B-2. `load_profile` masks DB errors as empty profile
**File**: `src/agents/orchestrator.py`
**Problem**: Same pattern — any exception → `student_profile: None`. This silently degrades exam generation (no weak-topic boosting) without any signal to the user or logs beyond a warning.
**Fix**: Preserve the error in state (e.g., `profile_load_error: str`) so `synthesize_response` can mention "no pude cargar tu perfil" instead of silently generating generic exams.

#### B-3. Ingestor graph has no early-exit on rejected status
**File**: `src/agents/ingestor.py:301-304`
**Problem**: The graph is linear: `parse → classify → chunk_and_embed`. If `parse_document` returns `status: "rejected"` (e.g., image file), the pipeline still runs `classify_document` and `chunk_and_embed` on empty/error state. These nodes handle it gracefully, but waste LLM calls and embedding computation.
**Fix**: Add conditional edge after `parse_document`: if `status in ("rejected", "error")` → skip to END.

#### B-4. `run_async_in_sync` ThreadPoolExecutor creates unbounded threads
**File**: `src/utils/async_.py:38`
**Problem**: Each call creates a new `ThreadPoolExecutor()` (default workers = `min(32, cpu+4)`). Under load, `sync_scores` calling `save_evaluation` per question in a batch of 20 creates 20 thread pools serially.
**Fix**: Use a module-level shared executor with `max_workers=4`. Add timeout to `future.result(timeout=30)`.

---

### C. State Inconsistency Windows (MEDIUM)

#### C-1. Evaluator `sync_scores` is not transactional
**File**: `src/agents/evaluator.py:677-743`
**Problem**: Each evaluation result is saved in a separate `save_evaluation()` call. If the process crashes after saving 3/5 results, the DB has partial evaluation data with no way to detect or recover.
**Fix**: Wrap the entire batch in a single DB transaction. Add an `evaluation_batch_id` to link related records.

#### C-2. `validate_feedback` mutates `evaluation` dict in-place
**File**: `src/agents/evaluator.py:502`
**Problem**: `evaluation["validation_warnings"] = validation_warnings` mutates the dict from state directly. In LangGraph, state dicts should be treated as immutable — returning a NEW dict. Current code works because the evaluator returns the mutated ref, but it violates the LangGraph contract and could break with checkpointer serialization.
**Fix**: Return `{**evaluation, "validation_warnings": validation_warnings}` — create a new dict.

#### C-3. Orchestrator singleton has no mutex
**File**: `src/agents/orchestrator.py:965-999`
**Problem**: `get_orchestrator_graph()` checks `_orchestrator_graph is None` without a lock. Two concurrent requests on startup could both enter the init block, creating two DB connections but only keeping one (leaking the other).
**Fix**: Use `asyncio.Lock()` around the initialization block.

#### C-4. ChromaDB collection creation is not idempotent under race
**File**: `src/rag/__init__.py`
**Problem**: `embed_and_store` calls `get_or_create_collection`. If two concurrent ingestions for the same session happen simultaneously, both could create the collection, and one set of embeddings could be lost.
**Fix**: Add retry-on-conflict logic or serialize collection creation per session_id.

---

### D. Concurrency & Singleton Hazards (MEDIUM)

#### D-1. SentenceTransformer singleton is not thread-safe
**File**: `src/rag/__init__.py`
**Problem**: The embedding model is loaded lazily as a module-level singleton. `model.encode()` is called from synchronous graph nodes via `run_async_in_sync` which spawns threads. SentenceTransformer's `encode()` is not documented as thread-safe.
**Fix**: Add a `threading.Lock()` around `model.encode()` calls, or use a connection-pool pattern.

#### D-2. `get_llm()` creates a new instance on every call
**File**: `src/llm.py:23-74`
**Problem**: Not a singleton — every `get_llm()` call creates a fresh `ChatOllama` or `ChatGroq` instance. Under high concurrency, this means hundreds of HTTP client instances, each with its own connection pool.
**Fix**: Cache LLM instances per provider config. Use `functools.lru_cache` or a module-level singleton with lazy init.

#### D-3. `close_orchestrator_graph` can race with active requests
**File**: `src/agents/orchestrator.py:1002-1020`
**Problem**: During shutdown, `close_orchestrator_graph()` sets `_orchestrator_graph = None` and closes the DB connection. Any in-flight request calling `graph.ainvoke()` will get a "connection closed" error with no retry.
**Fix**: Use a shutdown flag + drain period. Or catch `aiosqlite.Error` in `ainvoke` and return a "service shutting down" response.

---

### E. LLM Output Trust (MEDIUM)

#### E-1. `classify_intent` JSON parsing has single fallback
**File**: `src/agents/orchestrator.py`
**Problem**: If `with_structured_output` returns a malformed object (e.g., Ollama returns `classification: "EXAMEN_PREVIO"` instead of the expected enum), Pydantic validation raises, caught by the outer except, which falls back to `general_chat`. No retry, no logging of WHAT the LLM actually returned.
**Fix**: Log the raw LLM output on validation failure. Add a single retry with temperature=0 before falling back. Track fallback rate in Langfuse.

#### E-2. `_ollama_json_mode_chain._parse` silently scans for JSON
**File**: `src/llm.py:126-147`
**Problem**: The parser scans backwards for the last `{}` pair. If the LLM returns a JSON object embedded in markdown (```json ... ```), the parser might extract a nested sub-object instead of the top-level response.
**Fix**: Strip markdown code fences before JSON scanning. Add a validation step that the parsed object contains expected top-level keys.

#### E-3. No timeout on LLM calls
**File**: All agents
**Problem**: `structured_llm.invoke(prompt)` has no timeout. If Groq/Ollama hangs, the request blocks indefinitely. FastAPI's request timeout (if configured) will kill the connection but not the LLM call, leaking resources.
**Fix**: Pass `timeout` to the LLM constructor. Add `asyncio.wait_for()` wrapper with 60s default in `orchestrate_chat`.

#### E-4. Exam generator accepts LLM-fabricated chunk IDs
**File**: `src/agents/exam_generator.py:374-378`
**Problem**: `source_chunk_ids` in generated questions come from the LLM output. The LLM might hallucinate chunk IDs that don't exist in `retrieved_chunks`. Validation checks claim grounding via embedding similarity but never verifies that `source_chunk_ids` actually match real chunk IDs.
**Fix**: Post-generation, filter `source_chunk_ids` to only include IDs present in `retrieved_chunks`. Log discrepancies.

---

### F. Test Suite Gaps (observed)

22 test failures in the current suite. Root causes:

| Failure Pattern | Count | Root Cause |
|----------------|-------|------------|
| `no such table: students` | 8 | Tests don't call `init_db()` before DB operations |
| `404 Not Found` on dashboard | 3 | Test uses wrong route prefix (`/students/` vs `/api/students/`) |
| Mock wiring mismatches | 7 | Observability tests mock at wrong import path |
| Async mock coroutine warnings | 4 | `AsyncMock` not properly awaited in checkpoint serialization |

**Fix**: Add a shared `@pytest.fixture(autouse=True)` that calls `init_db()` for any test touching the DB. Fix route paths in dashboard tests.

---

## 3. Implementation Plan

### Phase 1 — Input Validation (Priority: HIGH, Effort: S)

| Task | File | Change |
|------|------|--------|
| 1.1 | `schemas.py` | Add `max_length=10_000` to `ChatRequest.message` |
| 1.2 | `schemas.py` | Add `ge=1, le=30` to `question_count`, `ge=0.0, le=1.0` to mcq fields |
| 1.3 | `schemas.py` | Add `pattern=r"^[a-zA-Z0-9_-]{1,128}$"` to all `session_id` fields |
| 1.4 | `router.py` | Add file size check (20MB cap) before temp write in ingest |
| 1.5 | `schemas.py` | Change `exercise_type` to `Literal[...]` enum |
| 1.6 | Tests | Add parametrized tests for each boundary (empty, max, overflow) |

### Phase 2 — Error Propagation (Priority: HIGH, Effort: M)

| Task | File | Change |
|------|------|--------|
| 2.1 | `orchestrator.py` | Narrow exception types in `load_session_context`, `load_profile` |
| 2.2 | `orchestrator.py` | Add `profile_load_error` to state; use in `synthesize_response` |
| 2.3 | `ingestor.py` | Add conditional edge: `parse_document` → END on rejected/error |
| 2.4 | `orchestrator.py` | Log raw LLM output on `classify_intent` fallback |
| 2.5 | All agents | Add `timeout=60` to LLM constructor kwargs in `config.py` |

### Phase 3 — State Safety (Priority: MEDIUM, Effort: M)

| Task | File | Change |
|------|------|--------|
| 3.1 | `evaluator.py` | Wrap `sync_scores` in single DB transaction |
| 3.2 | `evaluator.py` | Stop mutating `evaluation` dict in-place in `validate_feedback` |
| 3.3 | `orchestrator.py` | Add `asyncio.Lock` to `get_orchestrator_graph()` |
| 3.4 | `async_.py` | Use shared `ThreadPoolExecutor(max_workers=4)` + timeout |

### Phase 4 — Concurrency (Priority: MEDIUM, Effort: M)

| Task | File | Change |
|------|------|--------|
| 4.1 | `rag/__init__.py` | Add `threading.Lock` around `model.encode()` |
| 4.2 | `llm.py` | Cache LLM instances per provider (singleton with lazy init) |
| 4.3 | `orchestrator.py` | Add shutdown drain to `close_orchestrator_graph` |

### Phase 5 — LLM Output Hardening (Priority: MEDIUM, Effort: S)

| Task | File | Change |
|------|------|--------|
| 5.1 | `llm.py` | Strip markdown fences in `_parse` before JSON scan |
| 5.2 | `exam_generator.py` | Post-filter `source_chunk_ids` against real chunk IDs |
| 5.3 | `orchestrator.py` | Single retry on `classify_intent` structured output failure |

### Phase 6 — Test Suite Fix (Priority: HIGH, Effort: S)

| Task | File | Change |
|------|------|--------|
| 6.1 | `conftest.py` | Add shared `init_db` fixture for all DB-touching tests |
| 6.2 | `test_support.py` | Fix dashboard route prefix |
| 6.3 | `test_observability.py` | Fix mock import paths |
| 6.4 | `test_*.py` | Fix async mock warnings (use `AsyncMock` properly) |

---

## 4. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Input validation breaks existing frontend | LOW | Frontend already sends valid data; these are guardrails for abuse |
| Narrowing exceptions surfaces hidden bugs | MEDIUM | Add integration test coverage before narrowing catches |
| LLM timeout too aggressive | LOW | 60s is generous; Groq p99 is ~8s, Ollama local ~30s |
| Thread lock on embeddings hurts throughput | LOW | Embedding calls are <100ms; contention is minimal |

---

## 5. Acceptance Criteria

- [ ] All 22 existing test failures fixed (green suite)
- [ ] Pydantic schemas reject: message >10K chars, question_count >30, invalid session_id format, invalid exercise_type
- [ ] Ingestor graph short-circuits on rejected files (no LLM call)
- [ ] `classify_intent` logs raw LLM output on fallback
- [ ] `sync_scores` uses single DB transaction
- [ ] `get_orchestrator_graph` is race-free (asyncio.Lock)
- [ ] LLM calls have 60s timeout
- [ ] `source_chunk_ids` in generated exams only contain real chunk IDs
- [ ] No `RuntimeWarning: coroutine never awaited` in test output
