# Epic 12: E2E Testing Framework + Memory, Profile & Stats Hardening

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.4 long-term memory, §6.1 RF-11, RF-13, §8 plan de evaluación)
**Source Epics:** [epic-09-profile-bootstrap.md](./epic-09-profile-bootstrap.md), [epic-06-support-agent.md](./DONE-epic-06-support-agent.md)
**Delivery window:** Entrega 3 — hardening sprint before defense (29/06/2026)

## Context

Epic 9 delivered the session lifecycle with three-layer memory (LangGraph checkpoint / SQLite+ChromaDB session / SQLite global profile), per-session profiles, and a frontend dashboard. Epic 11 refined topic extraction to cover full documents. Manual testing on the dev branch revealed several regressions in the memory and profile pipeline:

- **Long-term memory**: Profile data (topic scores, weak topics, session history) is not reliably persisted across evaluation cycles. The `update_student_profile` tool runs but some evaluation paths skip the profile update, causing the dashboard to show stale or empty data.
- **Profile aggregation**: Per-session profile aggregation sometimes returns empty `topicScores` even when evaluations exist in that session's DB rows. The aggregation SQL may miss rows due to `session_id` mismatch or missing joins.
- **Dashboard stats**: Frontend dashboard shows `--%` average and `0` topics covered after completing an exam because `getDashboard` returns an empty profile or times out on the aggregation query.
- **No E2E tests**: The project has 105 unit tests and 16 integration tests — all backend-only. Zero tests cover the full stack: frontend UI → API → agent workflow → LLM → persistence → dashboard. Regressions like the ones above are only caught manually.
- **No deterministic test mode for LLMs**: Integration tests use real LLMs (Ollama/Groq), making them non-deterministic, slow, and expensive to run. There is no mock seed mode that produces reproducible LLM outputs for fast E2E pipelines.
- **PRD Case 4 uncovered**: "Second session prioritizes weak topics" has no automated test and the feature itself is marked "not yet implemented" in test docs.

This epic hardens the memory/profile/stats pipeline, adds a Playwright E2E test suite covering the full stack, introduces a deterministic test mode for LLM calls, and ensures the system is defense-ready for Monday's review.

## Scope

**In scope**
- Fix long-term memory persistence: ensure `update_student_profile` is called on every evaluation path (exam submit, exercise submit, chat evaluation)
- Fix per-session profile aggregation SQL: correct `session_id` filtering in `get_topic_scores` and `compute_weak_topics`
- Fix dashboard data pipeline: ensure `getDashboard` returns fresh, aggregated data after exam completion
- Implement PRD Case 4: second session prioritizes weak topics from previous sessions
- Select and integrate Playwright as the E2E testing framework for frontend→backend→LLM flows
- Create deterministic LLM mock seed mode: configurable flag that replays pre-recorded LLM responses for reproducible E2E tests
- Write E2E test suite covering 4 core flows: ingest → exam → evaluate → dashboard
- Create manual LLM test checklist: step-by-step procedure a human (or LLM) can run to validate the full system
- Write E2E test fixtures: seeded ChromaDB collection, pre-loaded SQLite profile, mock LLM response seeds
- Update `tests_documentation.md` with E2E test inventory and framework docs

**Out of scope**
- Unit test coverage for frontend components (Jest + React Testing Library — separate epic)
- Visual regression testing (Percy/Chromatic)
- Performance/load testing
- CI/CD pipeline integration (GitHub Actions stays manual for now)
- Multi-browser matrix (Chromium only for defense; Firefox/WebKit as stretch)
- OCR math pipeline testing (deferred per PRD)
- Real-time streaming test assertions (SSE/streaming verification)

## Functional Requirements

### Memory Hardening

- **MEMFIX-01** The `update_student_profile` tool MUST be called on every evaluation completion path: exam submission via `/api/evaluate`, exercise submission, and chat-based evaluation.
- **MEMFIX-02** The `evaluate_answer` tool in the Evaluator agent MUST trigger `update_student_profile` after scoring, not rely on the Orchestrator to do it post-hoc.
- **MEMFIX-03** If `update_student_profile` fails (DB error, missing student), the evaluation result MUST still be returned to the user — profile update is best-effort, non-blocking.
- **MEMFIX-04** A failed profile update MUST be logged at WARNING level with the specific DB error and the evaluation that triggered it.

### Profile Aggregation Fixes

- **PROFIX-01** `get_topic_scores(student_id, session_id=None)` MUST return topic scores from ALL sessions when `session_id` is None (global aggregation for dashboard).
- **PROFIX-02** `get_topic_scores(student_id, session_id=uuid)` MUST return topic scores filtered to exactly that session (per-session profile).
- **PROFIX-03** `compute_weak_topics(student_id)` MUST aggregate across all sessions and return topics where the latest score is below the weak threshold (default: <6.0).
- **PROFIX-04** `get_recent_sessions(student_id)` MUST return sessions ordered by most recent evaluation, not creation date, to reflect actual user activity.

### Dashboard Data Pipeline

- **DASH-01** `GET /api/students/{student_id}/dashboard` MUST return a `StudentProfile` with non-empty `topicScores`, `weakTopics`, and `sessionHistory` when at least one evaluation exists.
- **DASH-02** The dashboard endpoint MUST complete in under 500ms (p95) for a student with up to 50 evaluations across 10 sessions.
- **DASH-03** If no evaluations exist, the endpoint MUST return a valid empty profile (all fields present, all collections empty) — never a 500 or timeout.
- **DASH-04** The frontend dashboard MUST handle the empty-profile case gracefully: show "No data yet" states, not spinner forever or crash.

### PRD Case 4 — Weak Topic Prioritization

- **PRIO-01** The ExamGenerator MUST receive the student's weak topics (from `get_student_summary`) and bias question generation toward those topics.
- **PRIO-02** When a student has weak topics from a previous session, at least 60% of generated exam questions MUST target those weak topics.
- **PRIO-03** When a student has no prior evaluations (empty profile), the exam MUST use uniform topic distribution — no bias.
- **PRIO-04** The prioritization MUST be verifiable: the exam response includes a `topic_distribution` field showing how many questions target each topic.

### E2E Testing Framework

- **E2E-01** Playwright MUST be integrated as the E2E testing framework with TypeScript test files under `front/e2e/`.
- **E2E-02** Playwright config MUST target the local dev servers (`localhost:3000` frontend, `localhost:8000` backend) with Chromium headless as default.
- **E2E-03** E2E tests MUST run with a single command: `cd front && npx playwright test` (mock mode) or `E2E_LIVE_LLM=true npx playwright test` (real LLM mode).
- **E2E-04** The framework MUST support dual-mode operation: `mock` mode (deterministic, fast, no LLM cost) and `live` mode (real LLM calls, catches response-quality bugs).
- **E2E-05** Mock mode MUST use a FastAPI dependency override that intercepts LLM calls and returns pre-recorded responses based on SHA256 prompt hashing — sub-30s suite, 100% reproducible.
- **E2E-06** Mock seed files MUST be committed as JSON fixtures under `front/e2e/fixtures/` — one seed file per flow, regenerated with `E2E_RECORD_MODE=true`.
- **E2E-07** Live mode MUST pass `E2E_LIVE_LLM=true` to the backend, which disables the mock override and routes all LLM calls to the real provider (Ollama/Groq per `.env`).

### Real LLM E2E Mode

- **LIVE-01** Live mode assertions MUST be tolerance-based: "exam has 3–7 questions" (not exactly 5), "score ≥ 3" (not ≥ 8), "response contains topic keyword" (not exact string match).
- **LIVE-02** Live mode MUST include retry logic on transient LLM failures: up to 3 retries with 2s exponential backoff on timeout/rate-limit errors.
- **LIVE-03** Live mode MUST capture a Playwright screenshot at each major step (upload, exam render, submit, results, dashboard) for post-mortem debugging.
- **LIVE-04** Each live-mode test MUST log: LLM provider used, total LLM call count, total wall time, and any retry events — printed to stdout at test end.
- **LIVE-05** Live mode suite (4 flows) MUST complete in under 5 minutes with a real LLM provider (p95).
- **LIVE-06** Live mode tests MUST be runnable independently: `npx playwright test --grep "@live"` to target only live-mode variants.

### E2E Test Flows

- **E2E-FLOW-01** **Ingest → Exam flow**: Navigate to app → create session "IA 2026" → upload test PDF → verify topic tree renders → request exam on topic "Agentes Inteligentes" → verify 5 questions render → answer all questions → submit → verify results page shows scores.
- **E2E-FLOW-02** **Profile persistence flow**: Complete E2E-FLOW-01 → navigate to dashboard → verify stats cards show 1 session, topics covered > 0, average > 0% → verify topic chart renders → verify weak topics list renders.
- **E2E-FLOW-03** **Session lifecycle flow**: Create session → verify sidebar shows it → upload file → verify file list updates → switch to another session → verify context changes → delete session → verify it disappears.
- **E2E-FLOW-04** **Weak topic prioritization**: Seed profile with weak topic "RAG/Chunking" (score 3) and strong topic "Agentes/Tipos" (score 9) → request exam → verify ≥60% questions target "RAG/Chunking" → verify topic_distribution in response.

### Manual Test Checklist

- **MAN-01** A `MANUAL_TEST_CHECKLIST.md` MUST be created at the repo root with step-by-step instructions for a human or LLM to validate the full system end-to-end.
- **MAN-02** The checklist MUST cover: backend health check, frontend load, session creation, PDF ingest, topic tree verification, exam generation, exam answering, evaluation, dashboard verification, session switching, and session deletion.
- **MAN-03** Each checklist item MUST include: the action to perform, the expected result, and how to verify it (what to look for in UI/network/logs).
- **MAN-04** The checklist MUST include a "Quick smoke test" subset (5 items, <3 minutes) for rapid pre-demo validation.

## Non-Functional Requirements

- **MEMFIX-NFR-01** Profile update on evaluation MUST NOT add more than 100ms to the evaluation endpoint latency (p95) — it runs as a background task or fire-and-forget.
- **PROFIX-NFR-01** Dashboard aggregation query MUST complete in under 500ms (p95) for typical student data (50 evaluations, 10 sessions).
- **E2E-NFR-01** Mock-mode suite (4 flows) MUST complete in under 60 seconds on a developer laptop.
- **E2E-NFR-02** Mock-mode tests MUST be deterministic: 10 consecutive runs produce identical pass/fail results.
- **E2E-NFR-03** Mock seed mode MUST NOT require environment variable changes for E2E — it activates via a test-only FastAPI dependency override (`E2E_TEST_MODE=true`).
- **E2E-NFR-04** E2E tests MUST produce Playwright trace files on failure for debugging (`trace: 'on-first-retry'`).
- **E2E-NFR-05** Live-mode tests MUST skip gracefully (not fail) when the configured LLM provider is unreachable — log the skip reason and continue remaining tests.
- **E2E-NFR-06** Seed file recording (`E2E_RECORD_MODE=true`) MUST produce valid, replayable seeds — a recorded seed run through mock mode MUST produce the same test pass/fail result.

## Test Coverage

### New E2E Tests (Playwright — `front/e2e/`)

Each flow has a mock variant and a live variant. Mock runs on every commit; live runs pre-defense/nightly.

| # | Test | Mode | Flow | What It Proves |
|---|------|------|------|----------------|
| 1 | `ingest-exam-flow.spec.ts` | mock | Ingest → Exam → Evaluate | Full pipeline structure: file upload, topic tree, exam render, answer submission, results display |
| 1L | `ingest-exam-flow.spec.ts` | `@live` | Ingest → Exam → Evaluate | Real LLM: actual classification, topic extraction, exam generation quality, evaluation scoring validity |
| 2 | `profile-persistence.spec.ts` | mock | Exam → Dashboard | Profile update plumbing: evaluation triggers profile write, dashboard reads it back |
| 2L | `profile-persistence.spec.ts` | `@live` | Exam → Dashboard | Real LLM: end-to-end data flows with actual scores, dashboard reflects real exam results |
| 3 | `session-lifecycle.spec.ts` | mock | Session CRUD + file tracking | Session sidebar: create, switch, delete work; file list updates after upload |
| 3L | `session-lifecycle.spec.ts` | `@live` | Session CRUD + file tracking | Real ingest: actual PDF parsing, topic tree generation, file metadata persistence verified |
| 4 | `weak-topic-prioritization.spec.ts` | mock | Profile → Exam bias | Weak topics fed to generator, topic_distribution field present in response |
| 4L | `weak-topic-prioritization.spec.ts` | `@live` | Profile → Exam bias | Real LLM: verifies generator actually biases toward weak topics with real profile data |

### New Unit/Integration Tests (Backend)

- Unit test: `update_student_profile` is called on exam submission path (mocked tool)
- Unit test: `update_student_profile` is called on exercise submission path (mocked tool)
- Unit test: `update_student_profile` failure does not block evaluation result
- Unit test: `get_topic_scores` with `session_id=None` aggregates across all sessions
- Unit test: `get_topic_scores` with `session_id=uuid` filters to that session
- Unit test: `compute_weak_topics` returns topics below threshold (6.0)
- Unit test: `get_recent_sessions` orders by most recent evaluation, not creation
- Unit test: Dashboard endpoint returns valid empty profile when no evaluations exist
- Unit test: ExamGenerator receives weak topics in prompt when profile has them
- Unit test: ExamGenerator uses uniform distribution when profile is empty
- Unit test: `topic_distribution` field present in exam response
- Integration test: Full flow with real SQLite — evaluate → check profile updated → dashboard returns data
- Integration test: PRD Case 4 — second session exam prioritizes weak topics (real LLM)

### Updated PRD Test Case Coverage

| PRD # | Status Before | Status After |
|-------|--------------|--------------|
| 4 | ⏳ Not yet implemented | ✅ Integration test + E2E test |

## User Stories

### US-12.1: Fix evaluation → profile persistence (MEMFIX)
- **As a** student who just completed an exam
- **I want** my scores to be saved to my profile automatically
- **So that** my dashboard and future sessions reflect my progress
- **Acceptance criteria:**
  - After submitting exam answers via `/api/evaluate`, `update_student_profile` is called
  - Topic scores appear in `GET /api/students/{student_id}/dashboard` within 2 seconds
  - If profile update fails, evaluation result is still returned — user is not blocked
  - Failed profile updates are logged at WARNING with DB error and evaluation context
  - Exercise submission also triggers profile update
  - Chat-based evaluation also triggers profile update
- **Dependencies:** Epic 6 (`update_student_profile`, `upsert_topic_scores`)
- **Maps to:** MEMFIX-01 through MEMFIX-04, RF-11

### US-12.2: Fix profile aggregation SQL (PROFIX)
- **As a** developer debugging the dashboard
- **I want** profile aggregation queries to return correct, complete data
- **So that** the frontend always shows accurate stats
- **Acceptance criteria:**
  - `get_topic_scores(student_id)` aggregates scores from all sessions
  - `get_topic_scores(student_id, session_id)` filters to one session only
  - `compute_weak_topics(student_id)` returns topics with latest score < 6.0
  - `get_recent_sessions(student_id)` orders by most recent evaluation timestamp
  - All queries complete in <100ms for typical data volume
  - Unit tests cover each function with in-memory SQLite
- **Dependencies:** `back/src/memory/schema.py`
- **Maps to:** PROFIX-01 through PROFIX-04

### US-12.3: Fix dashboard data pipeline (DASH)
- **As a** student viewing my progress page
- **I want** to see real stats immediately after completing an exam
- **So that** I can track my improvement
- **Acceptance criteria:**
  - Dashboard shows non-zero session count, topics covered, and average score after one exam
  - Empty state renders cleanly (no spinner forever, no crash) when no evaluations exist
  - Dashboard endpoint responds in <500ms (p95)
  - Stats cards, topic chart, weak topics, and session history all render with real data
  - Active session name appears in dashboard header when a session is selected
- **Dependencies:** US-12.1, US-12.2
- **Maps to:** DASH-01 through DASH-04, RF-13

### US-12.4: Implement weak topic prioritization (PRIO)
- **As a** returning student with known weak areas
- **I want** exam questions to focus on topics I struggle with
- **So that** my study time is spent where it matters most
- **Acceptance criteria:**
  - ExamGenerator receives weak topics from student profile
  - When weak topics exist, ≥60% of exam questions target them
  - When no prior evaluations, uniform topic distribution used
  - Exam response includes `topic_distribution` field for verification
  - PRD Case 4 is now tested and verified
- **Dependencies:** US-12.1, Epic 3 (ExamGenerator)
- **Maps to:** PRIO-01 through PRIO-04, PRD Case 4

### US-12.5: Integrate Playwright E2E framework
- **As a** developer shipping for defense
- **I want** a robust E2E testing framework that covers the full stack
- **So that** regressions in UI→API→LLM flow are caught automatically
- **Acceptance criteria:**
  - Playwright installed as dev dependency in `front/package.json`
  - `playwright.config.ts` configured with Chromium headless, baseURL `localhost:3000`
  - `front/e2e/` directory with TypeScript test files
  - `npm run test:e2e` script in `front/package.json`
  - Test fixtures directory `front/e2e/fixtures/` with mock LLM seeds
  - Trace files generated on failure
  - README section documenting how to run E2E tests
- **Dependencies:** `front/package.json`
- **Maps to:** E2E-01 through E2E-05

### US-12.6: Create deterministic LLM mock mode
- **As a** test author
- **I want** LLM calls to return reproducible, pre-recorded responses during E2E tests
- **So that** tests are fast, deterministic, and don't require API keys
- **Acceptance criteria:**
  - Backend has a test mode activated by `E2E_TEST_MODE=true` env var or FastAPI dependency override
  - In test mode, `src/llm.py` factory returns a mock that matches prompts to seed files via hash
  - Seed files are JSON: `{"prompt_hash": "abc123", "response": "...", "tool_calls": [...]}`
  - One seed file per flow: `ingest-exam-seed.json`, `profile-seed.json`, etc.
  - Non-matching prompts raise a clear error: "No seed for prompt hash X — run in record mode to generate"
  - Optional record mode (`E2E_RECORD_MODE=true`) saves real LLM responses to seed files for later replay
- **Dependencies:** `back/src/llm.py`, `back/src/config.py`
- **Maps to:** E2E-04 through E2E-06

### US-12.7: Write E2E test suite — mock mode (4 flows)
- **As a** team preparing for defense
- **I want** fast, deterministic E2E tests that prove structural correctness
- **So that** every commit catches regressions in UI→API→DB plumbing
- **Acceptance criteria:**
  - `ingest-exam-flow.spec.ts`: upload PDF → verify topic tree renders → request exam → verify 5 questions render → answer all → submit → verify results page
  - `profile-persistence.spec.ts`: complete mock exam → navigate to dashboard → verify stats cards, topic chart, weak topics, session history all render
  - `session-lifecycle.spec.ts`: create session → sidebar shows it → upload file → file list updates → switch session → delete session
  - `weak-topic-prioritization.spec.ts`: seed weak topic profile → request exam → verify topic_distribution field present
  - All 4 pass consistently (10/10 runs)
  - Tests use deterministic mock mode — no real LLM calls
  - Suite completes in <60 seconds
- **Dependencies:** US-12.5, US-12.6, US-12.1 through US-12.4
- **Maps to:** E2E-FLOW-01 through E2E-FLOW-04 (mock variants)

### US-12.8: Write E2E test suite — live LLM mode (4 flows)
- **As a** team validating before defense
- **I want** the same 4 flows tested with real LLM calls
- **So that** I catch bugs from malformed LLM responses, timeouts, hallucinations, and scoring anomalies
- **Acceptance criteria:**
  - Live mode activated via `E2E_LIVE_LLM=true` on backend, `@live` tag on test
  - All tolerance-based assertions: question count 3–7, score ≥ 3, topic keyword present
  - Retry on transient LLM failures: 3 attempts, 2s exponential backoff
  - Screenshot captured at each step: upload, exam render, submit, results, dashboard
  - Each test logs: LLM provider, call count, wall time, retry events
  - Suite completes in <5 minutes with real Ollama/Groq
  - Tests skip gracefully if LLM provider unreachable (don't fail CI on network issues)
  - Run independently: `npx playwright test --grep "@live"`
- **Dependencies:** US-12.7 (same test files, dual-mode variants)
- **Maps to:** LIVE-01 through LIVE-06

### US-12.9: Create manual test checklist
- **As a** developer or LLM agent validating the system
- **I want** a clear, step-by-step checklist to manually verify every feature
- **So that** I can smoke-test before the defense and document known issues
- **Acceptance criteria:**
  - `MANUAL_TEST_CHECKLIST.md` at repo root with 15-20 items covering all flows
  - Each item: action, expected result, verification method
  - Quick smoke test subset (5 items, <3 minutes)
  - Includes backend health check, frontend load, and all agent workflows
  - Written so an LLM agent can follow it (deterministic assertions, not subjective judgments)
  - Links to relevant API endpoints and UI pages
- **Dependencies:** —
- **Maps to:** MAN-01 through MAN-04

### US-12.10: Full regression pass + documentation
- **As a** developer merging this epic
- **I want** all existing tests to pass and new E2E tests to validate the hardened system
- **So that** the defense demo has zero known regressions
- **Acceptance criteria:**
  - All 105 existing unit tests pass
  - All 16 existing integration tests pass
  - All 4 mock-mode E2E tests pass (10/10 deterministic runs)
  - All 4 live-mode E2E tests pass (at least 1 run before defense)
  - New backend unit/integration tests for memory fixes pass
  - `tests_documentation.md` updated: E2E test inventory, Playwright setup, dual-mode docs
  - `AGENTS.md` updated: E2E test commands, mock/live mode usage
  - `ruff` format/lint clean
  - `npm run build` succeeds (no TypeScript errors)
- **Dependencies:** US-12.1 through US-12.9
- **Maps to:** All requirements

## Technical Notes

### Playwright Setup

```bash
cd front
npm install -D @playwright/test
npx playwright install chromium
```

```typescript
// playwright.config.ts
import { defineConfig } from '@playwright/test';

const LIVE_LLM = process.env.E2E_LIVE_LLM === 'true';
const RECORD_MODE = process.env.E2E_RECORD_MODE === 'true';

export default defineConfig({
  testDir: './e2e',
  timeout: LIVE_LLM ? 120000 : 30000,       // 2min per test in live mode
  retries: LIVE_LLM ? 2 : 1,                 // more retries for flaky LLM
  use: {
    baseURL: 'http://localhost:3000',
    headless: true,
    screenshot: LIVE_LLM ? 'on' : 'only-on-failure',  // screenshot every step in live
    trace: 'on-first-retry',
  },
  // Tag live tests so they can be run independently
  grep: process.env.E2E_LIVE_ONLY ? /@live/ : undefined,
  webServer: [
    {
      command: 'cd ../back && uv run uvicorn src.main:app --port 8000',
      port: 8000,
      reuseExistingServer: !RECORD_MODE,  // fresh server for recording seeds
      env: {
        E2E_TEST_MODE: LIVE_LLM ? 'false' : 'true',
        E2E_LIVE_LLM: LIVE_LLM ? 'true' : 'false',
        E2E_RECORD_MODE: RECORD_MODE ? 'true' : 'false',
      },
    },
  ],
});
```

### Dual-Mode Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      E2E Test (Playwright)                    │
│  playwright clicks → frontend fetch → backend API             │
└────────────────────────┬─────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  FastAPI App        │
              │  reads env:         │
              │  E2E_LIVE_LLM?      │
              └──────┬──────┬───────┘
                     │      │
          ┌──────────▼┐  ┌──▼──────────────┐
          │ LIVE MODE  │  │ MOCK MODE       │
          │ (real LLM) │  │ (seed replay)   │
          └──────────┬─┘  └──┬──────────────┘
                     │        │
          ┌──────────▼┐  ┌───▼──────────────┐
          │ Ollama /  │  │ MockLLM           │
          │ Groq      │  │ prompt → SHA256   │
          │ real call │  │ → seed JSON file  │
          └──────────┬─┘  └───┬──────────────┘
                     │         │
          ┌──────────▼┐  ┌───▼──────────────┐
          │ Real      │  │ Pre-recorded      │
          │ response  │  │ response + tools  │
          │ (variable)│  │ (deterministic)   │
          └───────────┘  └───────────────────┘

Record mode (E2E_RECORD_MODE=true):
  Live LLM call → save prompt_hash + response → seed JSON file
  Then replay with mock mode for deterministic CI.
```

### Deterministic Mock Mode (E2E_TEST_MODE=true, E2E_LIVE_LLM=false)

```python
# back/src/llm_test.py (new)
import hashlib, json
from pathlib import Path

SEEDS_DIR = Path(__file__).parent.parent.parent / "front" / "e2e" / "fixtures"

class MockLLM:
    """Returns pre-recorded responses matched by prompt hash."""
    
    def __init__(self, seed_file: str):
        with open(SEEDS_DIR / seed_file) as f:
            self.seeds = {s["prompt_hash"]: s for s in json.load(f)}
    
    async def ainvoke(self, messages, **kwargs):
        prompt_text = json.dumps(messages, sort_keys=True)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]
        
        seed = self.seeds.get(prompt_hash)
        if not seed:
            raise ValueError(
                f"No seed for prompt hash {prompt_hash}. "
                f"Run with E2E_RECORD_MODE=true to generate seeds."
            )
        return MockResponse(seed["response"], seed.get("tool_calls", []))
```

### Seed File Format

```json
// front/e2e/fixtures/ingest-exam-seed.json
[
  {
    "prompt_hash": "a1b2c3d4e5f6",
    "label": "classify_document — apunte_teorico",
    "response": "{\"classification\": \"apunte_teorico\", \"confidence\": 0.95, ...}",
    "tool_calls": []
  },
  {
    "prompt_hash": "b2c3d4e5f6a1",
    "label": "extract_topics — segment 1",
    "response": "{\"topics\": [\"Agentes inteligentes\", \"Taxonomía de agentes\"], ...}",
    "tool_calls": []
  }
]
```

### Profile Update — Where the Fix Goes

The core bug: `update_student_profile` is NOT called in all evaluation paths.

**Current state (broken):**
- `/api/evaluate` → Evaluator agent → returns scores → Orchestrator sometimes calls `update_student_profile`
- If Orchestrator path skips it (error, early return, recursion limit), profile is never updated

**Fix — Observer pattern in Evaluator:**
```python
# back/src/agents/evaluator.py — after scoring completes
async def _persist_results(state: EvaluatorState) -> dict:
    """Fire-and-forget profile update after evaluation."""
    try:
        await update_student_profile(
            student_id=state["student_id"],
            topic_scores=_extract_topic_scores(state),
        )
    except Exception as exc:
        logger.warning("Profile update failed (non-blocking): %s", exc)
    return {}  # Don't block the graph
```

Add this as a terminal node in the Evaluator graph, AFTER the batch evaluation loop completes. It runs regardless of whether the Orchestrator later does its own update — upsert is idempotent.

### Memory State — Three-Layer Verification Checklist

| Layer | What | Verification |
|-------|------|-------------|
| Short-term (LangGraph) | Last N chat messages | Chat → "what did I just ask?" → agent references prior message |
| Session (SQLite+ChromaDB) | Files, per-session scores | Dashboard → per-session profile endpoint → topic scores |
| Long-term (SQLite global) | Weak topics, session history, preferences | Dashboard → global dashboard endpoint → aggregated stats |

### Files Affected

| File | Change |
|------|--------|
| `back/src/agents/evaluator.py` | Add `_persist_results` terminal node, wire into graph |
| `back/src/agents/exam_generator.py` | Add weak topic prioritization in generation prompt |
| `back/src/memory/schema.py` | Fix `get_topic_scores` aggregation, `compute_weak_topics` threshold, `get_recent_sessions` ordering |
| `back/src/api/router.py` | Ensure dashboard endpoint handles empty profile gracefully |
| `back/src/llm.py` | Add `get_llm()` dual-mode override (mock vs live) |
| `back/src/llm_test.py` | **NEW** — MockLLM + SeedMatcher + RecordMode |
| `back/src/config.py` | Add `E2E_TEST_MODE`, `E2E_LIVE_LLM`, `E2E_RECORD_MODE` settings |
| `front/package.json` | Add `@playwright/test`, `test:e2e`, `test:e2e:live` scripts |
| `front/playwright.config.ts` | **NEW** — Dual-mode Playwright configuration |
| `front/e2e/*.spec.ts` | **NEW** — 4 E2E test files (mock + @live variants) |
| `front/e2e/fixtures/*.json` | **NEW** — Mock LLM seed files (recorded from real runs) |
| `MANUAL_TEST_CHECKLIST.md` | **NEW** — Manual test procedure |
| `tests_documentation.md` | Update: E2E section, dual-mode docs |
| `AGENTS.md` | Update: E2E test commands, mock/live mode usage |

### Implementation Order

```
Phase 1 — Memory fixes (backend only, testable immediately):
  US-12.1 (profile persistence)
  US-12.2 (aggregation SQL)
  US-12.3 (dashboard pipeline)

Phase 2 — Feature completion:
  US-12.4 (weak topic prioritization)

Phase 3 — E2E infrastructure:
  US-12.5 (Playwright setup)
  US-12.6 (dual-mode mock/live LLM)

Phase 4 — Mock E2E tests:
  US-12.7 (4 mock-mode flows — fast, deterministic)

Phase 5 — Record seeds + Live E2E tests:
  Run E2E_RECORD_MODE=true → record real LLM responses → commit seed files
  US-12.8 (4 live-mode flows — run with @live tag for pre-defense validation)

Phase 6 — Documentation:
  US-12.9 (manual checklist)
  US-12.10 (regression pass + docs update)
```

Phases 1 and 2 are independent of 3-5 and can be verified with existing backend tests. Phases 3-5 depend on 1-2 being stable to record correct seed files.

### Live Mode Test Example

```typescript
// front/e2e/ingest-exam-flow.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Ingest → Exam → Evaluate', () => {
  test('mock mode — structural flow', async ({ page }) => {
    // ... strict assertions: exactly 5 questions, exact score value
  });

  test('@live real LLM — quality validation', async ({ page }) => {
    test.slow();  // 2min timeout

    await page.goto('/');
    // Create session
    await page.click('[data-testid="new-session-btn"]');
    await page.fill('[data-testid="session-name-input"]', 'IA 2026 Live');
    await page.click('[data-testid="session-create-confirm"]');

    // Upload PDF
    const fileChooser = page.locator('input[type="file"]');
    await fileChooser.setInputFiles('./fixtures/apunteAgentes_IA2007.pdf');

    // Tolerance-based assertions for real LLM
    await expect(page.locator('[data-testid="topic-tree"]')).toBeVisible({ timeout: 30000 });

    // Request exam
    await page.fill('[data-testid="chat-input"]', 'dame un examen de agentes inteligentes');
    await page.click('[data-testid="send-btn"]');

    // Real LLM: question count may vary
    const questions = page.locator('[data-testid="exam-question"]');
    const count = await questions.count();
    expect(count).toBeGreaterThanOrEqual(3);
    expect(count).toBeLessThanOrEqual(7);

    // Answer all questions (best-effort for MCQ)
    for (let i = 0; i < count; i++) {
      const radio = questions.nth(i).locator('input[type="radio"]').first();
      if (await radio.isVisible()) await radio.click();
    }

    await page.click('[data-testid="submit-exam-btn"]');

    // Results: at least some scores should be non-zero
    await expect(page.locator('[data-testid="results-container"]')).toBeVisible({ timeout: 30000 });
    const scoreText = await page.locator('[data-testid="total-score"]').textContent();
    expect(scoreText).toBeTruthy();

    // Screenshot for manual inspection
    await page.screenshot({ path: `test-results/live-ingest-exam-${Date.now()}.png`, fullPage: true });
  });
});
```

### package.json Scripts

```json
{
  "scripts": {
    "test:e2e": "npx playwright test",
    "test:e2e:live": "E2E_LIVE_LLM=true npx playwright test --grep @live",
    "test:e2e:record": "E2E_LIVE_LLM=true E2E_RECORD_MODE=true npx playwright test --grep @live",
    "test:e2e:mock": "npx playwright test --grep-invert @live"
  }
}
```

### Risk: Seed File Drift

When backend prompts change (e.g., system prompt edits), the SHA256 hash changes and mock seeds become stale. Mitigation:
- `E2E_RECORD_MODE=true` regenerates all seeds from real LLM responses
- CI workflow: record mode before test mode, or maintain seed files as part of PR review
- Seed files include a `label` field so drift is human-readable
