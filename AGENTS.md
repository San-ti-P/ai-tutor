# AGENTS.md — Tutor Academico Personal

Multi-agent LLM system for adaptive university exam preparation. 6 specialized agents coordinate via LangGraph to ingest academic material, generate personalized exams/exercises, evaluate answers, and track student progress across sessions.

**Stack**: Python 3.12+ · LangGraph · LangChain · FastAPI · ChromaDB · SQLite · Langfuse · Next.js 15 · React 19 · Tailwind 4
**Course**: IA 2026 — UTN Santa Fe (CIDISI)
**Deliveries**: 08/06 (concept) → 22/06 (MVP) → 29/06 (complete + defense)

---

## Quick Start

```bash
# Backend (uses uv for package management)
cd back
uv sync
uv run uvicorn src.main:app --reload

# Frontend
cd front
npm install
npm run dev
```

---

## Architecture

```
ai-tutor/
├── back/                  # FastAPI + LangGraph + ChromaDB
│   ├── src/
│   │   ├── agents/        # One file per agent (orchestrator, ingestor, exam_generator, ...)
│   │   ├── tools/         # Tool definitions (ingest_document, retrieve_chunks, evaluate_answer, ...)
│   │   ├── rag/           # ChromaDB setup, chunking, embedding, retrieval
│   │   ├── memory/        # SQLite schema, student profile CRUD
│   │   ├── api/           # FastAPI routes + Pydantic schemas
│   │   ├── observability/ # Langfuse setup + span helpers
│   │   ├── config.py      # Settings from .env (pydantic-settings)
│   │   └── main.py        # FastAPI app entry point
│   ├── tests/             # pytest suite (12 test cases from PRD section 8)
│   └── pyproject.toml     # uv project config + dependencies
├── front/                 # Next.js 15 App Router
│   └── src/               # Chat UI, exam renderer, file upload, dashboard
├── epics/                 # 8 epic docs — implementation breakdown per agent
├── init_PRD.md            # Product requirements — single source of truth
└── .agents/skills/        # Project skills (see .atl/skill-registry.md)
```

### Agent Roles

| Agent | Loop Pattern | Primary Tools |
|-------|-------------|---------------|
| **Orchestrator** | Plan-and-Execute | Routes to specialized agents, session state |
| **Ingestor** | ReAct | `ingest_document`, `ocr_math_extract` |
| **ExamGenerator** | ReAct + Tools | `retrieve_chunks`, `generate_exam` |
| **ExerciseGenerator** | ReAct + Tools | `retrieve_chunks`, `generate_exercise` |
| **Evaluator** | Chain-of-Thought | `evaluate_answer` |
| **Support Agent** | Reactive | `update_student_profile`, `get_student_summary` |

### Key Flows

1. **Ingestion**: Upload → markitdown → classify → OCR math → chunk → ChromaDB
2. **Exam generation**: User request → student profile → retrieve chunks → generate → validate
3. **Evaluation**: Student answer → (OCR if image) → evaluate → score + feedback → update profile

---

## Conventions

### Python (Backend)

- LangGraph nodes return **partial state dicts** — never mutate state directly
- Use `Annotated[list, operator.add]` reducers for accumulating lists in StateGraph
- Tool functions: clear name, typed args (Pydantic), docstring describing what it does
- Async all I/O: FastAPI endpoints, ChromaDB queries, LLM calls
- Env vars for all secrets — `.env` is gitignored, `.env.example` committed
- Use `langfuse.observe()` decorator on agent entry points
- `ruff` for linting/formatting

### TypeScript (Frontend)

- Use `next/image`, `next/font` — never native `<img>`
- Server Components default; `'use client'` only for interactivity
- `generateMetadata` for SEO; `generateStaticParams` for static paths
- Suspense + `Promise.all` to avoid data waterfalls
- `notFound()`, `redirect()` for server-side error handling

### General

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`
- Backend entities in `back/src/agents/` and `back/src/tools/` — no cross-agent imports outside defined APIs
- RAG pipeline: Load → Split → Embed → Store → Retrieve → Generate
- Same embedding model for indexing AND querying; don't mix

---

## Testing

```bash
# ── Backend ──────────────────────────────────────────
cd back
uv run pytest tests/ -v                    # Unit tests (fast, no external deps)
uv run pytest tests/ -v -k "test_ingest"   # Per module
uv run pytest tests/ -v --cov=src          # With coverage
uv run pytest tests/ -v -m integration     # Integration tests (needs real LLM)

# ── Frontend ─────────────────────────────────────────
cd front
npx playwright test                       # E2E mock mode (fast, deterministic, CI)
E2E_LIVE_LLM=true npx playwright test --grep @live   # E2E live mode (real LLM, pre-defense)
npx playwright test --grep-invert @live   # E2E mock only (skip live)
npx playwright test --list                # List all 16 E2E tests

# ── Record E2E seeds ─────────────────────────────────
# Start backend in record mode, then run:
cd back
E2E_RECORD_MODE=true E2E_LIVE_LLM=true uv run python scripts/record_e2e_seeds.py
# Seeds saved to front/e2e/fixtures/recorded-seed.json
```

### Test Tiers

| Tier | Command | Runs on | Requires |
|------|---------|---------|----------|
| Unit | `pytest tests/ -v` | Every commit | Nothing external — LLMs/embeddings mocked |
| Integration | `pytest tests/ -v -m integration` | Manually or CI with secrets | Real LLM (Ollama/Groq) + real embeddings + real PDF |
| **E2E Mock** | `npx playwright test` | Every commit | Backend + Frontend running — LLM calls mocked via seeds |
| **E2E Live** | `E2E_LIVE_LLM=true npx playwright test --grep @live` | Pre-defense, nightly | Real LLM (Ollama/Groq) + backend + frontend |

**Integration tests** use real external resources: real LLM calls, real SentenceTransformer embeddings, real ChromaDB, or real PDF files. Skipped by default (`addopts = "-m 'not integration'"` in `pyproject.toml`).

**E2E tests** (Playwright) exercise the full stack: browser → frontend → API → agents. Mock mode replays pre-recorded LLM seeds (sub-40s, deterministic). Live mode calls real LLMs with tolerance-based assertions (catches response-quality bugs).

### Test Documentation

Full test inventory, PRD case coverage mapping, fixture catalog, and rules for adding tests are documented in [`tests_documentation.md`](../tests_documentation.md). **When adding or modifying an integration test or a PRD-mapped test, update that document.**

12 test cases required (PRD section 8): 5 happy path, 4 edge cases, 3 adversarial. Tests use in-memory SQLite and ChromaDB where possible — no external services needed for unit tests.

---

## Skill Registry

Project skills live in `.agents/skills/`. The registry at `.atl/skill-registry.md` maps every skill to its trigger and provides compact rules for sub-agents. Skills auto-activate when their trigger matches the task.

Key skills by layer:

| Layer | Skills |
|-------|--------|
| Agent orchestration | langgraph-fundamentals, langgraph-persistence, langgraph-human-in-the-loop, langchain-fundamentals, langchain-middleware, multi-agent-orchestration |
| RAG | langchain-rag |
| Backend | fastapi-python, python-best-practices |
| Frontend | next-best-practices |
| Architecture | clean-architecture, framework-selection |
| Observability | langfuse |
| Dependencies | langchain-dependencies |

---

## Guardrails

| Risk | Mitigation |
|------|-----------|
| Hallucinated exam questions | Post-generation: verify each fact has chunk with score > threshold. Regenerate up to 3x, then skip |
| Infinite agent loop | Max 15 iterations per task. Terminate and return partial result |
| Low-confidence OCR | Threshold 0.85 — request user confirmation before using output |
| Non-academic material | Ingestor rejects files without apunte/examen structure |
| Inconsistent evaluations | LLM-as-judge: second pass validates Evaluator output (sampling) |

---

## References

| Document | Purpose |
|----------|---------|
| `init_PRD.md` | Full product requirements, architecture decisions, acceptance criteria |
| `epics/epic-01-orchestrator.md` | Orchestrator implementation plan |
| `epics/epic-02-ingestor.md` | Document ingestion + RAG setup |
| `epics/epic-03-exam-generator.md` | Exam generation flow |
| `epics/epic-04-exercise-generator.md` | Exercise generation |
| `epics/epic-05-evaluator.md` | Answer evaluation + scoring |
| `epics/epic-06-support-agent.md` | Student profile + progress |
| `epics/epic-07-ui.md` | Frontend architecture |
| `epics/epic-08-observability.md` | Langfuse integration + test suite |
| `.atl/skill-registry.md` | Full skill catalog with compact rules |
| `tests_documentation.md` | Test inventory, PRD coverage map, integration test rules |
