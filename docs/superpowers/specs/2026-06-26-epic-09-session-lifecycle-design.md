# Design: Epic 9 — Session Lifecycle & Study Session Model

**Date:** 2026-06-26
**Status:** Draft
**Source PRD:** [init_PRD.md](../../init_PRD.md) §4.4, RNF-02
**Extends:** [epic-09-profile-bootstrap.md](../../epics/epic-09-profile-bootstrap.md)

## Context

Epic 9 originally scoped only Profile Bootstrap — loading the student profile at session start. User requirements have expanded to a full study session lifecycle: sessions act as named "subjects" (materias), each with its own RAG namespace, file tracking, short-term agent memory, and per-session progress tracking. A global student profile aggregates across all sessions, while each session maintains its own material and topic progress.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Session model | Flat (Approach A): session = subject | Aligns with user definition "sesión = materia". Simpler than two-level Subject+ChatThread model. |
| Session naming | Named: name + optional description | User explicitly chose named sessions over anonymous/date-based. |
| File persistence | Metadata only in DB | Chunks already in ChromaDB. No need to keep raw files on disk. |
| Short-term memory | LangGraph checkpoint (last N interactions) + session tools | Built-in checkpointing already exists via AsyncSqliteSaver. Extend with query tools. |
| Profile split | Global (all sessions) + per-session (filtered) | Global for preferences/overall stats; per-session for material-specific progress. |

## Data Model

### Schema Changes

```sql
-- sessions: add name + description
ALTER TABLE sessions ADD COLUMN name TEXT NOT NULL DEFAULT '';
ALTER TABLE sessions ADD COLUMN description TEXT DEFAULT '';

-- ingested_documents: link to session
ALTER TABLE ingested_documents ADD COLUMN session_id TEXT 
    REFERENCES sessions(id);
```

### Entity Relationships

```
students (global profile)
  ├── id TEXT PK
  ├── preferences_json TEXT
  ├── created_at TEXT
  │
  └── sessions (one per subject)
       ├── id TEXT PK (= session_id)
       ├── student_id TEXT FK → students.id
       ├── name TEXT ("IA 2026", "Cálculo")
       ├── description TEXT (optional)
       ├── started_at, ended_at, status
       │
       ├── ingested_documents
       │    ├── id TEXT PK
       │    ├── session_id TEXT FK → sessions.id
       │    ├── file_name, classification
       │    ├── topics_json, chunks_count, ingested_at
       │
       ├── evaluations (existing, filtered by session_id)
       └── topic_scores (existing, filtered by session_id)
```

### Profile Access Patterns

| Endpoint | Scope | Description |
|----------|-------|-------------|
| `GET /api/profile/{student_id}` | Global | Aggregated stats across ALL sessions |
| `GET /api/sessions/{session_id}/profile` | Per-session | Progress in THIS subject only |
| `GET /api/dashboard/{student_id}` | Global | Full dashboard with session history |

## Memory Architecture

Three-layer memory model:

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Short-term (LangGraph checkpoint)         │
│  ─────────────────────────────────────────────      │
│  • Last N interactions via AsyncSqliteSaver          │
│  • Thread per session (thread_id = session_id)       │
│  • classify_intent receives conversation history     │
│  • Already EXISTS — just needs to be UTILIZED        │
├─────────────────────────────────────────────────────┤
│  LAYER 2: Session (SQLite + ChromaDB)               │
│  ─────────────────────────────────────────────      │
│  • Uploaded files (ingested_documents)               │
│  • Topic progress (topic_scores filtered by session)│
│  • Completed exams (evaluations)                     │
│  • Accessed via AGENT TOOLS (not injected in prompt) │
├─────────────────────────────────────────────────────┤
│  LAYER 3: Long-term (SQLite students)               │
│  ─────────────────────────────────────────────      │
│  • Global profile: preferences, aggregate stats      │
│  • Loaded at session bootstrap (Epic 9 original)     │
│  • Updated after each evaluation                     │
└─────────────────────────────────────────────────────┘
```

## Orchestrator Graph Changes

### Current Graph

```
START → classify_intent → [route] → plan_composite → execute_step → synthesize_response → END
```

### New Graph

```
START → load_profile → load_session_context → classify_intent → [route] → ... → synthesize_response → END
```

### New Nodes

**load_profile** (Epic 9 original — US-9.1 to US-9.5):
- Calls `get_student_summary` tool with `student_id = session_id`
- Populates `OrchestratorState.student_profile`
- Falls back to `{}` on failure (non-blocking)
- Logs WARNING on fallback

**load_session_context** (Epic 9 extension — US-9.9):
- Queries `ingested_documents` for this session
- Queries `topic_scores` for this session
- Populates `OrchestratorState.session_context` (new field)
- Available for `classify_intent` and `synthesize_response`

### State Schema Additions

```python
class OrchestratorState(TypedDict):
    # ... existing fields ...
    student_profile: dict | None       # populated by load_profile
    session_context: dict | None       # NEW: files, progress for this session
    messages_history: list[dict]       # NEW: last N interactions from checkpoint
```

### classify_intent Prompt Enrichment

The classification prompt now receives:
1. Current user message
2. Last N conversation messages (from LangGraph checkpoint)
3. `student_profile` (weak topics, preferences) — if available
4. `session_context` (files uploaded, topic progress) — if available

### New Agent Tools

| Tool | Source | Returns |
|------|--------|---------|
| `list_session_files` | `ingested_documents` table | `[{id, file_name, classification, topics, chunks_count, ingested_at}]` |
| `get_session_progress` | `topic_scores` + `evaluations` filtered | `{topics: [{name, score, evaluations_count}], weak_topics: [...]}` |
| `get_student_summary` | Existing tool (Support Agent) | Global profile data |

## API Endpoints

### New

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List all sessions for the user (query param: `student_id`) |
| `POST` | `/api/sessions` | Create a new session `{name, description?}` → returns session with generated UUID |
| `GET` | `/api/sessions/{id}` | Get session details (name, files count, progress summary) |
| `DELETE` | `/api/sessions/{id}` | Delete a session and its associated files |
| `GET` | `/api/sessions/{id}/files` | List files uploaded to this session |
| `GET` | `/api/sessions/{id}/profile` | Per-session topic scores, weak topics, exam history |

### Modified

| Method | Path | Change |
|--------|------|--------|
| `POST` | `/api/ingest` | Now inserts row into `ingested_documents` table (metadata persistence) |
| `POST` | `/api/chat` | No API change — orchestrator internally uses checkpoint history |

## Frontend Architecture

### Component Tree

```
ChatPage
├── SessionSidebar
│   ├── SessionCreateButton → SessionCreateModal
│   └── SessionList
│       └── SessionItem (name, file count, progress, active indicator)
├── SessionFileList
│   └── FileItem (name, classification badge, topics, date)
├── UploadDropzone (existing, modified)
└── ChatMessageList + ChatInput (existing, unchanged)
```

### Hook: useStudySession (replaces useSession)

```typescript
interface UseStudySession {
  sessions: Session[];
  activeSession: Session | null;
  isLoading: boolean;
  createSession: (name: string, description?: string) => Promise<Session>;
  switchSession: (id: string) => void;
  deleteSession: (id: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
}
```

Active session ID stored in `localStorage` under `ai-tutor-active-session`.

### Data Flow

```
User creates session → POST /api/sessions → session saved in DB → sidebar updates
User switches session → localStorage updated → chat + file list reload for new session
User uploads file → POST /api/ingest → ingested_documents populated → SessionFileList refreshes via GET
User chats → POST /api/chat → orchestrator loads profile + session context → personalized response
```

## User Stories

### Original Epic 9 (preserved)

| ID | Story | Status |
|----|-------|--------|
| US-9.1 | Load profile at session start | Preserved |
| US-9.2 | Graceful fallback on profile failure | Preserved |
| US-9.3 | Personalized intent classification | Preserved |
| US-9.4 | Personalized response synthesis | Preserved |
| US-9.5 | Student identity resolution | Preserved |

### Extended Epic 9 (new)

| ID | Story | Dependencies |
|----|-------|-------------|
| US-9.6 | Session model: DB migration + CRUD API | schema.py, router.py |
| US-9.7 | File metadata persistence | US-9.6 |
| US-9.8 | Short-term memory via LangGraph checkpoint | Orchestrator (US-9.1-9.5) |
| US-9.9 | Session context for agent (tools + node) | US-9.7, US-9.8 |
| US-9.10 | Per-session profile endpoint | US-9.6 |
| US-9.11 | Session sidebar (frontend) | US-9.6 (API) |
| US-9.12 | File list in session (frontend) | US-9.7, US-9.11 |
| US-9.13 | Integration tests + documentation | All above |

## Implementation Order

```
US-9.1..9.5 (profile bootstrap — existing, needs implementation)
    │
    ▼
US-9.6 (DB sessions + API CRUD)
    │
    ├──▶ US-9.7 (file persistence) ──▶ US-9.12 (frontend files)
    │
    ├──▶ US-9.8 (short-term memory)
    │         │
    │         ▼
    │    US-9.9 (session context for agent)
    │
    ├──▶ US-9.10 (per-session profile)
    │
    └──▶ US-9.11 (sidebar frontend)
              │
              ▼
         US-9.13 (integration + tests)
```

## Files Affected

| File | Change |
|------|--------|
| `back/src/memory/schema.py` | ALTER sessions, ingested_documents; new query functions |
| `back/src/agents/orchestrator.py` | New nodes: load_profile, load_session_context; state additions |
| `back/src/tools/orchestrate_chat.py` | Extract history from checkpoint; pass to initial state |
| `back/src/tools/` | New tools: list_session_files.py, get_session_progress.py |
| `back/src/api/router.py` | New endpoints + ingest modification |
| `back/src/api/schemas.py` | New Pydantic models for sessions |
| `front/src/hooks/useSession.ts` → `useStudySession.ts` | Replace with session management hook |
| `front/src/components/upload/UploadDropzone.tsx` | Minor: accept active session prop |
| `front/src/components/upload/UploadFileList.tsx` → `SessionFileList.tsx` | Show persisted files |
| `front/src/components/layout/SessionSidebar.tsx` | NEW: session management sidebar |
| `front/src/app/page.tsx` | Integrate sidebar + file list + useStudySession |
| `front/src/lib/api.ts` | New API client methods |
| `front/src/lib/types.ts` | New types: Session, SessionProfile |
| `epics/epic-09-profile-bootstrap.md` | Update scope to include extended stories |
| `docs/superpowers/specs/2026-06-26-epic-09-session-lifecycle-design.md` | This document |

## Risks

| Risk | Mitigation |
|------|-----------|
| LangGraph checkpoint history may be empty on first message | Default to empty list; no special handling needed |
| `ingested_documents` table exists but was never populated | Migration is additive (ALTER TABLE), no data loss |
| Multiple sessions share same student_id → ChromaDB collections must be per-session | Already the case: collection name includes session_id |
| Frontend session switching may cause state flicker | Use Suspense + loading states per session switch |
