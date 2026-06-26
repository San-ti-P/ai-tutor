# Epic 9: Session Lifecycle & Profile Bootstrap

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.4, RNF-02)
**Design doc:** [docs/superpowers/specs/2026-06-26-epic-09-session-lifecycle-design.md](../docs/superpowers/specs/2026-06-26-epic-09-session-lifecycle-design.md)
**Delivery window:** Entrega 3 — extends Profile Bootstrap with full session lifecycle (named subjects, file tracking, short-term memory, session CRUD)

## Context

Epic 1 (Orchestrator) US-1.1 requires the Orchestrator to load the student profile at session bootstrap with a non-blocking fallback to an empty profile. The Orchestrator implementation (verified 2026-06-18) does not load profiles — `student_profile` is always initialized to `None`. The Support Agent (Epic 6) already implements `get_student_summary` and the SQLite profile layer, but the Orchestrator never calls it.

Beyond profile loading, the system lacks a proper study session model:
- **Sessions are anonymous**: a UUID is generated but sessions have no name, description, or user-facing identity.
- **Files are ephemeral**: uploaded files are processed and deleted — no metadata is persisted, no file list exists.
- **No short-term memory**: the agent doesn't remember previous interactions within a session. LangGraph checkpointing exists but isn't leveraged.
- **No session context**: the agent has no tools to query what files are loaded, what topics have been covered, or what progress exists in the current session.

This epic addresses all four gaps: sessions become named "subjects" (materias), files are tracked in the DB, the agent gains short-term memory via LangGraph checkpoints and session-query tools, and the frontend gains session management UI (sidebar, file list, session switching).

## Scope

**In scope**
- Orchestrator loads student profile via the Support Agent's `get_student_summary` tool at session bootstrap
- Non-blocking fallback: profile-load failure does not abort the session; an empty profile is used instead
- Profile data flows into `OrchestratorState.student_profile` for downstream nodes
- `classify_intent` and `synthesize_response` use profile context for personalization (when available)
- Error is logged and surfaced in observability (Langfuse) when fallback fires
- Session restore: existing profile is reloaded when a known `session_id` is provided
- **NEW** Sessions as named entities: `sessions` table gains `name` and `description` columns
- **NEW** Session CRUD API: create, list, get, delete sessions
- **NEW** File metadata persistence: `ingested_documents` table populated on upload
- **NEW** File listing endpoint per session
- **NEW** Short-term memory: LangGraph checkpoint history injected into `classify_intent` prompt
- **NEW** Session context node: `load_session_context` loads files + progress for the current session
- **NEW** Agent tools: `list_session_files`, `get_session_progress`
- **NEW** Per-session profile endpoint: topic scores and progress filtered by session_id
- **NEW** Frontend session sidebar: create, switch, delete sessions
- **NEW** Frontend file list: shows uploaded files per active session

**Out of scope**
- Profile creation and updates (Epic 6 US-6.1, US-6.3)
- Score tracking and weak-topic computation (Epic 6 US-6.2, US-6.4)
- Dashboard data endpoint (Epic 6 US-6.8)
- UI rendering of profile data (Epic 7)
- Raw file persistence on disk (metadata only)
- Multi-user authentication
- Real-time collaboration

## Functional Requirements

- **PROF-01** The Orchestrator MUST attempt to load the student profile via `get_student_summary` at session bootstrap, before intent classification.
- **PROF-02** If the profile load fails (database error, missing student, tool exception), the Orchestrator MUST fall back to an empty profile (`{}`) and continue processing.
- **PROF-03** The fallback MUST be logged at WARNING level with the exception detail, and the event MUST be visible in Langfuse traces.
- **PROF-04** The loaded profile MUST be available in `OrchestratorState.student_profile` for all downstream nodes.
- **PROF-05** `classify_intent` MUST include relevant profile context (weak topics, preferences) in the classification prompt when a profile is present.
- **PROF-06** `synthesize_response` MUST use profile context to personalize the response tone and content when a profile is present.
- **PROF-07** The profile load MUST NOT add more than 200ms to the session bootstrap latency (p95).

### Session Model (new)

- **SESS-01** The `sessions` table MUST support `name` (required, non-empty) and `description` (optional) columns.
- **SESS-02** `POST /api/sessions` MUST create a new session with a generated UUID, name, and optional description, linked to the requesting student.
- **SESS-03** `GET /api/sessions` MUST return all sessions for a given `student_id`, ordered by most recent first.
- **SESS-04** `GET /api/sessions/{id}` MUST return session details including file count and progress summary.
- **SESS-05** `DELETE /api/sessions/{id}` MUST remove the session and cascade-delete associated ingested_documents rows.

### File Persistence (new)

- **FILE-01** On successful ingest, the endpoint MUST insert a row into `ingested_documents` with id, file_name, classification, topics_json, chunks_count, session_id, and ingested_at.
- **FILE-02** `GET /api/sessions/{id}/files` MUST return all files ingested for that session, ordered by most recent first.
- **FILE-03** File metadata persistence MUST NOT add more than 50ms to the ingest endpoint latency (p95).

### Short-Term Memory (new)

- **MEM-01** `classify_intent` MUST receive the last N conversation messages (default N=10) from the LangGraph checkpoint state when available.
- **MEM-02** On the first message of a session (empty checkpoint), `classify_intent` MUST work without conversation history.
- **MEM-03** `synthesize_response` MUST include conversation history in its prompt for contextual continuity.
- **MEM-04** The `OrchestratorState` MUST include a `messages_history` field populated from the checkpoint.

### Session Context Tools (new)

- **CTX-01** `load_session_context` node MUST run after `load_profile` and before `classify_intent`.
- **CTX-02** The node MUST populate `session_context` with files uploaded and topic progress for the current session.
- **CTX-03** `list_session_files` tool MUST be callable by the agent and return the file list for the current session.
- **CTX-04** `get_session_progress` tool MUST be callable by the agent and return topic scores and weak topics filtered by session_id.

### Per-Session Profile (new)

- **PSP-01** `GET /api/sessions/{id}/profile` MUST return topic scores, weak topics, exam count, and average score filtered to that session.
- **PSP-02** The endpoint MUST return 404 if the session does not exist.

### Frontend (new)

- **UI-01** The chat page MUST show a collapsible sidebar listing all user sessions with create/switch/delete actions.
- **UI-02** The chat page MUST show a file list with all files uploaded to the active session.
- **UI-03** Switching sessions MUST reload the chat context (messages, files) for the new session.
- **UI-04** Creating a session MUST prompt for a name (required) and optional description via a modal.
- **UI-05** The active session MUST be persisted in localStorage and restored on page reload.

## Non-Functional Requirements

- **PROF-NFR-01** Profile load is a read-only operation — no side effects on the database.
- **PROF-NFR-02** Profile load failure MUST NOT propagate to the user as an error; the user experience is unchanged.
- **PROF-NFR-03** The empty-profile fallback path MUST be covered by a unit test with a mocked tool failure.
- **SESS-NFR-01** Session CRUD operations MUST complete within 200ms (p95).
- **FILE-NFR-01** File metadata persistence MUST NOT add more than 50ms to ingest latency (p95).
- **MEM-NFR-01** Conversation history extraction from LangGraph checkpoint MUST NOT add more than 100ms to chat latency (p95).

## Test Coverage

- Unit test: profile loads successfully → `student_profile` is populated in state.
- Unit test: profile load fails (tool raises) → `student_profile` is `{}`, session continues, WARNING logged.
- Unit test: profile is `None` (new student) → `student_profile` is `{}`, session continues.
- Unit test: `classify_intent` prompt includes profile context when profile is present.
- Unit test: `classify_intent` prompt does not include profile context when profile is empty.
- Unit test: `synthesize_response` personalizes output when profile is present.
- Integration test: two sessions with the same student_id load the same profile from SQLite.
- **NEW** Unit test: session CRUD (create, list, get, delete) — all operations
- **NEW** Unit test: file metadata persisted on ingest, retrievable via API
- **NEW** Unit test: `load_session_context` populates session_context correctly
- **NEW** Unit test: `classify_intent` receives messages_history from checkpoint
- **NEW** Unit test: `classify_intent` works without history on first message
- **NEW** Unit test: per-session profile returns topic scores filtered by session_id
- **NEW** Integration test: full session lifecycle (create → upload → chat → profile)

## User Stories

### US-9.1: Load profile at session start
- **As a** student starting a session
- **I want** the Orchestrator to load my profile before processing my request
- **So that** my exam generation and responses are personalized from the first message
- **Acceptance criteria:**
  - `load_profile` node runs before `classify_intent`
  - Profile is fetched via `get_student_summary` tool
  - `student_profile` in state is populated with the returned data
- **Dependencies:** Epic 6 US-6.5 (`get_student_summary` tool)
- **Maps to:** Epic 1 US-1.1, §4.4 long-term memory

### US-9.2: Graceful fallback on profile failure
- **As a** student whose profile data is corrupted or unavailable
- **I want** the system to continue working with an empty profile
- **So that** I can still use the tutor without being blocked by a database issue
- **Acceptance criteria:**
  - Any exception from `get_student_summary` is caught
  - `student_profile` is set to `{}`
  - A WARNING is logged with the exception message
  - The session proceeds normally (intent classification, routing, execution)
  - No error is returned to the user
- **Dependencies:** US-9.1
- **Maps to:** Epic 1 US-1.1 (non-blocking fallback), RNF-02

### US-9.3: Personalized intent classification
- **As a** returning student with weak topics in my profile
- **I want** the Orchestrator to consider my profile when classifying my intent
- **So that** "quiz me" is understood in the context of what I need to study
- **Acceptance criteria:**
  - When `student_profile` is non-empty, `classify_intent` includes weak topics and preferences in the LLM prompt
  - When `student_profile` is empty, the prompt works without profile context
  - Classification accuracy is not degraded by the absence of a profile
- **Dependencies:** US-9.1, Epic 1 US-1.2
- **Maps to:** §4.1 Orchestrator intent detection, §5 all flows

### US-9.4: Personalized response synthesis
- **As a** returning student
- **I want** responses to reference my progress and weak areas
- **So that** the interaction feels tailored to my situation
- **Acceptance criteria:**
  - When `student_profile` is non-empty, `synthesize_response` includes profile context in the synthesis prompt
  - When `student_profile` is empty, synthesis works without profile context
- **Dependencies:** US-9.1, Epic 1 US-1.3
- **Maps to:** §4.2 Orchestrator routing, §5 all flows

### US-9.5: Student identity resolution
- **As a** developer
- **I want** a clear mapping between `session_id` and `student_id`
- **So that** the profile load knows which student to query
- **Acceptance criteria:**
  - Either `student_id` is added to `ChatRequest`, or a deterministic derivation from `session_id` is documented
  - The mapping is consistent across sessions
  - New students get a created profile on first evaluation (delegated to Epic 6)
- **Dependencies:** Epic 6 US-6.1
- **Maps to:** §4.4 long-term memory, RNF-02

### US-9.6: Session model — DB migration + CRUD API
- **As a** student with multiple subjects to study
- **I want** to create named study sessions for each subject
- **So that** each subject has its own RAG material, progress, and context
- **Acceptance criteria:**
  - `sessions` table has `name` (NOT NULL) and `description` columns
  - `ingested_documents` has `session_id` FK
  - `POST /api/sessions` creates session returning `{id, name, description, created_at}`
  - `GET /api/sessions?student_id=X` lists sessions ordered by most recent
  - `GET /api/sessions/{id}` returns session detail + file count + progress summary
  - `DELETE /api/sessions/{id}` cascades to ingested_documents
- **Dependencies:** schema.py, router.py
- **Maps to:** SESS-01 through SESS-05

### US-9.7: File metadata persistence
- **As a** student who uploaded study material
- **I want** to see a list of files I've uploaded to this session
- **So that** I know what material is available for the tutor
- **Acceptance criteria:**
  - `/api/ingest` inserts row into `ingested_documents` after successful processing
  - `GET /api/sessions/{id}/files` returns `[{id, file_name, classification, topics, chunks_count, ingested_at}]`
  - Files from temp directory are NOT deleted before metadata is persisted
  - Metadata persistence does not add significant latency (p95 < 50ms)
- **Dependencies:** US-9.6
- **Maps to:** FILE-01 through FILE-03

### US-9.8: Short-term memory via LangGraph checkpoint
- **As a** student chatting with the tutor
- **I want** the tutor to remember our recent conversation
- **So that** follow-up questions are understood in context
- **Acceptance criteria:**
  - `classify_intent` receives last N messages (default 10) from LangGraph checkpoint
  - First message in a session works without history (empty list)
  - `synthesize_response` includes conversation context in prompt
  - `OrchestratorState` includes `messages_history: list[dict]`
  - `orchestrate_chat` extracts history from checkpoint and passes to initial state
- **Dependencies:** US-9.1 (orchestrator graph compiled with checkpointer)
- **Maps to:** MEM-01 through MEM-04

### US-9.9: Session context for the agent
- **As a** tutor agent
- **I want** to know what files are loaded and what topics have been covered in this session
- **So that** I can give relevant, contextual responses
- **Acceptance criteria:**
  - `load_session_context` node runs after `load_profile`, before `classify_intent`
  - Node populates `session_context` dict with `files: [...]` and `progress: {...}`
  - `list_session_files` tool is callable and returns current session file list
  - `get_session_progress` tool returns topic scores + weak topics filtered by session_id
  - `classify_intent` prompt includes session context when available
  - `synthesize_response` prompt includes session context when available
- **Dependencies:** US-9.7, US-9.8
- **Maps to:** CTX-01 through CTX-04

### US-9.10: Per-session profile endpoint
- **As a** frontend component
- **I want** to fetch progress data for a specific session
- **So that** I can show per-subject statistics
- **Acceptance criteria:**
  - `GET /api/sessions/{id}/profile` returns topic scores, weak topics, exam count, average score
  - Data is filtered by `session_id` (not aggregated across all sessions)
  - Returns 404 for non-existent session
- **Dependencies:** US-9.6
- **Maps to:** PSP-01, PSP-02

### US-9.11: Session sidebar (frontend)
- **As a** student using the app
- **I want** a sidebar showing all my study sessions
- **So that** I can switch between subjects easily
- **Acceptance criteria:**
  - `SessionSidebar` component renders list of sessions with name, file count, progress indicator
  - "New session" button opens modal with name (required) + description (optional) fields
  - Clicking a session switches to it: updates active indicator, reloads chat + files
  - Active session is highlighted with accent color
  - Delete button removes session with confirmation
  - `useStudySession` hook replaces `useSession` — manages sessions array + active session
  - Active session id persisted in `localStorage`
- **Dependencies:** US-9.6 (API)
- **Maps to:** UI-01 through UI-05

### US-9.12: File list in session (frontend)
- **As a** student in an active session
- **I want** to see what files I've uploaded to this subject
- **So that** I can verify the tutor has my material
- **Acceptance criteria:**
  - `SessionFileList` component shows files for the active session (fetched from API)
  - Each file shows: name, classification badge, topic tags, upload date
  - File list refreshes when switching sessions
  - File list updates after successful upload (optimistic + refetch)
  - Empty state: "No files uploaded yet" message
- **Dependencies:** US-9.7 (API), US-9.11 (active session)
- **Maps to:** UI-02, UI-03

### US-9.13: Integration and regression tests
- **As a** developer merging this epic
- **I want** comprehensive test coverage for all new functionality
- **So that** session lifecycle works end-to-end with zero regressions
- **Acceptance criteria:**
  - Unit tests for all new schema functions (session CRUD, file queries, per-session profile)
  - Unit tests for new orchestrator nodes (load_profile, load_session_context)
  - Unit tests for new tools (list_session_files, get_session_progress)
  - Integration test: create session → upload file → chat with memory → verify profile
  - All existing tests pass (no regressions)
  - `tests_documentation.md` updated with new test cases
  - `ruff` format/lint clean
- **Dependencies:** US-9.6 through US-9.12
- **Maps to:** All requirements

## Technical Notes

### Three-Layer Memory Architecture

```
Layer 1 — Short-term (LangGraph checkpoint):
  • Last N interactions via AsyncSqliteSaver
  • Thread per session (thread_id = session_id)
  • Already EXISTS via get_orchestrator_graph()

Layer 2 — Session (SQLite + ChromaDB):
  • Uploaded files (ingested_documents)
  • Topic progress (topic_scores filtered by session_id)
  • Accessed via AGENT TOOLS

Layer 3 — Long-term (SQLite students):
  • Global profile: preferences, aggregate stats
  • Loaded at session bootstrap (load_profile node)
```

### Orchestrator Graph (updated)

```
START → load_profile → load_session_context → classify_intent → [route] → ... → synthesize_response → END
```

### State Schema Additions

```python
class OrchestratorState(TypedDict):
    # ... existing: session_id, user_message, intent, confidence, plan,
    #     current_step, results, errors, response, status, iteration_count
    student_profile: dict | None       # populated by load_profile
    session_context: dict | None       # NEW: {files: [...], progress: {...}}
    messages_history: list[dict]       # NEW: last N messages from checkpoint
```

### Files Affected

See [design doc](../docs/superpowers/specs/2026-06-26-epic-09-session-lifecycle-design.md) for complete file list and implementation order.
