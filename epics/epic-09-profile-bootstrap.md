# Epic 9: Profile Bootstrap

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.4, RNF-02)
**Delivery window:** Post-MVP — addresses the profile-load gap surfaced in Epic 1 verification

## Context

Epic 1 (Orchestrator) US-1.1 requires the Orchestrator to load the student profile at session bootstrap with a non-blocking fallback to an empty profile. The Orchestrator implementation (verified 2026-06-18) does not load profiles — `student_profile` is always initialized to `None`. The Support Agent (Epic 6) already implements `get_student_summary` and the SQLite profile layer, but the Orchestrator never calls it. This epic closes that integration gap: the Orchestrator must fetch the profile at session start, make it available to all downstream nodes for personalization, and degrade gracefully when the profile is missing or the database is unreachable.

## Scope

**In scope**
- Orchestrator loads student profile via the Support Agent's `get_student_summary` tool at session bootstrap
- Non-blocking fallback: profile-load failure does not abort the session; an empty profile is used instead
- Profile data flows into `OrchestratorState.student_profile` for downstream nodes
- `classify_intent` and `synthesize_response` use profile context for personalization (when available)
- Error is logged and surfaced in observability (Langfuse) when fallback fires
- Session restore: existing profile is reloaded when a known `session_id` is provided

**Out of scope**
- Profile creation and updates (Epic 6 US-6.1, US-6.3)
- Score tracking and weak-topic computation (Epic 6 US-6.2, US-6.4)
- Dashboard data endpoint (Epic 6 US-6.8)
- UI rendering of profile data (Epic 7)

## Functional Requirements

- **PROF-01** The Orchestrator MUST attempt to load the student profile via `get_student_summary` at session bootstrap, before intent classification.
- **PROF-02** If the profile load fails (database error, missing student, tool exception), the Orchestrator MUST fall back to an empty profile (`{}`) and continue processing.
- **PROF-03** The fallback MUST be logged at WARNING level with the exception detail, and the event MUST be visible in Langfuse traces.
- **PROF-04** The loaded profile MUST be available in `OrchestratorState.student_profile` for all downstream nodes.
- **PROF-05** `classify_intent` MUST include relevant profile context (weak topics, preferences) in the classification prompt when a profile is present.
- **PROF-06** `synthesize_response` MUST use profile context to personalize the response tone and content when a profile is present.
- **PROF-07** The profile load MUST NOT add more than 200ms to the session bootstrap latency (p95).

## Non-Functional Requirements

- **PROF-NFR-01** Profile load is a read-only operation — no side effects on the database.
- **PROF-NFR-02** Profile load failure MUST NOT propagate to the user as an error; the user experience is unchanged.
- **PROF-NFR-03** The empty-profile fallback path MUST be covered by a unit test with a mocked tool failure.

## Technical Notes

- The profile load happens in a new node (`load_profile`) that runs BEFORE `classify_intent` in the Orchestrator graph.
- Graph topology change: `START → load_profile → classify_intent → ...` (insert one node at the front).
- `load_profile` calls the existing `get_student_summary` tool from `src/tools/`.
- The `student_id` must be derived from the `session_id` or passed explicitly in the chat request. Current `ChatRequest` has `session_id` but no `student_id` — this epic MUST resolve how the student identity is obtained (likely: `student_id` is extracted from `session_id` or added to the request schema).
- The profile is stored in `OrchestratorState.student_profile` as a dict (already in the schema).

## Test Coverage

- Unit test: profile loads successfully → `student_profile` is populated in state.
- Unit test: profile load fails (tool raises) → `student_profile` is `{}`, session continues, WARNING logged.
- Unit test: profile is `None` (new student) → `student_profile` is `{}`, session continues.
- Unit test: `classify_intent` prompt includes profile context when profile is present.
- Unit test: `classify_intent` prompt does not include profile context when profile is empty.
- Unit test: `synthesize_response` personalizes output when profile is present.
- Integration test: two sessions with the same student_id load the same profile from SQLite.

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
