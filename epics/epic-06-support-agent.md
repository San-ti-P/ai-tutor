# Epic 6: Support Agent

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §3.2, §4.2, §4.4, §5.3, §6.1, §8)
**Delivery window:** Entrega 2 partial (profile schema) → Entrega 3 (full adaptation)

## Context

The Support Agent owns the student's long-term profile: history, per-topic scores, preferences, weak areas. It exposes two tools (`update_student_profile`, `get_student_summary`) that the other agents call to personalize their work. It also drives the dashboard data and the adaptive study plan for the next session.

## Scope

**In scope**
- Persisting the user profile in SQLite (scores, history, preferences, weak topics)
- Tracking scores by topic across sessions
- Updating the profile after each evaluation
- Generating study focus recommendations
- Exposing `get_student_summary` to other agents
- Managing user preferences (RF-12)
- Supporting the "second session prioritizes weak topics" behavior (test case 4)

**Out of scope**
- Generating study material (Epics 3, 4)
- Evaluating answers (Epic 5)
- Building the knowledge base (Epic 2)
- Rendering the dashboard UI (Epic 7)

## Functional Requirements

- **SUP-01** Persist the student profile across sessions in SQLite.
- **SUP-02** Track per-topic scores across sessions.
- **SUP-03** Update the profile after each evaluation with the per-topic scores.
- **SUP-04** Generate study focus recommendations (topics with score < 6) for the next session.
- **SUP-05** Expose `get_student_summary` for personalization.
- **SUP-06** Manage user preferences: question types, difficulty, count, topic include/exclude.

## Non-Functional Requirements

- **SUP-NFR-01** No external database server; SQLite is sufficient (source PRD §4.5).
- **SUP-NFR-02** Profile reads must be fast enough to not block the agent loop.

## Technical Notes

- Tools: `update_student_profile`, `get_student_summary` (source PRD §3.2).
- Storage: SQLite / JSON persistent (source PRD §4.4 long-term memory).
- Reactive loop (source PRD §4.2).

## Test Coverage

- Source PRD §8 case 4 (second session prioritizes weak topics) covers this epic.

## User Stories

### US-6.1: Persist user profile
- **As a** student
- **I want** my profile (identity, history, scores, preferences) to persist between sessions
- **So that** I don't lose progress across logins
- **Acceptance criteria:**
  - Profile data is stored in SQLite
  - A new session can be created for a new user
  - An existing user is recognized on return
- **Dependencies:** —
- **Maps to:** RF-11, §4.4 long-term memory

### US-6.2: Track per-topic scores
- **As a** Support Agent
- **I want** to record scores broken down by topic
- **So that** weak topics can be identified precisely
- **Acceptance criteria:**
  - Each evaluation produces a {topic, score} pair
  - The profile maintains a running history of scores per topic
  - The current score per topic is queryable
- **Dependencies:** US-6.1, Epic 5 US-5.7
- **Maps to:** RF-11, §5.3 step 5

### US-6.3: Update profile after evaluation
- **As a** system
- **I want** the profile to be updated automatically after each evaluation
- **So that** the data is always fresh
- **Acceptance criteria:**
  - `update_student_profile` is called with the per-topic scores
  - The update is atomic
  - The timestamp of the last update is recorded
- **Dependencies:** US-6.2, Epic 5 US-5.7
- **Maps to:** §5.3 step 5

### US-6.4: Generate study focus recommendations
- **As a** student
- **I want** the system to recommend what to study next
- **So that** I focus on what I am weakest at
- **Acceptance criteria:**
  - Topics with score < 6 are flagged as weak
  - The recommendation list is sorted by score ascending
  - At least the top 3 weak topics are exposed
- **Dependencies:** US-6.2
- **Maps to:** §4.2 Support Agent "genera recomendaciones de foco de estudio"

### US-6.5: Expose get_student_summary
- **As a** any agent
- **I want** to call `get_student_summary` to get the full profile
- **So that** I can personalize my behavior
- **Acceptance criteria:**
  - Tool returns: identity, current scores per topic, preferences, weak topics, session history
  - Tool is read-only and side-effect free
- **Dependencies:** US-6.1
- **Maps to:** §3.2 `get_student_summary`, §5.2 step 2

### US-6.6: Manage user preferences
- **As a** student
- **I want** to configure my exam preferences
- **So that** the generated exams match my course format
- **Acceptance criteria:**
  - User can set: question types, difficulty, count per exam, topic include/exclude
  - Preferences are stored in the profile
  - Preferences are read by the ExamGenerator via `get_student_summary`
- **Dependencies:** US-6.1
- **Maps to:** RF-12

### US-6.7: Multi-session adaptation
- **As a** student who studied yesterday
- **I want** my next exam to focus on yesterday's weak topics
- **So that** I get targeted practice
- **Acceptance criteria:**
  - The second session's exam contains a higher proportion of weak topics (test case 4)
  - The behavior is deterministic for a given profile
- **Dependencies:** US-6.2, US-6.4, Epic 3 US-3.5
- **Maps to:** §8 case 4

### US-6.8: Dashboard data
- **As a** UI (Epic 7)
- **I want** an endpoint or query that returns the data needed for the dashboard
- **So that** the UI can render progress charts
- **Acceptance criteria:**
  - Returns: per-topic score history, total sessions, weak topics, preferences
  - Latency p95 < 300ms
- **Dependencies:** US-6.2
- **Maps to:** RF-13 (data side), §4.2 "alimenta el dashboard de progreso"
