# Epic 7: User Interface

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.5, §5, §6.1 RF-13, §10.1)
**Delivery window:** Entrega 2 (skeleton) → Entrega 3 (full)

## Context

The UI is the student's surface. It hosts the chat, the file upload, the exam renderer, the evaluation feedback, and the progress dashboard. It is built on Next.js + React + Tailwind per the source PRD. It talks to the FastAPI backend that runs the agents. Per the source PRD, the production target is the Next.js web UI; the Streamlit mention is kept only as a historical prototype note.

## Scope

**In scope**
- Chat interface
- File upload component
- Interactive exam renderer
- Evaluation feedback view
- Progress dashboard
- OCR confirmation prompts *(deferred — see US-7.6)*
- LaTeX rendering
- Configuration panel for preferences
- FastAPI endpoints the UI calls (server side; co-located with the backend repo structure)

**Out of scope**
- Agent logic (Epics 1-6)
- Tracing infrastructure (Epic 8)
- Authentication / multi-user management (out of scope per source PRD §2.2)

## Functional Requirements

- **UI-01** Provide a chat interface for natural-language requests to the Orchestrator.
- **UI-02** Allow the user to upload one or more files (PDF, PNG/JPG, TXT) for ingestion.
- **UI-03** Render an exam interactively (MCQ + open-answer) and submit answers.
- **UI-04** Display the evaluation result with per-question breakdown, score, and feedback.
- **UI-05** Display a progress dashboard: per-topic scores, history chart, weak topics.
- **UI-06** *(Deferred)* Show OCR low-confidence confirmation prompts with the extracted LaTeX.
- **UI-07** Render LaTeX and math expressions correctly.
- **UI-08** Provide a configuration panel for user preferences.
- **UI-09** Expose FastAPI endpoints for all of the above.

## Non-Functional Requirements

- **UI-NFR-01** UI must be understandable without documentation for a university student (RNF-04).
- **UI-NFR-02** UI must be responsive on a standard laptop browser.
- **UI-NFR-03** Every backend response should include trace identifiers (RNF-05).
- **UI-NFR-04** LaTeX rendering must be reliable for typical math expressions.

## Technical Notes

- Frontend: Next.js + React + Tailwind (source PRD §4.5).
- Backend: FastAPI (source PRD §4.5) — endpoints live under `/backend` in the repo.
- File upload uses a multipart endpoint, streamed for large PDFs.

## Test Coverage

- End-to-end demo flow per source PRD §11 ("Defensa oral con demostración en vivo") covers this epic.

## User Stories

### US-7.1: Chat interface
- **As a** student
- **I want** a chat UI to send natural-language requests
- **So that** I can drive the Orchestrator
- **Acceptance criteria:**
  - User can type a message and send it
  - The assistant's reply is rendered with markdown (headings, lists, code)
  - The session id is persisted in the client across reloads
- **Dependencies:** Epic 1 US-1.3 (Orchestrator routing)
- **Maps to:** §5.1, §5.2 entry points

### US-7.2: File upload
- **As a** student
- **I want** to upload one or more files (PDF, image, TXT)
- **So that** the material is ingested
- **Acceptance criteria:**
  - Drop zone and file picker both work
  - Multiple files can be uploaded in one batch
  - The UI shows a per-file status (pending, ingesting, done, error)
- **Dependencies:** Epic 2 US-2.1
- **Maps to:** §5.1 step 1

### US-7.3: Interactive exam renderer
- **As a** student
- **I want** to take an exam in the UI
- **So that** I can submit my answers
- **Acceptance criteria:**
  - MCQ questions render with radio buttons
  - Open questions render with a textarea
  - The user can navigate between questions
  - The exam is submitted as a single batch
- **Dependencies:** Epic 3 US-3.9
- **Maps to:** §5.2 step 7

### US-7.4: Evaluation feedback view
- **As a** student
- **I want** to see detailed feedback after submitting
- **So that** I can learn from the corrections
- **Acceptance criteria:**
  - Per-question: my answer, the correct answer, the score, the feedback
  - Total score is displayed prominently
  - Suggestions link to the relevant topics
- **Dependencies:** Epic 5 US-5.8
- **Maps to:** §5.3 step 6

### US-7.5: Progress dashboard
- **As a** student
- **I want** to see my progress over time
- **So that** I know whether I am improving
- **Acceptance criteria:**
  - Per-topic score history chart
  - Total sessions and total questions answered
  - List of weak topics with links to start a focused exam
- **Dependencies:** Epic 6 US-6.8
- **Maps to:** RF-13

### US-7.6: OCR confirmation prompts *(DEFERRED)*
- **As a** student
- **I want** to be asked to confirm OCR output when confidence is low
- **So that** bad OCR does not pollute my knowledge base
- **Acceptance criteria:**
  - The extracted LaTeX is shown in an editable field
  - The user can confirm, edit, or reject
  - Only confirmed content is added to the index
- **Dependencies:** Epic 2 US-2.4, deferred to post-MVP
- **Maps to:** §5.1 step 5, §8 case 9

### US-7.7: LaTeX rendering
- **As a** student
- **I want** math expressions to render correctly in the UI
- **So that** I can read them
- **Acceptance criteria:**
  - Inline and block LaTeX render reliably
  - Common math notation (fractions, sums, integrals) is supported
- **Dependencies:** —
- **Maps to:** §4.5 ("preserva tablas y listas" + math)

### US-7.8: Configuration panel
- **As a** student
- **I want** to set my exam preferences in the UI
- **So that** generated exams match my course format
- **Acceptance criteria:**
  - Form for: question types, difficulty, count, topic include/exclude
  - Changes are saved to the profile via the Support Agent
- **Dependencies:** Epic 6 US-6.6
- **Maps to:** RF-12

### US-7.9: FastAPI endpoints
- **As a** frontend
- **I want** typed HTTP endpoints for every UI action
- **So that** the frontend and the agent backend stay decoupled
- **Acceptance criteria:**
  - Endpoints: chat, upload, exam request/submit, exercise request/submit, profile, dashboard
  - OpenAPI schema is generated and published
  - All endpoints return trace identifiers
- **Dependencies:** All agent epics
- **Maps to:** §4.5 FastAPI, §10.1 `/backend`
