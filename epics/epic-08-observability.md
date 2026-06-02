# Epic 8: Observability and Quality

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §7, §9, §10.3)
**Delivery window:** Entrega 2 (LLM-call logging) → Entrega 3 (full Langfuse)

## Context

Observability is cross-cutting. Every agent and tool call must leave a trace so we can debug, measure, and evaluate. The platform is Langfuse (self-hosted or cloud) per the source PRD. The test suite (12 cases in §8) is also part of this epic because it is how we prove the system meets the requirements.

## Scope

**In scope**
- Langfuse integration setup
- Tracing of LLM calls (model, prompt, response, tokens, latency)
- Tracing of tool calls (name, input, output, status, time)
- Tracing of RAG retrievals (query, top-K, scores, chunks selected)
- Tracing of evaluation spans
- Aggregate metrics (tokens, cost, latency, tool success rate, average scores per topic)
- Test suite of 12 cases from the source PRD §8

**Out of scope**
- Agent logic (Epics 1-6)
- UI (Epic 7)

## Functional Requirements

- **OBS-01** Integrate Langfuse as the trace backend.
- **OBS-02** Emit a root span per execution with session id, user id, task type.
- **OBS-03** Emit an LLM Call span for every LLM invocation with model, prompt, response, tokens, latency, cost.
- **OBS-04** Emit a Tool Call span for every tool invocation with name, input, output, time, status.
- **OBS-05** Emit a RAG Retrieval span with query, top-K, scores, and selected chunks.
- **OBS-06** Emit an Evaluation span with question, base answer, student answer, score, justification.
- **OBS-07** Aggregate per-execution metrics: total steps, tokens, cost, latency, tool success rate, average scores per topic.

## Non-Functional Requirements

- **OBS-NFR-01** Tracing must not block the agent loop (async emission).
- **OBS-NFR-02** Traces must include the full RNF-05 trail per response.

## Technical Notes

- Platform: Langfuse (open-source), self-hosted or cloud.
- LLM-as-judge sampling: integrated in the Evaluator (Epic 5 US-5.5).

## Test Coverage

- All 12 test cases from source PRD §8 are run and documented as part of this epic.

## User Stories

### US-8.1: Langfuse integration
- **As a** developer
- **I want** Langfuse configured and connected to the agent backend
- **So that** traces are recorded
- **Acceptance criteria:**
  - Langfuse is running (self-hosted or cloud) and reachable from the backend
  - A test run produces at least one trace visible in the dashboard
- **Dependencies:** —
- **Maps to:** §9, RF-14

### US-8.2: LLM call traces
- **As a** developer
- **I want** every LLM call to produce a trace with model, prompt, response, tokens, latency
- **So that** I can debug and measure cost
- **Acceptance criteria:**
  - Each LLM call emits a span with all the fields above
  - Input/output tokens are recorded
  - Latency is recorded
- **Dependencies:** US-8.1
- **Maps to:** §9 LLM Call span, RNF-05

### US-8.3: Tool call traces
- **As a** developer
- **I want** every tool call to produce a trace with name, input, output, time, status
- **So that** I can debug tool failures
- **Acceptance criteria:**
  - Each tool invocation emits a span with all the fields above
  - Status (success / error) is recorded
- **Dependencies:** US-8.1
- **Maps to:** §9 Tool Call span

### US-8.4: RAG retrieval traces
- **As a** developer
- **I want** every RAG retrieval to produce a trace with query, top-K, scores, selected chunks
- **So that** I can verify grounding
- **Acceptance criteria:**
  - Each `retrieve_chunks` call emits a span with the query, top-K results, scores, and selected chunk ids
- **Dependencies:** US-8.1, Epic 2 US-2.9
- **Maps to:** §9 RAG Retrieval span

### US-8.5: Evaluation spans
- **As a** developer
- **I want** every evaluation to produce a trace with question, base answer, student answer, score, justification
- **So that** grading decisions are auditable
- **Acceptance criteria:**
  - Each evaluation emits an Evaluation span with all the fields above
- **Dependencies:** US-8.1, Epic 5 US-5.1
- **Maps to:** §9 Evaluation span

### US-8.6: Aggregate metrics
- **As a** project owner
- **I want** to see aggregate metrics per execution: total steps, tokens, cost, latency, tool success rate, average scores per topic
- **So that** I can monitor cost and quality
- **Acceptance criteria:**
  - Metrics are computed per execution and stored
  - A dashboard view shows them
- **Dependencies:** US-8.2 through US-8.5
- **Maps to:** §9 metrics box, RF-14

### US-8.7: Test suite (12 cases)
- **As a** project owner
- **I want** the 12 test cases from §8 to be implemented and runnable
- **So that** we can prove the system meets the requirements
- **Acceptance criteria:**
  - 12 cases implemented in pytest (per source PRD §10.1 /tests)
  - Happy path, edge case, and adversarial categories all covered
  - LLM-as-judge concordance tests are included for subjective criteria
  - The suite is run as part of the CI / pre-coloquio checklist
- **Dependencies:** Epics 1-7
- **Maps to:** §8 full test matrix, §10.3 Entrega 3 "Suite de casos de prueba ejecutada y documentada"
