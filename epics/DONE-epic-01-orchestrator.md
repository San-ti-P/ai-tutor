# Epic 1: Orchestrator Agent

**Status:** Done
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §4.1, §4.2, §5)
**Delivery window:** Foundational — must be in place before any other agent can be invoked (Entrega 2 onwards)

## Context

The Orchestrator is the central coordinator of the multi-agent tutor. Every user request enters through it. It must classify intent, route to the right specialized agent (Ingestor, ExamGenerator, ExerciseGenerator, Evaluator, Support), and — for multi-step tasks — apply the Plan-and-Execute pattern. It also owns session state and the global iteration guardrail that protects against infinite loops.

## Scope

**In scope**
- Intent classification from user messages
- Routing to the appropriate specialized agent
- Plan-and-Execute loop for composite tasks (e.g., "ingest this PDF and quiz me")
- Session state management (load, persist, restore)
- Iteration limit guardrail (15 steps max per task — source PRD §7)
- Parallel read-only tool calls when independent
- Partial result handling when a delegated agent fails

**Out of scope**
- Knowledge base construction (Epic 2)
- Generating study material (Epics 3, 4)
- Answer evaluation (Epic 5)
- Updating user profile (Epic 6)
- UI rendering (Epic 7)
- Tracing infrastructure (Epic 8)

## Functional Requirements

- **ORCH-01** Accept a user message and identify its primary intent.
- **ORCH-02** Route the request to the matching agent and return its response.
- **ORCH-03** For multi-step tasks, plan the sequence of agent invocations, execute them, and aggregate the result.
- **ORCH-04** Persist session state across the conversation; reload on the next turn.
- **ORCH-05** Enforce a hard cap of 15 iterations per task (§7) and return a partial response with an error flag if exceeded.
- **ORCH-06** Allow concurrent invocations of independent read-only tools (e.g., parallel `retrieve_chunks`).
- **ORCH-07** On a delegated agent failure, return a partial response identifying which step failed.

## Non-Functional Requirements

- **ORCH-NFR-01** Routing decision latency: p95 < 500ms for simple intents.
- **ORCH-NFR-02** Session state must survive process restarts (persisted via the memory layer).
- **ORCH-NFR-03** All decisions must be traced (see Epic 8: Observability).
- **ORCH-NFR-04** Code must follow modular architecture (RNF-06) and use .env for all secrets (RNF-07).

## Technical Notes

- Implementation platform: LangGraph state graph (source PRD §4.1).
- ReAct loop delegated to each specialized agent (source PRD §4.2).
- Plan-and-Execute: planner LLM step + executor loop (source PRD §4.1).
- Session state persisted via the Support Agent's memory layer (see Epic 6).
- Intent classes: `ingest | generate_exam | generate_exercise | evaluate | query_profile | general_chat | composite`.

## Test Coverage

- Source PRD §8 case 4 (second session prioritizes weak topics) is verified end-to-end through Orchestrator + Support + ExamGenerator.

## User Stories

### US-1.1: Session bootstrap
- **As a** student starting a study session
- **I want** the Orchestrator to load my profile and create a session id
- **So that** subsequent requests in the session can be personalized and the state survives disconnects
- **Acceptance criteria:**
  - First request returns a new `session_id`
  - Subsequent requests with that `session_id` reload existing state
  - Profile load is non-blocking; failure falls back to an empty profile
- **Dependencies:** Epic 6 US-6.1 (Support Agent profile persistence)
- **Maps to:** §4.4 long-term memory, RNF-02 (cross-session continuity)

### US-1.2: Intent classifier
- **As a** Orchestrator
- **I want** to classify the user's message into a small fixed set of intents
- **So that** I can route the request to the right specialized agent
- **Acceptance criteria:**
  - Classifier outputs one of {`ingest`, `generate_exam`, `generate_exercise`, `evaluate`, `query_profile`, `general_chat`, `composite`}
  - F1 ≥ 0.85 on a labeled validation set of at least 50 representative messages
  - Confidence below threshold falls back to `general_chat` (asks the user for clarification)
- **Dependencies:** —
- **Maps to:** §4.1 Orchestrator intent detection, §5 (all flows enter here)

### US-1.3: Agent router
- **As a** Orchestrator
- **I want** to dispatch a classified request to the matching agent
- **So that** each request is handled by the most qualified agent
- **Acceptance criteria:**
  - Each intent is dispatched to its dedicated agent (or to the relevant tool directly)
  - The agent's response is returned to the caller
  - A `general_chat` intent is answered inline by the Orchestrator
- **Dependencies:** US-1.2; Epic 2 (Ingestor), Epic 3 (ExamGenerator), Epic 4 (ExerciseGenerator), Epic 5 (Evaluator), Epic 6 (Support)
- **Maps to:** §4.2 Orchestrator "decide qué agente activar en función de la intención del usuario"

### US-1.4: Plan-and-Execute for composite tasks
- **As a** student
- **I want** to give a multi-step request (e.g., "ingest this PDF and quiz me")
- **So that** the system plans the steps and executes them in order
- **Acceptance criteria:**
  - A composite intent produces a plan with explicit steps
  - Each step is executed and the result is aggregated
  - The plan and per-step results are persisted in the session
- **Dependencies:** US-1.3
- **Maps to:** §4.1 Plan-and-Execute box, §5 all flows

### US-1.5: Iteration guardrail
- **As a** system operator
- **I want** the Orchestrator to enforce a hard cap of 15 iterations per task
- **So that** a runaway agent loop cannot hang the system
- **Acceptance criteria:**
  - After 15 tool calls, the loop exits with `status="incomplete"`
  - The partial response identifies which step was the last to complete
  - The cap is configurable via .env
- **Dependencies:** —
- **Maps to:** §7 "Loop infinito del agente" guardrail

### US-1.6: Parallel read tool calls
- **As a** Orchestrator
- **I want** to invoke independent read-only tools in parallel within a turn
- **So that** multi-topic retrievals don't serialize needlessly
- **Acceptance criteria:**
  - A plan with N independent `retrieve_chunks` calls completes in roughly 1/N of the serial time
  - The outputs are merged by topic before the next step
- **Dependencies:** Epic 2 US-2.9 (retrieve_chunks tool)
- **Maps to:** §3.2 `retrieve_chunks (Todos)`

### US-1.7: Partial result on agent failure
- **As a** student
- **I want** a partial response when one step of a composite task fails
- **So that** I still get value from the parts that did succeed
- **Acceptance criteria:**
  - A failure in step k returns the results of steps 1..k-1 plus a structured error for step k
  - The session state reflects the partial completion
- **Dependencies:** US-1.3
- **Maps to:** §7 "Loop infinito del agente" (return partial response with flag)

### US-1.8: Modular code & secrets hygiene
- **As a** developer
- **I want** the Orchestrator module to be self-contained, typed, and use .env for secrets
- **So that** the codebase stays maintainable and no key leaks in commits
- **Acceptance criteria:**
  - All secrets read from environment variables
  - No hardcoded API keys
  - Module exposes a single public entry point
- **Dependencies:** —
- **Maps to:** RNF-06, RNF-07
