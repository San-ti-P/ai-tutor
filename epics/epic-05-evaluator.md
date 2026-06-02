# Epic 5: Evaluator Agent

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §3.2, §4.2, §5.3, §6.1, §7, §8)
**Delivery window:** Entrega 3 (per source PRD §10)

## Context

The Evaluator grades student responses. It is invoked after a student submits answers (text or image) to an exam or exercise. It uses the question, the base answer, and the relevant RAG chunks to produce a score (0-10), a justification, and concrete feedback. It also feeds scores back to the Support Agent for profile update.

## Scope

**In scope**
- Scoring free-text answers on a 0-10 scale with reasoning
- Identifying conceptual errors in the student's answer
- Providing actionable improvement suggestions
- Cross-referencing student answer with base answer and RAG chunks
- LLM-as-judge self-validation (sampling)
- Handling non-evaluable answers (other language, gibberish, off-topic) without crashing
- Sending per-topic scores to the Support Agent for profile update

**Out of scope**
- Generating the exam questions (Epic 3)
- OCR of answer images (Epic 2)
- Updating the user profile (Epic 6)
- Rendering the feedback UI (Epic 7)

## Functional Requirements

- **EVAL-01** Score each free-text answer on a 0-10 scale with a written justification.
- **EVAL-02** Identify the conceptual errors in the student's answer.
- **EVAL-03** Provide concrete suggestions for review (specific topics or chunks to revisit).
- **EVAL-04** Use the question, the base answer, and the relevant RAG chunks as the grading context.
- **EVAL-05** Trigger an LLM-as-judge second pass on a sample of evaluations to catch inconsistencies.
- **EVAL-06** If an answer is non-evaluable (other language, gibberish, off-topic), return a structured "cannot evaluate" response instead of erroring.

## Non-Functional Requirements

- **EVAL-NFR-01** Anti-hallucination: feedback must reference real chunks, not invented content.
- **EVAL-NFR-02** All evaluations must be traced (Epic 8).
- **EVAL-NFR-03** The LLM-as-judge pass must run on a configurable sample rate (default 10%).

## Technical Notes

- Tool: `evaluate_answer` (source PRD §3.2).
- Loop: Chain-of-Thought (source PRD §4.2).
- Inputs: {pregunta, respuesta_base, respuesta_estudiante, chunks_relevantes} (source PRD §5.3 step 3).
- Outputs: score (0-10), justificación, errores conceptuales, sugerencias (source PRD §5.3 step 4).

## Test Coverage

- Source PRD §8 cases 3, 8, 12 cover this epic.

## User Stories

### US-5.1: Score free-text answer 0-10
- **As a** student
- **I want** my free-text answer to receive a numeric score (0-10) with a written justification
- **So that** I know how I did and why
- **Acceptance criteria:**
  - Each answer receives a score in [0, 10]
  - A justification explains how the score was derived
  - Scoring is consistent across reruns of the same input (within tolerance)
- **Dependencies:** Epic 3 US-3.6 (base answers)
- **Maps to:** RF-10, §5.3 step 4

### US-5.2: Identify conceptual errors
- **As a** student
- **I want** the feedback to point out the specific conceptual errors I made
- **So that** I know what to study, not just that I was wrong
- **Acceptance criteria:**
  - Each incorrect or partially correct answer lists the specific error(s)
  - Errors reference the relevant concept or chunk
- **Dependencies:** US-5.1
- **Maps to:** RF-10, §5.3 step 4

### US-5.3: Provide improvement suggestions
- **As a** student
- **I want** concrete suggestions on what to review
- **So that** my next study session is targeted
- **Acceptance criteria:**
  - Suggestions include specific topics or chunks to revisit
  - Suggestions are derived from the errors and the RAG context
- **Dependencies:** US-5.2
- **Maps to:** §5.3 step 4

### US-5.4: Cross-reference with RAG chunks
- **As a** system operator
- **I want** the Evaluator to grade against the base answer and the relevant RAG chunks
- **So that** the grading is anchored in the same material that generated the question
- **Acceptance criteria:**
  - The grading context includes: question, base answer, top-K relevant chunks
  - Feedback references these chunks
- **Dependencies:** Epic 2 US-2.9
- **Maps to:** §5.3 step 3

### US-5.5: LLM-as-judge self-validation
- **As a** system operator
- **I want** a second LLM pass to validate a sample of evaluations
- **So that** inconsistent grading is flagged for review
- **Acceptance criteria:**
  - A configurable percentage of evaluations is re-graded by a second LLM
  - Disagreements above threshold are marked `requires_review=true` and surfaced in the UI
- **Dependencies:** US-5.1
- **Maps to:** §7 "Respuestas del evaluador inconsistentes", §8 case 3 (LLM-as-judge concordance)

### US-5.6: Handle non-evaluable answers
- **As a** student
- **I want** a graceful response when my answer is not evaluable (other language, gibberish, off-topic)
- **So that** the system does not crash or return a fake score
- **Acceptance criteria:**
  - Non-evaluable input yields a structured response, not a 500 error
  - The response explains why it could not evaluate and suggests the next step
- **Dependencies:** US-5.1
- **Maps to:** §8 case 12

### US-5.7: Send scores to the Support Agent
- **As a** Support Agent
- **I want** to receive per-topic scores after each evaluation
- **So that** the student profile reflects the latest performance
- **Acceptance criteria:**
  - After each evaluation, the Orchestrator calls `update_student_profile` with the per-topic scores
  - The update is atomic with the evaluation result
- **Dependencies:** Epic 6 US-6.3 (update_student_profile)
- **Maps to:** §5.3 step 5

### US-5.8: End-to-end evaluation flow
- **As a** student
- **I want** to submit answers and see detailed feedback
- **So that** I close the practice loop
- **Acceptance criteria:**
  - Submit → score → feedback → profile update works end-to-end
  - The UI shows the breakdown per question
- **Dependencies:** US-5.1 through US-5.7, Epic 7 US-7.4
- **Maps to:** §5.3 full flow, §8 cases 3 and 8
