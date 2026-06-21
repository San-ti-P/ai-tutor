# Epic 4: ExerciseGenerator Agent

**Status:** Done
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §3.2, §4.2, §6.1, §7)
**Delivery window:** Entrega 3 (per source PRD §10)

## Context

The ExerciseGenerator produces complex practical exercises (problem-solving) that go beyond the recall style of exams. It can seed the exercise data with values grounded in the ingested material (e.g., a physics problem that uses a real formula from the notes).

## Scope

**In scope**
- Generating complex practical exercises
- Grounding exercise data in the ingested material
- Anti-hallucination guardrail (same as ExamGenerator)
- Providing a model solution for the student to compare against

**Out of scope**
- MCQ or open-answer exam questions (Epic 3)
- Evaluating student answers (Epic 5)
- Knowledge base construction (Epic 2)

## Functional Requirements

- **EX-01** Generate practical exercises of higher complexity than exam questions.
- **EX-02** Ground exercise data in chunks from the material; the agent must not invent facts.
- **EX-03** Provide a model solution with the exercise.

## Non-Functional Requirements

- **EX-NFR-01** Anti-hallucination guardrail applies (same as ExamGenerator).
- **EX-NFR-02** All generations must be traced (Epic 8).

## Technical Notes

- Tool: `generate_exercise` (source PRD §3.2).
- ReAct + Tools loop per source PRD §4.2.

## Test Coverage

- Reuses the anti-hallucination test patterns from Epic 3 (no dedicated test cases in §8).

## User Stories

### US-4.1: Generate complex practical exercise
- **As a** student
- **I want** to receive a practical exercise (problem to solve) on a given topic
- **So that** I can practice applying the material, not just recalling it
- **Acceptance criteria:**
  - Exercise has a clear statement, given data, and a question to answer
  - The exercise is more complex than a typical exam question (multi-step, requires reasoning)
- **Dependencies:** Epic 2 US-2.9
- **Maps to:** RF-09, §3.2 `generate_exercise`

### US-4.2: Ground exercise data in material
- **As a** student
- **I want** the exercise's data and context to come from my notes
- **So that** the practice is relevant to my course
- **Acceptance criteria:**
  - Exercise references real formulas, definitions, or scenarios from the material
  - Source chunks are attached to the exercise output
- **Dependencies:** US-4.1
- **Maps to:** RF-09, §3.2 `generate_exercise`

### US-4.3: Anchor exercise to source chunks (anti-hallucination)
- **As a** system operator
- **I want** the same anti-hallucination guardrail as for exams
- **So that** practical exercises are not invented either
- **Acceptance criteria:**
  - Same chunk-anchoring check as Epic 3 US-3.3
  - Failing exercises are regenerated up to 3 times
- **Dependencies:** US-4.2
- **Maps to:** RNF-03

### US-4.4: Provide model solution
- **As a** student
- **I want** a model solution to compare my work against after I attempt the exercise
- **So that** I can self-correct when the Evaluator is not (yet) available for the exercise
- **Acceptance criteria:**
  - Each exercise ships with a model solution
  - The model solution references the source chunks
- **Dependencies:** US-4.1
- **Maps to:** §3.2 `generate_exercise`
