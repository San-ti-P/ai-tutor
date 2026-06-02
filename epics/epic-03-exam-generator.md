# Epic 3: ExamGenerator Agent

**Status:** Draft
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §3.2, §4.2, §5.2, §6.1, §7, §8)
**Delivery window:** Entrega 2 (per source PRD §10)

## Context

The ExamGenerator produces practice exams from the ingested material. It uses the RAG retriever (built in Epic 2) to ground every question, and respects the student profile (built in Epic 6) to prioritize weak topics and honor preferences. It also enforces the anti-hallucination guardrail: every fact in every question must have a source chunk associated.

## Scope

**In scope**
- Generating MCQ (multiple choice) questions
- Generating open-answer questions
- Anchoring each question to source chunks (anti-hallucination)
- Honoring user preferences (difficulty, quantity, type, topic include/exclude)
- Prioritizing weak topics from the student profile
- Providing base answers for the Evaluator
- Handling the "topic not in material" case gracefully

**Out of scope**
- Building the knowledge base (Epic 2)
- Evaluating student answers (Epic 5)
- Updating the student profile (Epic 6)
- Rendering the exam UI (Epic 7)
- Generating complex practical exercises (Epic 4)

## Functional Requirements

- **EXAM-01** Generate exams with at least two question types: MCQ and open-answer.
- **EXAM-02** Every question must be grounded in chunks from the ingested material; the agent must not invent facts.
- **EXAM-03** Respect user preferences: question types, difficulty, count, topic include/exclude.
- **EXAM-04** When the student profile indicates weak topics, weight the exam toward those topics.
- **EXAM-05** For each question, provide a base answer that the Evaluator can use.
- **EXAM-06** If a requested topic is not present in the material, return a clear message with suggestions of close topics.

## Non-Functional Requirements

- **EXAM-NFR-01** A 10-question exam must be generated in under 30 seconds end-to-end (RNF-01).
- **EXAM-NFR-02** Anti-hallucination guardrail: each fact in a question must have a source chunk with similarity above the threshold; otherwise the question is regenerated up to 3 times.
- **EXAM-NFR-03** All generations must be traced (Epic 8).

## Technical Notes

- Tools: `generate_exam`, `retrieve_chunks` (owned by Ingestor), `get_student_summary` (owned by Support).
- ReAct + Tools loop per source PRD §4.2.
- Plan-and-Execute for the composite flow per source PRD §5.2.

## Test Coverage

- Source PRD §8 cases 2, 7, 11 cover this epic.

## User Stories

### US-3.1: Generate MCQ questions
- **As a** student
- **I want** the exam to include multiple-choice questions
- **So that** I can practice quick-recall format typical of university exams
- **Acceptance criteria:**
  - Each MCQ has a stem, 3-5 options, and exactly one correct option
  - Options include plausible distractors derived from the material
  - The correct answer is annotated and the source chunk is recorded
- **Dependencies:** Epic 2 US-2.9 (retrieve_chunks)
- **Maps to:** RF-07, §5.2 step 5

### US-3.2: Generate open-answer questions
- **As a** student
- **I want** the exam to include open-answer questions
- **So that** I can practice explaining concepts, not just recognizing them
- **Acceptance criteria:**
  - Each open question has a clear prompt and an expected base answer
  - The base answer is grounded in the material and references the source chunk
  - The prompt is open-ended (not a yes/no question)
- **Dependencies:** Epic 2 US-2.9
- **Maps to:** RF-07, §5.2 step 5

### US-3.3: Anchor questions to source chunks (anti-hallucination)
- **As a** system operator
- **I want** every fact in every question to be traceable to a source chunk
- **So that** the agent never invents content (RNF-03)
- **Acceptance criteria:**
  - Post-generation check: every atomic claim in the question is matched to a chunk with score above threshold
  - Failing questions are regenerated up to 3 times; then omitted with a notification
  - The exam output includes the chunk reference per question
- **Dependencies:** US-3.1, US-3.2
- **Maps to:** §7 "Alucinación en preguntas generadas", RNF-03, §8 case 11

### US-3.4: Respect user preferences
- **As a** student
- **I want** to configure the exam style
- **So that** the practice matches my course format
- **Acceptance criteria:**
  - User can set: question types, difficulty, count, topic include/exclude
  - Generated exam conforms to those settings
  - Unspecified settings fall back to safe defaults
- **Dependencies:** Epic 6 US-6.6 (preferences persistence)
- **Maps to:** RF-12, §5.2 step 1

### US-3.5: Prioritize weak topics
- **As a** student with low scores on specific topics
- **I want** the exam to focus on those topics
- **So that** my study time is spent where it matters
- **Acceptance criteria:**
  - The agent loads `get_student_summary` before planning
  - Topics with score < 6 receive higher weight in the topic distribution
  - The final topic mix is recorded in the session state
- **Dependencies:** Epic 6 US-6.5 (get_student_summary)
- **Maps to:** RF-11, §8 case 4

### US-3.6: Provide base answers for the Evaluator
- **As a** Evaluator
- **I want** each question to ship with a base answer and source chunks
- **So that** I can grade the student's response against a known good answer
- **Acceptance criteria:**
  - Each exam question output includes: prompt, type, base_answer, source_chunk_ids
  - The base answer is sufficient to grade against (covers the key points)
- **Dependencies:** US-3.1, US-3.2
- **Maps to:** §5.2 step 5, §5.3 step 3

### US-3.7: Handle missing-topic case
- **As a** student
- **I want** clear feedback when I ask for a topic that is not in the material
- **So that** I know the agent didn't silently invent content
- **Acceptance criteria:**
  - When no chunk matches the requested topic above threshold, the agent returns a structured "topic not found" message
  - The message includes up to 3 suggested close topics from the index
- **Dependencies:** Epic 2 US-2.9, US-2.5
- **Maps to:** §8 case 7, RNF-03

### US-3.8: Performance budget
- **As a** student
- **I want** a 10-question exam to be generated quickly
- **So that** I can iterate between practice and review
- **Acceptance criteria:**
  - 10-question exam completes in under 30 seconds end-to-end (RNF-01)
  - Latency is broken down per phase (retrieval, generation, validation) in traces
- **Dependencies:** US-3.1 through US-3.6
- **Maps to:** RNF-01

### US-3.9: End-to-end exam flow
- **As a** student
- **I want** to request, take, and submit an exam from the UI
- **So that** I can complete a practice round
- **Acceptance criteria:**
  - Request → render → submit works end-to-end
  - The exam is stored in the session state until submission
- **Dependencies:** US-3.1 through US-3.8, Epic 7 US-7.3
- **Maps to:** §5.2 full flow, §8 case 2
