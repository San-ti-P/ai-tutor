# PRD vs Implementation — Gap Analysis

This document provides a systematic gap analysis comparing the specifications outlined in the Product Requirements Document (init_PRD.md) against the actual implementation of the AI Tutor system. The analysis evaluates agent architectures, tool definitions, functional and non-functional requirements, guardrails, test coverage, and frontend-backend integration.

---

## 1. Agents and Loop Patterns

The Product Requirements Document specifies a dual architectural pattern for agent execution. The Orchestrator is intended to follow a Plan-and-Execute pattern to handle multi-step academic tasks, while the specialized agents (Ingestor, ExamGenerator, and ExerciseGenerator) are designated to operate under the ReAct (Reasoning + Acting) loop pattern. The Evaluator is defined to use Chain-of-Thought reasoning, and the Support Agent is described as a reactive handler.

In the actual codebase, there is a significant structural deviation in the loop patterns of the specialized agents:

### The Orchestrator Agent
The Orchestrator agent aligns with the Plan-and-Execute requirements. It is implemented as a LangGraph StateGraph that classifies user intent via a structured LLM output (the `IntentClassification` schema). For simple queries, it bypasses complex planning and assigns a single tool execution step. For multi-step queries (intents classified as `composite`), it invokes an LLM planner to construct a `CompositePlan` containing an ordered list of tools to run. The Orchestrator then executes these steps sequentially, tracking progress in its shared state.

### The Ingestor Agent
The Ingestor agent deviates from the specified ReAct model. Rather than running an autonomous loop that dynamically reasons and selects actions based on input, it is implemented as a deterministic, linear StateGraph. The pipeline executes three nodes sequentially: `parse_document`, `classify_document`, and `chunk_and_embed`. While this linear design ensures high reliability and predictable execution, it does not employ the iterative ReAct reasoning loop outlined in the PRD.

### The Exam and Exercise Generators
Both the ExamGenerator and ExerciseGenerator graphs utilize a deterministic topology instead of a ReAct loop. In a standard ReAct agent, the LLM decides dynamically when to call the vector store to search for documents. In this codebase, the retrieval phase is hardcoded as a structural node (`retrieve_relevant_chunks` in `exam_generator.py`) that runs before the question generation node. The generation node then takes these pre-retrieved chunks and outputs the final questions. This pipeline structure eliminates the need for tool-calling loops, reducing latency and avoiding infinite agent loops, but it represents a clear architectural divergence from the PRD.

### The Evaluator Agent
The Evaluator agent adheres to the Chain-of-Thought requirement. It runs a detailed 8-node StateGraph that processes candidate answers, performs semantic comparison against the material, and checks for translation or character set compatibility. It executes a structured reflection step to evaluate answer accuracy and generates a final score (0-10) with suggestions.

### The Support Agent
The Support Agent is implemented as a reactive component. It updates student profile metrics in the SQLite database and retrieves performance data to inform the retrieval process.

### Summary of Agent Loop Status
- Orchestrator: Plan-and-Execute is fully implemented.
- Ingestor: Non-ReAct; runs as a linear, deterministic pipeline.
- ExamGenerator: Non-ReAct; retrieval and generation are structured as fixed sequential nodes.
- ExerciseGenerator: Non-ReAct; uses the same fixed pipeline layout as the ExamGenerator.
- Evaluator: Chain-of-Thought is implemented via multi-stage graph nodes and structured prompting.
- Support Agent: Reactive; functional via state transitions and database helpers.

---

## 2. Tool Implementations

The PRD defines a set of eight core tools that act as the interface between agents and the underlying data layer. Below is an analysis of how these tools are implemented in the codebase:

### ingest_document
This tool is fully implemented. It encapsulates the Ingestor's StateGraph execution, taking a local file path, running Microsoft MarkItDown to parse the contents, classifying the document type, and writing the resulting embeddings to ChromaDB.

### retrieve_chunks
This tool is fully implemented. It interfaces directly with ChromaDB to execute similarity searches. It supports retrieval configurations such as session-based collection scoping and similarity distance thresholds.

### generate_exam and generate_exercise
These tools are fully implemented. They wrap the execution of their respective LangGraph subgraphs, returning structured JSON outputs representing exams and exercises.

### evaluate_answer
This tool is fully implemented. It wraps the Evaluator's StateGraph and performs granular grading, returning scores, conceptual error lists, and suggestions.

### update_student_profile and get_student_summary
These tools are fully implemented. They handle SQLite database operations to persist user performance statistics, recalculate weak topics, and load student history for subsequent sessions.

### ocr_math_extract
This tool is not implemented. The mathematical OCR extraction pipeline has been deferred. In `ingestor.py`, the `parse_document` function explicitly rejects image file formats (PNG, JPG, JPEG) and returns a rejection status. The file-level documentation notes that this feature is marked as post-MVP. As a result, the tool does not exist in the active graph, causing a functional gap.

### Undocumented Helper Tools
The implementation contains helper functions that act as internal tools but are not explicitly registered as standalone agents in the PRD:
- `extract_topics`: Uses regex patterns to identify subjects mentioned in natural language prompts.
- `validate_claim_grounding`: Compares generated exam questions against source chunks using cosine similarity to prevent hallucinated questions.
- `orchestrate_chat`: Provides a generic chat routing function.

---

## 3. Functional Requirements

### RF-01: Accept PDF, PNG/JPG, and TXT File Formats
There is a partial gap in file format support. PDF and TXT file parsing is operational using Microsoft MarkItDown. However, image uploads (PNG, JPG, JPEG) are rejected by the Ingestor agent. The system cannot ingest visual resources or photos of notes.

### RF-02: Automatic Document Classification
This requirement is met. The Ingestor agent uses structured LLM output to classify incoming materials into four categories: `apunte_teorico`, `examen_previo`, `ejercicio_resuelto`, and `no_academico`. If a document is classified as `no_academico`, the system rejects it, preventing non-academic files from contaminating the vector database.

### RF-03: Incremental Ingestion
This requirement is met. ChromaDB collections are persisted locally in `./data/chroma`. The system appends new documents to the active session collection without wiping or reprocessing previously ingested files.

### RF-04: OCR Mathematical Extraction to LaTeX
This is a complete gap. The system lacks the capability to recognize math equations in images or translate them into LaTeX syntax, as the image parser is not active.

### RF-05: Low-Confidence OCR Confirmation Flow
This is a complete gap. Because the mathematical OCR pipeline is disabled, there is no interface or backend logic to detect low-confidence parsing (confidence score below 0.85) or present correction prompts to the user.

### RF-06: Hierarchical Topic Index and Search
This requirement is met in the backend. The system defines a `ThematicIndex` class that models topics as nested dictionary trees using slash-separated paths (e.g., "Math/Algebra"). It supports deep-merging to combine topic trees during incremental ingestion. However, there is a gap on the frontend, as there is no user interface to visualize or navigate this hierarchy.

### RF-07: Multiple Choice (MCQ) and Open-Answer Questions
This requirement is met in the backend. The generator outputs structured JSON containing questions of both types, including prompt texts, selectable options for MCQs, and detailed reference answers.

### RF-08: Question Grounding in Ingested Material
This requirement is met. The system implements a validation guardrail (`validate_claim_grounding`) that calculates cosine similarity between the generated question text and the retrieved document chunks. If a question fails to meet the similarity threshold, the generator rejects it and attempts regeneration up to three times.

### RF-09: Complex Practical Exercises
This requirement is met. The ExerciseGenerator creates multi-step exercises with specific context variables, solution procedures, and chunk references.

### RF-10: Score (0-10) and Conceptual Feedback
This requirement is met. The Evaluator agent returns a structured JSON payload that includes numerical scores, detailed justifications, list of conceptual errors, and suggestions for study.

### RF-11: Cross-Session Memory and Weak Topic Prioritization
This requirement is met. Student profiles are saved in SQLite. The system runs a background calculation to determine weak topics based on historical scores. During retrieval, these weak topics are passed as a priority list, and the retrieval function doubles the search weight for chunks matching those topics.

### RF-12: User Preferences Configuration
There is a gap. The backend schema supports configurations for question count, difficulty, and topic exclusion. However, the frontend generation page (`front/src/app/exam/page.tsx`) has all preference inputs disabled, meaning users cannot customize their exams.

### RF-13: Dashboard Interface
There is a gap. The backend provides an endpoint (`/students/{id}/dashboard`) to retrieve performance statistics. The frontend dashboard page (`front/src/app/dashboard/page.tsx`) consists of a static template with hardcoded "0" values and placeholder text.

### RF-14: LLM and Tool Call Tracing
This requirement is met. The codebase is integrated with Langfuse. Using the `@observe` decorator and custom callbacks, the system records root session details, LLM prompts and responses, token usage, latency, tool arguments, and vector retrieval scores.

---

## 4. Non-Functional Requirements

### RNF-01: Exam Generation Latency under 30 Seconds
This has not been systematically validated. The response time depends heavily on the external LLM provider (Ollama or Groq). Under normal usage, generating a 5-question exam takes 5 to 15 seconds, but a 10-question exam with multiple retrieval stages may approach the 30-second limit.

### RNF-02: Ingestion Speed under 2 Minutes for 50 Pages
This is unverified. Single-page PDFs process in under 5 seconds, but large files have not been benchmarked. Performance is bound by the local SentenceTransformer embedding speed.

### RNF-03: Anti-Hallucination Guardrail
This is met. The validation check prevents the model from generating questions that are unsupported by the vector database.

### RNF-04: Usability without Technical Documentation
There is a gap. Because the frontend components are static and input fields are disabled, the system cannot be used by an end-user in its current state.

### RNF-05: Traceability
This is met. Every execution registers a full trace in Langfuse containing inputs, outputs, and intermediate states.

### RNF-06: Modular Architecture
This is met. The backend maintains clean separation of concerns across directory structures.

### RNF-07: Secure Key Management
This is met. API keys and configuration constants are loaded from environment variables using `Pydantic-Settings`.

---

## 5. Guardrail Status

- **Question Hallucination**: Active. Verified via claim-level cosine similarity checks.
- **Infinite Loops**: Active. Enforced with a 15-iteration limit on agent executions.
- **OCR Validation**: Inactive. Blocked by the deferred OCR pipeline.
- **Non-Academic Content**: Active. Automatically rejected during the ingestion classification node.
- **Evaluation Inconsistency**: Active. A secondary LLM-as-judge reviews 10% of evaluations to check for grading errors.

---

## 6. Test Case Verification (12 Required Cases)

- **Test 1: Ingest Well-Formatted PDF**: Fully verified via `test_ingestor.py`.
- **Test 2: Generate 5-Question Exam**: Fully verified via `test_exam_generator.py`.
- **Test 3: Evaluate Correct Answer**: Fully verified via `test_evaluator.py`.
- **Test 4: Prioritize Weak Topics**: Partially verified. The logic exists in the retriever, but there is no dedicated integration test asserting that the second session correctly prioritizes historical weak topics.
- **Test 5: Incremental Ingestion**: Fully verified. ChromaDB collection updates are covered in tests.
- **Test 6: PDF with Tables/Equations (OCR 80%)**: Blocked. Cannot run due to the lack of image and equation OCR.
- **Test 7: Exam Request on Missing Topic**: Fully verified. The system returns suggestions for matching topics.
- **Test 8: Partially Correct Answer**: Partially verified. Evaluator logic processes partial credit, but no test asserts a specific score range (5-7/10).
- **Test 9: Handwritten Photo with Low OCR**: Blocked. Image upload is disabled.
- **Test 10: Non-Academic Document Ingestion**: Fully verified. Rejection of non-academic text is tested.
- **Test 11: Exam Request on Unrelated Topic**: Fully verified. Tested under missing topic constraints.
- **Test 12: Answer in Different Language**: Fully verified. Handled via the language evaluation safety checks.

---

## 7. Integration and API Gaps

The primary gap in the system is the connection between the frontend UI and the backend API, along with several mock endpoints:

### Endpoint Prefix Mismatch
The backend application prefix is configured as `/api` in `main.py`. However, the frontend API client (`api.ts`) defines endpoints without this prefix (e.g., calling `/chat` instead of `/api/chat`). Running the application in this configuration causes all frontend API calls to fail with 404 errors.

### exam/generate Endpoint
The backend route `POST /api/exam/generate` contains a stub implementation. It accepts the request but returns a hardcoded schema with an empty list of questions. It does not invoke the ExamGenerator subgraph. The generator can only be triggered via the chat agent.

### profile/{id} Endpoint
The backend route `GET /api/profile/{session_id}` returns a static dictionary. The actual student profile loading and rendering logic is not connected.

### Frontend Skeleton
The frontend layout is defined, but all interactive components are empty shells:
- `front/src/components/chat/`: Contains no files.
- `front/src/components/dashboard/`: Contains no files.
- `front/src/components/upload/`: Contains no files.
- `front/src/components/exam/`: Contains no files.
- The UI contains placeholder text indicating that these features will be available soon.
