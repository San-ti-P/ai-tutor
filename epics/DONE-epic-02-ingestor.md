# Epic 2: Ingestor Agent

**Status:** Done
**Source PRD:** [../init_PRD.md](../init_PRD.md) (esp. §3.2, §4.2, §4.3, §4.4, §5.1, §6.1, §7, §8)
**Delivery window:** Entrega 2 (per source PRD §10)

## Context

The Ingestor is the gateway for all study material. It receives uploaded files (PDFs, plain text), converts them to structured Markdown, classifies them, extracts hierarchical topics, builds the RAG index, and exposes the read-side tool (`retrieve_chunks`) to all other agents. It also enforces the guardrail that rejects non-academic content.

This epic owns the RAG module (ChromaDB collection, chunking strategy, embeddings, thematic index) because the Ingestor is the only writer; all other agents are consumers.

## Scope

**In scope**
- File parsing (PDF, TXT) into Markdown
- Document classification (apunte teórico / examen previo / ejercicio resuelto)
- Thematic index extraction (tree of topics) per document
- Standalone `extract_topics` tool — any agent can analyze a file's content
- Semantic chunking + embedding + insertion into ChromaDB
- Incremental ingestion (new files added, existing chunks untouched)
- Rejection of non-academic content
- The `retrieve_chunks` tool (read-side; used by all agents)

**Deferred (scope-restricted — see US-2.3, US-2.4)**
- OCR of mathematical expressions to LaTeX (US-2.3)
- User confirmation prompt on low OCR confidence (US-2.4)
- Image file support (PNG/JPG) — depends on OCR pipeline
- `extract_answer_from_image` in the Evaluator (Epic 5)

**Out of scope**
- Generating study material (Epics 3, 4)
- Evaluating answers (Epic 5)
- Updating the user profile (Epic 6)
- UI for upload confirmation (Epic 7)

## Functional Requirements

- **ING-01** Accept files in PDF and TXT formats. Images (PNG/JPG) are rejected with a clear "deferred" message.
- **ING-02** Classify each document into one of: apunte teórico, examen previo, ejercicio resuelto.
- **ING-03** Ingest incrementally: add new chunks without reprocessing existing material; merge thematic indices.
- **ING-04** *(Deferred)* Extract mathematical expressions from images and represent them in LaTeX.
- **ING-05** *(Deferred)* If OCR confidence is below the configured threshold, display the extracted LaTeX to the user and require explicit confirmation before proceeding.
- **ING-06** Build a hierarchical thematic index of the document and merge it into the global session index.
- **ING-07** Reject files that do not look like academic material; surface a clear error to the user.
- **ING-08** Expose a `retrieve_chunks` tool that takes a topic or query and returns the top-K relevant chunks with metadata.
- **ING-09** Expose an `extract_topics` tool that any agent can call to analyze text or a file and return its hierarchical topic structure.

## Non-Functional Requirements

- **ING-NFR-01** Ingest a PDF of up to 50 pages in under 2 minutes (RNF-02).
- **ING-NFR-02** Chunking must be semantic-by-section with fallback to 512-token chunks with 64-token overlap.
- **ING-NFR-03** Embedding model choice must favor local execution to avoid latency and cost (§4.3).
- **ING-NFR-04** Vector store is ChromaDB, local and persistent; collections separated by session/materia.

## Technical Notes

- File-to-Markdown: markitdown (source PRD §4.5).
- Vector store: ChromaDB + LangChain.
- Topic extraction: LLM via Ollama (ChatOllama) with structured output (Pydantic schema).
- RAG infrastructure: per §4.3.
- Episodic memory: ChromaDB vector store is the source of truth for ingested material (source PRD §4.4).
- OCR math *(deferred)*: Mathpix API as primary, pix2tex (local) as fallback.
- Image support *(deferred)*: blocked on OCR pipeline.

## Ingestor Graph (simplified, OCR deferred)

```
START → parse_document → classify_document → chunk_and_embed → END
```

The OCR nodes (`run_ocr_if_needed`, `check_ocr_confidence`) are removed from the active graph. Image files are rejected at `parse_document` with a descriptive error.

## Test Coverage

- Source PRD §8 cases 1, 5, 6, 9, 10 cover this epic directly.
- Case 6 (Math PDF OCR) and case 9 (Low OCR confidence) are deferred alongside the OCR scope.
- New tests required for `extract_topics` tool.

## User Stories

### US-2.1: File-to-Markdown conversion
- **As a** student uploading study material
- **I want** PDFs and TXT files to be converted to structured Markdown
- **So that** the rest of the pipeline can process them uniformly
- **Acceptance criteria:**
  - PDF input produces Markdown preserving headings, tables, and lists
  - TXT input is read as-is
  - Image input is rejected with a "not yet supported" message (OCR deferred)
- **Dependencies:** —
- **Maps to:** RF-01, §4.5 markitdown, §5.1 step 2

### US-2.2: Document classification
- **As a** system
- **I want** each uploaded file to be classified into one of three classes
- **So that** downstream agents can apply class-specific behavior
- **Acceptance criteria:**
  - Each file receives a label: `apunte_teorico | examen_previo | ejercicio_resuelto`
  - Confidence is reported alongside the label
  - Files below confidence are surfaced to the user for manual labeling
- **Dependencies:** US-2.1
- **Maps to:** RF-02, §5.1 step 3

### US-2.3: OCR math extract *(DEFERRED)*
- **As a** student uploading photos of notes
- **I want** the math expressions to be extracted as LaTeX
- **So that** they can be indexed and later used to generate questions
- **Acceptance criteria:**
  - Image input produces LaTeX for each detected expression
  - The OCR engine is configurable (cloud or local)
  - Each extracted expression has an associated confidence score
- **Dependencies:** US-2.1, deferred to post-MVP
- **Maps to:** RF-04, §3.2 `ocr_math_extract`, §4.5 OCR stack

### US-2.4: OCR low-confidence confirmation *(DEFERRED)*
- **As a** student
- **I want** to be asked to confirm when OCR confidence is low
- **So that** bad OCR does not pollute the knowledge base
- **Acceptance criteria:**
  - Threshold is configurable (default 0.85)
  - On low confidence, the UI shows the extracted LaTeX and asks for confirmation
  - Only confirmed extractions are added to the index
- **Dependencies:** US-2.3, Epic 7 US-7.6 (UI confirmation), deferred to post-MVP
- **Maps to:** RF-05, §7 OCR guardrail, §8 case 9

### US-2.5: Thematic index extraction
- **As a** Orchestrator or ExamGenerator
- **I want** each document to contribute a tree of topics to the global index
- **So that** retrieval can be filtered by topic before similarity search
- **Acceptance criteria:**
  - Each document yields a tree of topics (LLM-assisted)
  - The tree is merged into the global session index
  - Retrievals can request chunks from a specific topic
- **Dependencies:** US-2.1
- **Maps to:** RF-06, §4.3 thematic index, §5.1 step 5

### US-2.6: Semantic chunking and embedding
- **As a** system
- **I want** documents to be split into semantic chunks and embedded
- **So that** they can be retrieved by similarity
- **Acceptance criteria:**
  - Chunks respect section/topic boundaries when detectable
  - Fallback chunk size: 512 tokens with 64-token overlap
  - Embeddings are computed and stored in ChromaDB
- **Dependencies:** US-2.5
- **Maps to:** RF-06, §4.3 chunking and embeddings, §5.1 step 6

### US-2.7: Incremental ingestion
- **As a** student
- **I want** to add more files to the same knowledge base
- **So that** the system keeps up with new material across the course
- **Acceptance criteria:**
  - A second upload adds chunks without modifying previous chunks
  - The thematic index is merged, not replaced
  - Total chunk count grows monotonically
- **Dependencies:** US-2.6
- **Maps to:** RF-03, §4.3 "Actualización Incremental", §8 case 5

### US-2.8: Reject non-academic content
- **As a** system
- **I want** files that are not academic material to be rejected
- **So that** the knowledge base stays clean
- **Acceptance criteria:**
  - The Ingestor flags a rejection when the document is not academic
  - A random-text adversarial input is rejected (no chunks added)
  - The user gets a clear, human-readable explanation
- **Dependencies:** US-2.2
- **Maps to:** §7 "Material fuera de dominio académico", §8 case 10

### US-2.9: retrieve_chunks tool
- **As a** any specialized agent (Exam, Exercise, Evaluator, Orchestrator)
- **I want** to call a `retrieve_chunks` tool with a topic or query
- **So that** I can ground my generation in the ingested material
- **Acceptance criteria:**
  - Tool takes a query and an optional topic filter
  - Returns top-K (K=5-8, configurable) chunks with metadata and similarity scores
  - Chunks respect the topic filter before similarity search
- **Dependencies:** US-2.6
- **Maps to:** RF-06, §3.2 `retrieve_chunks (Todos)`, §4.3 Retriever

### US-2.10: extract_topics tool
- **As a** any agent (Orchestrator, ExamGenerator, ExerciseGenerator)
- **I want** to call an `extract_topics` tool with text or a file path
- **So that** I can understand what topics a document covers without running the full ingestion pipeline
- **Acceptance criteria:**
  - Tool accepts `text` or `file_path` (at least one required)
  - Returns a one-sentence summary of the content
  - Returns a flat list of topics (3-15)
  - Returns a hierarchical topic tree for drill-down
- **Dependencies:** US-2.1
- **Maps to:** §3.2 new tool, ING-09

### US-2.11: End-to-end ingest flow
- **As a** student
- **I want** to upload a PDF and see the system confirm what it ingested
- **So that** I know the material is ready for generating exams
- **Acceptance criteria:**
  - A well-formatted PDF ingest yields: classification, topic list, chunk count, status
  - The UI shows a confirmation screen with the detected topics and chunk count
  - Image uploads are rejected with a clear "not yet supported" message
- **Dependencies:** US-2.1 through US-2.7, US-2.9, Epic 7 US-7.2
- **Maps to:** §5.1 full flow, §8 case 1
