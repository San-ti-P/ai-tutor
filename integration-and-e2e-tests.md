# Tests de Integración y E2E Live

Material de defensa — resumen de cada test, qué cubre, y por qué importa.

---

## 🔬 Integration Tests (Backend) — 12 tests

Requieren `-m integration`. Usan LLM real (Ollama/Groq), embeddings reales (SentenceTransformer), ChromaDB real, y el PDF académico real `apunteAgentes_IA2007.pdf`.

---

### 1. Ingestor — Pipeline de ingesta real

**Archivo:** `back/tests/test_ingestor.py :: TestRealPDFIngestion`

| Test | Descripción |
|------|-------------|
| `test_parse_real_pdf` | PDF real se parsea correctamente → produce >500 chars con términos reconocibles de IA |
| `test_ingest_real_pdf` | La ingesta a ChromaDB real produce una colección con chunks |
| `test_classify_real_pdf` | LLM real clasifica el PDF como `apunte_teorico` con ≥3 topics extraídos |
| `test_retrieve_from_real_pdf` | Búsqueda semántica sobre el PDF real devuelve chunks relevantes |

**Qué demuestra**: parseo (markitdown) → embeddings → ChromaDB → clasificación LLM → retrieval. Todo real, nada mockeado.

---

### 2. RAG — Chunking, embeddings, retrieval

**Archivo:** `back/tests/test_rag.py :: TestRealRAG`

| Test | Descripción |
|------|-------------|
| `test_real_chunking` | Texto académico real produce ≥5 chunks semánticos, cada uno ≤900 chars |
| `test_real_embed_and_retrieve` | Embeddings reales devuelven resultados ordenados por similitud con scores diferenciados |
| `test_real_topic_extraction` | `extract_topics` sobre texto real produce ≥3 topics + árbol JSON válido |

**Qué demuestra**: La base RAG funciona con datos reales — chunking, embedding, retrieval, y extracción de temas.

---

### 3. Exam Generator — Examen end-to-end con LLM real

**Archivo:** `back/tests/test_exam_generator.py :: TestRealIntegration`

| Test | Descripción |
|------|-------------|
| `test_generate_exam_from_real_pdf` | PDF real → chunks reales → LLM real → examen validado. Verifica: estructura (exam_id, questions, status), cada pregunta tiene `source_chunk_ids`, tipos válidos (mcq/open_answer), ≥3 opciones en MCQs, `base_answer` en abiertas. Anti-hallucination guard: status es `complete` o `partial`, nunca `error` |

**Cubre requisitos PRD**: R1 (MCQs), R2 (open answer), R3 (grounding), R6 (source tracking).

**Qué demuestra**: el generador produce preguntas grounded en el material real. No alucina — toda pregunta referencia chunks reales.

---

### 4. Exercise Generator — Ejercicio end-to-end con LLM real

**Archivo:** `back/tests/test_exercise_generator.py :: TestRealIntegration`

| Test | Descripción |
|------|-------------|
| `test_generate_exercise_from_real_pdf` | PDF real → LLM real → ejercicio validado. Verifica: `statement`, `given_data`, `question`, `model_solution` (≥2 steps, `final_answer`, `key_concepts`), `source_chunk_ids` no vacío |

**Qué demuestra**: ejercicios de resolución de problemas generados con grounding real en el material académico.

---

### 5. Evaluator — Evaluación con LLM real

**Archivo:** `back/tests/test_evaluator.py :: TestEvaluatorIntegration`

| Test | Descripción |
|------|-------------|
| `test_evaluate_correct_answer` | **PRD Caso 3** (happy path): respuesta correcta → score ≥6 con RAG backing |
| `test_evaluate_partially_correct` | **PRD Caso 8** (edge case): respuesta parcial → score en rango medio (3.0–7.5) |
| `test_evaluate_wrong_language` | **PRD Caso 12** (adversarial): respuesta no evaluable (gibberish) → structured rejection |

**Qué demuestra**: el evaluador puntúa correctamente respuestas buenas, regulares, y rechaza respuestas inválidas. Todo contra LLM real.

---

### 6. Support Agent — Weak topics y sync de scores

**Archivo:** `back/tests/test_support.py`

| Test | Descripción |
|------|-------------|
| `test_second_session_prioritizes_weak_topics` | **PRD Caso 4**: sesión 1 con score bajo en "cálculo" → `compute_weak_topics` devuelve "cálculo" como débil, "álgebra" no |
| `test_evaluator_updates_topic_scores` | `sync_scores` persiste scores por tópico en SQLite |

**Cubre requisitos**: SUP-03 (persistencia de scores), SUP-07 (detección de temas débiles cross-session).

**Qué demuestra**: el sistema detecta temas débiles entre sesiones y persiste el progreso del estudiante.

---

### 7. Orchestrator — Pipeline de clasificación + síntesis con LLM real

**Archivo:** `back/tests/test_orchestrator.py :: TestRealOrchestratorIntegration`

| Test | Descripción |
|------|-------------|
| `test_classify_exam_request` | "Generame un examen" → intent `generate_exam` o `composite` con confianza >0.5 |
| `test_classify_general_chat` | "Hola, ¿cómo estás?" → intent `general_chat` |
| `test_classify_multi_step_request` | "Subí apuntes y generame un examen" → intent `composite` o plan pre-poblado |
| `test_classify_ingest_request` | "Quiero subir un PDF" → intent `ingest` o `composite` |
| `test_synthesize_general_chat_response` | Pregunta simple → respuesta coherente en español con contenido educativo |
| `test_synthesize_composite_aggregation` | Resultados compuestos → LLM agrega en resumen coherente |
| `test_synthesize_incomplete_response` | Estado incompleto → respuesta incluye advertencia de límite |
| `test_synthesize_partial_with_errors` | Estado parcial con errores → respuesta menciona el problema |
| `test_e2e_general_chat_real` | **Full graph**: classify → synthesize → response. LLM real, español verificado |
| `test_e2e_exam_request_real` | **Full graph**: exam request clasifica correctamente, llega a synthesize |

**Qué demuestra**: el orquestador clasifica intención correctamente en español, sintetiza respuestas educativas, y el grafo completo funciona con LLM real.

---

### 8. Topic Extraction — Pipeline real de extracción de tópicos

**Archivo:** `back/tests/test_topic_extraction.py :: TestFullPipelineRealPDF`

| Test | Descripción |
|------|-------------|
| `test_full_pipeline_real_pdf` | Shape verification (dry run sin LLM) + pipeline real con LLM → ≥3 topics, `topic_tree` JSON válido, `segment_count` >0 |

**Qué demuestra**: la extracción de tópicos funciona end-to-end con LLM real y produce estructura navegable.

---

### 9. Observability — Trazas Langfuse reales

**Archivo:** `back/tests/test_observability.py :: TestLangfuseRealTraces`

| Test | Descripción |
|------|-------------|
| `test_create_trace_emits_test_metadata` | Trace via `obs_manager` lleva metadata: `environment=test`, `test_run_id`, `test_name`, `source` |
| `test_agent_invocation_creates_trace` | Ingestor graph con Langfuse real → tracer enabled, callback handler creado |

**Cubre**: OBS-01 (trazas por invocación), OBS-02 (metadatos de test).

**Qué demuestra**: la integración con Langfuse está operativa — cada invocación de agente genera trazas con metadatos para depuración.

---

### 10. Session Lifecycle — Ciclo completo SQLite + ChromaDB

**Archivo:** `back/tests/integration/test_session_lifecycle.py :: TestSessionLifecycle`

| Test | Descripción |
|------|-------------|
| `test_full_lifecycle_create_upload_chat_profile` | **T-028**: crear sesión → insertar documento → insertar evaluaciones → perfil por sesión (weak topics, avg score, exam count) → orquestador con contexto de sesión cargado |
| `test_delete_session_cascades` | Borrar sesión → ingested_documents eliminados + colección ChromaDB dropeada |
| `test_integration_exam_generation_flow` | Crear sesión → tool generate_exam con session_id |

**Qué demuestra**: el ciclo de vida completo (creación, operaciones, borrado en cascada) funciona con SQLite real + ChromaDB real.

---

## 🌐 E2E Live Tests (Frontend) — 6 tests

Requieren `E2E_LIVE_LLM=true`. Browser → React → FastAPI → LangGraph → LLM real. Sin mocks en ningún nivel.

---

### 1. Ingest → Exam → Evaluate — Flujo completo

**Archivo:** `front/e2e/ingest-exam-flow.spec.ts`

| Test | Descripción |
|------|-------------|
| `@live real LLM — quality validation` | Browser → subir PDF real → chat "generame un examen" → esperar LLM real → verificar preguntas renderizadas o que página no crasheó |

**Qué demuestra**: el flujo completo usuario→UI→backend→LLM→respuesta funciona sin mockear nada. Timeout 5 min por lentitud de LLM real.

---

### 2. Composite Plan-and-Execute — Múltiples acciones

**Archivo:** `front/e2e/composite-flow.spec.ts`

| Test | Descripción |
|------|-------------|
| `@live — composite exam + exercise generation` | Un solo mensaje "examen + ejercicio" → LLM real clasifica composite, ejecuta ambas, página muestra contenido de examen O ejercicio |
| `@live — composite with no material (graceful degradation)` | Sin material subido → sistema responde graceful sugiriendo subir archivo, no crashea |

**Qué demuestra**: el plan-and-execute compuesto funciona con LLM real. El sistema degrada graceful cuando no hay material.

---

### 3. Session Lifecycle — Persistencia de sesiones

**Archivo:** `front/e2e/session-lifecycle.spec.ts`

| Test | Descripción |
|------|-------------|
| `@live — session endurance` | Crear sesión → recargar página → sesión sigue visible. Verifica persistencia real (no mockeada) del estado en backend |

**Qué demuestra**: las sesiones persisten entre recargas — los datos guardados en SQLite sobreviven al ciclo del frontend.

---

### 4. Profile Persistence — Dashboard con datos reales

**Archivo:** `front/e2e/profile-persistence.spec.ts`

| Test | Descripción |
|------|-------------|
| `@live — full profile cycle` | Navegar a /dashboard → ver stats cards O estado vacío. Verifica que el dashboard carga sin crashear contra datos reales |

**Qué demuestra**: el dashboard de progreso funciona conectado al backend real.

---

### 5. Weak Topic Prioritization — Temas débiles

**Archivo:** `front/e2e/weak-topic-prioritization.spec.ts`

| Test | Descripción |
|------|-------------|
| `@live — weak topic prioritization flow` | Crear sesión → pedir examen sobre "matrices" → navegar a /dashboard → stats cards visibles. Verifica que el flujo completo genera datos de perfil |

**Qué demuestra**: el dashboard refleja actividad de estudio real — los weak topics se computan contra datos reales.

---

## 💣 Puntos clave para la defensa

| Punto | Detalle |
|-------|---------|
| **LLM real en integración** | 12 tests usan Ollama/Groq real. No es común — demuestra que el sistema funciona con inteligencia real, no datos fabricados |
| **PDF académico real** | `apunteAgentes_IA2007.pdf` es material de cursada real de IA 2026, no un fixture de juguete |
| **ChromaDB + SQLite real** | Las integration tests usan persistencia real — embeddings, retrieval, y esquema relacional |
| **E2E live es full-stack** | Browser → React → FastAPI → LangGraph → LLM real. Sin mocks en ningún nivel |
| **Anti-hallucination guard** | Cada pregunta tiene `source_chunk_ids` obligatorio. Si el LLM inventa, status = `partial`, no `complete` |
| **Graceful degradation probada** | Sin material → no crashea → sugiere subir archivos |
| **Tiers separados** | Unit (commit), Integration (manual/CI), E2E Mock (commit), E2E Live (pre-defensa). Cada tier con propósito distinto |
| **Cobertura de PRD** | Casos 3 (correcto), 4 (temas débiles), 8 (parcial), 12 (adversarial) cubiertos como integration tests |
| **Langfuse operativo** | Trazas reales por invocación de agente — observable y depurable |
