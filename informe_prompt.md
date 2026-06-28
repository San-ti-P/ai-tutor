# Prompt para Generación del Informe Técnico — AI Tutor (TP 2 IA 2026)

Eres un asistente académico. Tu tarea es redactar un **informe técnico completo** sobre el proyecto **"AI Tutor — Tutor Académico Personal"**, siguiendo estrictamente el formato definido en el **Anexo I** del documento `docs/TP_2-Agente.pdf`. Usa exclusivamente la información provista en este prompt. No inventes datos, métricas ni resultados que no estén documentados aquí.

---

## Especificaciones de formato

- **Idioma**: español
- **Extensión**: ~15–20 páginas
- **Tipo de letra**: Times New Roman 12pt, interlineado 1.5, márgenes 2.5 cm
- **Formato final**: PDF
- **Tono**: académico, formal, tercera persona
- **Citas y referencias**: estilo APA-like (numeradas entre corchetes [1], [2], etc.)
- **Figuras y tablas**: numeradas secuencialmente, con leyenda explicativa al pie
- **Ecuaciones**: numeradas a la derecha entre paréntesis

---

## Estructura del informe (según Anexo I)

### Encabezado
```
Nombre del TP: AI Tutor — Tutor Académico Personal
Nro. de Grupo: [COMPLETAR]
Nombre y Apellido integrante1 - e-mail
Nombre y Apellido integrante2 - e-mail
Nombre y Apellido integrante3 - e-mail
```

### Resumen (70–150 palabras)
Sintetizar:
- Aplicación desarrollada: sistema multi-agente basado en LLMs para preparación adaptativa de exámenes universitarios
- Problema concreto: estudiantes necesitan práctica personalizada con retroalimentación inmediata y seguimiento de progreso
- Estado de resolución: implementado funcionalmente con 6 agentes especializados, RAG sobre material académico, evaluación automática
- Resultados obtenidos: sistema operativo con frontend web, 12 casos de prueba definidos y ejecutados, pipeline de ingestion funcional

### 1. Introducción
Desarrollar:
- **Área de aplicación**: inteligencia artificial generativa aplicada a educación universitaria, agentes autónomos basados en LLMs
- **Problema**: estudiantes de IA 2026 en UTN Santa Fe (CIDISI) necesitan una herramienta que les permita practicar para exámenes con ejercicios personalizados, recibir retroalimentación inmediata y hacer seguimiento de sus áreas débiles
- **Contexto del TP 2**: segundo trabajo práctico de la materia Inteligencia Artificial, enfocado en diseño e implementación de agentes inteligentes basados en LLMs con RAG, Tools y Memoria
- **Objetivo del agente**: asistir al estudiante en la preparación de exámenes ingiriendo material de estudio, generando exámenes y ejercicios personalizados, evaluando respuestas y registrando el progreso
- Incluir **Figura 1**: diagrama de alto nivel del sistema (Frontend ↔ Backend FastAPI ↔ Agentes LangGraph ↔ ChromaDB/SQLite)
- Describir cómo sigue el informe (secciones 2, 3, 4)

### 2. Solución

#### 2.1 Arquitectura General del Agente
Describir con **Figura 2** (diagrama de componentes y flujo de control):

El sistema sigue una arquitectura **multi-agente orquestada** con 6 agentes especializados sobre LangGraph, expuestos mediante una API FastAPI y consumidos por un frontend Next.js.

**Stack tecnológico**:
| Capa | Tecnología |
|------|-----------|
| Orquestación | LangGraph 1.x + LangChain 1.x (Python 3.12+) |
| Backend | FastAPI + Uvicorn, uv package manager |
| LLM | Multi-proveedor: Ollama, Groq, OpenCode Go, OpenAI, Anthropic-compatible |
| Vector Store | ChromaDB (PersistentClient, cosine similarity) |
| Embeddings | SentenceTransformer (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, local) |
| Persistencia | SQLite via aiosqlite (students, sessions, evaluations, topic_scores, ingested_documents) |
| Observabilidad | Langfuse (traces, spans, callback handler) |
| Parsing | Microsoft markitdown (PDF → Markdown) |
| NLP | NLTK (Spanish stopwords + SnowballStemmer) |
| Frontend | Next.js 15 + React 19 + Tailwind 4 + TypeScript 5 |
| Gráficos | Recharts |
| Renderizado matemático | KaTeX |
| Testing E2E | Playwright |
| Linting | ruff (Python), ESLint (TypeScript) |

**Agentes y sus patrones de loop**:

| Agente | Archivo | Loop | Nodos del grafo | Herramientas principales |
|--------|---------|------|-----------------|--------------------------|
| **Orchestrator** | `back/src/agents/orchestrator.py` | Plan-and-Execute | 7 nodos: load_profile → load_session_context → classify_intent → [plan_composite \| execute_step] → check_iteration_limit → synthesize_response | Todas las tools del sistema |
| **Ingestor** | `back/src/agents/ingestor.py` | Pipeline lineal | 3 nodos: parse_document → classify_document → chunk_and_embed | ingest_document, extract_topics |
| **ExamGenerator** | `back/src/agents/exam_generator.py` | Pipeline determinístico | 4 nodos: retrieve → generate → validate → [retry hasta 3x \| format_exam] | generate_exam, retrieve_chunks, validate_claim_grounding |
| **ExerciseGenerator** | `back/src/agents/exercise_generator.py` | Pipeline determinístico | 4 nodos: retrieve → generate → validate → [retry \| format] | generate_exercise, retrieve_chunks |
| **Evaluator** | `back/src/agents/evaluator.py` | Chain-of-Thought | 8 nodos: prepare → check_evaluability → evaluate_answer → validate_feedback → [llm_judge] → build_feedback → next_question → [loop \| sync_scores] | evaluate_answer, validate_claim_grounding |
| **Support Agent** | `back/src/agents/support.py` | Reactivo (templates hardcodeados, sin LLM) | 4 nodos: fetch_profile → fetch_history → compute_progress → generate_response | get_student_summary, update_student_profile, get_session_progress |

**Archivos de herramientas**:
| Tool | Archivo | Propósito |
|------|---------|-----------|
| `ingest_document` | `tools/__init__.py` | Wrapper del Ingestor |
| `retrieve_chunks` | `tools/__init__.py` | Búsqueda semántica en ChromaDB |
| `generate_exam` | `tools/__init__.py` | Wrapper del ExamGenerator |
| `generate_exercise` | `tools/__init__.py` | Wrapper del ExerciseGenerator |
| `evaluate_answer` | `tools/__init__.py` | Wrapper del Evaluator |
| `get_student_summary` | `tools/get_student_summary.py` | Agregación de perfil (solo lectura) |
| `update_student_profile` | `tools/update_student_profile.py` | Upsert de preferencias + scores |
| `extract_topics` | `tools/__init__.py` | Pipeline de extracción de tópicos |
| `validate_claim_grounding` | `tools/validate_claim_grounding.py` | Validación anti-alucinación: cosine similarity claims vs chunks |
| `orchestrate_chat` | `tools/orchestrate_chat.py` | Ruteo de chat al orquestador |
| `query_material` | `tools/query_material.py` | Consulta RAG directa |
| `list_session_files` | `tools/list_session_files.py` | Archivos ingeridos en sesión |
| `get_session_progress` | `tools/get_session_progress.py` | Progreso por sesión |

#### 2.2 System Prompt y Estrategia de Prompting

- No hay un único system prompt monolítico; cada agente tiene su propio prompt específico definido en los prompts de LangGraph.
- El **Orchestrator** usa un prompt que establece su rol como coordinador principal, con instrucciones para clasificar la intención del usuario (ingestión, generación de examen, evaluación, consulta de perfil) y planificar los pasos necesarios.
- El **Ingestor** usa un prompt que le indica clasificar documentos como "apunte", "examen", "ejercicio" o "no_academico", y extraer temas mediante segmentación por encabezados con unificación vía Jaccard + NLP en español.
- El **ExamGenerator** y **ExerciseGenerator** incluyen en sus prompts instrucciones para generar preguntas fundamentadas exclusivamente en los chunks recuperados, con formato MCQ u open-answer, y validar cada claim contra el contenido fuente.
- El **Evaluator** usa un prompt de Chain-of-Thought con criterios explícitos de evaluación, escalas de puntuación, y un paso de LLM-as-judge para consistencia.
- El **Support Agent** no usa LLM: responde con templates determinísticos a consultas de perfil y progreso.

#### 2.3 Tools y Function Calling

Listar las 13 herramientas definidas (ver tabla en 2.1), describiendo para cada una:
- **Schema**: tipo de argumentos (Pydantic models)
- **Validaciones**: chequeos de tipos, límites de puntuación [0-10], rechazo de imágenes, etc.
- **Manejo de errores**: propagación estructurada de errores al Orchestrator; el Orchestrator reintenta o solicita aclaración al usuario según el tipo de error

El Orchestrator decide qué tool(es) invocar según la intención clasificada. Las tools wrapper (ingest_document, generate_exam, generate_exercise, evaluate_answer) internamente ejecutan el sub-grafo de LangGraph del agente correspondiente.

#### 2.4 Loop del Agente

**Describir y justificar la elección para cada agente**:

- **Orchestrator — Plan-and-Execute**: adecuado para coordinar múltiples agentes especializados. Primero clasifica la intención, luego genera un plan de pasos, ejecuta cada paso delegando al agente correspondiente, y finalmente sintetiza la respuesta. El límite duro de 15 iteraciones previene loops infinitos.

- **Ingestor — Pipeline lineal**: la PRD especificaba ReAct, pero se optó por un pipeline determinístico (parse → classify → chunk → embed) porque la ingestion es una tarea estructurada sin necesidad de razonamiento iterativo. Esto mejora performance y confiabilidad del proceso.

- **ExamGenerator y ExerciseGenerator — Pipeline determinístico**: similar al Ingestor, la PRD especificaba ReAct pero se implementó como pipeline con loop de validación (retry hasta 3x). La generación de preguntas sigue un proceso lineal bien definido; el valor está en la validación post-generación, no en la iteración reactiva.

- **Evaluator — Chain-of-Thought**: el más complejo del sistema (8 nodos). El CoT es adecuado porque la evaluación requiere múltiples pasos de razonamiento encadenados (evaluabilidad → puntuación → validación → LLM-judge → feedback → siguiente pregunta). Incluye un paso de LLM-as-judge que muestrea el 30% de las evaluaciones para detectar inconsistencias (discrepancia > 2.0 puntos dispara revisión).

- **Support Agent — Reactivo (templates)**: no usa LLM. Responde consultas de perfil y progreso mediante templates determinísticos. Esto garantiza respuestas instantáneas, sin riesgo de alucinación, para consultas que son puramente de agregación de datos.

#### 2.5 Memoria

El sistema implementa **memoria conversacional** (short-term) y **memoria persistente** (long-term):

- **Short-term**: el historial completo de la conversación se mantiene en el state de LangGraph durante la sesión. El Orchestrator inyecta el contexto relevante (últimos N mensajes) en cada llamada al LLM.

- **Long-term (SQLite)**: esquema con 5 tablas:
  - `students`: id, name, preferences (JSON), global_strengths, global_weaknesses
  - `sessions`: id, student_id, name, status, created_at, updated_at
  - `evaluations`: id, session_id, question, answer, score, feedback, created_at
  - `topic_scores`: id, student_id, session_id, topic_path, score, count, last_updated
  - `ingested_documents`: id, session_id, filename, collection_name, classification, topic_count, status

- **Política de actualización**: las evaluaciones actualizan los `topic_scores` de forma incremental. Los temas con score < 6.0 se consideran "débiles" (weak topics). El perfil del estudiante se recalcula agregando scores de todas las sesiones usando el `ThematicIndex` (árbol jerárquico de temas con merge profundo para ingestion incremental).

- **WAL mode + foreign keys** en SQLite para concurrencia segura.

#### 2.6 RAG (Retrieval-Augmented Generation)

**Pipeline de escritura (Write path)**:
1. Upload de PDF → `markitdown` convierte a Markdown
2. Extracción de tópicos (Epic 11): segmentación por encabezados → LLM por segmento → unificación con Jaccard + NLP en español → construcción de árbol jerárquico
3. Clasificación del documento (LLM con preview de 3000 chars + tópicos detectados): categorías "apunte", "examen", "ejercicio", "no_academico"
4. Chunking: `RecursiveCharacterTextSplitter` con 512 tokens de chunk y 64 de overlap
5. Embedding: SentenceTransformer `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones) ejecutado localmente
6. Almacenamiento: colecciones de ChromaDB por sesión, con cosine distance

**Pipeline de lectura (Read path)**:
- `retrieve(query, collection_name, top_k=5, topic_filter)` → búsqueda por similitud semántica + filtro opcional de tópico por prefijo
- Weak-topic boosting: los chunks de tópicos con score < 6.0 reciben peso 2x en el retrieval

**ThematicIndex**: árbol jerárquico de temas con paths separados por "/", que permite merge profundo al ingerir incrementalmente múltiples documentos en una misma sesión.

**Nota**: imágenes (PNG/JPG) son rechazadas. La extracción OCR de matemática fue diferida post-MVP por limitaciones de tiempo.

#### 2.7 Guardrails y Validación

| Riesgo | Mecanismo | Ubicación |
|--------|-----------|-----------|
| Alucinación en preguntas de examen | Validación post-generación claim-level con cosine similarity. Cada afirmación debe tener un chunk fuente con score > umbral. Reintento hasta 3x; si falla, se omite la pregunta. | `exam_generator.py`, `validate_claim_grounding.py` |
| Loop infinito | Límite duro de 15 iteraciones por tarea. Al alcanzarlo, se termina y retorna resultado parcial. | `orchestrator.py:check_iteration_limit` |
| Contenido no académico | El Ingestor clasifica con LLM y rechaza documentos `no_academico` | `ingestor.py:classify_document` |
| Evaluaciones inconsistentes | LLM-as-judge: segunda pasada de evaluación en el 30% de los casos. Discrepancia > 2.0 puntos dispara revisión. | `evaluator.py:llm_judge` |
| Respuestas sin sentido (gibberish) | Heurística determinística de ratio de vocales + análisis de conjunto de caracteres. Respuestas no evaluables reciben score 0 y feedback explicativo. | `evaluator.py:check_evaluability` |
| Imágenes no soportadas | Rechazo con mensaje descriptivo. OCR diferido. | `ingestor.py:parse_document` |
| Errores de LLM provider | Manejo de excepciones con reintentos y propagación controlada al Orchestrator | `config.py`, todos los agentes |

#### 2.8 Observabilidad

El sistema usa **Langfuse** como plataforma de observabilidad:

- `ObservabilityManager` singleton con inicialización lazy del cliente Langfuse
- Decorador `@observe()` en todos los entry points de tools y operaciones RAG
- `CallbackHandler` de LangChain inyectado en los configs de los grafos
- `propagate_attributes()` para herencia de contexto sesión/usuario en spans anidados
- Tipos de span: `span`, `tool`, `embedding`, `retriever`
- Logging de: llamadas al LLM (prompts y respuestas), invocaciones a tools (entrada y salida), decisiones del Orchestrator (intención clasificada, pasos del plan)

### 3. Resultados

#### 3.1 Casos de prueba definidos

Los 12 casos de prueba requeridos por la PRD (sección 8) cubren:
- **5 happy path**: ingestión exitosa de PDF, generación de examen MCQ, generación de ejercicio open-answer, evaluación de respuesta correcta, consulta de perfil de progreso
- **4 edge cases**: documento sin estructura de apunte, pregunta fuera del material ingerido, respuesta vacía, sesión sin documentos ingeridos
- **3 adversarial**: prompt injection para generar contenido no académico, subida de archivo no-PDF, texto sin sentido como respuesta

Cobertura: 7/12 con modelos reales, 3/12 mock-only, 2/12 diferidos (OCR de imágenes).

Los 10 requisitos TXR (Epic 11, extracción de tópicos) están todos cubiertos por tests.

#### 3.2 Estrategia de testing

| Tier | Comando | Alcance |
|------|---------|---------|
| **Unit** | `pytest tests/ -v` | 105 tests en 22 archivos. LLMs/embeddings mockeados. |
| **Integration** | `pytest tests/ -v -m integration` | 16 tests con LLM real + embeddings real + PDF real |
| **E2E Mock** | `npx playwright test` | Full stack con seeds pre-grabadas (determinístico, <40s) |
| **E2E Live** | `E2E_LIVE_LLM=true npx playwright test --grep @live` | Real LLM calls con aserciones basadas en tolerancia |

**E2E tests (Playwright)**: 16 tests que ejercitan el flujo completo: browser → frontend → API → agentes. Modo mock para CI (rápido y determinístico), modo live para pre-defensa (detecta bugs de calidad de respuesta).

#### 3.3 Trazas representativas de ejecución

[Incluir capturas de pantalla o fragmentos de trazas de Langfuse mostrando]:
- Invocación al LLM del Orchestrator clasificando una intención y generando un plan
- Llamada a la tool `ingest_document`: input (archivo PDF), output (documento clasificado + chunks + colección ChromaDB)
- Llamada a `generate_exam`: input (tema, tipo, cantidad), output (preguntas generadas con referencias a chunks fuente)
- Llamada a `evaluate_answer`: ciclo completo con evaluación, feedback y actualización de perfil
- Span jerárquico en Langfuse mostrando la traza completa de una interacción

#### 3.4 API endpoints expuestos

| Endpoint | Método | Propósito |
|----------|--------|-----------|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Chat principal (classify intent → route → synthesize) |
| `/api/sessions` | GET/POST | Listar/crear sesiones |
| `/api/sessions/{id}` | GET/DELETE/PATCH | Sesión CRUD + rename |
| `/api/sessions/{id}/files` | GET | Archivos ingeridos en sesión |
| `/api/sessions/{id}/profile` | GET | Progreso por sesión |
| `/api/ingest` | POST (multipart) | Upload y procesamiento de documentos |
| `/api/exam/generate` | POST | Generar examen |
| `/api/exercise/generate` | POST | Generar ejercicio |
| `/api/evaluate` | POST | Enviar y evaluar respuestas |
| `/api/profile/{id}` | GET | Perfil del estudiante |
| `/api/profile/{id}/preferences` | PUT | Actualizar preferencias |
| `/api/students/{id}/dashboard` | GET | Datos del dashboard |

#### 3.5 Frontend — Páginas implementadas

| Ruta | Componentes clave |
|------|-------------------|
| `/` | ChatInput, ChatMessageList, ChatMessage, ExamWidget, SessionSidebar |
| `/exam` | ExamRenderer, ExamForm, QuestionNavigator |
| `/results` | EvaluationView con feedback por pregunta |
| `/dashboard` | StatsCards, TopicChart, WeakTopics, SessionHistory |
| `/settings` | Preferencias de examen configurables |
| Upload | UploadDropzone, UploadFileList, TopicTree, SessionFileList |

#### 3.6 Métricas agregadas

[NOTA: si hay métricas reales disponibles, incluirlas aquí. Si no, indicar que están pendientes de recolección en la ejecución final previa a la defensa.]
- Cantidad de pasos promedio por tarea del Orchestrator
- Tokens consumidos por tipo de operación
- Tiempo de respuesta promedio por endpoint
- Cobertura de tests

### 4. Conclusiones

Redactar conclusiones que aborden:

- **Logros**: implementación funcional de un sistema multi-agente completo con 6 agentes especializados, RAG funcional sobre material académico, evaluación automática con Chain-of-Thought, frontend web interactivo, pipeline completo de ingestion → generación → evaluación → seguimiento

- **Decisiones de diseño acertadas**:
  - Pipeline lineal en lugar de ReAct para ingestion y generación (más rápido, más confiable)
  - Chain-of-Thought para evaluación (proceso que naturalmente requiere múltiples pasos)
  - Support Agent sin LLM (determinístico, sin alucinaciones para consultas de datos)
  - Uso de SentenceTransformer local (sin dependencia de APIs externas para embeddings)
  - Multi-proveedor LLM (flexibilidad para usar modelos locales gratuitos o cloud)

- **Desviaciones respecto a la PRD**:
  - ReAct reemplazado por pipelines lineales en Ingestor, ExamGenerator y ExerciseGenerator
  - OCR de matemática diferido post-MVP
  - Algunos endpoints del frontend son stubs; dashboard con valores hardcodeados

- **Trabajo futuro**:
  - Implementar OCR para extracción de fórmulas matemáticas
  - Completar integración frontend-backend (endpoints pendientes)
  - Implementar ReAct en agentes que se beneficiarían de razonamiento iterativo
  - Agregar soporte para más formatos de documento (DOCX, HTML, URLs)
  - Implementar re-ranker en el pipeline RAG para mejorar precisión
  - Agregar más casos de prueba adversariales

- **Valoración general**: el sistema demuestra la viabilidad de agentes basados en LLMs para educación personalizada. La arquitectura modular permite extender funcionalidades sin reescribir componentes existentes. La combinación de RAG + evaluación automática + seguimiento de progreso ofrece una experiencia de aprendizaje completa.

### 5. Referencias

Incluir las referencias recomendadas del Anexo I más las relevantes al proyecto:

1. Russell, S., Norvig, P.: Artificial Intelligence: A Modern Approach. 4th edition, Pearson (2020)
2. Yao, S., et al.: ReAct: Synergizing Reasoning and Acting in Language Models. ICLR (2023)
3. Lewis, P., et al.: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS (2020)
4. Shinn, N., et al.: Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS (2023)
5. Schick, T., et al.: Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS (2023)
6. Wang, L., et al.: A Survey on Large Language Model based Autonomous Agents. Frontiers of Computer Science (2024)
7. Anthropic: Building Effective Agents (2024). https://www.anthropic.com/research/building-effective-agents
8. LangGraph Documentation. https://langchain-ai.github.io/langgraph/
9. LangChain Documentation. https://python.langchain.com/
10. ChromaDB Documentation. https://docs.trychroma.com/
11. Langfuse Documentation. https://langfuse.com/docs
12. SentenceTransformers Documentation. https://www.sbert.net/
---

## Fuentes de información del proyecto (para referencia del escritor)

### Documentos de planificación
- `docs/TP_2-Agente.pdf` — Consigna oficial del TP 2 (objetivo, etapas, criterios de evaluación, formato de entrega, Anexo I)
- `init_PRD.md` — Product Requirements Document: definición del problema, arquitectura, 6 agentes, flujos clave, criterios de aceptación
- `gap_analysis.md` — Análisis de gaps vs PRD (185 líneas): desviaciones documentadas, trabajo pendiente
- `rag.md` — Documentación detallada del pipeline RAG
- `tests_documentation.md` — Inventario de tests, cobertura de casos PRD, catálogo de fixtures
- `MANUAL_TEST_CHECKLIST.md` — Checklist de pruebas manuales

### Documentación de agentes (epics/)
- `epics/epic-01-orchestrator.md` — Plan de implementación del Orchestrator
- `epics/epic-02-ingestor.md` — Document ingestion + RAG setup
- `epics/epic-03-exam-generator.md` — Flujo de generación de exámenes
- `epics/epic-04-exercise-generator.md` — Generación de ejercicios
- `epics/epic-05-evaluator.md` — Evaluación de respuestas + scoring
- `epics/epic-06-support-agent.md` — Perfil de estudiante + progreso
- `epics/epic-07-ui.md` — Arquitectura del frontend
- `epics/epic-08-observability.md` — Integración Langfuse + test suite
- `epics/epic-09-profile-bootstrap.md` — Ciclo de vida de sesiones, perfiles por sesión
- `epics/epic-10-code-refactoring.md` — Arquitectura limpia, organización
- `epics/epic-11-topic-extraction.md` — Pipeline de extracción de tópicos
- `epics/epic-12-e2e-testing.md` — Playwright E2E, hardening de memoria
- `epics/epic-13-robustness.md` — Validación, propagación de errores, concurrencia

### Código fuente
- `back/src/agents/orchestrator.py` (1019 líneas) — 7 nodos, Plan-and-Execute
- `back/src/agents/ingestor.py` (306 líneas) — Pipeline lineal 3 nodos
- `back/src/agents/exam_generator.py` (693 líneas) — Pipeline 4 nodos con validación
- `back/src/agents/exercise_generator.py` (515 líneas) — Pipeline 4 nodos
- `back/src/agents/evaluator.py` (845 líneas) — 8 nodos, Chain-of-Thought, LLM-as-judge
- `back/src/agents/support.py` (325 líneas) — Reactivo, template-based
- `back/src/api/router.py` (733 líneas) — 14 endpoints FastAPI
- `back/src/config.py` (193 líneas) — Pydantic-settings, 5 providers LLM, modos E2E
- `back/src/memory/schema.py` (504 líneas) — Esquema SQLite, 5 tablas, migraciones
- `back/src/rag/` — Módulo RAG: chunking, embeddings, retrieval, ChromaDB
- `back/src/tools/` — 13 tools definidas como funciones Pydantic
- `back/src/observability/` — Langfuse: singleton, decoradores, span propagation
- `front/src/` — Next.js 15 App Router: 5 páginas, hooks, componentes
- `front/e2e/` — Playwright E2E tests

### Estados de implementación (según epics)
- Epics 01-06 y 08: DONE (agentes core + observabilidad)
- Epic 07 (UI): Draft
- Epic 09 (Profile Bootstrap): Active
- Epic 10 (Code Refactoring): DONE
- Epic 11 (Topic Extraction): Active
- Epic 12 (E2E Testing): Active
- Epic 13 (Robustness): Active

---

## Instrucciones finales para el escritor

1. **No inventes**: cada dato, métrica y afirmación debe provenir de la información aquí provista. Si un dato no está disponible, indícalo explícitamente como "pendiente de medición" o "no documentado".
2. **Estructura rigurosa**: seguí exactamente el orden del Anexo I. Si necesitás agregar subsecciones, hacelo dentro de las secciones principales (2. Solución es la más extensa).
3. **Figuras y tablas**: generá descripciones textuales de qué debe contener cada figura/tabla. Si no podés generar imágenes, describí el diagrama en texto y marcá `[FIGURA X: descripción]`.
4. **Tono académico**: tercera persona, voz pasiva cuando corresponda, terminología técnica precisa.
5. **Extensión**: el informe completo debe ocupar 15-20 páginas en el formato especificado.
6. **Referencias**: todas las referencias del Anexo I deben aparecer. Agregá solo referencias que efectivamente se usen en el texto.
7. **Revisión**: antes de entregar, verificá que cada criterio de evaluación de la consigna (sección "Criterios de evaluación") esté cubierto en el informe.
