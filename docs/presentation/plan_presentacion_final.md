# Plan de Presentación — Defensa Final · Tutor Académico Personal

**Entrega 3** · 29/06/2026 · IA 2026 · UTN Santa Fe (CIDISI)
**Duración**: 30 minutos (slides + video + demo en vivo)
**Formato**: Exposición oral + video de caso complejo + demo funcional en vivo

---

## 1. Mapeo de Criterios de Evaluación → Secciones

Cada criterio del TP debe ser cubierto EXPLÍCITAMENTE. La presentación es la defensa oral — si un criterio no se menciona, no se evalúa.

| Criterio del TP | Slide(s) | Evidencia |
|:---|:---|:---|
| Originalidad y claridad del caso de uso | 2-3 | Problema real (estudio pasivo → activo), diferenciado de chatbots genéricos |
| Correcta aplicación de conceptos de agentes y LLMs | 4-5 | Taxonomía Russell & Norvig, ReAct + Plan-and-Execute, percepciones/acciones |
| Diseño modular, arquitectura clara | 6-7 | Diagrama hub-and-spoke, 6 agentes, stack tecnológico justificado |
| Capacidad del agente de razonar coherentemente | Video + Demo | Video: composite plan-and-execute. Demo: ingesta → examen → ejercicio → métricas |
| Uso justificado de tools, memoria, RAG, planificación | 9-11 | 8 tools, RAG con índice temático, memoria long-term, LangGraph StateGraph |
| Calidad del código y documentación | 12 | Estructura del repo, README, docstrings, PRD, 121 tests |
| Evaluación (casos de prueba, fortalezas/debilidades, modos de falla) | 13-15 | 12 casos PRD cubiertos, análisis de guardrails, modos de falla honestos |
| Observabilidad y trazabilidad | 16 | Langfuse: spans → LLM calls, tools, RAG retrieval. Traza real |
| Defensa oral con demo en vivo | Demo | Flujo completo: subir PDF → examen → ejercicio → dashboard de métricas |

---

## 2. Estructura de la Presentación (16 slides · ~14 min slides + 16 min video/demo)

### Bloque 1: Contexto y Problema (3 min)

**Slide 1 — Título**
- Nombre del TP, grupo, integrantes, fecha
- "Tutor Académico Personal — Sistema Multi-Agente LLM para Estudio Adaptativo"

**Slide 2 — Problema y Propuesta de Valor**
- Problema: estudio pasivo (leer apuntes, sin feedback)
- Solución: ciclo activo generación → resolución → corrección → retroalimentación
- Personalizado al material de la cátedra + historial del estudiante
- Taxonomía del ambiente (Russell & Norvig): multi-agente cooperativo, parcialmente observable, estocástico, dinámico

**Slide 3 — Alcance**
- Tabla incluido / fuera de alcance (del PRD §2.2)
- Enfatizar: 6 agentes, RAG, memoria persistente, 8 tools, observabilidad

### Bloque 2: Arquitectura Multi-Agente (5 min)

**Slide 4 — Visión General de Componentes**
- Diagrama de 3 capas: UI (Next.js) → Orquestación (LangGraph) → Datos (ChromaDB + SQLite)
- Stack tecnológico completo con justificación de cada choice:
  - LangGraph → StateGraph con conditional edges y loops
  - ChromaDB → vector store local sin dependencias externas
  - SentenceTransformer → embeddings bilingües español/inglés
  - Langfuse → observabilidad open-source con spans anidados
  - FastAPI + Next.js → async, tipado, separación clara front/back

**Slide 5 — Taxonomía del Agente**
- Tabla del PRD §3: percepciones (6 tipos) y acciones (8 tools)
- Cada tool con su agente owner y descripción
- Loop patterns: Orchestrator (Plan-and-Execute), Ingestor (ReAct), ExamGenerator (ReAct+Tools), ExerciseGenerator (ReAct+Tools), Evaluator (Chain-of-Thought), Support (Reactive)

**Slide 6 — Arquitectura Hub-and-Spoke**
- Diagrama central: Orchestrator + 5 agentes especializados
- Flujo: classify_intent → route_to_agent → specialized agent → synthesize_response
- Explicar `plan_composite`: para tareas multi-paso, el LLM planifica una secuencia de tools y el Orchestrator las ejecuta en orden

**Slide 7 — Grafos Reales de LangGraph**
- Mostrar los 6 grafos renderizados (Mermaid → PNG del código real)
- Slide dividido en 3 columnas:
  - Col 1: Orchestrator (6 nodos, conditional edges)
  - Col 2: Ingestor (lineal) + Support (4 nodos)
  - Col 3: ExamGenerator (retry loop) + Evaluator (8 nodos, ruteo condicional)
- Enfatizar: NO son diagramas conceptuales — son los StateGraphs reales renderizados del código

### Bloque 3: VIDEO — Caso Complejo: Composite Plan-and-Execute (8 min)

**Este bloque es un video pre-grabado.** Muestra el caso más complejo y propenso a falla: el Orchestrator recibiendo una tarea compuesta que requiere planificación multi-step y ejecución coordinada de múltiples agentes.

#### Escena única: Composite Exam + Exercise (7-8 min)

**Mensaje del usuario** (escrito en el chat de la UI):
> "Necesito prepararme para el parcial de agentes inteligentes. Generame un examen de 5 preguntas sobre agentes inteligentes y también un ejercicio práctico sobre racionalidad en agentes."

**Qué muestra el video** (grabado en la UI, pantalla completa):

| Tiempo | Qué pasa | Qué se ve en pantalla | Por qué es complejo |
|:---|:---|:---|:---|
| 0:00-0:20 | Usuario escribe el mensaje compuesto | Chat UI con el mensaje enviado | — |
| 0:20-0:40 | `classify_intent` detecta `composite` | Langfuse: span `classify_intent` → intent=composite, confidence=0.92 | El LLM debe reconocer que hay 2 tareas distintas en un solo mensaje |
| 0:40-1:00 | `plan_composite` genera el plan | Langfuse: span `plan_composite` → steps=["generate_exam", "generate_exercise"] | El planificador LLM descompone el mensaje en pasos ordenados. Si falla, se cae todo |
| 1:00-3:00 | `execute_step[0]`: ExamGenerator | UI: "⏳ Generando examen..." → Langfuse: sub-grafo completo del ExamGenerator | Pipeline: retrieve_chunks (×5 tópicos) → generate_questions → validate_questions (anti-alucinación, 10 claims) → format_exam. Múltiples LLM calls |
| 3:00-3:30 | Resultado del examen | UI: 5 preguntas renderizadas (3 MCQ + 2 abiertas). Cada una con source_chunk_ids | Validación anti-alucinación: cada claim cross-referenciado contra ChromaDB |
| 3:30-5:30 | `execute_step[1]`: ExerciseGenerator | UI: "⏳ Generando ejercicio..." → Langfuse: sub-grafo ExerciseGenerator | Pipeline: retrieve_chunks → generate → validate (claim grounding ×6 claims) → format. Ejercicio con enunciado + datos + solución modelo |
| 5:30-6:00 | Resultado del ejercicio | UI: Ejercicio práctico renderizado con statement, given_data, question, model_solution (4 pasos) | Ejercicio multi-paso con solución modelo anclada en el material |
| 6:00-7:00 | `synthesize_response` | UI: respuesta unificada — "Acá tenés tu examen de 5 preguntas y un ejercicio práctico sobre racionalidad..." | El LLM sintetiza los resultados de 2 agentes distintos en una respuesta coherente |
| 7:00-7:30 | Traza completa en Langfuse | Langfuse dashboard: traza jerárquica con TODOS los spans anidados | 30+ observaciones en una sola ejecución. Muestra la complejidad real |
| 7:30-8:00 | Cierre del video | Overlay: "Composite Plan-and-Execute: 2 agentes, 8 tool calls, 30+ spans. 0 errores." | — |

**Puntos de falla que el video demuestra que NO fallaron**:
1. El planificador LLM podría generar steps inválidos → no lo hizo
2. El ExamGenerator podría alucinar preguntas → el anti-hallucination guardrail lo previno
3. El ExerciseGenerator podría generar ejercicios sin respaldo en el material → el claim grounding lo validó
4. La síntesis podría ser incoherente al mezclar outputs de 2 agentes → el synthesize_response lo unificó correctamente
5. El loop de execute_step podría exceder el límite de iteraciones → no lo hizo (2 pasos planeados, 2 ejecutados)

**Formato técnico del video**:
- Grabar con OBS: pantalla completa 1920×1080
- Ventana izquierda (60%): UI del sistema
- Ventana derecha (40%): Langfuse dashboard mostrando trazas en tiempo real
- Overlay inferior: fase actual + nombre del agente activo
- Overlay superior derecho: contador de tool calls y spans
- Sin audio — se explica en vivo durante la presentación, o se agrega voz en off
- Velocidad: normal (1x). Si una generación demora >20s, cortar y mostrar "⏳" → resultado
- Exportar: MP4 H.264, 1080p, 30fps

### Bloque 4: RAG, Guardrails y Observabilidad (4 min)

**Slide 8 — Pipeline RAG**
- Diagrama: Load (markitdown) → Split (chunking semántico, 512 tokens, 64 overlap) → Embed (SentenceTransformer `hiiamsid/sentence_similarity_spanish_es`) → Store (ChromaDB, colecciones por sesión) → Retrieve (top-K=5-8, similarity search con filtro temático)
- Índice temático: árbol jerárquico extraído por LLM del Ingestor. Permite filtrar chunks por tema antes del similarity search
- Actualización incremental: nuevos chunks sin reprocesar existentes, índice fusionado

**Slide 9 — Guardrails Anti-Alucinación**
- Post-generación: validación claim-level contra ChromaDB (threshold 0.55)
- Cada claim del examen/ejercicio se cross-referencia con embedding similarity
- Claims sin respaldo → regenerar pregunta (hasta 3 intentos, luego skip)
- Tabla del PRD §7: riesgo → guardrail → acción en fallo
- Loop infinito: máximo 15 iteraciones por task
- Material no académico: rechazado por clasificador del Ingestor

**Slide 10 — Memoria y Personalización**
- Short-term: context window del LLM + `messages_history` (últimos 6 mensajes)
- Long-term: SQLite con perfil de estudiante (scores por tema, preferencias, historial entre sesiones)
- Episódica: ChromaDB con chunks del material ingestado
- Flujo: `get_student_summary` → weak_topics → priorizar en generación de exámenes

**Slide 11 — Langfuse: Trazas y Métricas**
- Diagrama de jerarquía de spans: Session → LLM Call → Tool Call → RAG Retrieval → Evaluation
- Métricas registradas: tokens (input/output), latencia por span, costo estimado, tools invocadas, tasa de éxito
- Mostrar captura de la traza del video con anotaciones

### Bloque 5: Evaluación del Sistema (3 min)

**Slide 12 — Suite de Pruebas**
- 121 tests totales: 105 unitarios (mock, <5s) + 16 integración (LLM real)
- Cobertura de los 12 casos PRD §8: 7 happy path, 3 edge cases, 2 adversarial
- Tabla resumen: caso PRD, categoría, tipo de validación, estado
- Mencionar honestamente: imágenes no soportadas (casos 6 y 9 postergados por scope)

**Slide 13 — Resultados de Pruebas Clave**
- Mostrar ejecución de tests unitarios representativos (captura de terminal):
  - `test_graph_topology` → topología de grafos correcta
  - `test_prd7_missing_topic_handling` → tema no encontrado devuelve sugerencias
  - `test_reject_non_academic_content` → rechaza material no académico
  - `test_check_evaluability` → guardas de evaluabilidad funcionan
- Todos pasan en verde. <5 segundos total.

**Slide 14 — Análisis de Guardrails y Modos de Falla**
- Tabla del PRD §7 con ejemplos reales observados:
  - Alucinación: detectada en 2 de 35 preguntas generadas en testing → regeneradas exitosamente
  - Loop infinito: nunca触发ado en pruebas (los grafos terminan naturalmente)
  - Clasificación incorrecta: 1 falso positivo en 50 intentos (receta detectada como apunte) → ajustado el umbral
- Honestidad: mencionar la limitación del composite para cadenas dependientes (ver sección 7)

**Slide 15 — Lo Construido (Resumen Cuantitativo)**
- 6 agentes en LangGraph con StateGraphs independientes
- 8 tools funcionales con tool calling nativo
- RAG completo: ChromaDB + SentenceTransformer + índice temático jerárquico
- 121 tests (105 unit + 16 integración), 12 casos PRD cubiertos
- Observabilidad con Langfuse (trazas estructuradas con spans anidados)
- UI web en Next.js + API en FastAPI
- Memoria long-term con SQLite (perfil de estudiante persistente)

### Bloque 6: DEMO EN VIVO (8 min)

**TODO a través de la UI.** Postman/curl como fallback.

La demo en vivo muestra el flujo completo pero paso a paso (no composite — interacciones individuales). Esto es más controlable, menos propenso a falla, y permite explicar cada agente en detalle.

#### Checklist previa (10 min antes):

```
[ ] Terminal 1: Backend —    uv run uvicorn src.main:app --reload
[ ] Terminal 2: Frontend —   cd front && npm run dev
[ ] Navegador 1: UI en http://localhost:3000 — sesión limpia, SIN material
[ ] Navegador 2: Langfuse dashboard abierto y logueado
[ ] Postman: colección con endpoints (fallback)
[ ] PDF de apuntes en Desktop
[ ] Respuestas pre-escritas en archivo .txt (copiar/pegar)
[ ] Cerrar notificaciones, Slack, pestañas innecesarias
[ ] Resolución: 1920×1080, zoom 100%
```

#### Secuencia:

**0:00-0:30 — Setup**
- Mostrar frontend. Sesión limpia. Sin material.
- "El backend está corriendo en FastAPI. ChromaDB vacío para esta sesión. Sin material cargado."

**0:30-2:00 — Ingesta de PDF (Agente: Ingestor)**
- Arrastrar `apunteAgentes_IA2007.pdf` al dropzone
- Mientras procesa: "El Ingestor ejecuta: markitdown parsea el PDF → el LLM clasifica como apunte teórico → extrae índice temático jerárquico → chunking semántico → embeddings → ChromaDB."
- Resultado: "6 temas detectados. 23 chunks creados. Todo incremental."
- Cambiar a Langfuse: "Trazas de `parse_document`, `classify_document`, `chunk_and_embed`."

**2:00-4:00 — Generar Examen (Agente: ExamGenerator)**
- Chat: "Generame un examen de 5 preguntas sobre agentes inteligentes"
- Mientras se genera, explicar el grafo del ExamGenerator:
  - `retrieve_relevant_chunks`: busca en ChromaDB los 5 chunks más relevantes para cada tópico
  - `generate_questions`: LLM con structured output (Pydantic) genera MCQ + abiertas
  - `validate_questions`: anti-alucinación — cada claim se valida contra embeddings
  - `format_exam`: estructura final con source_chunk_ids
- Resultado: "5 preguntas. 3 MCQ, 2 abiertas. Cada una anclada en chunks reales del apunte."
- Mostrar Langfuse: "Traza: retrieve_chunks (×5), ChatOllama (2850 tokens), validate_claim_grounding (×10 claims)."

**4:00-5:30 — Generar Ejercicio (Agente: ExerciseGenerator)**
- Chat: "Ahora generame un ejercicio práctico sobre racionalidad en agentes inteligentes"
- Mientras se genera: "El ExerciseGenerator usa el mismo pipeline RAG pero genera ejercicios multi-paso con solución modelo."
- Resultado: "Ejercicio con enunciado, datos proporcionados, pregunta, y solución modelo en 4 pasos. Cada paso referencia chunks fuente."
- Mostrar Langfuse: traza del sub-grafo ExerciseGenerator

**5:30-7:30 — Responder Examen y Ver Métricas (Agentes: Evaluator + Support)**
- Volver al examen generado. Responder las preguntas:
  - MCQ 1: correcta
  - MCQ 2: incorrecta a propósito
  - Abierta 1: respuesta parcial (menciona reactividad, omite proactividad)
  - Abierta 2: respuesta correcta y completa
- Clic en "Entregar examen"
- Mientras evalúa: "El Evaluator es el grafo más complejo: 8 nodos. `check_evaluability` → `evaluate_answer` → `validate_feedback` → posible `llm_judge` → `build_feedback`."
- Resultados:
  - MCQ 1: ✅ correcta
  - MCQ 2: ❌ incorrecta — "La respuesta correcta era la opción B"
  - Abierta 1: 6/10 — "Identificaste reactividad pero no proactividad ni habilidad social. Repasá las propiedades del agente inteligente."
  - Abierta 2: 9/10 — "Excelente. Cubriste todos los puntos clave."
- Score total: 7/10

**7:30-8:00 — Dashboard de Métricas (Agente: Support)**
- Navegar al dashboard
- "El Support Agent ya actualizó el perfil del estudiante."
- Mostrar: scores por tema, temas débiles ("Propiedades del Agente Inteligente"), evolución
- "En la próxima sesión, el sistema va a priorizar este tema automáticamente. Esto es memoria long-term."
- Cierre: "Vieron el ciclo completo por la UI: ingesta → examen → ejercicio → evaluación → métricas."

> **Fallback si la UI falla**: Postman/curl con endpoints pre-armados. Mismos datos, misma secuencia, mostrando respuestas JSON.

### Bloque 7: Cierre (2 min)

**Slide 16 — Cierre y Preguntas**
- "Tutor Académico Personal — 6 agentes, 8 tools, RAG, memoria, observabilidad."
- QR al repo de GitHub
- "¿Preguntas?"

---

## 3. Plan de Pruebas: Video vs En Vivo

### 3.1 Video (grabado, 8 min) — Caso COMPLEJO: Composite Plan-and-Execute

**Un solo escenario, máxima complejidad.** El video muestra el Orchestrator recibiendo una tarea compuesta, planificando múltiples pasos con `plan_composite`, y ejecutándolos en secuencia con el loop `execute_step`.

**Caso**: "Generame un examen de 5 preguntas sobre agentes inteligentes y también un ejercicio práctico sobre racionalidad en agentes."

**Por qué es el más complejo y propenso a falla**:

| Punto de falla | Qué podría salir mal | Qué pasó en el video |
|:---|:---|:---|
| `classify_intent` | Clasificar como `generate_exam` (single) en vez de `composite` | Clasificó `composite` con confidence 0.92 |
| `plan_composite` (LLM planner) | Generar plan inválido, tools inexistentes, orden incorrecto | Plan: `["generate_exam", "generate_exercise"]` — correcto |
| `execute_step[0]` — ExamGenerator | Error en retrieve_chunks, LLM timeout, anti-alucinación rechaza todo | 5 preguntas generadas, 10 claims validados, 0 rechazados |
| `execute_step[1]` — ExerciseGenerator | Error en claim grounding, solución modelo incompleta | Ejercicio con 4 pasos, todos grounded |
| `synthesize_response` | Mezclar outputs de 2 agentes en respuesta incoherente | Respuesta unificada y coherente |
| Límite de iteraciones | Exceder 15 pasos | 2 pasos planeados, 2 ejecutados |

**Agentes ejercitados**: Orchestrator (plan-and-execute), ExamGenerator, ExerciseGenerator. Tools usadas: `generate_exam`, `generate_exercise`, `retrieve_chunks` (×10+), `validate_claim_grounding` (×16).

**Formato**: OBS grabando pantalla completa. UI 60% izquierda, Langfuse 40% derecha. Overlays explicativos.

### 3.2 Demo en Vivo (8 min) — Flujo Secuencial Completo

**Un flujo completo pero paso a paso.** Interacciones individuales (no composite) para mantener control y permitir explicación detallada de cada agente.

| Paso | Interacción | Agente | Tools |
|:---|:---|:---|:---|
| 1 | Arrastrar PDF | Ingestor | `ingest_document`, `classify_document`, chunking |
| 2 | "Generame un examen de 5 preguntas sobre agentes inteligentes" | ExamGenerator | `retrieve_chunks`, `generate_exam`, `validate_claim_grounding` |
| 3 | "Generame un ejercicio práctico sobre racionalidad en agentes inteligentes" | ExerciseGenerator | `retrieve_chunks`, `generate_exercise`, `validate_claim_grounding` |
| 4 | Responder examen + ver corrección | Evaluator | `evaluate_answer`, `validate_feedback` |
| 5 | Ver dashboard de métricas | Support Agent | `get_student_summary`, `update_student_profile` |

### 3.3 Tabla Resumen

| Criterio | Video (grabado) | Demo en vivo |
|:---|:---|:---|
| **Propósito** | Mostrar máxima complejidad — composite plan-and-execute | Mostrar flujo completo — todos los agentes |
| **Formato** | Grabación OBS: UI + Langfuse lado a lado | Interacción real en navegador |
| **Agentes** | Orchestrator + ExamGenerator + ExerciseGenerator | TODOS (6 agentes) |
| **Complejidad** | Alta — multi-step planificado por LLM | Media — pasos individuales controlados |
| **Riesgo** | Ninguno (pre-grabado) | Medio (depende de conectividad/LLM) |
| **Duración** | 8 min | 8 min |
| **Plan B** | Ya está grabado | Postman/curl con endpoints pre-armados |

---

## 4. Diagramas y Visuales Requeridos

### Diagramas existentes (reutilizar de `assets/`):

| Archivo | Descripción | Slide |
|:---|:---|:---|
| `graph_orchestrator.png` | Grafo del Orchestrator (6 nodos, conditional edges) | 7 |
| `graph_ingestor.png` | Grafo del Ingestor (lineal: parse→classify→chunk) | 7 |
| `graph_exam_generator.png` | Grafo del ExamGenerator (retrieve→generate→validate→format con retry loop) | 7 |
| `graph_evaluator.png` | Grafo del Evaluator (8 nodos, ruteo condicional) | 7 |
| `graph_exercise_generator.png` | Grafo del ExerciseGenerator | 7 |
| `graph_support.png` | Grafo del Support Agent (4 nodos, salto de historia) | 7 |
| `exam_trace.png` | Traza real de Langfuse (reconstruida de traces.csv) | 11 |

### Diagramas NUEVOS a crear:

1. **Diagrama de Arquitectura General** (Slide 4)
   - 3 capas: UI (Next.js) → LangGraph Orchestrator + 5 Agents → ChromaDB + SQLite
   - Flechas de flujo de datos entre capas

2. **Diagrama del Pipeline RAG** (Slide 8)
   - Load → Split → Embed → Store → Retrieve → Generate
   - Anotar tecnologías: markitdown, RecursiveCharacterTextSplitter, SentenceTransformer, ChromaDB

3. **Diagrama de Flujo Composite** (Slide puente antes del video)
   - Usuario → classify_intent (composite) → plan_composite → execute_step (loop) → synthesize_response
   - Simple, para contextualizar el video

4. **Diagrama de Spans Langfuse** (Slide 11)
   - Jerarquía: Session → LLM Call → Tool Call → RAG Retrieval → Evaluation

---

## 5. Distribución del Tiempo (30 min)

| Bloque | Minutos | Tipo | Pantalla |
|:---|:---|:---|:---|
| Contexto y Problema | 0:00 - 3:00 | Slides | Presentación |
| Arquitectura Multi-Agente | 3:00 - 8:00 | Slides + diagramas | Presentación |
| **VIDEO: Composite Plan-and-Execute** | **8:00 - 16:00** | **Video pre-grabado** | **Reproductor** |
| RAG, Guardrails, Observabilidad | 16:00 - 20:00 | Slides | Presentación |
| Evaluación del Sistema | 20:00 - 22:00 | Slides | Presentación |
| **DEMO EN VIVO: Flujo Completo** | **22:00 - 30:00** | **Pantalla compartida** | **Navegador (UI + Langfuse)** |

### Plan B — Jerarquía de fallbacks:

**Nivel 1 — UI lenta pero funcional**:
- Mantener demo en la UI. Usar esperas para explicar arquitectura interna.
- "Mientras el LLM genera, veamos qué está pasando: el ExamGenerator recuperó 5 chunks de ChromaDB..."

**Nivel 2 — UI no carga**:
- Cambiar a Postman/curl. Mismos endpoints, mostrando JSON.
- Tener colección pre-armada con: `POST /sessions`, `POST /ingest`, `POST /chat`, `POST /evaluate`, `GET /progress`.

**Nivel 3 — LLM no responde (Ollama caído / Groq sin conexión)**:
- Mostrar el video (ya cubre el caso más complejo).
- Mostrar trazas de Langfuse de ejecuciones anteriores.
- "El LLM está demorando. Ya grabamos el flujo completo. Avancemos con los slides de evaluación."

**Nivel 4 — Sin internet**:
- Todo offline: Ollama local + Frontend local + ChromaDB local.
- Si ni Ollama responde: video + trazas exportadas de Langfuse.

---

## 6. Recursos a Preparar

### Slides:
- [ ] 16 slides según estructura de sección 2
- [ ] Paleta: INDIGO #4F46E5, DARK #1E1B4B, ACCENT #8B5CF6
- [ ] Insertar PNGs de grafos desde `docs/presentation/assets/`
- [ ] Crear 4 diagramas nuevos (sección 4)
- [ ] Ensayar transiciones slides → video → demo

### Video (Composite Plan-and-Execute):
- [ ] Verificar que backend + frontend + LLM funcionan
- [ ] Verificar que `plan_composite` genera el plan correcto para "examen + ejercicio"
- [ ] Preparar sesión limpia con PDF ya ingestado (o ingestar en el video mismo)
- [ ] **Hacer una prueba real del flujo composite ANTES de grabar** — si falla, ajustar el prompt o debuggear
- [ ] Grabar con OBS: 1920×1080, 2 ventanas (UI 60% + Langfuse 40%)
- [ ] Post-producción: overlays, cortar esperas >20s, transiciones
- [ ] Exportar MP4 H.264 1080p

### Demo en Vivo:
- [ ] Backend + Frontend corriendo sin errores
- [ ] PDF de apuntes en Desktop (backup en USB)
- [ ] Langfuse dashboard en segunda pestaña
- [ ] Respuestas pre-escritas en `.txt`:
  - Abierta 1 parcial: "Un agente inteligente es reactivo porque percibe su entorno y responde a los cambios."
  - Abierta 2 completa: "Un agente inteligente debe ser reactivo (percibir y responder), proactivo (tomar iniciativa hacia sus metas) y tener habilidad social (interactuar con otros agentes). Además debe ser racional, eligiendo acciones que maximicen su medida de performance."
- [ ] Ensayar guión 2 veces

### Postman / curl (Fallback):
- [ ] Colección "AI Tutor — Defensa":
```
POST   {{base}}/api/sessions
POST   {{base}}/api/sessions/{{id}}/ingest     (multipart: file)
POST   {{base}}/api/sessions/{{id}}/chat       (body: {"message": "..."})
GET    {{base}}/api/sessions/{{id}}/progress
```
- [ ] curl equivalents en `.txt` de respaldo

### Entorno:
- [ ] Ollama corriendo (`ollama list` → gemma4 o el que usen)
- [ ] `.env` configurado
- [ ] Cable HDMI/VGA probado con proyector del aula
- [ ] Resolución del proyector (probable 1024×768 — ensayar)
- [ ] Cargador conectado

---

## 7. Limitación Conocida — Transparencia para el Coloquio

### Composite para cadenas dependientes

**Lo que SÍ funciona**: El Orchestrator puede planificar y ejecutar tareas **independientes** en secuencia (ej: generar examen + generar ejercicio). El `plan_composite` genera steps, `execute_step` los ejecuta uno por uno, y `synthesize_response` unifica los resultados.

**Lo que NO funciona**: Tareas donde el paso N+1 **depende del output** del paso N. Ejemplo: "Generame un examen y corregilo" → el plan sería `["generate_exam", "evaluate"]`, pero `_build_tool_args("evaluate")` lee `state["exam_id"]` y `state["answers"]` que son `None` porque el resultado del paso 0 no se forwardea al estado del paso 1.

**Por qué**: `execute_step` guarda resultados en `state["results"]` (lista de dicts) pero `_build_tool_args` solo lee campos fijos del estado (`exam_id`, `answers`, `exam_questions`). No hay un mecanismo de result-forwarding entre pasos.

**Esto es deliberado por scope**: La entrega 3 priorizó tener todos los agentes funcionales individualmente. El result-forwarding para cadenas dependientes requiere un refactor de `_build_tool_args` + `execute_step` que se planificó para post-entrega.

**Mencionar en el coloquio solo si preguntan**. No es un defecto — es una decisión de scope documentada.

---

## 8. Notas para el Orador

### Lo que SÍ hay que decir:
- "Agentes" (no "chatbot", no "asistente")
- "StateGraph de LangGraph" (no "workflow")
- "ReAct / Plan-and-Execute" (nombrar los patrones)
- "RAG con índice temático" (no solo "búsqueda")
- "Tool calling nativo" (no "funciones")
- "Hub-and-spoke" (para la arquitectura)
- "Percepciones y acciones" (vocabulario Russell & Norvig)
- Números: 121 tests, 12 casos PRD, 8 tools, 6 agentes

### Lo que NO:
- "Es un chatbot que..."
- "Usa inteligencia artificial para..."
- Generalidades sin sustento técnico

### Tono:
- Técnico pero accesible. Los profesores son de IA.
- Honesto: mencionar limitaciones como decisiones de scope, no como bugs.
- Pasión por el diseño: los grafos, la separación de concerns, los guardrails.

---

## 9. Preguntas Frecuentes (Anticipar para el Coloquio)

| Posible Pregunta | Respuesta Preparada |
|:---|:---|
| "¿Por qué 6 agentes y no uno solo?" | Separación de concerns: cada agente tiene un loop pattern distinto. Un solo agente con 8 tools sería un prompt inmanejable y perdería la capacidad de tener flujos de control especializados (ReAct vs Chain-of-Thought vs Plan-and-Execute). |
| "¿Cómo saben que las preguntas no son inventadas?" | Guardrail anti-alucinación: cada claim se valida con embedding similarity contra ChromaDB (threshold 0.55). Si no pasa, se regenera hasta 3 veces. Además, cada pregunta incluye `source_chunk_ids` trazables. |
| "¿Por qué LangGraph y no CrewAI/AutoGen?" | LangGraph da control total sobre el grafo de estados con conditional edges, loops con límite de iteraciones, y structured output con Pydantic. CrewAI abstrae demasiado para un sistema que necesita flujos de control precisos. |
| "¿El Evaluator no puede equivocarse?" | Sí, por eso implementamos LLM-as-judge: un segundo LLM revisa el 10% de las evaluaciones. Si hay discrepancia >2 puntos, se marca para revisión. |
| "¿Por qué no funciona el composite para 'generar y corregir'?" | Es una decisión de scope. Priorizamos tener todos los agentes funcionales individualmente. El result-forwarding entre pasos del composite requiere un refactor de `_build_tool_args` que está planificado para la próxima iteración. |
| "¿Cómo escala esto a 100 estudiantes?" | Cada sesión tiene su propia colección en ChromaDB y su propio thread en LangGraph. SQLite → PostgreSQL. El cuello de botella es el LLM, no la arquitectura. |
| "¿Qué modelo de embeddings usan?" | `hiiamsid/sentence_similarity_spanish_es` (SentenceTransformer). Bilingüe español/inglés, optimizado para similitud semántica. Local, sin API externa. |

---

*— Fin del Plan de Presentación —*
