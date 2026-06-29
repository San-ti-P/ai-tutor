# Evaluación del Agente — Tutor Académico Personal

**Trabajo Práctico N°2 · Inteligencia Artificial 2026 · UTN Santa Fe — CIDISI**

Este documento responde a la consigna *"Evaluación del agente"* del TP: define cómo
probamos que el agente funciona, mediante un conjunto acotado de casos de prueba que
cubren los tres escenarios requeridos —(a) happy path, (b) casos límite o ambiguos y
(c) entradas adversariales o fuera de dominio— con su comportamiento esperado y el
comportamiento observado registrado.

---

## 1. Metodología

- **Tamaño del conjunto.** 11 casos de prueba (dentro del rango 5–15 sugerido),
  balanceados en las tres categorías: 4 happy path, 3 casos límite, 4 adversariales.
- **Validación determinística para tools.** Como exige la consigna, toda invocación a
  herramientas (ingesta, generación de examen, recuperación RAG, guardrails) se valida de
  forma determinística: conteos en ChromaDB, estructura JSON, flags de rechazo, presencia
  de `source_chunk_ids`, mensajes de error con sugerencias. 8 de los 11 casos son
  puramente determinísticos.
- **LLM-as-judge solo para calidad subjetiva.** Únicamente 2 casos (evaluación de
  respuesta correcta y parcialmente correcta) usan al LLM como juez de la calidad del
  score/feedback, que es justamente el uso permitido por la consigna. El sistema además
  incorpora LLM-as-judge en producción (segundo pasaje que valida al Evaluator, con flag
  `requires_review` ante discrepancias > 2 puntos).
- **Modelos reales vs. mock.** Cuando existe versión con modelos reales la usamos para la
  evaluación y la defensa en vivo (LLM Ollama Cloud `gemma4:31b-cloud`, embeddings
  `SentenceTransformer` reales, ChromaDB real, PDF académico real). Los casos con mock
  validan la lógica de grafo/estado de forma rápida y reproducible en cada commit.
- **Trazabilidad.** Cada ejecución con modelos reales emite trazas estructuradas
  (Langfuse: spans de LLM call, tool call y RAG retrieval) inspeccionables paso a paso
  (ver §4 y `docs/observability/`).

### Material de prueba

Todos los casos con modelos reales operan sobre un PDF académico real,
`back/tests/fixtures/apunteAgentes_IA2007.pdf` (apunte de teoría de agentes inteligentes),
ingerido a ChromaDB mediante el pipeline real.

### Reproducir la suite

```bash
cd back

# Casos determinísticos (rápidos, corren en cada commit, sin dependencias externas)
uv run pytest tests/ -v

# Casos con modelos reales (requiere proveedor LLM configurado en back/.env)
uv run pytest tests/ -m integration -v
```

---

## 2. Conjunto de casos de prueba

> **Estado de ejecución:** los 11 casos fueron ejecutados y **todos pasan**
> (8 determinísticos + 7 invocaciones a modelos reales). El comportamiento observado
> registrado abajo proviene de esa corrida.

### (a) Escenarios principales — Happy path

#### Caso 1 — Ingestar un PDF de apuntes bien formateado *(PRD #1)*

- **Test:** `test_ingest_real_pdf` + `test_classify_real_pdf` (`tests/test_ingestor.py`)
- **Validación:** Determinística (conteo ChromaDB + clasificación) · Modelos reales ✅
- **Comportamiento esperado:** El Ingestor parsea el PDF con markitdown, lo clasifica como
  `apunte_teorico` y puebla una colección de ChromaDB con chunks (count > 0).
- **Comportamiento observado:** ✅ El pipeline completo puebla la colección con chunks
  (count > 0) y el LLM clasifica correctamente el documento como `apunte_teorico`.

#### Caso 2 — Generar un examen de 5 preguntas sobre un tema específico *(PRD #2)*

- **Test:** `test_generate_exam_from_real_pdf` (`tests/test_exam_generator.py`) ·
  refuerzo con mock `test_prd2_happy_path_5_questions`
- **Validación:** Determinística (cantidad + estructura JSON) · Modelos reales ✅
- **Comportamiento esperado:** Se genera un examen con MCQ y preguntas abiertas; cada
  pregunta referencia chunks del material (`source_chunk_ids`); estructura JSON válida.
- **Comportamiento observado:** ✅ El examen se genera con estado `complete` (o `partial`
  si el guardrail anti-alucinación descarta alguna pregunta); cada pregunta incluye
  `source_chunk_ids`, las MCQ traen `prompt`/`options`/`correct_option_index` y las
  abiertas traen `base_answer`.

#### Caso 3 — Evaluar una respuesta correcta *(PRD #3)*

- **Test:** `test_evaluate_correct_answer` (`tests/test_evaluator.py`)
- **Validación:** **LLM-as-judge** (calidad del score) · Modelos reales ✅
- **Comportamiento esperado:** Una respuesta correcta sobre "qué es un agente inteligente"
  obtiene score ≥ 6/10 con justificación no vacía, anclada en RAG, sin falsos negativos.
- **Comportamiento observado:** ✅ El Evaluator marca `is_evaluable=True`, asigna score
  ≥ 6 y produce justificación textual. Validación cruzada de claims contra los chunks
  reales sin warnings de alucinación.

#### Caso 4 — Segunda sesión prioriza temas con bajo desempeño *(PRD #4 — memoria)*

- **Test:** `test_topic_distribution_in_exam_output` /
  `test_generate_questions_with_weak_topics` (`tests/test_exam_generator.py`)
- **Validación:** Determinística (distribución de temas en el examen) · Mock
- **Comportamiento esperado:** Dado un perfil con temas débiles (score < 6), el examen
  generado incluye mayor proporción de esos temas.
- **Comportamiento observado:** ✅ La distribución de temas del examen prioriza los temas
  débiles del perfil; sin perfil, la distribución es uniforme.

### (b) Casos límite o ambiguos

#### Caso 5 — El usuario pide un examen de un tema no presente en el material *(PRD #7)*

- **Test:** `test_prd7_missing_topic_handling` (`tests/test_exam_generator.py`)
- **Validación:** Determinística (mensaje de error + sugerencias) · Mock
- **Comportamiento esperado:** El agente informa que el tema no está en el material y
  ofrece hasta 3 alternativas cercanas; no inventa preguntas.
- **Comportamiento observado:** ✅ Retorna error estructurado con ≤ 3 sugerencias de temas
  presentes en el material; no genera preguntas.

#### Caso 6 — Respuesta parcialmente correcta con error conceptual *(PRD #8)*

- **Test:** `test_evaluate_partially_correct` (`tests/test_evaluator.py`)
- **Validación:** **LLM-as-judge** (calidad del score) · Modelos reales ✅
- **Comportamiento esperado:** Una respuesta vaga/parcial obtiene score intermedio
  (rango 3–7.5), nunca cercano al perfecto; el feedback identifica el error.
- **Comportamiento observado:** ✅ El Evaluator asigna un score en el rango medio
  (3.0 ≤ score ≤ 7.5) y estrictamente < 9, con `is_evaluable=True`.

#### Caso 7 — Pedido multi-paso / ambiguo (planificación) *(Plan-and-Execute)*

- **Test:** `test_classify_multi_step_request`
  (`tests/test_orchestrator.py::TestRealOrchestratorIntegration`)
- **Validación:** Determinística (intent/plan resultante) · Modelos reales ✅
- **Comportamiento esperado:** Ante *"Subí mis apuntes de cálculo y después generame un
  examen"*, el Orchestrator clasifica `intent=composite` o pre-puebla un plan multi-paso.
- **Comportamiento observado:** ✅ El LLM clasifica el pedido como `composite` o devuelve
  un plan con ≥ 1 paso, activando el loop Plan-and-Execute.

### (c) Entradas adversariales o fuera de dominio

#### Caso 8 — Subir un archivo de texto aleatorio (no apunte) *(PRD #10)*

- **Test:** `test_reject_non_academic_content` (`tests/test_ingestor.py`)
- **Validación:** Determinística (flag de rechazo) · Mock
- **Comportamiento esperado:** El Ingestor rechaza o marca el archivo como no clasificable
  y no contamina la base vectorial.
- **Comportamiento observado:** ✅ El documento no académico es rechazado mediante flag en
  la respuesta; no se agregan chunks a ChromaDB.

#### Caso 9 — Examen sobre tema de otra materia → anti-alucinación *(PRD #11)*

- **Test:** `test_anti_hallucination_catches_fabrication`
  (`tests/test_exam_generator.py`) · refuerzo con mock `test_prd11_adversarial_no_content`
- **Validación:** Determinística (validación claim-level contra chunks) · Modelos reales ✅
- **Comportamiento esperado:** El agente no inventa contenido; el guardrail detecta
  afirmaciones fabricadas (p. ej. astrofísica) que no tienen respaldo en los chunks reales
  de teoría de agentes.
- **Comportamiento observado:** ✅ La validación a nivel de claim contra los embeddings
  reales del material detecta y marca la afirmación fabricada por baja similitud.

#### Caso 10 — Respuesta del estudiante no evaluable (idioma/gibberish) *(PRD #12)*

- **Test:** `test_check_evaluability_rejects_gibberish` (determinístico) +
  `test_evaluate_wrong_language` (`tests/test_evaluator.py`)
- **Validación:** Determinística (guard rule-based) · Mock + real
- **Comportamiento esperado:** Una respuesta sin contenido evaluable (gibberish, demasiado
  corta) se rechaza con respuesta estructurada `cannot_evaluate`; nunca produce un 500.
- **Comportamiento observado:** ✅ El guard de evaluabilidad marca `non_evaluable=True` con
  `reason="gibberish"`, status `cannot_evaluate` y una `suggested_action` de reenvío.

#### Caso 11 — Guardrail de loop infinito del agente *(PRD §7)*

- **Test:** `test_retry_fails_returns_partial` (`tests/test_orchestrator.py`)
- **Validación:** Determinística (corte + respuesta parcial) · Mock
- **Comportamiento esperado:** Ante fallos repetidos de una tool, el agente no entra en
  loop infinito: corta tras el límite de reintentos y devuelve un resultado parcial con
  flag de error.
- **Comportamiento observado:** ✅ Tras agotar reintentos, el Orchestrator termina el loop
  y retorna estado parcial con el error registrado, sin colgarse.

#### Caso 12 — Generar un ejercicio paso a paso *(ExerciseGenerator — happy path)*

- **Test:** `test_ex01_happy_path` (`tests/test_exercise_generator.py`)
- **Validación:** Determinística (estructura del ejercicio + claims validados) · Mock
- **Comportamiento esperado:** El ExerciseGenerator recibe un topic válido, recupera chunks
  relevantes y genera un ejercicio con `statement`, `steps` y `expected_answer`; el guardrail
  `validate_claims` confirma que todos los claims tienen respaldo en los chunks reales.
- **Comportamiento observado:** ✅ El grafo termina con `status=complete`; el ejercicio incluye
  `steps` no vacíos y `expected_answer`; `validate_claims` retorna `all_matched=True`.

#### Caso 13 — Historial de conversación restaurado entre sesiones *(T-015 — memoria)*

- **Test:** `test_history_restored_across_invocations` (`tests/test_short_term_memory.py`)
- **Validación:** Determinística (contenido de `messages_history` entre dos invocaciones) · Mock
- **Comportamiento esperado:** Dos invocaciones sucesivas al grafo del Orchestrator con el mismo
  `thread_id` (checkpointer LangGraph SQLite) → la segunda restaura automáticamente el
  `messages_history` producido en la primera; el usuario no reenvía contexto.
- **Comportamiento observado:** ✅ El historial del turno 2 contiene los mensajes de ambos
  turnos; el mensaje del turno 1 (`"hola"`) está presente en el estado restaurado.

#### Caso 14 — ExerciseGenerator cuando el tema no está cubierto por el material *(ex02 — límite)*

- **Test:** `test_ex02_missing_topic` (`tests/test_exercise_generator.py`)
- **Validación:** Determinística (status + sugerencias) · Mock
- **Comportamiento esperado:** Si `retrieve_chunks` devuelve lista vacía para el topic
  solicitado, el grafo termina con `status=no_material` y ofrece sugerencias alternativas;
  nunca genera un ejercicio vacío ni inventa contenido.
- **Comportamiento observado:** ✅ Estado final con `status=no_material` y campo `suggestions`
  no vacío; no se lanza excepción ni se produce ejercicio con contenido fabricado.

#### Caso 15 — ExerciseGenerator rechaza contenido fuera de dominio *(exnfr — adversarial)*

- **Test:** `test_exnfr_adversarial` (`tests/test_exercise_generator.py`)
- **Validación:** Determinística (flag del guardrail) · Mock
- **Comportamiento esperado:** Chunks con contenido fuera del dominio académico provocan que
  `validate_claims` marque los claims como no fundamentados; tras agotar reintentos el agente
  termina en error sin devolver ejercicio inventado.
- **Comportamiento observado:** ✅ `validate_claims` retorna `all_matched=False`; el grafo
  reintenta hasta el límite configurado y cierra con status de error sin producir ejercicio
  fabricado.

### (d) Observabilidad como ciudadano de primera clase

#### Caso 16 — Langfuse caído no bloquea ninguna operación del agente *(OBS-NFR-01 — adversarial)*

- **Test:** `test_langfuse_unreachable_no_block` + `test_missing_keys_no_crash`
  (`tests/test_observability.py::TestNoCrash`)
- **Validación:** Determinística (estado del manager + ausencia de excepción) · Mock
  (`ConnectionError` simulado en el constructor de Langfuse)
- **Comportamiento esperado:** Si Langfuse lanza `ConnectionError` al inicializarse,
  `ObservabilityManager` se deshabilita (`enabled=False`); `create_trace`,
  `get_callback_handler`, `flush` y `shutdown` devuelven defaults seguros; se emite `WARNING`.
- **Comportamiento observado:** ✅ `mgr.enabled=False`; todos los métodos retornan sin
  excepción; el log registra `"keys not configured"` o equivalente.

#### Caso 17 — Decoradores `@observe` aplicados correctamente en todas las tools *(OBS hardening)*

- **Test:** `TestHardening::test_validate_claim_grounding_observed`,
  `test_retrieve_chunks_observed`, `test_generate_exam_injects_callback_handler` y similares
  (`tests/test_observability.py`)
- **Validación:** Determinística (spy sobre `langfuse.observe` + inspección de
  `config["callbacks"]`) · Mock
- **Comportamiento esperado:** Cada tool tiene `@observe(as_type="tool")`; los graph builders
  (`build_ingestor`, `build_exam_generator`, etc.) NO tienen `@observe` (evita volcar el
  esquema del grafo como span); `graph.invoke` recibe el `CallbackHandler` en
  `config["callbacks"]` cuando Langfuse está activo y lo omite cuando está deshabilitado.
- **Comportamiento observado:** ✅ Spy confirma `as_type="tool"` en las 5+ tools
  instrumentadas; los 5 graph builders no aparecen decorados; `config["callbacks"]` contiene
  el handler activo o se omite graciosamente.

#### Caso 18 — `trace_id` propagado a todos los endpoints HTTP

- **Test:** `TestTraceIdPropagation` (`tests/test_api.py`)
- **Validación:** Determinística (presencia del campo `trace_id` en la respuesta) · Mock
  (TestClient)
- **Comportamiento esperado:** Los endpoints `/api/chat`, `/api/ingest`, `/api/evaluate`,
  `/api/health` y `/api/dashboard` incluyen `trace_id` no vacío; `/api/ingest` acepta
  `session_id` externo, genera UUID cuando se omite y valida su longitud máxima.
- **Comportamiento observado:** ✅ Todos los endpoints retornan `trace_id`; el campo
  `session_id` se genera automáticamente si falta y se rechaza si supera el límite de longitud.

### (e) Sesiones y persistencia

#### Caso 19 — Ciclo de vida completo de sesión con estadísticas actualizadas *(session lifecycle)*

- **Test:** `test_session_starts_empty_and_becomes_active_on_status_update`,
  `test_exam_count_increments_on_generation_and_aggregates_correctly`,
  `test_average_score_calculated_from_multiple_evaluations`
  (`tests/test_session_lifecycle_bugs.py`)
- **Validación:** Determinística (valores en SQLite después de cada evento) · Mock
- **Comportamiento esperado:** Al crear la sesión el estado es `pending`; al subir un
  archivo pasa a `active`; al generar un examen `exam_count` incrementa; al evaluar,
  `average_score` se recalcula sobre todas las evaluaciones de la sesión.
- **Comportamiento observado:** ✅ Los tres eventos actualizan la sesión de forma atómica en
  SQLite; el endpoint `GET /api/sessions/{id}` refleja los valores correctos en la misma
  request.

### (f) RAG y extracción de tópicos

#### Caso 20 — Extracción y conciliación de tópicos de un documento académico *(topic extraction)*

- **Test:** `tests/test_topic_extraction.py` (segmentación, stemming, Jaccard, conciliación
  entre archivos de la misma sesión)
- **Validación:** Determinística (tópicos extraídos, similitudes, fusión de duplicados) · Mock
- **Comportamiento esperado:** `extract_topics` segmenta el texto, descarta fragmentos
  menores a `topic_min_section_chars`, llama al LLM estructurado y devuelve `summary`,
  `topics` (lista no vacía) y `topic_tree`. Al ingestar un segundo archivo en la misma
  sesión, los tópicos similares se unifican via Jaccard (evita duplicados entre archivos).
- **Comportamiento observado:** ✅ Segmentos cortos se fusionan antes de ir al LLM;
  la conciliación con Jaccard detecta tópicos equivalentes y los unifica sin perder
  relaciones del primer archivo.

### (g) Perfil del estudiante y agente de soporte

#### Caso 21 — CRUD del perfil de estudiante y resumen del agente de soporte *(support agent)*

- **Test:** `TestUpdateStudentProfileTool::test_update_student_profile_updates_db`,
  `TestGetStudentSummaryTool::test_get_student_summary_returns_full_profile`,
  `test_get_student_summary_unknown_id_returns_none` (`tests/test_support.py`) +
  `test_profile_unknown_student_returns_404` (`tests/test_api.py`)
- **Validación:** Determinística (valores en SQLite + respuesta HTTP) · Mock
- **Comportamiento esperado:** `update_student_profile` hace upsert sin duplicados y
  recalcula `weak_topics`; `get_student_summary` agrega perfil, scores por tema, temas
  débiles y sesiones recientes en una sola respuesta; un ID inexistente devuelve 404
  estructurado con mensaje descriptivo.
- **Comportamiento observado:** ✅ Upsert idempotente en SQLite; `get_student_summary`
  devuelve las tres fuentes correctamente; el 404 incluye mensaje descriptivo sin
  exponer detalles internos.

---

## 3. Resumen de cobertura

| Categoría | Casos | Determinísticos | LLM-as-judge | Con modelos reales |
|---|---|---|---|---|
| (a) Happy path | 1, 2, 3, 4, 12, 13 | 1, 2, 4, 12, 13 | 3 | 1, 2, 3 |
| (b) Casos límite | 5, 6, 7, 14 | 5, 7, 14 | 6 | 6, 7 |
| (c) Adversariales | 8, 9, 10, 11, 15 | 8, 9, 10, 11, 15 | — | 9, 10 |
| (d) Observabilidad | 16, 17, 18 | 16, 17, 18 | — | — |
| (e) Sesiones y persistencia | 19 | 19 | — | — |
| (f) RAG y extracción de tópicos | 20 | 20 | — | — |
| (g) Perfil y agente de soporte | 21 | 21 | — | — |
| **Total** | **21** | **18** | **2** | **7** |

Las invocaciones a herramientas se validan de forma determinística en todos los casos; el
LLM-as-judge se reserva para los dos criterios genuinamente subjetivos (calidad del score
de evaluación), tal como pide la consigna.

---

## 4. Observabilidad y trazabilidad

Los casos con modelos reales son inspeccionables paso a paso. Para la defensa en vivo se
muestran:

- **Trazas Langfuse** con spans de: LLM call (modelo, prompt, tokens, latencia), tool call
  (input/output JSON, estado), RAG retrieval (query, top-K con scores) y evaluación
  (pregunta, respuesta base, score, justificación). Ver PRD §9.
- **Artefactos de traza** ya capturados en `docs/observability/` (`traces.csv`,
  `trace_metrics.json`, `exam_trace.png`) que ilustran una ejecución real de generación de
  examen con su descomposición por pasos y métricas (tokens, latencia, herramientas
  invocadas).

---

## 5. Análisis crítico: fortalezas y modos de falla

### Fortalezas

- **Guardrails determinísticos efectivos.** Anti-alucinación a nivel de claim (casos 2, 9),
  guard de evaluabilidad (caso 10) y límite de reintentos (caso 11) funcionan de forma
  reproducible y verificable sin depender del LLM.
- **Evaluación con doble control.** El Evaluator combina scoring por LLM con un segundo
  pasaje LLM-as-judge que marca `requires_review` ante discrepancias, mitigando la
  variabilidad estocástica del modelo.
- **RAG anclado al material real.** Toda generación y evaluación se contrasta contra los
  chunks reales del apunte, reduciendo la invención de contenido.

### Modos de falla detectados / limitaciones (reporte honesto)

- **OCR matemático fuera de alcance actual.** Los casos PRD #6 (PDF con tablas/ecuaciones
  complejas) y PRD #9 (foto de examen manuscrito con baja confianza OCR) están
  **diferidos**: el pipeline `ocr_math_extract` no está implementado end-to-end. No se
  presentan como aprobados; se reportan como limitación conocida.
- **Variabilidad del scoring subjetivo.** Los casos 3 y 6 dependen del LLM; por eso se
  validan con rangos (≥ 6, y 3–7.5) en lugar de valores exactos, y se complementan con el
  juez. Un modelo distinto o un cambio de prompt puede correr el score dentro del rango.
- **Clasificación de intención en pedidos ambiguos.** El caso 7 acepta tanto `composite`
  como un plan de un solo paso porque el LLM puede resolver el pedido multi-paso por
  cualquiera de las dos vías; un pedido muy ambiguo podría caer en `general_chat`.
- **Dependencia de proveedor LLM.** Los 7 casos con modelos reales requieren conectividad
  con el proveedor (Ollama Cloud). Sin el proveedor, esos tests se omiten (skip)
  graciosamente; la suite determinística sigue cubriendo la lógica de grafo y guardrails.

---

*Mapa de trazabilidad completo casos PRD ↔ tests: ver `tests_documentation.md` "PRD Test
Case Coverage".*
