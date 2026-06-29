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

---

## 3. Resumen de cobertura

| Categoría | Casos | Determinísticos | LLM-as-judge | Con modelos reales |
|---|---|---|---|---|
| (a) Happy path | 1, 2, 3, 4 | 1, 2, 4 | 3 | 1, 2, 3 |
| (b) Casos límite | 5, 6, 7 | 5, 7 | 6 | 6, 7 |
| (c) Adversariales | 8, 9, 10, 11 | 8, 9, 10, 11 | — | 9, 10 |
| **Total** | **11** | **8** | **2** | **7** |

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

*Mapa de trazabilidad completo casos PRD ↔ tests: ver `tests_documentation.md` §"PRD Test
Case Coverage".*
