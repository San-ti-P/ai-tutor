**PRODUCT REQUIREMENTS DOCUMENT**

**Tutor Académico Personal**

*Sistema de Agentes LLM para Estudio Adaptativo*

**Trabajo Práctico N°2**  ·  Inteligencia Artificial 2026  ·  UTN Santa Fe — CIDISI

Entrega Final: 29/06/2026    Versión: 1.0    Fecha: 01/06/2026

| Proyecto | Tutor Académico Personal con Agentes LLM |
| :---- | :---- |
| Materia | Inteligencia Artificial — UTN Santa Fe (CIDISI) |
| Entrega 1 | Definición conceptual — 08/06/2026 |
| Entrega 2 | Avances en implementación — 22/06/2026 |
| Entrega 3 | Implementación completa \+ Coloquio — 29/06/2026 |
| Framework LLM | LLM SDK (modelo y SDK a definir) + LangGraph |
| UI | React \+ Next.js (web) / Streamlit (prototipo rápido) |

# **1\.  Resumen Ejecutivo**

Este documento define los requerimientos, arquitectura y criterios de aceptación del Tutor Académico Personal, un sistema de agentes inteligentes basados en Modelos de Lenguaje (LLMs) diseñado para asistir a estudiantes universitarios en la preparación de exámenes. El sistema combina técnicas de Retrieval-Augmented Generation (RAG), memoria persistente entre sesiones, herramientas (Tools) especializadas y una arquitectura multi-agente con roles diferenciados.

El agente ingiere material académico (apuntes de cátedra, exámenes anteriores), lo estructura en un índice temático consultable, y genera artefactos de estudio personalizados: exámenes de opción múltiple, preguntas de respuesta libre y ejercicios prácticos. Un agente evaluador corrige las respuestas del estudiante, produce feedback detallado e identifica áreas de debilidad para adaptar el plan de estudio en sesiones futuras.

| Propuesta de Valor Reemplazar el estudio pasivo por un ciclo activo de generación → resolución → corrección → retroalimentación, personalizado al material propio de la cátedra y al historial de desempeño del estudiante. |
| :---- |

# **2\.  Objetivo y Alcance**

## **2.1  Objetivo del Sistema**

Construir un agente inteligente autónomo capaz de: (a) ingerir y estructurar material académico en una base de conocimiento consultable; (b) generar artefactos de estudio (exámenes, cuestionarios, flashcards, ejercicios) adaptados a los contenidos y al perfil de desempeño del usuario; (c) evaluar respuestas del estudiante y proveer feedback accionable; (d) mantener un modelo del progreso del usuario entre sesiones para adaptar el foco de estudio.

## **2.2  Alcance Funcional**

| ✅  Incluido en el Alcance | ❌  Fuera del Alcance |
| :---- | :---- |
| Ingesta de PDFs, imágenes y texto plano | Generación de resúmenes automáticos de libros |
| Generación de exámenes y ejercicios | Corrección gramatical / estilo de redacción |
| Evaluación de respuestas libres con LLM | Integración con plataformas LMS (Moodle, etc.) |
| OCR y parsing de expresiones matemáticas | Generación de contenido de libros externos |
| Memoria persistente entre sesiones | Soporte multiusuario / multi-cátedra simultáneo |
| Panel de progreso y estadísticas | Modo offline sin API de LLM |
| Preferencias de estilo de examen por usuario | Certificación formal o acreditación académica |

# **3\.  Ambiente del Agente**

El ambiente en que el agente se desenvuelve es semi-estructurado, episódico y parcialmente observable. A continuación se caracteriza según la taxonomía de Russell & Norvig:

| Dimensión | Clasificación | Justificación |
| :---- | :---- | :---- |
| Observabilidad | Parcialmente observable | El agente conoce el material ingestado y el historial, pero no el estado cognitivo real del estudiante |
| Agentes | Multi-agente (cooperativo) | Ingestor, ExamGenerator, ExerciseGenerator, Evaluator, SupportAgent coordinados |
| Determinismo | Estocástico | Las respuestas del LLM son no deterministas; el estudiante puede responder de formas imprevistas |
| Episodicidad | Semi-episódico | Cada sesión es independiente pero el estado del estudiante persiste entre sesiones |
| Dinamismo | Dinámico | El conocimiento del usuario evoluciona; el material puede actualizarse durante el curso |
| Continuidad | Discreto | Interacciones en turnos (conversación estructurada) |

## **3.1  Percepciones**

| Percepción | Descripción |
| :---- | :---- |
| Archivos subidos | PDFs, imágenes (fotos de apuntes), texto plano de apuntes y exámenes previos |
| Respuestas del estudiante | Texto libre o selección de opción múltiple como respuesta a preguntas generadas |
| Imágenes de exámenes resueltos | Fotos de hojas con ejercicios resueltos a mano, incluyendo fórmulas matemáticas |
| Preferencias del usuario | Configuración: tipo de preguntas preferidas, temas a priorizar, nivel de dificultad |
| Historial de sesiones | Puntajes, temas consultados y áreas con bajo desempeño de sesiones anteriores |
| Mensajes de chat | Instrucciones en lenguaje natural del estudiante para guiar la generación |

## **3.2  Acciones / Tools**

| Tool | Agente Owner | Descripción |
| :---- | :---- | :---- |
| ingest\_document | Ingestor | Parsea y clasifica un archivo subido; actualiza el índice RAG incremental |
| retrieve\_chunks | Todos | Busca fragmentos relevantes en la base vectorial dado un tema o query |
| generate\_exam | ExamGenerator | Crea un examen con preguntas de opción múltiple y respuesta libre sobre un tema |
| generate\_exercise | ExerciseGenerator | Genera ejercicios prácticos complejos con enunciado y datos de contexto |
| evaluate\_answer | Evaluator | Compara respuesta del estudiante con respuesta base y el material RAG; devuelve score y feedback |
| ocr\_math\_extract | Ingestor / UI | Extrae texto y expresiones matemáticas de imágenes; genera LaTeX |
| update\_student\_profile | SupportAgent | Actualiza estadísticas, áreas débiles y preferencias del usuario en la BD |
| get\_student\_summary | SupportAgent | Recupera el perfil completo del estudiante para personalizar la generación |

# **4\.  Arquitectura del Sistema**

## **4.1  Visión General de Componentes**

El sistema se organiza en tres capas principales: la capa de interfaz de usuario (UI), la capa de orquestación de agentes y la capa de datos persistentes. Los agentes se comunican a través de un bus de mensajes interno (estado compartido de LangGraph).

| Loop del Agente — Estrategia ReAct \+ Plan-and-Execute Cada agente sigue el patrón ReAct (Reasoning \+ Acting): emite un pensamiento interno, decide qué herramienta invocar, recibe el resultado de la herramienta, razona sobre él y decide el siguiente paso o entrega la respuesta final. El Orchestrator aplica Plan-and-Execute para tareas multi-paso (como generar un examen completo: planificar temas → recuperar chunks → generar preguntas → validar → formatear). |
| :---- |

## **4.2  Agentes y Responsabilidades**

| Agente | Tipo de Loop | Responsabilidades Principales |
| :---- | :---- | :---- |
| Orchestrator | Plan-and-Execute | Coordina el flujo entre agentes; mantiene el estado de la sesión; decide qué agente activar en función de la intención del usuario |
| Ingestor Agent | ReAct | Parsea archivos, clasifica contenido (apuntes/examen/ejercicio), construye/actualiza el índice RAG de forma incremental, confirma con el usuario en casos de baja confianza en OCR |
| ExamGenerator | ReAct \+ Tools | Genera exámenes con MCQ y preguntas abiertas; usa retrieve\_chunks; respeta preferencias del usuario; proporciona respuestas base para el Evaluator |
| ExerciseGenerator | ReAct \+ Tools | Genera ejercicios prácticos complejos orientados a la resolución; puede basar los datos en contexto del material ingestado |
| Evaluator Agent | Chain-of-Thought | Corrige respuestas libres contrastando con la respuesta base y los chunks RAG; produce score (0-10), justificación y sugerencias de repaso |
| Support Agent | Reactive | Actualiza y consulta el perfil del estudiante; genera recomendaciones de foco de estudio; alimenta el dashboard de progreso |

## **4.3  Módulo RAG**

| Componente RAG | Decisión de Diseño |
| :---- | :---- |
| Chunking | Chunking semántico: se divide por sección/tema detectada en el índice del documento. Fallback: chunks de 512 tokens con 64 de overlap |
| Embeddings | Modelo de embeddings (a definir; opción local preferida para evitar latencia y costo) |
| Vector Store | ChromaDB (local, persistente). Colecciones separadas por sesión/materia |
| Índice Temático | Árbol de temas extraído por el Ingestor (LLM-assisted). Permite filtrar chunks por tema antes del similarity search |
| Retriever | Top-K similarity search (K=5-8). Query construida dinámicamente por el agente en función del tema requerido |
| Actualización Incremental | Al ingerir un nuevo archivo, se agregan los nuevos chunks sin re-procesar los existentes. El índice temático se fusiona |

## **4.4  Memoria**

| Tipo de Memoria | Implementación | Contenido |
| :---- | :---- | :---- |
| Short-term (conversacional) | Context window del LLM | Historial del turno actual: mensajes, resultados de tools, razonamiento parcial |
| Long-term (perfil de usuario) | SQLite / JSON persistente | Historial de sesiones, scores por tema, preferencias, temas con debilidad identificada |
| Episódica (material) | ChromaDB vectorial | Chunks del material ingestado; metadatos: tipo de documento, tema, fecha de ingesta |

## **4.5  Stack Tecnológico**

| Capa | Tecnología | Justificación |
| :---- | :---- | :---- |
| Modelo LLM | LLM (modelo y SDK a definir) | Capacidad de razonamiento, seguimiento de instrucciones, tool calling nativo; buen desempeño en matemática y ciencias |
| Orquestación | LangGraph (Python) | Soporte nativo para grafos de agentes con estado persistente; fácil de visualizar y debuggear |
| RAG / Vector Store | ChromaDB \+ LangChain | Open-source, local, sin dependencias externas; fácil integración con LangChain loaders |
| OCR Matemático | Mathpix API / pix2tex (local) | Estado del arte en reconocimiento de fórmulas LaTeX; Mathpix como fallback confiable |
| Parsing de documentos | markitdown (Microsoft) | Convierte PDF/DOCX/PPTX a Markdown estructurado; preserva tablas y listas |
| UI Web | Next.js \+ React \+ Tailwind | SPA moderna; soporte para upload de archivos, chat, dashboard; deploy sencillo |
| Backend API | FastAPI (Python) | Async, tipado, OpenAPI automático; fácil integración con LangGraph y ChromaDB |
| Persistencia | SQLite (perfil usuario) | Sin necesidad de servidor de BD; suficiente para datos de un grupo reducido de usuarios |
| Observabilidad | Langfuse (open-source) | Trazas estructuradas de LLM calls \+ tool calls; dashboard web; LLM-as-judge integrado |

# **5\.  Flujos Principales del Sistema**

## **5.1  Flujo de Ingesta**

| Paso | Actor | Descripción |
| :---- | :---- | :---- |
| 1 | Usuario → UI | El usuario sube uno o más archivos (PDF, imagen, texto) al sistema |
| 2 | Ingestor Agent | Invoca ingest\_document: ejecuta markitdown para convertir a Markdown |
| 3 | Ingestor Agent | Clasifica el documento: apunte teórico / examen previo / ejercicio resuelto (LLM) |
| 4 | Ingestor Agent | Si detecta imágenes con fórmulas matemáticas: invoca ocr\_math\_extract |
| 5 | Ingestor Agent \+ UI | Si confianza OCR \< umbral (0.85): muestra el LaTeX extraído al usuario para confirmación |
| 6 | Ingestor Agent | Extrae el índice temático del documento (LLM: genera árbol de temas) |
| 7 | Ingestor Agent | Realiza chunking semántico y agrega embeddings a ChromaDB (incremental) |
| 8 | Support Agent | Actualiza el índice temático global de la sesión de estudio |
| 9 | UI | Muestra confirmación: temas detectados, cantidad de chunks, materiales clasificados |

## **5.2  Flujo de Generación de Examen**

| Paso | Actor | Descripción |
| :---- | :---- | :---- |
| 1 | Usuario → UI | El usuario solicita un examen (puede especificar temas, dificultad, cantidad de preguntas) |
| 2 | Orchestrator | Invoca get\_student\_summary para obtener el perfil; ajusta instrucción de generación |
| 3 | ExamGenerator | Plan: lista de temas a cubrir en el examen (priorizando áreas débiles del perfil) |
| 4 | ExamGenerator | Para cada tema: invoca retrieve\_chunks; construye contexto RAG |
| 5 | ExamGenerator | Invoca generate\_exam con el contexto; genera MCQ \+ preguntas abiertas con respuestas base |
| 6 | ExamGenerator | Valida que las preguntas estén ancladas en el material (guardrail anti-alucinación) |
| 7 | UI | Renderiza el examen en modo interactivo; el usuario puede resolverlo en la interfaz |

## **5.3  Flujo de Evaluación**

| Paso | Actor | Descripción |
| :---- | :---- | :---- |
| 1 | Usuario → UI | El usuario entrega sus respuestas (texto o imagen de hoja resuelta) |
| 2 | Ingestor Agent | Si es imagen: ocr\_math\_extract → confirmar con usuario si baja confianza |
| 3 | Evaluator | Para cada respuesta libre: invoca evaluate\_answer con {pregunta, respuesta\_base, respuesta\_estudiante, chunks\_relevantes} |
| 4 | Evaluator | Produce: score (0-10), análisis de correctitud, errores conceptuales detectados, sugerencias |
| 5 | Support Agent | Registra resultados: invoca update\_student\_profile con scores por tema |
| 6 | UI | Muestra corrección detallada, puntaje total y recomendaciones de repaso |
| 7 | Support Agent | Actualiza el plan de estudio adaptativo para la próxima sesión |

# **6\.  Requerimientos del Sistema**

## **6.1  Requerimientos Funcionales**

| ID | Módulo | Descripción | Prioridad |
| :---- | :---- | :---- | :---- |
| RF-01 | Ingesta | El sistema debe aceptar archivos PDF, PNG/JPG y TXT como material de estudio | Alta |
| RF-02 | Ingesta | El Ingestor debe clasificar automáticamente el documento en: apunte teórico, examen previo, ejercicio resuelto | Alta |
| RF-03 | Ingesta | La ingesta debe ser incremental: nuevos archivos se agregan sin reprocesar el material existente | Alta |
| RF-04 | OCR | El sistema debe extraer expresiones matemáticas de imágenes y representarlas en LaTeX | Alta |
| RF-05 | OCR | Si la confianza del OCR es baja (\<0.85), el sistema debe solicitar confirmación al usuario antes de proceder | Media |
| RF-06 | RAG | El sistema debe estructurar el material en un índice temático jerárquico y permitir búsqueda por tema o búsqueda global | Alta |
| RF-07 | ExamGen | El ExamGenerator debe producir exámenes con al menos 2 tipos de preguntas: MCQ y respuesta libre | Alta |
| RF-08 | ExamGen | Las preguntas deben estar basadas en el material ingestado; el agente no debe inventar hechos no presentes en los apuntes | Alta |
| RF-09 | ExerciseGen | El ExerciseGenerator debe generar ejercicios prácticos de mayor complejidad que los exámenes teóricos | Alta |
| RF-10 | Evaluador | El Evaluator debe puntuar respuestas libres en escala 0-10 y proveer feedback con errores conceptuales identificados | Alta |
| RF-11 | Memoria | El sistema debe recordar el desempeño del usuario entre sesiones y priorizar temas con bajo rendimiento en futuras generaciones | Alta |
| RF-12 | Preferencias | El usuario debe poder configurar: tipo de preguntas, nivel de dificultad, temas a incluir/excluir, cantidad de preguntas por examen | Media |
| RF-13 | Dashboard | La UI debe mostrar: historial de sesiones, scores por tema, gráfica de evolución y temas recomendados para repasar | Media |
| RF-14 | Observabilidad | El sistema debe registrar trazas de todas las llamadas al LLM y todas las invocaciones a tools (inputs/outputs) | Alta |

## **6.2  Requerimientos No Funcionales**

| ID | Categoría | Descripción |
| :---- | :---- | :---- |
| RNF-01 | Rendimiento | La generación de un examen de 10 preguntas no debe superar 30 segundos de latencia extremo a extremo |
| RNF-02 | Rendimiento | La ingesta de un PDF de hasta 50 páginas debe completarse en menos de 2 minutos |
| RNF-03 | Confiabilidad | El agente no debe producir preguntas basadas en información no presente en el material ingestado (guardrail de alucinación) |
| RNF-04 | Usabilidad | La UI debe ser comprensible sin documentación adicional para un estudiante universitario |
| RNF-05 | Trazabilidad | Cada respuesta del agente debe tener asociada la traza completa de herramientas usadas y chunks recuperados |
| RNF-06 | Mantenibilidad | El código debe seguir una arquitectura modular con separación clara entre agentes, tools y capa de datos |
| RNF-07 | Seguridad | Las API keys no deben estar hardcodeadas; deben gestionarse mediante variables de entorno (.env) |

# **7\.  Guardrails y Validaciones**

| Riesgo | Mecanismo de Guardrail | Acción en caso de fallo |
| :---- | :---- | :---- |
| Alucinación en preguntas generadas | Post-generación: verificar que cada hecho de la pregunta tenga chunk RAG asociado con score \> umbral | Regenerar la pregunta; si falla 3 veces, omitir y notificar |
| Loop infinito del agente | Límite de iteraciones máximas por task: 15 pasos | Terminar el loop; devolver respuesta parcial con flag de error |
| OCR con baja confianza | Threshold de confianza configurable (default: 0.85) | Solicitar confirmación explícita del usuario antes de usar el output |
| Material fuera de dominio académico | El Ingestor clasifica el documento; rechaza archivos que no tengan estructura de apunte/examen | Notificar al usuario que el archivo no es material académico válido |
| Respuestas del evaluador inconsistentes | LLM-as-judge: un segundo pasaje del LLM valida la corrección realizada por el Evaluator (sampling) | Marcar evaluación como 'requiere revisión'; mostrar al usuario |

# **8\.  Plan de Evaluación del Agente**

Se construirá un conjunto de casos de prueba cubriendo los tres escenarios requeridos por el TP: happy path, casos límite y entradas adversariales.

| \# | Categoría | Caso de Prueba | Comportamiento Esperado | Validación |
| :---- | :---- | :---- | :---- | :---- |
| 1 | Happy Path | Ingestar PDF de apuntes bien formateado | Clasificación correcta, índice temático con al menos 3 temas, chunks generados | Determinística: verificar ChromaDB count \> 0 |
| 2 | Happy Path | Generar examen de 5 preguntas sobre un tema específico | 5 preguntas con respuestas base, todas referenciando chunks del material | Determinística: verificar cantidad y estructura JSON |
| 3 | Happy Path | Evaluar respuesta correcta a pregunta de cálculo | Score ≥ 8/10, feedback positivo, sin falsos negativos | LLM-as-judge: concordancia con evaluación humana |
| 4 | Happy Path | Segunda sesión prioriza temas con bajo score | El examen generado incluye mayor proporción de temas con score \<6 de sesión anterior | Determinística: verificar topic distribution en examen |
| 5 | Happy Path | Ingesta incremental de segundo PDF | Nuevos chunks agregados, chunks anteriores sin modificar, índice fusionado | Determinística: contar chunks antes y después |
| 6 | Caso Límite | PDF con tablas y ecuaciones complejas | OCR correctamente extrae al menos el 80% del contenido matemático | LLM-as-judge: comparación con transcripción manual |
| 7 | Caso Límite | Usuario pide examen de tema no presente en el material | El agente informa que el tema no está en el material; ofrece alternativas cercanas | Determinística: verificar mensaje de error \+ sugerencias |
| 8 | Caso Límite | Respuesta parcialmente correcta con error conceptual menor | Score entre 5-7/10; feedback identifica el error específico | LLM-as-judge: concordancia con evaluación humana |
| 9 | Caso Límite | Foto de examen escrito a mano con letra difícil de leer | Sistema detecta baja confianza OCR y solicita confirmación al usuario | Determinística: verificar que se emite prompt de confirmación |
| 10 | Adversarial | Subir un archivo de texto aleatorio (no apunte) | El Ingestor rechaza o marca como no clasificable; no contamina la BD vectorial | Determinística: verificar flag de rechazo en respuesta |
| 11 | Adversarial | Solicitar examen sobre tema de otra materia no ingestada | El agente no inventa contenido; informa ausencia de material para ese tema | LLM-as-judge: detectar si hay alucinación en preguntas |
| 12 | Adversarial | Respuesta del estudiante en otro idioma | El Evaluator procesa y evalúa correctamente; o informa que no puede evaluar | Determinística: verificar que no retorna error 500; respuesta coherente |

# **9\.  Observabilidad y Trazabilidad**

Se utilizará Langfuse (self-hosted o cloud) para registrar trazas estructuradas de todas las ejecuciones del agente. Cada traza debe incluir los siguientes spans:

* Span raíz: identificador de sesión, usuario, tipo de tarea (ingesta / generar examen / evaluar / ejercicio)

* LLM Call span: modelo, prompt completo (system \+ user), respuesta, tokens consumidos, latencia, costo estimado

* Tool Call span: nombre de la tool, input JSON, output JSON, tiempo de ejecución, estado (éxito / error)

* RAG Retrieval span: query, top-K resultados con scores, chunks seleccionados y metadatos asociados

* Evaluation span (si aplica): pregunta, respuesta estudiante, respuesta base, score asignado, justificación del evaluador

| Métricas a Registrar Por cada ejecución: número total de pasos del agente, tokens consumidos (input/output), costo estimado en USD, latencia total y por agente, herramientas invocadas (frecuencia y tasa de éxito), scores de evaluación promedio por tema. |
| :---- |

# **10\.  Entregables y Cronograma**

| Entrega | Fecha | Componentes a Presentar |
| :---- | :---- | :---- |
| 1ª Entrega | 08/06/2026 | Este PRD completo. Diagrama de arquitectura. Definición de agentes, tools, ambiente, percepciones y acciones. Decisiones tecnológicas justificadas. |
| 2ª Entrega | 22/06/2026 | Implementación base funcional del Ingestor \+ RAG (ChromaDB operativo). ExamGenerator con al menos 2 tools funcionando. Flujo básico de chat con LangGraph. Logging de LLM calls. Esqueleto ejecutable de la UI. |
| 3ª Entrega | 29/06/2026 | Sistema completo e integrado. Todos los agentes funcionales. UI interactiva con dashboard de progreso. Suite de casos de prueba ejecutada y documentada. Observabilidad con Langfuse. Informe técnico en PDF. README con instrucciones. Defensa oral en vivo. |

## **10.1  Estructura del Repositorio GitHub**

| Ruta | Contenido |
| :---- | :---- |
| /backend | FastAPI app, agentes LangGraph, tools, módulo RAG |
| /backend/agents | Código de cada agente (ingestor.py, exam\_generator.py, etc.) |
| /backend/tools | Definición y lógica de cada Tool |
| /backend/rag | ChromaDB setup, chunking, embedding, retrieval |
| /backend/memory | Perfil de usuario, SQLite schema, update/read funciones |
| /frontend | Next.js app: chat UI, dashboard, upload de archivos |
| /tests | Suite de casos de prueba (pytest), LLM-as-judge scripts |
| /docs | Informe técnico PDF, diagramas de arquitectura |
| README.md | Instrucciones de instalación y ejecución detalladas |
| .env.example | Variables de entorno requeridas (sin valores reales) |
| requirements.txt / package.json | Dependencias Python y Node declaradas |

# **11\.  Criterios de Aceptación del TP**

La siguiente tabla mapea los criterios de evaluación definidos en el enunciado del TP a los componentes de este PRD, indicando el nivel mínimo esperado y la evidencia que se presentará en el coloquio.

| Criterio del TP | Nivel Mínimo | Evidencia / Componente del PRD |
| :---- | :---- | :---- |
| Originalidad y claridad del caso de uso | Bueno | PRD Sección 1-2: caso de uso concreto (tutor académico), diferenciado de ejemplos genéricos |
| Correcta aplicación de conceptos de agentes y LLMs | Excelente | Secciones 3-4: ambiente caracterizado, percepciones/acciones definidas, loops ReAct y Plan-and-Execute |
| Diseño modular, arquitectura clara | Excelente | Sección 4: diagrama de componentes, agentes con responsabilidades claras, stack justificado |
| Capacidad del agente de razonar coherentemente | Bueno | Demostración en vivo: generación de examen adaptativo \+ evaluación con feedback coherente |
| Uso justificado de tools, memoria, RAG y planificación | Excelente | Secciones 3.2, 4.3, 4.4: ≥8 tools definidas, RAG con índice temático, memoria long-term, loop multi-step |
| Calidad del código y documentación | Bueno | GitHub: estructura modular, README, requirements.txt, .env.example, docstrings |
| Evaluación realizada por el grupo (casos de prueba) | Bueno | Sección 8: 12 casos de prueba con 3 categorías; LLM-as-judge para criterios subjetivos |
| Observabilidad y trazabilidad | Bueno | Sección 9: Langfuse con spans detallados para LLM calls, tools y retrieval; métricas de tokens y costo |
| Defensa oral con demostración en vivo | Bueno | Demo: ingesta de PDF → generación de examen → resolución → evaluación con feedback → dashboard de progreso |

# **12\.  Referencias**

1. Russell, S., Norvig, P.: Artificial Intelligence: A Modern Approach. 4th edition, Pearson (2020) — capítulos sobre agentes inteligentes

2. Yao, S., et al.: ReAct: Synergizing Reasoning and Acting in Language Models. ICLR (2023). https://arxiv.org/abs/2210.03629

3. Lewis, P., et al.: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS (2020). https://arxiv.org/abs/2005.11401

4. Shinn, N., et al.: Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS (2023). https://arxiv.org/abs/2303.11366

5. Schick, T., et al.: Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS (2023). https://arxiv.org/abs/2302.04761

6. Wang, L., et al.: A Survey on Large Language Model based Autonomous Agents. Frontiers of Computer Science (2024). https://arxiv.org/abs/2308.11432

7. Anthropic: Building Effective Agents (2024). https://www.anthropic.com/research/building-effective-agents

8. Model Context Protocol (MCP) Specification. https://modelcontextprotocol.io

9. LangGraph Documentation. https://langchain-ai.github.io/langgraph/

10. Langfuse: Open Source LLM Engineering Platform. https://langfuse.com/docs

11. markitdown: Microsoft Open Source Document Converter. https://github.com/microsoft/markitdown

*— Fin del Documento —*

