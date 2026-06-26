"""Orchestrator Agent — Plan-and-Execute with hub-and-spoke routing."""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import settings
from src.llm import get_llm as _get_llm, get_structured_llm

logger = logging.getLogger(__name__)

Intent = Literal[
    "ingest",
    "retrieve",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
    "general_chat",
    "composite",
]

_SINGLE_TOOL_INTENTS: set[str] = {
    "ingest",
    "retrieve",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
}


class IntentClassification(BaseModel):
    intent: Intent = Field(description="The classified intent category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class CompositePlan(BaseModel):
    steps: list[str] = Field(description="Ordered list of tool names to execute")


# ── Tool wiring ──────────────────────────────────────────────────────────────
# Must be built after tool imports (avoid circular imports with tools/__init__.py)

TOOL_MAP: dict[str, object] = {}


def _init_tool_map() -> dict[str, object]:
    """Lazy-init TOOL_MAP to avoid circular imports at module level."""
    global TOOL_MAP
    if TOOL_MAP:
        return TOOL_MAP
    from src.tools import (  # noqa: F811
        evaluate_answer,
        generate_exam,
        generate_exercise,
        ingest_document,
        query_material,
    )
    from src.tools.get_student_summary import get_student_summary

    TOOL_MAP = {
        "ingest": ingest_document,
        "retrieve": query_material,
        "generate_exam": generate_exam,
        "generate_exercise": generate_exercise,
        "evaluate": evaluate_answer,
        "query_profile": get_student_summary,
    }
    return TOOL_MAP


class OrchestratorState(TypedDict):
    session_id: str
    user_message: str
    intent: Intent
    confidence: float
    plan: list[str]
    current_step: int
    results: list[dict]
    errors: list[dict]
    response: str
    status: str  # "pending" | "complete" | "incomplete" | "partial"
    iteration_count: int
    student_profile: dict | None


def classify_intent(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Classify user message into one of 8 intents with confidence score.

    On low confidence (< settings.classification_confidence_threshold), forces
    general_chat. On any exception, returns general_chat with confidence=0.0.
    Single-tool intents pre-populate plan with the tool name.
    """
    message = state["user_message"]
    profile = state.get("student_profile")

    try:
        structured = get_structured_llm(IntentClassification)

        prompt = (
            "Sos un clasificador de intents para un tutor académico. "
            "Clasificá el mensaje del usuario en UNA de estas 8 categorías:\n\n"
            "  - ingest: el usuario quiere SUBIR apuntes, PDFs, documentos\n"
            "  - retrieve: preguntar/consultar sobre el contenido de apuntes o documentos YA SUBIDOS\n"
            "  - generate_exam: pide generar un examen\n"
            "  - generate_exercise: pide un ejercicio práctico\n"
            "  - evaluate: pide que corrijan/evalúen una respuesta\n"
            "  - query_profile: pregunta por su perfil, progreso, temas débiles\n"
            "  - general_chat: saludos, charla casual, preguntas NO académicas, "
            "preguntas sobre cómo usar el sistema\n"
            "  - composite: pide VARIAS tareas distintas en el mismo mensaje\n\n"
            "REGLAS CRÍTICAS:\n"
            "1. Si es saludo, cortesía, o NO contiene tarea académica explícita → "
            "general_chat con confidence >= 0.95\n"
            "2. Ante la duda entre dos intents, elegí general_chat\n"
            "3. ingest SOLO si el usuario quiere SUBIR un archivo; retrieve SOLO si pregunta por contenido ya subido\n"
            "4. composite SOLO si hay 2+ tareas distintas y explícitas (no inferidas)\n\n"
            "EJEMPLOS:\n"
            '"Hola" → {"intent": "general_chat", "confidence": 0.99}\n'
            '"Buenos días" → {"intent": "general_chat", "confidence": 0.99}\n'
            '"Gracias" → {"intent": "general_chat", "confidence": 0.99}\n'
            '"¿Qué hace esta app?" → {"intent": "general_chat", "confidence": 0.95}\n'
            '"Quiero ver mi progreso" → {"intent": "query_profile", "confidence": 0.95}\n'
            '"Generame un examen de derivadas" → {"intent": "generate_exam", "confidence": 0.95}\n'
            '"Subime este PDF" → {"intent": "ingest", "confidence": 0.95}\n'
            '"¿Qué dice el archivo sobre derivadas?" → {"intent": "retrieve", "confidence": 0.95}\n'
            '"Generame un examen y corregilo" → {"intent": "composite", "confidence": 0.90}\n\n'
            f"Mensaje del usuario: {message!r}\n"
        )
        if profile:
            weak = profile.get("weak_topics", [])
            if weak:
                prompt += f"Contexto del estudiante (temas débiles): {weak}\n"
        prompt += "Respondé SOLO con un objeto JSON con claves 'intent' y 'confidence'."

        invoke_kwargs = {"config": config} if config is not None else {}
        result = structured.invoke(prompt, **invoke_kwargs)
        intent = result.intent
        confidence = result.confidence

        # Low-confidence fallback
        if confidence < settings.classification_confidence_threshold:
            intent = "general_chat"

        # Pre-populate plan for single-tool intents
        plan: list[str] = []
        if intent in _SINGLE_TOOL_INTENTS:
            plan = [intent]

        return {"intent": intent, "confidence": confidence, "plan": plan}

    except Exception:
        logger.exception("classify_intent failed, falling back to general_chat")
        return {"intent": "general_chat", "confidence": 0.0, "plan": []}


def route_to_agent(state: OrchestratorState) -> str:
    """Route classified intent to the matching agent node.

    composite → plan_composite
    general_chat → synthesize_response
    Any single-tool intent → execute_step
    """
    intent = state["intent"]
    if intent == "composite":
        return "plan_composite"
    if intent == "general_chat":
        return "synthesize_response"
    return "execute_step"


def plan_composite(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Plan steps for composite (multi-step) tasks.

    Uses LLM planner to generate an ordered list of tool names.
    Strips tools not in TOOL_MAP. Empty plan → treated as general_chat downstream.
    """
    message = state["user_message"]
    tool_map = _init_tool_map()
    tool_descriptions = "\n".join(
        f"- {name}: {getattr(tool, 'description', '')}" for name, tool in tool_map.items()
    )

    try:
        structured = get_structured_llm(CompositePlan)

        prompt = (
            "Sos un planificador de tareas académicas. El usuario quiere:\n"
            f'"{message}"\n\n'
            "Herramientas disponibles:\n"
            f"{tool_descriptions}\n\n"
            "Generá una lista ORDENADA de nombres de herramientas a ejecutar. "
            "Solo usá herramientas de la lista. Respondé SOLO con un objeto JSON "
            'con clave "steps". '
            'Ejemplo: {"steps": ["generate_exam", "evaluate"]}'
        )

        invoke_kwargs = {"config": config} if config is not None else {}
        result = structured.invoke(prompt, **invoke_kwargs)
        plan = result.steps

        # Strip invalid tool names
        valid_plan = [step for step in plan if step in tool_map]

        return {"plan": valid_plan}

    except Exception:
        logger.exception("plan_composite failed")
        return {"plan": []}


def _extract_topics(message: str) -> list[str]:
    """Extract topic keywords from user message using simple heuristics.

    Looks for phrases like 'sobre X', 'acerca de X', 'de X', 'temas de X'.
    Falls back to ['general'] if no topics detected.
    """
    import re

    patterns = [
        r"(?:sobre|acerca\s+de|de)\s+(?:los\s+)?(?:temas?\s+(?:de\s+)?)?['\"]?([^,.;!?\d]+?)['\"]?(?:\s+(?:con|y|,|\.|$|con\s+))",
        r"(?:gener[áa]\w*\s+(?:un\s+)?(?:examen|ejercicio)\s+(?:sobre|acerca\s+de|de)\s+)(.+?)(?:\s+(?:con|de|y|,|\.|$|\d+\s+preguntas))",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            topic_text = match.group(1).strip().rstrip(".")
            if topic_text and len(topic_text) > 2:
                # Split on ' y ', ' e ', ', ' for multiple topics
                topics = re.split(r"\s+(?:y|e)\s+|,\s*", topic_text)
                return [t.strip() for t in topics if len(t.strip()) > 2]

    return ["general"]


def _extract_difficulty(message: str) -> str:
    """Extract difficulty from user message. Defaults to 'medium'."""
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["fácil", "facil", "easy", "básico", "basico"]):
        return "easy"
    if any(w in msg_lower for w in ["difícil", "dificil", "hard", "avanzado"]):
        return "hard"
    return "medium"


def _extract_question_count(message: str) -> int:
    """Extract question count from user message. Defaults to 5."""
    import re

    match = re.search(r"(\d+)\s*(?:preguntas?|questions?)", message, re.IGNORECASE)
    if match:
        count = int(match.group(1))
        return max(1, min(count, 20))  # clamp 1-20
    return 5


def _build_tool_args(tool_name: str, state: OrchestratorState) -> dict:
    """Build argument dict for a tool call from the current state.

    Each tool gets the fields it needs extracted from shared state.
    """
    args: dict = {"session_id": state["session_id"]}
    profile = state.get("student_profile")

    if tool_name == "ingest":
        # ingest_document needs file_path, session_id
        pass  # file_path not in orchestrator state; tool will use its own
    elif tool_name == "retrieve":
        args["query"] = state["user_message"]
        args["top_k"] = 5
    elif tool_name == "generate_exam":
        args["topics"] = _extract_topics(state["user_message"])
        args["difficulty"] = _extract_difficulty(state["user_message"])
        args["question_count"] = _extract_question_count(state["user_message"])
        args["mcq_ratio"] = 0.5
        if profile:
            args["student_profile"] = profile
    elif tool_name == "generate_exercise":
        msg_topics = _extract_topics(state["user_message"])
        args["topic"] = msg_topics[0] if msg_topics else (
            profile.get("weak_topics", ["general"])[0] if profile else "general"
        )
        args["difficulty"] = _extract_difficulty(state["user_message"])
        args["exercise_type"] = "problem_solving"
        if profile:
            args["student_profile"] = profile
    elif tool_name == "evaluate":
        args["exam_id"] = ""
        args["answers"] = []
        args["student_id"] = ""
    elif tool_name == "query_profile":
        args["student_id"] = state.get("session_id", "unknown")

    return args


async def _invoke_tool_with_retry(tool, args: dict, step: int) -> dict:
    """Invoke a tool with exactly one retry on failure.

    Returns the tool result on success. Raises ValueError on double-failure.
    """
    try:
        return await tool.ainvoke(args)
    except Exception as first_err:
        logger.warning("Step %d failed, retrying once: %s", step, first_err)
        try:
            return await tool.ainvoke(args)
        except Exception as second_err:
            logger.exception("Step %d failed after retry", step)
            raise ValueError(f"Tool '{tool.name}' failed after retry: {second_err}") from second_err


async def execute_step(state: OrchestratorState) -> dict:
    """Execute the current step in the plan.

    Resolves tool from TOOL_MAP, builds args, invokes with one retry, records result.
    Increments current_step and iteration_count. On failure, records error.
    """
    plan = state["plan"]
    current = state["current_step"]
    iteration = state["iteration_count"]
    results = state.get("results", [])
    errors = state.get("errors", [])

    tool_map = _init_tool_map()

    # Safeguard: empty plan
    if not plan or current >= len(plan):
        return {"current_step": current, "iteration_count": iteration}

    tool_name = plan[current]
    tool = tool_map.get(tool_name)

    if tool is None:
        logger.warning("Tool '%s' not found in TOOL_MAP", tool_name)
        return {
            "errors": errors + [
                {
                    "step": current,
                    "tool": tool_name,
                    "error": f"Tool '{tool_name}' not found",
                }
            ],
            "status": "partial",
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }

    try:
        args = _build_tool_args(tool_name, state)
        result = await _invoke_tool_with_retry(tool, args, current)
        return {
            "results": results + [{"step": current, "tool": tool_name, "result": result}],
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }
    except Exception as exc:
        logger.warning("execute_step failed for tool '%s': %s", tool_name, exc)
        return {
            "errors": errors + [{"step": current, "tool": tool_name, "error": str(exc)}],
            "status": "partial",
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }


def _is_academic_question(message: str) -> bool:
    """Determine whether *message* looks like an academic question.

    Uses deterministic regex/heuristics — no LLM call — to avoid
    probing retrieval for greetings, casual chat, or how-to-use-app
    questions.

    Returns True when:
    - Message length >= 10 characters.
    - Contains question-like patterns in Spanish (¿, ?, qué es, explica,
      definí, cómo se, cuál, etc.).
    - Does NOT match typical greeting/courtesy patterns.
    - Does NOT match meta/how-to-use-app patterns.
    """
    import re

    if not message or len(message.strip()) < 10:
        return False

    msg = message.strip().lower()

    # Exclude greetings, thanks, farewells
    greeting_patterns = [
        r"^(hola|buenos\s+d[ií]as|buenas\s+(tardes|noches)|chau|adios|gracias)",
        r"^(ok|okey|entendido|perfecto|genial|bien|bueno)[\s\!\.]*$",
        r"^(c[oó]mo\s+(est[aá]s|andas|va|andan))",
    ]
    for pat in greeting_patterns:
        if re.search(pat, msg):
            return False

    # Exclude meta / how-to-use-app questions
    meta_patterns = [
        r"c[oó]mo\s+(funciona|uso|subo|hago|puedo|se\s+usa)",
        r"qu[eé]\s+(puedes|pod[eé]s|hac[eé]s|sabes|sab[eé]s)\s+hacer",
        r"para\s+qu[eé]\s+(sirve|sirven|funciona)",
        r"ayuda|help|comando",
    ]
    for pat in meta_patterns:
        if re.search(pat, msg):
            return False

    # Academic question signals
    academic_signals = [
        r"[¿\?]",                    # Contains question marks
        r"qu[eé]\s+(es|son|significa)",  # "qué es/son/significa"
        r"explic[aá]",               # "explica/explicá"
        r"defin[ií]",                # "definí/define"
        r"c[oó]mo\s+(se\s+)?(calcula|resuelve|determina|obtiene|halla)",
        r"cu[aá]l\s+(es|son)",       # "cuál es/son"
        r"diferencia\s+entre",
        r"qu[eé]\s+(dice|dice\s+el|habla|trata|contiene)",
        r"en\s+qu[eé]\s+(consiste|se\s+basa)",
        r"mencion[aá]",              # "menciona/mencioná"
        r"describ[ií]",              # "describe/describí"
        r"caracter[ií]sticas?\s+de",
    ]
    for pat in academic_signals:
        if re.search(pat, msg):
            return True

    return False


def synthesize_response(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Combine results from agent executions into a final response.

    - general_chat: LLM synthesizes direct answer from user_message.
    - composite/single: LLM aggregates results into coherent narrative.
    - incomplete: prepends cap warning in Spanish.
    - partial: includes error summary.
    - On LLM failure: hardcoded Spanish apology + raw results.
    """
    import json

    from src.rag.policy import RAG_ONLY_SYSTEM_PROMPT, no_material_message

    intent = state["intent"]
    message = state["user_message"]
    results = state.get("results", [])
    errors = state.get("errors", [])
    status = state.get("status", "pending")

    # Detect cap hit: iteration_count >= max but steps not all done
    if (
        status == "pending"
        and state["iteration_count"] >= settings.max_iterations_per_task
        and state["current_step"] < len(state.get("plan", []))
    ):
        status = "incomplete"

    # Prepend cap warning for incomplete
    prefix = ""
    if status == "incomplete":
        prefix = (
            "⚠️ No pude completar todas las tareas solicitadas debido al límite de pasos. "
            "Esto es lo que logré hasta ahora:\n\n"
        )

    # Build prompt for LLM synthesis
    try:
        llm = _get_llm()

        if intent == "general_chat" and not results:
            # Academic probe: check if this looks like an academic question
            if _is_academic_question(message):
                from src.tools import query_material as _query_material

                try:
                    qm_result = _query_material.invoke({
                        "query": message,
                        "session_id": state["session_id"],
                        "top_k": 3,  # lighter probe
                    })

                    if qm_result.get("chunks_found", 0) > 0:
                        # RAG-grounded: synthesize with chunks
                        chunks_text = "\n\n".join(
                            qm_result.get("sources", [])
                        )
                        prompt = (
                            f"{RAG_ONLY_SYSTEM_PROMPT}\n\n"
                            f"Fragmentos del material:\n{chunks_text}\n\n"
                            f"Pregunta del estudiante: {message}\n\n"
                            "Respondé de forma clara y educativa en español, "
                            "citando los fragmentos relevantes."
                        )
                    else:
                        # No chunks — return canonical no-material message
                        return {
                            "response": no_material_message(),
                            "status": "complete",
                        }
                except Exception as exc:
                    logger.warning("Academic probe retrieval failed: %s", exc)
                    return {
                        "response": no_material_message(),
                        "status": "complete",
                    }
            else:
                prompt = (
                    f'El usuario preguntó: "{message}"\n'
                    "Respondé de forma clara y educativa en español."
                )
        else:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)
            errors_str = json.dumps(errors, ensure_ascii=False, indent=2) if errors else "ninguno"

            prompt = (
                f'Consulta original del usuario: "{message}"\n\n'
                f"Resultados de las herramientas ejecutadas:\n{results_str}\n\n"
                f"Errores encontrados:\n{errors_str}\n\n"
            )
            if status == "partial":
                prompt += (
                    "Algunas tareas no se completaron exitosamente. "
                    "Generá una respuesta que resuma lo logrado y mencione los errores. "
                    "Sé honesto pero alentador en español."
                )
            else:
                prompt += (
                    "Generá una respuesta coherente que resuma todos estos resultados "
                    "en español, de forma clara y educativa."
                )

        invoke_kwargs = {"config": config} if config is not None else {}
        response = llm.invoke(prompt, **invoke_kwargs)
        text = response.content if hasattr(response, "content") else str(response)

        final_status = "complete" if status == "pending" else status
        return {"response": prefix + text, "status": final_status}

    except Exception:
        logger.exception("synthesize_response LLM failed, using fallback")
        results_str = json.dumps(
            {"results": results, "errors": errors},
            ensure_ascii=False,
            indent=2,
        )
        return {
            "response": (
                "Mis disculpas, no pude generar una respuesta elaborada en este momento. "
                "Acá están los resultados sin procesar:\n\n"
                f"{results_str}"
            ),
            "status": "complete" if not errors else "partial",
        }


def check_iteration_limit(state: OrchestratorState) -> Literal["continue", "terminate"]:
    """Guardrail: enforce max iterations per task.

    Returns "terminate" if:
    - iteration_count >= settings.max_iterations_per_task
    - current_step >= len(plan) (all steps done)
    - status is "partial" (error already hit)
    Otherwise returns "continue".
    """
    if state.get("status") == "partial":
        return "terminate"
    if state["iteration_count"] >= settings.max_iterations_per_task:
        return "terminate"
    if state["current_step"] >= len(state["plan"]):
        return "terminate"
    return "continue"


def build_orchestrator() -> StateGraph:
    """Build and return the Orchestrator LangGraph."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("plan_composite", plan_composite)
    builder.add_node("execute_step", execute_step)
    builder.add_node("synthesize_response", synthesize_response)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_to_agent,
        ["plan_composite", "execute_step", "synthesize_response"],
    )
    builder.add_edge("plan_composite", "execute_step")
    builder.add_conditional_edges(
        "execute_step",
        check_iteration_limit,
        {"continue": "execute_step", "terminate": "synthesize_response"},
    )
    builder.add_edge("synthesize_response", END)

    return builder


# ── Singleton compilation ────────────────────────────────────────────────────

_orchestrator_graph: object | None = None
_orchestrator_db_conn: object | None = None


async def get_orchestrator_graph():
    """Return the module-level compiled Orchestrator graph singleton.

    Compiled ONCE with AsyncSqliteSaver for session persistence.
    Falls back to InMemorySaver if DB connection fails due to missing
    dependencies (ImportError), database errors (aiosqlite.Error), or
    filesystem issues (OSError). Unexpected errors propagate.
    """
    global _orchestrator_graph
    global _orchestrator_db_conn
    if _orchestrator_graph is None:
        from langgraph.checkpoint.memory import InMemorySaver

        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_dir = os.path.dirname(settings.sqlite_db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            conn = await aiosqlite.connect(settings.sqlite_db_path)
            checkpointer = AsyncSqliteSaver(conn)
            _orchestrator_db_conn = conn
            logger.info("Orchestrator using AsyncSqliteSaver at %s", settings.sqlite_db_path)
        except (ImportError, aiosqlite.Error, OSError) as exc:
            logger.warning(
                "AsyncSqliteSaver unavailable at %s: %s. Using InMemorySaver",
                settings.sqlite_db_path,
                exc,
            )
            checkpointer = InMemorySaver()

        _orchestrator_graph = build_orchestrator().compile(checkpointer=checkpointer)

    return _orchestrator_graph


async def close_orchestrator_graph():
    """Close the aiosqlite connection and reset the compiled graph singleton.

    After calling this, the next ``get_orchestrator_graph()`` call creates a
    fresh graph with a new database connection. Safe to call multiple times.
    """
    global _orchestrator_graph
    global _orchestrator_db_conn

    if _orchestrator_db_conn is not None:
        try:
            await _orchestrator_db_conn.close()
            logger.info("Closed orchestrator DB connection")
        except Exception:
            logger.exception("Error closing orchestrator DB connection")

    _orchestrator_graph = None
    _orchestrator_db_conn = None
