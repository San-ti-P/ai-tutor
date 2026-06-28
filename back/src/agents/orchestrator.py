"""Orchestrator Agent — Plan-and-Execute with hub-and-spoke routing."""

from __future__ import annotations

import asyncio
import logging
import operator
import os
import sqlite3
from typing import Annotated, Literal, NotRequired

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import settings
from src.llm import get_llm as _get_llm
from src.llm import get_structured_llm
from src.memory.schema import resolve_student_id

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
    from src.tools.update_student_profile import update_student_profile

    TOOL_MAP = {
        "ingest": ingest_document,
        "retrieve": query_material,
        "generate_exam": generate_exam,
        "generate_exercise": generate_exercise,
        "evaluate": evaluate_answer,
        "query_profile": get_student_summary,
        "update_student_profile": update_student_profile,
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
    student_id: NotRequired[str | None]
    session_context: NotRequired[dict | None]
    messages_history: NotRequired[Annotated[list, operator.add]]
    profile_load_error: NotRequired[str | None]


async def load_session_context(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Load session context: uploaded files and per-session progress.

    Runs after ``load_profile`` and before ``classify_intent``.
    Populates ``session_context`` with ``files`` (from ``list_session_files``)
    and ``progress`` (from ``get_session_profile``).

    Failures are caught individually so one broken data source does not
    block the other. A partial context is better than none.
    """
    from src.memory.schema import get_session_profile, list_session_files

    session_id = state["session_id"]

    # ── Files ────────────────────────────────────────────────────────────
    try:
        rows = await list_session_files(session_id)
        files = []
        for row in rows:
            topics: list[str] = []
            topics_json = row.get("topics_json")
            if topics_json:
                try:
                    import json as _json

                    topics = _json.loads(topics_json)
                except Exception:
                    pass
            files.append(
                {
                    "id": row["id"],
                    "file_name": row["file_name"],
                    "classification": row.get("classification") or "",
                    "topics": topics,
                    "chunks_count": row.get("chunks_count", 0),
                    "ingested_at": row["ingested_at"],
                    "session_id": row.get("session_id", session_id),
                }
            )
    except (sqlite3.OperationalError, aiosqlite.Error, KeyError):
        logger.exception("Failed to load session files for %s", session_id)
        files = []

    # ── Progress ─────────────────────────────────────────────────────────
    try:
        profile = await get_session_profile(session_id)
        progress = (
            {
                "topic_scores": profile.get("topic_scores", {}),
                "weak_topics": profile.get("weak_topics", []),
                "exam_count": profile.get("exam_count", 0),
                "average_score": profile.get("average_score"),
            }
            if profile
            else {
                "topic_scores": {},
                "weak_topics": [],
                "exam_count": 0,
                "average_score": None,
            }
        )
    except (sqlite3.OperationalError, aiosqlite.Error, KeyError):
        logger.exception("Failed to load session progress for %s", session_id)
        progress = {
            "topic_scores": {},
            "weak_topics": [],
            "exam_count": 0,
            "average_score": None,
        }

    return {"session_context": {"files": files, "progress": progress}}


async def load_profile(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Load the student profile at session bootstrap.

    Resolves ``student_id`` from the session row (or override), calls
    ``get_student_summary``, and stores the result in ``student_profile``.
    Any failure falls back to an empty dict so the session continues.
    """
    from src.tools.get_student_summary import get_student_summary

    session_id = state["session_id"]
    student_id_override = state.get("student_id")

    try:
        student_id = await resolve_student_id(session_id, student_id_override)
        profile = await get_student_summary.ainvoke({"student_id": student_id})
        if profile is None:
            return {"student_profile": {}, "student_id": student_id}
        return {"student_profile": profile, "student_id": student_id}
    except Exception:
        logger.exception(
            "Failed to load profile for session %s, falling back to empty profile",
            session_id,
        )
        return {
            "student_profile": {},
            "profile_load_error": (
                "No pude cargar tu perfil. "
                "Los resultados pueden no estar personalizados."
            ),
        }


def classify_intent(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Classify user message into one of 8 intents with confidence score.

    On low confidence (< settings.classification_confidence_threshold), forces
    general_chat. On any exception, returns general_chat with confidence=0.0.
    Single-tool intents pre-populate plan with the tool name.
    """
    import time

    message = state["user_message"]
    profile = state.get("student_profile")
    t0 = time.monotonic()

    try:
        structured = get_structured_llm(IntentClassification)

        prompt = (
            "Sos un clasificador de intents para un tutor académico. "
            "Clasificá el mensaje del usuario en UNA de estas 8 categorías:\n\n"
            "  - ingest: el usuario quiere SUBIR apuntes, PDFs, documentos\n"
            "  - retrieve: preguntar/consultar sobre el contenido de apuntes"
            " o documentos YA SUBIDOS\n"
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
            "3. ingest SOLO si el usuario quiere SUBIR un archivo; retrieve SOLO"
            " si pregunta por contenido ya subido\n"
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
        context_parts: list[str] = []
        if profile:
            weak = profile.get("weak_topics", [])
            if weak:
                context_parts.append(f"temas débiles: {weak}")
            prefs = profile.get("preferences")
            if prefs:
                context_parts.append(f"preferencias: {prefs}")
        session_context = state.get("session_context")
        if session_context:
            files = session_context.get("files")
            if files:
                context_parts.append(f"archivos cargados: {[f.get('file_name') for f in files]}")
            progress = session_context.get("progress")
            if progress:
                context_parts.append(f"progreso: {progress}")
        if context_parts:
            prompt += "Contexto del estudiante: " + "; ".join(context_parts) + "\n"

        history = state.get("messages_history", [])
        if history:
            recent = history[-10:]
            prompt += "Últimos mensajes de la conversación:\n"
            for h in recent:
                role = "usuario" if h.get("role") == "user" else "asistente"
                prompt += f"- {role}: {h.get('content', '')}\n"

        prompt += (
            "Respondé SOLO con un objeto JSON con claves 'intent' y 'confidence'. "
            "El output SIEMPRE debe ser en español."
        )

        invoke_kwargs = {"config": config} if config is not None else {}
        result = None

        # First attempt
        try:
            result = structured.invoke(prompt, **invoke_kwargs)
        except Exception as first_err:
            logger.warning(
                "[classify_intent] First attempt failed: %s. Retrying.",
                first_err,
            )
            # Retry once with a fresh structured LLM (temperature=0 implicit)
            retry_structured = get_structured_llm(IntentClassification)
            result = retry_structured.invoke(prompt, **invoke_kwargs)

        intent = result.intent
        confidence = result.confidence

        # Low-confidence fallback
        if confidence < settings.classification_confidence_threshold:
            intent = "general_chat"

        # Pre-populate plan for single-tool intents
        plan: list[str] = []
        if intent in _SINGLE_TOOL_INTENTS:
            plan = [intent]

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[classify_intent] COMPLETE | session=%s | intent=%s | confidence=%.2f | %dms",
            state["session_id"],
            intent,
            confidence,
            int(elapsed),
        )
        return {"intent": intent, "confidence": confidence, "plan": plan}

    except Exception:
        logger.exception(
            "classify_intent failed, falling back to general_chat. "
            "Message preview: %.200r",
            message,
        )
        return {"intent": "general_chat", "confidence": 0.0, "plan": []}


def route_to_agent(state: OrchestratorState) -> str:
    """Route classified intent to the matching agent node.

    composite → plan_composite
    general_chat → synthesize_response
    Any single-tool intent → execute_step
    """
    intent = state["intent"]
    if intent == "composite":
        target = "plan_composite"
    elif intent == "general_chat":
        target = "synthesize_response"
    else:
        target = "execute_step"
    logger.info(
        "[route] %s | session=%s | intent=%s",
        target,
        state.get("session_id", "?"),
        intent,
    )
    return target


def plan_composite(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Plan steps for composite (multi-step) tasks.

    Uses LLM planner to generate an ordered list of tool names.
    Strips tools not in TOOL_MAP. Empty plan → treated as general_chat downstream.
    """
    import time

    message = state["user_message"]
    t0 = time.monotonic()
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
            'con clave "steps". El output SIEMPRE debe ser en español. '
            'Ejemplo: {"steps": ["generate_exam", "evaluate"]}'
        )

        invoke_kwargs = {"config": config} if config is not None else {}
        result = structured.invoke(prompt, **invoke_kwargs)
        plan = result.steps

        # Strip invalid tool names
        valid_plan = [step for step in plan if step in tool_map]

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[plan_composite] COMPLETE | session=%s | steps=%s | %dms",
            state["session_id"],
            valid_plan,
            int(elapsed),
        )
        return {"plan": valid_plan}

    except Exception:
        logger.exception("plan_composite failed")
        return {"plan": []}


def _extract_topics(message: str) -> list[str]:
    """Extract topic keywords from user message using simple heuristics.

    Looks for phrases like 'sobre X', 'acerca de X', 'de X'.
    Returns ['general'] if no topics detected — caller should enrich
    from session context when available.
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
                topics = re.split(r"\s+(?:y|e)\s+|,\s*", topic_text)
                return [t.strip() for t in topics if len(t.strip()) > 2]

    return ["general"]


def _get_session_topics(session_context: dict | None) -> list[str]:
    """Extract all topics from ingested files in session context."""
    if not session_context:
        return []
    files = session_context.get("files", [])
    all_topics: list[str] = []
    for f in files:
        file_topics = f.get("topics", [])
        if isinstance(file_topics, list):
            all_topics.extend(file_topics)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in all_topics:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


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
        topics = _extract_topics(state["user_message"])
        # When user doesn't specify a topic, enrich from session files
        if topics == ["general"]:
            session_topics = _get_session_topics(state.get("session_context"))
            if session_topics:
                topics = session_topics[:20]  # Cap at 20 for token budget
        args["topics"] = topics
        args["difficulty"] = _extract_difficulty(state["user_message"])
        args["question_count"] = _extract_question_count(state["user_message"])
        args["mcq_ratio"] = 0.5
        if profile:
            args["student_profile"] = profile
    elif tool_name == "generate_exercise":
        topics = _extract_topics(state["user_message"])
        if topics == ["general"]:
            session_topics = _get_session_topics(state.get("session_context"))
            if session_topics:
                topics = session_topics[:5]
        args["topic"] = topics[0] if topics else "general"
        args["difficulty"] = _extract_difficulty(state["user_message"])
        args["exercise_type"] = "problem_solving"
        if profile:
            args["student_profile"] = profile
    elif tool_name == "evaluate":
        args["exam_id"] = ""
        args["answers"] = []
        resolved_sid = state.get("student_id") or state["session_id"]
        args["student_id"] = resolved_sid
    elif tool_name == "update_student_profile":
        resolved_sid = state.get("student_id") or state["session_id"]
        args["student_id"] = resolved_sid
        args["topic_scores"] = {}
        args["preferences"] = None
        if profile:
            prefs = profile.get("preferences", {})
            args["preferences"] = prefs
    elif tool_name == "query_profile":
        args["student_id"] = state.get("student_id") or state.get("session_id", "unknown")

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
    import time

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
    t0 = time.monotonic()
    logger.info(
        "[execute_step] START | session=%s | step=%d/%d | tool=%s",
        state["session_id"],
        current + 1,
        len(plan),
        tool_name,
    )

    if tool is None:
        logger.warning("Tool '%s' not found in TOOL_MAP", tool_name)
        return {
            "errors": errors
            + [
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
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[execute_step] COMPLETE | session=%s | step=%d | tool=%s | %dms",
            state["session_id"],
            current + 1,
            tool_name,
            int(elapsed),
        )
        return {
            "results": results + [{"step": current, "tool": tool_name, "result": result}],
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning(
            "execute_step failed for tool '%s' at step %d (%dms): %s",
            tool_name,
            current,
            int(elapsed),
            exc,
        )
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
        r"[¿\?]",  # Contains question marks
        r"qu[eé]\s+(es|son|significa)",  # "qué es/son/significa"
        r"explic[aá]",  # "explica/explicá"
        r"defin[ií]",  # "definí/define"
        r"c[oó]mo\s+(se\s+)?(calcula|resuelve|determina|obtiene|halla)",
        r"cu[aá]l\s+(es|son)",  # "cuál es/son"
        r"diferencia\s+entre",
        r"qu[eé]\s+(dice|dice\s+el|habla|trata|contiene)",
        r"en\s+qu[eé]\s+(consiste|se\s+basa)",
        r"mencion[aá]",  # "menciona/mencioná"
        r"describ[ií]",  # "describe/describí"
        r"caracter[ií]sticas?\s+de",
    ]
    for pat in academic_signals:
        if re.search(pat, msg):
            return True

    return False


def _build_enrichment_context(state: OrchestratorState) -> str:
    """Build a context block with profile, session_context, and messages_history.

    Returns an empty string when no enrichment data is available so the
    prompt stays lean for first-use / empty sessions.
    """
    parts: list[str] = []

    profile = state.get("student_profile")
    if profile:
        weak = profile.get("weak_topics", [])
        if weak:
            parts.append(f"Los temas más débiles del estudiante son: {', '.join(weak)}.")
        prefs = profile.get("preferences")
        if prefs:
            parts.append(f"Preferencias del estudiante: {prefs}.")

    session_context = state.get("session_context")
    if session_context:
        files = session_context.get("files", [])
        if files:
            names = [f.get("file_name", "?") for f in files]
            parts.append(f"Archivos cargados en esta sesión: {', '.join(names)}.")
            topics = {topic for f in files for topic in (f.get("topics") or [])}
            if topics:
                parts.append(f"Temas cubiertos por los archivos: {', '.join(sorted(topics))}.")
        progress = session_context.get("progress")
        if progress:
            weak = progress.get("weak_topics", [])
            if weak:
                parts.append(f"Temas débiles en esta sesión: {', '.join(weak)}.")
            avg = progress.get("average_score")
            if avg is not None:
                parts.append(f"Nota promedio en esta sesión: {avg:.1f}/10.")
            exam_count = progress.get("exam_count", 0)
            if exam_count:
                parts.append(f"Exámenes realizados en esta sesión: {exam_count}.")

    history = state.get("messages_history", [])
    if history:
        recent = history[-6:]  # last 3 exchanges
        if recent:
            lines = []
            for h in recent:
                role = "Estudiante" if h.get("role") == "user" else "Tutor"
                lines.append(f"- {role}: {h.get('content', '')}")
            parts.append("Últimos mensajes de la conversación:\n" + "\n".join(lines))

    if not parts:
        return ""

    return "\n\n".join(parts)


def synthesize_response(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    """Combine results from agent executions into a final response.

    - general_chat: LLM synthesizes direct answer from user_message.
    - composite/single: LLM aggregates results into coherent narrative.
    - incomplete: prepends cap warning in Spanish.
    - partial: includes error summary.
    - On LLM failure: hardcoded Spanish apology + raw results.
    """
    import json
    import time

    from src.rag.policy import RAG_ONLY_SYSTEM_PROMPT, no_material_message

    intent = state["intent"]
    message = state["user_message"]
    results = state.get("results", [])
    errors = state.get("errors", [])
    status = state.get("status", "pending")
    t0 = time.monotonic()

    logger.info(
        "[synthesize_response] START | session=%s | intent=%s | results=%d | errors=%d",
        state["session_id"],
        intent,
        len(results),
        len(errors),
    )

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

    # Prepend profile load error warning if present
    profile_load_error = state.get("profile_load_error")
    if profile_load_error:
        prefix = f"ℹ️ {profile_load_error}\n\n{prefix}"

    # Build prompt for LLM synthesis
    try:
        llm = _get_llm()

        if intent == "general_chat" and not results:
            # Academic probe: check if this looks like an academic question
            if _is_academic_question(message):
                from src.tools import query_material as _query_material

                try:
                    qm_result = _query_material.invoke(
                        {
                            "query": message,
                            "session_id": state["session_id"],
                            "top_k": 3,  # lighter probe
                        }
                    )

                    if qm_result.get("chunks_found", 0) > 0:
                        # RAG-grounded: synthesize with chunks
                        chunks_text = "\n\n".join(qm_result.get("sources", []))
                        prompt = (
                            f"{RAG_ONLY_SYSTEM_PROMPT}\n\n"
                            f"Fragmentos del material:\n{chunks_text}\n\n"
                            f"Pregunta del estudiante: {message}\n\n"
                            "Respondé SIEMPRE en español, de forma clara y educativa, "
                            "citando los fragmentos relevantes."
                        )
                    else:
                        # No chunks — return canonical no-material message
                        elapsed = (time.monotonic() - t0) * 1000
                        logger.info(
                            "[synthesize_response] COMPLETE | session=%s | no_material | %dms",
                            state["session_id"],
                            int(elapsed),
                        )
                        return {
                            "response": no_material_message(),
                            "status": "complete",
                        }
                except Exception as exc:
                    logger.warning("Academic probe retrieval failed: %s", exc)
                    elapsed = (time.monotonic() - t0) * 1000
                    logger.info(
                        "[synthesize_response] COMPLETE | session=%s | probe_failed | %dms",
                        state["session_id"],
                        int(elapsed),
                    )
                    return {
                        "response": no_material_message(),
                        "status": "complete",
                    }
            else:
                enrichment = _build_enrichment_context(state)
                prompt = (
                    f'El usuario preguntó: "{message}"\n'
                    "Respondé SIEMPRE en español, de forma clara y educativa."
                )
                if enrichment:
                    prompt = (
                        f"Contexto adicional sobre el estudiante y la sesión:\n{enrichment}\n\n"
                    ) + prompt
        else:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)
            errors_str = json.dumps(errors, ensure_ascii=False, indent=2) if errors else "ninguno"
            enrichment = _build_enrichment_context(state)

            prompt = (
                f'Consulta original del usuario: "{message}"\n\n'
                f"Resultados de las herramientas ejecutadas:\n{results_str}\n\n"
                f"Errores encontrados:\n{errors_str}\n\n"
            )
            if enrichment:
                prompt = (
                    f"Contexto adicional sobre el estudiante y la sesión:\n{enrichment}\n\n"
                ) + prompt
            if status == "partial":
                prompt += (
                    "Algunas tareas no se completaron exitosamente. "
                    "Generá una respuesta que resuma lo logrado y mencione los errores. "
                    "Respondé SIEMPRE en español. Sé honesto pero alentador."
                )
            else:
                prompt += (
                    "Generá una respuesta coherente que resuma todos estos resultados "
                    "de forma clara y educativa. Respondé SIEMPRE en español."
                )

        invoke_kwargs = {"config": config} if config is not None else {}
        response = llm.invoke(prompt, **invoke_kwargs)
        text = response.content if hasattr(response, "content") else str(response)

        final_status = "complete" if status == "pending" else status
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[synthesize_response] COMPLETE | session=%s | status=%s | %dms",
            state["session_id"],
            final_status,
            int(elapsed),
        )
        return {
            "response": prefix + text,
            "status": final_status,
            "messages_history": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefix + text},
            ],
        }

    except Exception:
        logger.exception("synthesize_response LLM failed, using fallback")
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[synthesize_response] COMPLETE | session=%s | fallback | %dms",
            state["session_id"],
            int(elapsed),
        )
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
        logger.info(
            "[check_iteration_limit] terminate | session=%s | reason=partial_error",
            state["session_id"],
        )
        return "terminate"
    if state["iteration_count"] >= settings.max_iterations_per_task:
        logger.info(
            "[check_iteration_limit] terminate | session=%s | reason=iteration_limit | iter=%d",
            state["session_id"],
            state["iteration_count"],
        )
        return "terminate"
    if state["current_step"] >= len(state["plan"]):
        logger.info(
            "[check_iteration_limit] terminate | session=%s | reason=plan_complete | steps=%d",
            state["session_id"],
            state["current_step"],
        )
        return "terminate"
    return "continue"


def build_orchestrator() -> StateGraph:
    """Build and return the Orchestrator LangGraph."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("load_profile", load_profile)
    builder.add_node("load_session_context", load_session_context)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("plan_composite", plan_composite)
    builder.add_node("execute_step", execute_step)
    builder.add_node("synthesize_response", synthesize_response)

    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "load_session_context")
    builder.add_edge("load_session_context", "classify_intent")
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
_orchestrator_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Return the module-level asyncio.Lock for singleton init.

    Must be a function (not module-level =) to avoid creating the lock
    outside a running event loop.
    """
    global _orchestrator_lock
    if _orchestrator_lock is None:
        _orchestrator_lock = asyncio.Lock()
    return _orchestrator_lock


async def get_orchestrator_graph():
    """Return the module-level compiled Orchestrator graph singleton.

    Compiled ONCE with AsyncSqliteSaver for session persistence.
    Falls back to InMemorySaver if DB connection fails due to missing
    dependencies (ImportError), database errors (aiosqlite.Error), or
    filesystem issues (OSError). Unexpected errors propagate.

    Serialized via asyncio.Lock — only one caller enters the init
    block, preventing double DB connections under concurrent startup.
    """
    global _orchestrator_graph
    global _orchestrator_db_conn

    lock = _get_lock()
    async with lock:
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

    Sets a 5-second drain period: in-flight `graph.ainvoke()` calls are
    given time to complete before the DB connection is closed.

    After calling this, the next ``get_orchestrator_graph()`` call creates a
    fresh graph with a new database connection. Safe to call multiple times.
    """
    global _orchestrator_graph
    global _orchestrator_db_conn

    # Drain in-flight requests: wait 5 seconds for active graph invocations
    if _orchestrator_graph is not None:
        logger.info("Draining in-flight orchestrator requests (5s)...")
        try:
            await asyncio.wait_for(
                asyncio.sleep(0), timeout=5.0
            )  # Allow async tasks to yield
        except asyncio.TimeoutError:
            pass
        # Brief drain sleep to let pending coroutines finish
        await asyncio.sleep(5)

    if _orchestrator_db_conn is not None:
        try:
            await _orchestrator_db_conn.close()
            logger.info("Closed orchestrator DB connection")
        except Exception:
            logger.exception("Error closing orchestrator DB connection")

    _orchestrator_graph = None
    _orchestrator_db_conn = None
