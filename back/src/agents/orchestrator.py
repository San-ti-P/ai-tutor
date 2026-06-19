"""Orchestrator Agent — Plan-and-Execute with hub-and-spoke routing."""

from __future__ import annotations

import logging
import operator
import os
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import settings

logger = logging.getLogger(__name__)

Intent = Literal[
    "ingest",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
    "general_chat",
    "composite",
]

_SINGLE_TOOL_INTENTS: set[str] = {
    "ingest",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
}


class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)


class CompositePlan(BaseModel):
    steps: list[str] = Field(description="Ordered tool names from TOOL_MAP")


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
    )
    from src.tools.get_student_summary import get_student_summary

    TOOL_MAP = {
        "ingest": ingest_document,
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
    results: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    response: str
    status: str  # "pending" | "complete" | "incomplete" | "partial"
    iteration_count: int
    student_profile: dict | None


def _get_llm():
    """Return a configured LLM instance for the current provider."""
    llm_cls, llm_kwargs = settings.llm_kwargs
    return llm_cls(**llm_kwargs)


def classify_intent(state: OrchestratorState) -> dict:
    """Classify user message into one of 7 intents with confidence score.

    On low confidence (< settings.classification_confidence_threshold), forces
    general_chat. On any exception, returns general_chat with confidence=0.0.
    Single-tool intents pre-populate plan with the tool name.
    """
    message = state["user_message"]
    profile = state.get("student_profile")

    try:
        llm = _get_llm()
        structured = llm.with_structured_output(IntentClassification)

        prompt = "Clasificá la siguiente consulta en una de estas categorías:\n"
        prompt += "- ingest: subir/apuntes/documentos\n"
        prompt += "- generate_exam: generar un examen\n"
        prompt += "- generate_exercise: generar un ejercicio práctico\n"
        prompt += "- evaluate: evaluar/corregir una respuesta\n"
        prompt += "- query_profile: consultar perfil/progreso\n"
        prompt += "- general_chat: charla general, saludo, pregunta no académica\n"
        prompt += "- composite: múltiples tareas combinadas\n\n"
        prompt += f"Consulta: {message}\n"
        if profile:
            weak = profile.get("weak_topics", [])
            if weak:
                prompt += f"Perfil del estudiante (temas débiles): {weak}\n"
        prompt += "Respondé SOLO con la clasificación en formato JSON."

        result = structured.invoke(prompt)
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


def plan_composite(state: OrchestratorState) -> dict:
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
        llm = _get_llm()
        structured = llm.with_structured_output(CompositePlan)

        prompt = (
            "Sos un planificador de tareas académicas. El usuario quiere:\n"
            f'"{message}"\n\n'
            "Herramientas disponibles:\n"
            f"{tool_descriptions}\n\n"
            "Generá una lista ORDENADA de nombres de herramientas a ejecutar. "
            "Solo usá herramientas de la lista. Respondé SOLO en formato JSON."
        )

        result = structured.invoke(prompt)
        plan = result.steps

        # Strip invalid tool names
        valid_plan = [step for step in plan if step in tool_map]

        return {"plan": valid_plan}

    except Exception:
        logger.exception("plan_composite failed")
        return {"plan": []}


def _build_tool_args(tool_name: str, state: OrchestratorState) -> dict:
    """Build argument dict for a tool call from the current state.

    Each tool gets the fields it needs extracted from shared state.
    """
    args: dict = {"session_id": state["session_id"]}
    profile = state.get("student_profile")

    if tool_name == "ingest":
        # ingest_document needs file_path, session_id
        pass  # file_path not in orchestrator state; tool will use its own
    elif tool_name == "generate_exam":
        args["topics"] = ["general"]
        args["difficulty"] = "medium"
        args["question_count"] = 5
        args["mcq_ratio"] = 0.5
        if profile:
            args["student_profile"] = profile
    elif tool_name == "generate_exercise":
        args["topic"] = profile.get("weak_topics", ["general"])[0] if profile else "general"
        args["difficulty"] = "medium"
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
            raise ValueError(
                f"Tool '{tool.name}' failed after retry: {second_err}"
            ) from second_err


async def execute_step(state: OrchestratorState) -> dict:
    """Execute the current step in the plan.

    Resolves tool from TOOL_MAP, builds args, invokes with one retry, appends result.
    Increments current_step and iteration_count. On failure, records error.
    """
    plan = state["plan"]
    current = state["current_step"]
    iteration = state["iteration_count"]

    tool_map = _init_tool_map()

    # Safeguard: empty plan
    if not plan or current >= len(plan):
        return {"current_step": current, "iteration_count": iteration}

    tool_name = plan[current]
    tool = tool_map.get(tool_name)

    if tool is None:
        logger.warning("Tool '%s' not found in TOOL_MAP", tool_name)
        return {
            "errors": [
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
            "results": [{"step": current, "tool": tool_name, "result": result}],
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }
    except Exception as exc:
        logger.warning("execute_step failed for tool '%s': %s", tool_name, exc)
        return {
            "errors": [{"step": current, "tool": tool_name, "error": str(exc)}],
            "status": "partial",
            "current_step": current + 1,
            "iteration_count": iteration + 1,
        }


def synthesize_response(state: OrchestratorState) -> dict:
    """Combine results from agent executions into a final response.

    - general_chat: LLM synthesizes direct answer from user_message.
    - composite/single: LLM aggregates results into coherent narrative.
    - incomplete: prepends cap warning in Spanish.
    - partial: includes error summary.
    - On LLM failure: hardcoded Spanish apology + raw results.
    """
    import json

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
            prompt = (
                f"El usuario preguntó: \"{message}\"\n"
                "Respondé de forma clara y educativa en español."
            )
        else:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)
            errors_str = json.dumps(errors, ensure_ascii=False, indent=2) if errors else "ninguno"

            prompt = (
                f"Consulta original del usuario: \"{message}\"\n\n"
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

        response = llm.invoke(prompt)
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


async def get_orchestrator_graph():
    """Return the module-level compiled Orchestrator graph singleton.

    Compiled ONCE with AsyncSqliteSaver for session persistence.
    Falls back to InMemorySaver if DB connection fails.
    """
    global _orchestrator_graph
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
            logger.info("Orchestrator using AsyncSqliteSaver at %s", settings.sqlite_db_path)
        except Exception:
            logger.warning(
                "AsyncSqliteSaver unavailable at %s, using InMemorySaver",
                settings.sqlite_db_path,
            )
            checkpointer = InMemorySaver()

        _orchestrator_graph = build_orchestrator().compile(checkpointer=checkpointer)

    return _orchestrator_graph
