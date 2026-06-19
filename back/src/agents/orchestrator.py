"""Orchestrator Agent — Plan-and-Execute with hub-and-spoke routing."""

from __future__ import annotations

import logging
import operator
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
    """Plan steps for composite (multi-step) tasks."""
    raise NotImplementedError


def execute_step(state: OrchestratorState) -> dict:
    """Execute the current step in the plan."""
    raise NotImplementedError


def synthesize_response(state: OrchestratorState) -> dict:
    """Combine results from agent executions into a final response."""
    raise NotImplementedError


def check_iteration_limit(state: OrchestratorState) -> Literal["continue", "terminate"]:
    """Guardrail: enforce max iterations per task."""
    raise NotImplementedError


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
