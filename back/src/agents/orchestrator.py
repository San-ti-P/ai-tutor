"""Orchestrator Agent — Plan-and-Execute with hub-and-spoke routing."""

import operator
from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

Intent = Literal[
    "ingest",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
    "general_chat",
    "composite",
]


class OrchestratorState(TypedDict):
    session_id: str
    user_message: str
    intent: Intent
    plan: list[str]
    current_step: int
    results: Annotated[list[dict], operator.add]
    response: str
    iteration_count: int


def classify_intent(state: OrchestratorState) -> dict:
    """Classify user message into one of 7 intents."""
    raise NotImplementedError


def route_to_agent(state: OrchestratorState) -> str:
    """Route classified intent to the matching agent node."""
    raise NotImplementedError


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
