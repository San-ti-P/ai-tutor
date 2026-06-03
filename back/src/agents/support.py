"""Support Agent — Reactive agent for student profile and progress queries."""

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class SupportState(TypedDict):
    session_id: str
    student_id: str
    query_type: str
    profile_data: dict | None
    session_history: list[dict]
    topic_scores: list[dict]
    response: str
    status: str


def fetch_student_profile(state: SupportState) -> dict:
    """Retrieve student profile data from SQLite."""
    raise NotImplementedError


def fetch_session_history(state: SupportState) -> dict:
    """Retrieve recent session history for the student."""
    raise NotImplementedError


def compute_progress_summary(state: SupportState) -> dict:
    """Compute progress summary across topics and sessions."""
    raise NotImplementedError


def generate_response(state: SupportState) -> dict:
    """Generate a natural language response about student progress."""
    raise NotImplementedError


def build_support_agent() -> StateGraph:
    """Build and return the Support Agent LangGraph."""
    builder = StateGraph(SupportState)

    builder.add_node("fetch_student_profile", fetch_student_profile)
    builder.add_node("fetch_session_history", fetch_session_history)
    builder.add_node("compute_progress_summary", compute_progress_summary)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "fetch_student_profile")
    builder.add_edge("fetch_student_profile", "fetch_session_history")
    builder.add_edge("fetch_session_history", "compute_progress_summary")
    builder.add_edge("compute_progress_summary", "generate_response")
    builder.add_edge("generate_response", END)

    return builder
