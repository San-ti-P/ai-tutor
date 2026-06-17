"""ExerciseGenerator Agent — ReAct + Tools for generating practice exercises."""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class ExerciseGeneratorState(TypedDict):
    session_id: str
    student_id: str
    topic: str
    difficulty: str
    exercise_type: str
    retrieved_chunks: Annotated[list[dict], operator.add]
    generated_exercise: dict
    validation_passed: bool
    retry_count: int
    exercise: dict
    status: str


def retrieve_relevant_chunks(state: ExerciseGeneratorState) -> dict:
    """Retrieve top-K relevant chunks from ChromaDB for the requested topic."""
    raise NotImplementedError


def generate_exercise(state: ExerciseGeneratorState) -> dict:
    """Generate a practice exercise using LLM with retrieved context."""
    raise NotImplementedError


def validate_exercise(state: ExerciseGeneratorState) -> dict:
    """Validate exercise is solvable and grounded in source material."""
    raise NotImplementedError


def should_retry(state: ExerciseGeneratorState) -> str:
    """Determine if exercise should be regenerated (max 3 retries)."""
    raise NotImplementedError


def format_exercise(state: ExerciseGeneratorState) -> dict:
    """Format validated exercise into final structure."""
    raise NotImplementedError


def build_exercise_generator() -> StateGraph:
    """Build and return the ExerciseGenerator LangGraph."""
    builder = StateGraph(ExerciseGeneratorState)

    builder.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
    builder.add_node("generate_exercise", generate_exercise)
    builder.add_node("validate_exercise", validate_exercise)
    builder.add_node("format_exercise", format_exercise)

    builder.add_edge(START, "retrieve_relevant_chunks")
    builder.add_edge("retrieve_relevant_chunks", "generate_exercise")
    builder.add_edge("generate_exercise", "validate_exercise")
    builder.add_conditional_edges(
        "validate_exercise",
        should_retry,
        {"retry": "generate_exercise", "done": "format_exercise"},
    )
    builder.add_edge("format_exercise", END)

    return builder
