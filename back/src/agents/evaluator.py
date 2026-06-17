"""Evaluator Agent — Chain-of-Thought for evaluating student answers."""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class EvaluatorState(TypedDict):
    session_id: str
    student_id: str
    question: dict
    student_answer: str
    answer_image: str | None
    ocr_extracted_text: str | None
    ocr_confidence: float
    evaluation: dict
    score: float
    feedback: str
    strengths: list[str]
    weaknesses: list[str]
    errors: Annotated[list[str], operator.add]
    status: str


def extract_answer_from_image(state: EvaluatorState) -> dict:
    """Run OCR on uploaded answer image to extract text and math expressions."""
    raise NotImplementedError


def evaluate_answer(state: EvaluatorState) -> dict:
    """Evaluate student answer against question using chain-of-thought reasoning."""
    raise NotImplementedError


def generate_feedback(state: EvaluatorState) -> dict:
    """Generate detailed feedback with strengths, weaknesses, and improvement suggestions."""
    raise NotImplementedError


def update_student_profile(state: EvaluatorState) -> dict:
    """Update student profile with new score and topic performance data."""
    raise NotImplementedError


def build_evaluator() -> StateGraph:
    """Build and return the Evaluator LangGraph."""
    builder = StateGraph(EvaluatorState)

    builder.add_node("extract_answer_from_image", extract_answer_from_image)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("generate_feedback", generate_feedback)
    builder.add_node("update_student_profile", update_student_profile)

    builder.add_edge(START, "extract_answer_from_image")
    builder.add_edge("extract_answer_from_image", "evaluate_answer")
    builder.add_edge("evaluate_answer", "generate_feedback")
    builder.add_edge("generate_feedback", "update_student_profile")
    builder.add_edge("update_student_profile", END)

    return builder
