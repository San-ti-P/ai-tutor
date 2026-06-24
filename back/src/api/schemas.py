"""Pydantic v2 request/response schemas mirroring frontend types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentEnum = Literal[
    "ingest",
    "generate_exam",
    "generate_exercise",
    "evaluate",
    "query_profile",
    "general_chat",
    "composite",
]

DifficultyEnum = Literal["easy", "medium", "hard"]

QuestionTypeEnum = Literal["mcq", "open"]

MessageRoleEnum = Literal["user", "assistant"]


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    intent: IntentEnum
    trace_id: str | None = None
    exam: dict | None = None  # Structured exam data for UI rendering (Epic 7 US-7.3)


class ExamPreferences(BaseModel):
    question_types: list[QuestionTypeEnum] = Field(alias="questionTypes")
    difficulty: DifficultyEnum
    question_count: int = Field(alias="questionCount")
    include_topics: list[str] = Field(alias="includeTopics")
    exclude_topics: list[str] = Field(alias="excludeTopics")

    model_config = {"populate_by_name": True}


class ExamRequest(BaseModel):
    session_id: str
    topic: str
    preferences: ExamPreferences


class ExamQuestion(BaseModel):
    id: str
    type: QuestionTypeEnum
    prompt: str
    options: list[str] | None = None
    base_answer: str | None = Field(default=None, alias="baseAnswer")
    source_chunk_ids: list[str] | None = Field(default=None, alias="sourceChunkIds")

    model_config = {"populate_by_name": True}


class Exam(BaseModel):
    id: str
    questions: list[ExamQuestion]
    topic: str
    difficulty: DifficultyEnum


class EvaluationResult(BaseModel):
    question_id: str = Field(alias="questionId")
    score: float
    justification: str
    conceptual_errors: list[str] = Field(alias="conceptualErrors")
    suggestions: list[str]
    is_evaluable: bool = Field(default=True, alias="isEvaluable")
    non_evaluable_reason: str = Field(default="", alias="nonEvaluableReason")
    requires_review: bool = Field(default=False, alias="requiresReview")
    judge_score: float | None = Field(default=None, alias="judgeScore")

    model_config = {"populate_by_name": True}


class EvaluationRequest(BaseModel):
    session_id: str
    exam_id: str
    answers: dict[str, str]


class StudentProfile(BaseModel):
    id: str
    topic_scores: dict[str, list[float]] = Field(alias="topicScores")
    weak_topics: list[str] = Field(alias="weakTopics")
    preferences: ExamPreferences
    session_count: int = Field(alias="sessionCount")

    model_config = {"populate_by_name": True}


class IngestResult(BaseModel):
    session_id: str = Field(alias="sessionId")
    status: str
    classification: str
    topics_detected: list[str] = Field(alias="topicsDetected")
    chunks_created: int = Field(alias="chunksCreated")
    classification_confidence: float | None = Field(default=None, alias="classificationConfidence")
    document_id: str | None = Field(default=None, alias="documentId")

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ExerciseRequest(BaseModel):
    session_id: str
    topic: str
    difficulty: DifficultyEnum = "medium"
    exercise_type: str = "problem_solving"


class ExerciseModelSolution(BaseModel):
    steps: list[str]
    final_answer: str
    key_concepts: list[str]


class Exercise(BaseModel):
    exercise_id: str = ""
    statement: str = ""
    given_data: str | None = None
    question: str = ""
    model_solution: ExerciseModelSolution = Field(default_factory=lambda: ExerciseModelSolution(
        steps=[], final_answer="", key_concepts=[]
    ))
    topics_covered: list[str] = []
    source_chunk_ids: list[str] | None = None
    topic_not_found: list[str] = []
    topic_suggestions: list[str] = []
    status: str = ""


class PreferencesUpdate(BaseModel):
    question_types: list[QuestionTypeEnum] = Field(alias="questionTypes")
    difficulty: DifficultyEnum
    question_count: int = Field(alias="questionCount")
    include_topics: list[str] = Field(alias="includeTopics")
    exclude_topics: list[str] = Field(alias="excludeTopics")

    model_config = {"populate_by_name": True}


class PreferencesStatus(BaseModel):
    status: str
    student_id: str = ""
    upserted_topics: int = 0
    errors: list[str] = []


class ApiResponse[T](BaseModel):
    data: T
    error: str | None = None
    trace_id: str | None = None
