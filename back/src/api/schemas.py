"""Pydantic v2 request/response schemas mirroring frontend types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentEnum = Literal[
    "ingest",
    "retrieve",
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
    student_id: str | None = None


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
    topic: str = Field(default="")
    difficulty: DifficultyEnum = Field(default="medium")

    model_config = {"populate_by_name": True}


class Exam(BaseModel):
    id: str
    questions: list[ExamQuestion]
    topic: str
    difficulty: DifficultyEnum
    status: str = Field(default="complete")
    warnings: list[str] = Field(default_factory=list)
    topic_not_found: list[str] = Field(default_factory=list, alias="topicNotFound")
    topic_suggestions: list[str] = Field(default_factory=list, alias="topicSuggestions")

    model_config = {"populate_by_name": True}


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
    exam_questions: list[ExamQuestion] | None = Field(default=None, alias="examQuestions")
    student_id: str | None = Field(default=None, alias="studentId")


class StudentProfile(BaseModel):
    id: str
    topic_scores: dict[str, list[float]] = Field(alias="topicScores")
    weak_topics: list[str] = Field(alias="weakTopics")
    preferences: ExamPreferences
    session_count: int = Field(alias="sessionCount")
    session_history: list[dict] = Field(default_factory=list, alias="sessionHistory")

    model_config = {"populate_by_name": True}


class IngestResult(BaseModel):
    session_id: str = Field(alias="sessionId")
    status: str
    classification: str
    topics_detected: list[str] = Field(alias="topicsDetected")
    topic_tree: dict | None = Field(default=None, alias="topicTree")
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


class ExerciseStepSchema(BaseModel):
    """A single step in the multi-step model solution (mirrors agent ExerciseStep)."""

    step_number: int = Field(alias="stepNumber")
    description: str
    result: str
    source_chunk_ids: list[str] = Field(default_factory=list, alias="sourceChunkIds")

    model_config = {"populate_by_name": True}


class ExerciseModelSolution(BaseModel):
    steps: list[ExerciseStepSchema]
    final_answer: str = Field(alias="finalAnswer")
    key_concepts: list[str] = Field(alias="keyConcepts")
    source_chunk_ids: list[str] = Field(default_factory=list, alias="sourceChunkIds")

    model_config = {"populate_by_name": True}


class Exercise(BaseModel):
    exercise_id: str = ""
    statement: str = ""
    given_data: str | None = None
    question: str = ""
    model_solution: ExerciseModelSolution = Field(
        default_factory=lambda: ExerciseModelSolution(steps=[], final_answer="", key_concepts=[])
    )
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


class SessionCreate(BaseModel):
    name: str
    description: str = ""
    student_id: str = Field(alias="studentId")

    model_config = {"populate_by_name": True}


class SessionRename(BaseModel):
    name: str
    description: str | None = None


class Session(BaseModel):
    id: str
    name: str
    description: str = ""
    student_id: str = Field(alias="studentId")
    created_at: str = Field(alias="createdAt")
    status: str = "active"
    file_count: int = Field(default=0, alias="fileCount")
    exam_count: int = Field(default=0, alias="examCount")
    average_score: float | None = Field(default=None, alias="averageScore")

    model_config = {"populate_by_name": True}


class SessionFile(BaseModel):
    id: str
    file_name: str = Field(alias="fileName")
    classification: str = ""
    topics: list[str] = Field(default_factory=list)
    topic_tree: dict | None = Field(default=None, alias="topicTree")
    chunks_count: int = Field(default=0, alias="chunksCount")
    ingested_at: str = Field(alias="ingestedAt")

    model_config = {"populate_by_name": True}


class SessionProfile(BaseModel):
    session_id: str = Field(alias="sessionId")
    topic_scores: dict[str, list[float]] = Field(default_factory=dict, alias="topicScores")
    weak_topics: list[str] = Field(default_factory=list, alias="weakTopics")
    exam_count: int = Field(default=0, alias="examCount")
    average_score: float | None = Field(default=None, alias="averageScore")

    model_config = {"populate_by_name": True}


class ApiResponse[T](BaseModel):
    data: T
    error: str | None = None
    trace_id: str | None = None
