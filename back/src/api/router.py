"""API router with all endpoints matching the frontend api.ts client."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, File, UploadFile

from src.api.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResult,
    Exam,
    ExamRequest,
    HealthResponse,
    IngestResult,
    StudentProfile,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    logger.info("Health check requested")
    return HealthResponse()


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat(request: ChatRequest) -> ApiResponse[ChatResponse]:
    logger.info("Chat request received for session %s", request.session_id)
    return ApiResponse(
        data=ChatResponse(
            response=(
                "Lo siento, los agentes del tutor aún no están implementados. "
                "Esta es una respuesta de infraestructura placeholder."
            ),
            intent="general_chat",
            trace_id=str(uuid.uuid4()),
        ),
        error=None,
    )


@router.post("/ingest", response_model=ApiResponse[IngestResult])
async def ingest(files: list[UploadFile] = File(...)) -> ApiResponse[IngestResult]:
    logger.info("Ingest request received with %d files", len(files))
    return ApiResponse(
        data=IngestResult(
            status="placeholder",
            classification="unknown",
            topics_detected=[],
            chunks_created=0,
        ),
        error=None,
    )


@router.post("/exam/generate", response_model=ApiResponse[Exam])
async def generate_exam(request: ExamRequest) -> ApiResponse[Exam]:
    logger.info("Exam generation requested for topic %s", request.topic)
    return ApiResponse(
        data=Exam(
            id=str(uuid.uuid4()),
            questions=[],
            topic=request.topic,
            difficulty=request.preferences.difficulty,
        ),
        error=None,
    )


@router.post("/evaluate", response_model=ApiResponse[list[EvaluationResult]])
async def evaluate(request: EvaluationRequest) -> ApiResponse[list[EvaluationResult]]:
    logger.info("Evaluation requested for exam %s", request.exam_id)
    return ApiResponse(
        data=[
            EvaluationResult(
                question_id=q_id,
                score=0.0,
                justification="Agentes de evaluación aún no implementados.",
                conceptual_errors=[],
                suggestions=[],
            )
            for q_id in request.answers
        ],
        error=None,
    )


@router.get("/profile/{session_id}", response_model=ApiResponse[StudentProfile])
async def get_profile(session_id: str) -> ApiResponse[StudentProfile]:
    logger.info("Profile requested for session %s", session_id)
    return ApiResponse(
        data=StudentProfile(
            id="placeholder-student",
            topic_scores={},
            weak_topics=[],
            preferences={
                "questionTypes": ["mcq"],
                "difficulty": "medium",
                "questionCount": 5,
                "includeTopics": [],
                "excludeTopics": [],
            },
            session_count=0,
        ),
        error=None,
    )
