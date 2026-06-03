"""API router with all endpoints matching the frontend api.ts client."""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.agents.ingestor import IngestorState, build_ingestor
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
    OcrExpression,
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


@router.post("/ingest", response_model=ApiResponse[list[IngestResult]])
async def ingest(files: list[UploadFile] = File(...)) -> ApiResponse[list[IngestResult]]:
    logger.info("Ingest request received with %d file(s)", len(files))
    results: list[IngestResult] = []
    request_session_id = str(uuid.uuid4())

    for file in files:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            # Build and invoke the Ingestor graph
            graph = build_ingestor().compile()
            session_id = request_session_id

            initial_state: IngestorState = {
                "session_id": session_id,
                "file_path": tmp_path,
                "file_type": "",
                "raw_text": "",
                "classification": "",
                "topics": [],
                "chunks_created": 0,
                "ocr_confidence": 0.0,
                "needs_ocr_confirmation": False,
                "errors": [],
                "status": "pending",
                "classification_confidence": 0.0,
                "ocr_expressions": [],
                "document_id": "",
                "chunk_ids": [],
            }

            result = await graph.ainvoke(initial_state)

            # Build low-confidence OCR expressions list if present
            low_ocr: list[OcrExpression] | None = None
            ocr_exprs: list[dict] = result.get("ocr_expressions", [])
            if ocr_exprs and result.get("needs_ocr_confirmation", False):
                low_ocr = [
                    OcrExpression(
                        expression=e.get("expression", ""),
                        confidence=float(e.get("confidence", 0.0)),
                    )
                    for e in ocr_exprs
                    if isinstance(e, dict)
                    and float(e.get("confidence", 1.0)) < 0.85
                ]
                if not low_ocr:
                    low_ocr = None

            results.append(
                IngestResult(
                    status=result.get("status", "unknown"),
                    classification=result.get("classification", ""),
                    topicsDetected=result.get("topics", []),
                    chunksCreated=result.get("chunks_created", 0),
                    classificationConfidence=result.get("classification_confidence"),
                    lowConfidenceOcr=low_ocr,
                    documentId=result.get("document_id"),
                )
            )
        except Exception:
            logger.exception("Ingest failed for %s", file.filename)
            results.append(
                IngestResult(
                    status="error",
                    classification="",
                    topicsDetected=[],
                    chunksCreated=0,
                )
            )
        finally:
            await file.close()
            Path(tmp_path).unlink(missing_ok=True)
    return ApiResponse(
        data=results,
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
