"""API router with all endpoints matching the frontend api.ts client."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.agents.ingestor import IngestorState, build_ingestor
from src.api.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResult,
    Exam,
    ExamPreferences,
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


@router.post("/ingest", response_model=ApiResponse[list[IngestResult]])
async def ingest(files: list[UploadFile] = File(...)) -> ApiResponse[list[IngestResult]]:
    logger.info("Ingest request received with %d file(s)", len(files))
    results: list[IngestResult] = []
    request_session_id = str(uuid.uuid4())

    # Compile once — the compiled graph is stateless and safe to reuse
    graph = build_ingestor().compile()

    for file in files:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
            tmp_path = tmp.name

        try:
            session_id = request_session_id

            initial_state: IngestorState = {
                "session_id": session_id,
                "file_path": tmp_path,
                "file_type": "",
                "raw_text": "",
                "classification": "",
                "classification_confidence": 0.0,
                "topics": [],
                "chunks_created": 0,
                "errors": [],
                "status": "pending",
                "document_id": "",
                "chunk_ids": [],
            }

            result = await graph.ainvoke(initial_state)

            results.append(
                IngestResult(
                    status=result.get("status", "unknown"),
                    classification=result.get("classification", ""),
                    topicsDetected=result.get("topics", []),
                    chunksCreated=result.get("chunks_created", 0),
                    classificationConfidence=result.get("classification_confidence"),
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

    from src.tools import evaluate_answer as _evaluate_tool

    # Convert dict[str, str] answers to list[dict] format expected by evaluator
    answers_list: list[dict] = []
    for question_id, student_answer in request.answers.items():
        answers_list.append(
            {
                "question_id": question_id,
                "question": "",  # Would come from exam data in full impl
                "base_answer": "",  # Would come from exam data
                "student_answer": student_answer,
                "source_chunk_ids": [],
                "topic": "",
                "difficulty": "medium",
            }
        )

    try:
        results = _evaluate_tool.invoke(
            {
                "session_id": request.session_id,
                "exam_id": request.exam_id,
                "answers": answers_list,
                "student_id": "",
            }
        )

        evaluation_results = [
            EvaluationResult(
                question_id=r.get("question_id", ""),
                score=r.get("score", 0.0),
                justification=r.get("justification", ""),
                conceptual_errors=r.get("conceptual_errors", []),
                suggestions=r.get("suggestions", []),
                is_evaluable=r.get("is_evaluable", True),
                non_evaluable_reason=r.get("non_evaluable_reason", ""),
                requires_review=r.get("requires_review", False),
                judge_score=r.get("judge_verdict", {}).get("score")
                if isinstance(r.get("judge_verdict"), dict)
                else None,
            )
            for r in results
        ]

        return ApiResponse(data=evaluation_results, error=None)

    except Exception as exc:
        logger.exception("Evaluation failed for exam %s", request.exam_id)
        return ApiResponse(
            data=[
                EvaluationResult(
                    question_id=q_id,
                    score=0.0,
                    justification=f"Evaluation error: {exc}",
                    conceptual_errors=[],
                    suggestions=[],
                )
                for q_id in request.answers
            ],
            error=str(exc),
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


@router.get("/students/{student_id}/dashboard", response_model=ApiResponse[StudentProfile])
async def get_dashboard(student_id: str) -> ApiResponse[StudentProfile]:
    """Return aggregated student progress for the dashboard UI (SUP-08).

    Aggregates topic scores, weak topics, preferences, and session count
    from local SQLite. Returns 404 for unknown student IDs.
    p95 < 300ms via indexed student_id queries.
    """
    from src.memory.schema import (
        compute_weak_topics,
        get_student_profile,
        get_topic_scores,
    )

    logger.info("Dashboard requested for student %s", student_id)

    profile = await get_student_profile(student_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    topic_scores_list = await get_topic_scores(student_id)
    weak_topics = await compute_weak_topics(student_id)

    # Convert list[dict] to dict[str, list[float]] for StudentProfile
    topic_scores_dict: dict[str, list[float]] = {}
    for ts in topic_scores_list:
        topic = ts["topic"]
        score = ts["score"]
        topic_scores_dict.setdefault(topic, []).append(score)

    # Convert DB preferences dict to ExamPreferences
    prefs = profile.get("preferences", {})
    exam_prefs = ExamPreferences(
        questionTypes=prefs.get("question_types", ["mcq"]),
        difficulty=prefs.get("difficulty", "medium"),
        questionCount=prefs.get("question_count", 5),
        includeTopics=prefs.get("include_topics", []),
        excludeTopics=prefs.get("exclude_topics", []),
    )

    return ApiResponse(
        data=StudentProfile(
            id=student_id,
            topicScores=topic_scores_dict,
            weakTopics=weak_topics,
            preferences=exam_prefs,
            sessionCount=profile.get("session_count", 0),
        ),
        error=None,
    )
