"""API router with all endpoints matching the frontend api.ts client."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    EvaluationRequest,
    EvaluationResult,
    Exam,
    ExamPreferences,
    ExamQuestion,
    ExamRequest,
    Exercise,
    ExerciseModelSolution,
    ExerciseRequest,
    IngestResult,
    PreferencesStatus,
    PreferencesUpdate,
    Session,
    SessionCreate,
    SessionRename,
    SessionFile,
    SessionProfile,
    StudentProfile,
)
from src.memory.schema import (
    create_session as _create_session,
)
from src.memory.schema import (
    delete_session as _delete_session,
)
from src.memory.schema import (
    ensure_student_exists as _ensure_student_exists,
)
from src.memory.schema import (
    get_session as _get_session,
)
from src.memory.schema import (
    insert_ingested_document as _insert_ingested_document,
)
from src.memory.schema import (
    list_session_files as _list_session_files,
)
from src.memory.schema import (
    list_sessions as _list_sessions,
)
from src.memory.schema import (
    rename_session as _rename_session,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    logger.info("Health check requested")
    return {"status": "ok", "version": "0.1.0", "trace_id": str(uuid.uuid4())}


@router.post("/chat", response_model=ApiResponse[ChatResponse])
async def chat(request: ChatRequest) -> ApiResponse[ChatResponse]:
    import time

    t0 = time.monotonic()
    logger.info("Chat request received for session %s", request.session_id)

    from src.tools import orchestrate_chat

    result = await orchestrate_chat.ainvoke(
        {
            "messages": [{"role": "user", "content": request.message}],
            "thread_id": request.session_id,
            "student_id": request.student_id,
        }
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Chat complete | session=%s | intent=%s | %dms",
        request.session_id, result["intent"], int(elapsed),
    )
    return ApiResponse(
        data=ChatResponse(
            response=result["response"],
            intent=result["intent"],
            trace_id=result["trace_id"],
            exam=result.get("exam"),
        ),
        error=None,
        trace_id=result.get("trace_id", str(uuid.uuid4())),
    )


def _validate_session_id(raw: str | None) -> str:
    """Validate and sanitise an incoming session_id.

    Accepts only non-empty, ≤ 64 chars, and UUID-like values (standard
    UUID regex or at least 32 hex chars with optional dashes). Returns
    a generated UUID4 when the input is invalid or absent.

    Rationale: avoids Langfuse OTEL baggage drop (>200 chars) and
    ChromaDB collection-name limits.
    """
    import re

    if not raw or not raw.strip():
        return str(uuid.uuid4())

    sid = raw.strip()
    if len(sid) > 64:
        logger.warning("session_id too long (%d chars), generating UUID", len(sid))
        return str(uuid.uuid4())

    # Standard UUID4 with dashes
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    if uuid_pattern.match(sid):
        return sid

    # At least 32 hex chars with optional dashes (loose UUID)
    hex_only = re.sub(r"-", "", sid)
    if re.match(r"^[0-9a-f]{32}$", hex_only, re.IGNORECASE):
        return sid

    logger.warning("session_id '%s' not UUID-like, generating UUID", sid)
    return str(uuid.uuid4())


@router.post("/sessions", response_model=ApiResponse[Session])
async def create_session_endpoint(request: SessionCreate) -> ApiResponse[Session]:
    """Create a new named study session."""
    logger.info("Creating session for student %s", request.student_id)
    await _ensure_student_exists(request.student_id)
    session = await _create_session(request.student_id, request.name, request.description)
    detail = await _get_session(session["id"])
    return ApiResponse(data=Session.model_validate(detail), error=None, trace_id=str(uuid.uuid4()))


@router.get("/sessions", response_model=ApiResponse[list[Session]])
async def list_sessions_endpoint(student_id: str) -> ApiResponse[list[Session]]:
    """List all sessions for a student ordered by most recent first."""
    logger.info("Listing sessions for student %s", student_id)
    rows = await _list_sessions(student_id)
    return ApiResponse(
        data=[Session.model_validate(row) for row in rows],
        error=None,
        trace_id=str(uuid.uuid4()),
    )


@router.get("/sessions/{session_id}", response_model=ApiResponse[Session])
async def get_session_endpoint(session_id: str) -> ApiResponse[Session]:
    """Get session details including file count and progress summary."""
    logger.info("Getting session %s", session_id)
    detail = await _get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return ApiResponse(data=Session(**detail), error=None, trace_id=str(uuid.uuid4()))


@router.delete("/sessions/{session_id}", response_model=ApiResponse[dict])
async def delete_session_endpoint(session_id: str) -> ApiResponse[dict]:
    """Delete a session and cascade its associated files."""
    logger.info("Deleting session %s", session_id)
    detail = await _get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    await _delete_session(session_id)
    return ApiResponse(data={"deleted": session_id}, error=None, trace_id=str(uuid.uuid4()))


@router.patch("/sessions/{session_id}", response_model=ApiResponse[Session])
async def rename_session_endpoint(session_id: str, body: SessionRename) -> ApiResponse[Session]:
    """Rename a session and optionally update its description."""
    logger.info("Renaming session %s to %s", session_id, body.name)
    detail = await _get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    updated = await _rename_session(session_id, body.name, body.description)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found after rename")
    return ApiResponse(data=Session.model_validate(updated), error=None, trace_id=str(uuid.uuid4()))


@router.get("/sessions/{session_id}/files", response_model=ApiResponse[list[SessionFile]])
async def get_session_files(session_id: str) -> ApiResponse[list[SessionFile]]:
    """List files uploaded to a session."""
    logger.info("Listing files for session %s", session_id)
    detail = await _get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    rows = await _list_session_files(session_id)
    files = []
    for row in rows:
        topics = []
        if row.get("topics_json"):
            try:
                topics = json.loads(row["topics_json"])
            except Exception:
                topics = []
        topic_tree = None
        if row.get("topic_tree_json"):
            try:
                topic_tree = json.loads(row["topic_tree_json"])
            except Exception:
                pass
        files.append(
            SessionFile(
                id=row["id"],
                fileName=row["file_name"],
                classification=row.get("classification", ""),
                topics=topics,
                topicTree=topic_tree,
                chunksCount=row.get("chunks_count", 0),
                ingestedAt=row["ingested_at"],
            )
        )
    return ApiResponse(data=files, error=None, trace_id=str(uuid.uuid4()))


@router.get("/sessions/{session_id}/profile", response_model=ApiResponse[SessionProfile])
async def get_session_profile_endpoint(session_id: str) -> ApiResponse[SessionProfile]:
    """Return per-session progress: topic scores, weak topics, exam count, avg score."""
    logger.info("Session profile requested for %s", session_id)
    from src.memory.schema import get_session_profile as _get_session_profile

    detail = await _get_session_profile(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return ApiResponse(
        data=SessionProfile.model_validate(detail),
        error=None,
        trace_id=str(uuid.uuid4()),
    )


@router.post("/ingest", response_model=ApiResponse[list[IngestResult]])
async def ingest(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(None),
) -> ApiResponse[list[IngestResult]]:
    import time

    t0 = time.monotonic()
    logger.info("Ingest request received with %d file(s)", len(files))
    results: list[IngestResult] = []

    # Validate or generate session_id
    effective_session_id = _validate_session_id(session_id)
    logger.info("Effective ingest session_id: %s", effective_session_id)

    # Use the tools layer — ingest_document wraps the full ingestion graph
    from src.tools import ingest_document as _ingest_tool

    for file in files:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
            tmp_path = tmp.name

        try:
            result = await _ingest_tool.ainvoke(
                {"file_path": tmp_path, "session_id": effective_session_id},
            )

            document_id = result.get("document_id") or str(uuid.uuid4())
            try:
                # Ensure the session row exists so ingested_documents FK succeeds.
                # Anonymous uploads generate a fresh session_id on the fly.
                session_detail = await _get_session(effective_session_id)
                if session_detail is None:
                    await _create_session(
                        student_id=effective_session_id,
                        name="",
                        description="",
                        session_id=effective_session_id,
                    )
                await _insert_ingested_document(
                    {
                        "id": document_id,
                        "file_name": file.filename or "unknown",
                        "classification": result.get("classification"),
                        "topics_json": json.dumps(result.get("topics", [])),
                        "topic_tree_json": result.get("topic_tree", "{}"),
                        "chunks_count": result.get("chunks_created", 0),
                        "session_id": effective_session_id,
                    }
                )
            except Exception:
                logger.exception("Failed to persist file metadata for %s", file.filename)

            # Parse topic_tree from JSON string to dict for API response
            topic_tree_raw = result.get("topic_tree", "{}")
            topic_tree = None
            if topic_tree_raw and topic_tree_raw != "{}":
                try:
                    topic_tree = json.loads(topic_tree_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            results.append(
                IngestResult(
                    sessionId=effective_session_id,
                    status=result.get("status", "unknown"),
                    classification=result.get("classification", ""),
                    topicsDetected=result.get("topics", []),
                    topicTree=topic_tree,
                    chunksCreated=result.get("chunks_created", 0),
                    classificationConfidence=result.get("classification_confidence"),
                    documentId=document_id,
                )
            )
        except Exception:
            logger.exception("Ingest failed for %s", file.filename)
            results.append(
                IngestResult(
                    sessionId=effective_session_id,
                    status="error",
                    classification="",
                    topicsDetected=[],
                    chunksCreated=0,
                )
            )
        finally:
            await file.close()
            Path(tmp_path).unlink(missing_ok=True)

    elapsed = (time.monotonic() - t0) * 1000
    ok = sum(1 for r in results if r.status != "error")
    fail = len(results) - ok
    logger.info(
        "Ingest complete | session=%s | files=%d | ok=%d | fail=%d | %dms",
        effective_session_id, len(files), ok, fail, int(elapsed),
    )
    return ApiResponse(
        data=results,
        error=None,
        trace_id=str(effective_session_id),
    )


@router.post("/exam/generate", response_model=ApiResponse[Exam])
async def generate_exam(request: ExamRequest) -> ApiResponse[Exam]:
    logger.info("Exam generation requested for topic %s", request.topic)

    from src.tools import generate_exam as _gen_exam_tool

    # Determine mcq_ratio from question types
    qtypes = request.preferences.question_types
    mcq_ratio = sum(1 for t in qtypes if t == "mcq") / max(len(qtypes), 1)

    result = await asyncio.to_thread(
        _gen_exam_tool.invoke,
        {
            "session_id": request.session_id,
            "topics": [request.topic],
            "difficulty": request.preferences.difficulty,
            "question_count": request.preferences.question_count,
            "mcq_ratio": mcq_ratio,
        },
    )

    # Map tool output to Exam model
    questions = []
    for q in result.get("questions", []):
        qtype = q.get("type", "open")
        if qtype == "open_answer":
            qtype = "open"
        # Compute base_answer for MCQ from correct_option_index
        base_answer = q.get("baseAnswer", q.get("base_answer"))
        if qtype == "mcq" and not base_answer:
            options = q.get("options", [])
            correct_idx = q.get("correct_option_index")
            if isinstance(correct_idx, int) and 0 <= correct_idx < len(options):
                base_answer = options[correct_idx]
        questions.append(
            ExamQuestion(
                id=q.get("id", str(uuid.uuid4())),
                type=qtype,
                prompt=q.get("prompt", ""),
                options=q.get("options"),
                baseAnswer=base_answer,
                sourceChunkIds=q.get("sourceChunkIds", q.get("source_chunk_ids")),
                topic=q.get("topic", ""),
                difficulty=q.get("difficulty", "medium"),
            )
        )

    return ApiResponse(
        data=Exam(
            id=result.get("exam_id", str(uuid.uuid4())),
            questions=questions,
            topic=request.topic,
            difficulty=request.preferences.difficulty,
            status=result.get("status", "complete"),
            warnings=result.get("warnings", []),
            topic_not_found=result.get("topic_not_found", []),
            topic_suggestions=result.get("topic_suggestions", []),
        ),
        error=None,
        trace_id=str(uuid.uuid4()),
    )


@router.post("/evaluate", response_model=ApiResponse[list[EvaluationResult]])
async def evaluate(request: EvaluationRequest) -> ApiResponse[list[EvaluationResult]]:
    logger.info("Evaluation requested for exam %s", request.exam_id)

    from src.tools import evaluate_answer as _evaluate_tool

    # Build lookup map from exam_questions (if provided) for cross-referencing
    question_map: dict[str, dict] = {}
    if request.exam_questions:
        for eq in request.exam_questions:
            question_map[eq.id] = {
                "question": eq.prompt,
                "base_answer": eq.base_answer or "",
                "topic": eq.topic,
                "difficulty": str(eq.difficulty),
                "source_chunk_ids": eq.source_chunk_ids or [],
            }

    # Convert dict[str, str] answers to list[dict] format expected by evaluator
    answers_list: list[dict] = []
    for question_id, student_answer in request.answers.items():
        if question_id in question_map:
            qdata = question_map[question_id]
            answers_list.append(
                {
                    "question_id": question_id,
                    "question": qdata["question"],
                    "base_answer": qdata["base_answer"],
                    "student_answer": student_answer,
                    "source_chunk_ids": qdata["source_chunk_ids"],
                    "topic": qdata["topic"],
                    "difficulty": qdata["difficulty"],
                }
            )
        else:
            if request.exam_questions:
                logger.warning(
                    "question_id '%s' not found in exam_questions, using empty placeholders",
                    question_id,
                )
            answers_list.append(
                {
                    "question_id": question_id,
                    "question": "",
                    "base_answer": "",
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
                questionId=r.get("question_id", ""),
                score=r.get("score", 0.0),
                justification=r.get("justification", ""),
                conceptualErrors=r.get("conceptual_errors", []),
                suggestions=r.get("suggestions", []),
                isEvaluable=r.get("is_evaluable", True),
                nonEvaluableReason=r.get("non_evaluable_reason", ""),
                requiresReview=r.get("requires_review", False),
                judgeScore=r.get("judge_verdict", {}).get("score")
                if isinstance(r.get("judge_verdict"), dict)
                else None,
            )
            for r in results
        ]

        return ApiResponse(data=evaluation_results, error=None, trace_id=str(uuid.uuid4()))

    except Exception as exc:
        logger.exception("Evaluation failed for exam %s", request.exam_id)
        return ApiResponse(
            data=[
                EvaluationResult(
                    questionId=q_id,
                    score=0.0,
                    justification=f"Evaluation error: {exc}",
                    conceptualErrors=[],
                    suggestions=[],
                )
                for q_id in request.answers
            ],
            error=str(exc),
            trace_id=str(uuid.uuid4()),
        )


@router.get("/profile/{session_id}", response_model=ApiResponse[StudentProfile])
async def get_profile(session_id: str) -> ApiResponse[StudentProfile]:
    logger.info("Profile requested for session %s", session_id)

    from src.tools.get_student_summary import get_student_summary as _summary_tool

    result = await _summary_tool.ainvoke({"student_id": session_id})
    if result is None:
        raise HTTPException(status_code=404, detail=f"Student '{session_id}' not found")

    # Build topic_scores dict from list[dict]
    topic_scores_dict: dict[str, list[float]] = {}
    for ts in result.get("topic_scores", []):
        topic = ts["topic"]
        score = ts["score"]
        topic_scores_dict.setdefault(topic, []).append(score)

    prefs = result.get("preferences", {})
    exam_prefs = ExamPreferences(
        questionTypes=prefs.get("question_types", ["mcq"]),
        difficulty=prefs.get("difficulty", "medium"),
        questionCount=prefs.get("question_count", 5),
        includeTopics=prefs.get("include_topics", []),
        excludeTopics=prefs.get("exclude_topics", []),
    )

    return ApiResponse(
        data=StudentProfile(
            id=result["id"],
            topicScores=topic_scores_dict,
            weakTopics=result.get("weak_topics", []),
            preferences=exam_prefs,
            sessionCount=result.get("session_count", 0),
        ),
        error=None,
        trace_id=str(uuid.uuid4()),
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
        get_enriched_session_history,
        get_student_profile,
        get_topic_scores,
    )

    logger.info("Dashboard requested for student %s", student_id)

    profile = await get_student_profile(student_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    topic_scores_list = await get_topic_scores(student_id)
    weak_topics = await compute_weak_topics(student_id)
    enriched_sessions = await get_enriched_session_history(student_id)

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
            sessionHistory=enriched_sessions,
        ),
        error=None,
        trace_id=str(uuid.uuid4()),
    )


@router.post("/exercise/generate", response_model=ApiResponse[Exercise])
async def generate_exercise_endpoint(request: ExerciseRequest) -> ApiResponse[Exercise]:
    """Generate a practice exercise via the ExerciseGenerator agent (REQ-API-004)."""
    logger.info("Exercise generation requested for topic %s", request.topic)

    from src.tools import generate_exercise as _gen_exercise_tool

    result = await asyncio.to_thread(
        _gen_exercise_tool.invoke,
        {
            "session_id": request.session_id,
            "topic": request.topic,
            "difficulty": request.difficulty,
            "exercise_type": request.exercise_type,
        },
    )

    ms = result.get("model_solution", {})
    return ApiResponse(
        data=Exercise(
            exercise_id=result.get("exercise_id", ""),
            statement=result.get("statement", ""),
            given_data=result.get("given_data"),
            question=result.get("question", ""),
            model_solution=ExerciseModelSolution(
                steps=ms.get("steps", []),
                final_answer=ms.get("final_answer", ""),
                key_concepts=ms.get("key_concepts", []),
                source_chunk_ids=ms.get("source_chunk_ids", []),
            ),
            topics_covered=result.get("topics_covered", []),
            source_chunk_ids=result.get("source_chunk_ids"),
            topic_not_found=result.get("topic_not_found", []),
            topic_suggestions=result.get("topic_suggestions", []),
            status=result.get("status", ""),
        ),
        error=None,
        trace_id=str(uuid.uuid4()),
    )


@router.put("/profile/{student_id}/preferences", response_model=ApiResponse[PreferencesStatus])
async def update_preferences(
    student_id: str,
    preferences: PreferencesUpdate,
) -> ApiResponse[PreferencesStatus]:
    """Persist exam preferences via the Support Agent (REQ-CONFIG-003)."""
    logger.info("Preferences update requested for student %s", student_id)

    from src.tools.update_student_profile import update_student_profile as _upsert_tool

    prefs_dict: dict = {
        "question_types": preferences.question_types,
        "difficulty": preferences.difficulty,
        "question_count": preferences.question_count,
        "include_topics": preferences.include_topics,
        "exclude_topics": preferences.exclude_topics,
    }

    result = await _upsert_tool.ainvoke(
        {
            "student_id": student_id,
            "topic_scores": {},
            "preferences": prefs_dict,
        }
    )

    return ApiResponse(
        data=PreferencesStatus(
            status=result.get("status", "ok"),
            student_id=result.get("student_id", student_id),
            upserted_topics=result.get("upserted_topics", 0),
            errors=result.get("errors", []),
        ),
        error=None,
        trace_id=str(uuid.uuid4()),
    )
