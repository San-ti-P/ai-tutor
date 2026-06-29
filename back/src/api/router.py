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
    ChatHistoryResponse,
    ChatMessageRecord,
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
    ExerciseStepSchema,
    ExerciseRequest,
    IngestResult,
    PreferencesStatus,
    PreferencesUpdate,
    Session,
    SessionCreate,
    SessionFile,
    SessionProfile,
    ExamEvaluationSummary,
    SessionRename,
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
    update_session_status as _update_session_status,
    insert_generated_exam as _insert_generated_exam,
)
from src.memory.schema import (
    save_chat_message as _save_chat_message,
    get_chat_messages as _get_chat_messages,
    get_chat_message_count as _get_chat_message_count,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_exam_from_raw(raw: dict, topic: str = "", difficulty: str = "medium") -> Exam:
    """Normalize raw exam dict from generate_exam tool into the Exam schema."""
    questions = []
    for q in raw.get("questions", []):
        qtype = q.get("type", "open")
        if qtype == "open_answer":
            qtype = "open"
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

    resolved_topic = topic or (raw.get("topics_covered") or [""])[0]
    resolved_difficulty = difficulty or (
        questions[0].difficulty if questions else "medium"
    )
    return Exam(
        id=raw.get("exam_id", str(uuid.uuid4())),
        questions=questions,
        topic=resolved_topic,
        difficulty=resolved_difficulty,
        status=raw.get("status", "complete"),
        warnings=raw.get("warnings", []),
        topic_not_found=raw.get("topic_not_found", []),
        topic_suggestions=raw.get("topic_suggestions", []),
        topic_distribution=raw.get("topic_distribution", {}),
    )


def _build_exercise_from_raw(raw: dict) -> Exercise:
    """Normalize raw exercise dict from generate_exercise tool into the Exercise schema."""
    ms = raw.get("model_solution", {}) or {}
    steps = []
    for step in ms.get("steps", []):
        steps.append(
            ExerciseStepSchema(
                step_number=step.get("step_number") or step.get("stepNumber") or 0,
                description=step.get("description", ""),
                result=step.get("result", ""),
                source_chunk_ids=step.get("source_chunk_ids") or step.get("sourceChunkIds") or [],
            )
        )
    return Exercise(
        exercise_id=raw.get("exercise_id") or raw.get("exerciseId") or "",
        statement=raw.get("statement", ""),
        given_data=raw.get("given_data") or raw.get("givenData"),
        question=raw.get("question", ""),
        model_solution=ExerciseModelSolution(
            steps=steps,
            final_answer=ms.get("final_answer") or ms.get("finalAnswer") or "",
            key_concepts=ms.get("key_concepts") or ms.get("keyConcepts") or [],
            source_chunk_ids=ms.get("source_chunk_ids") or ms.get("sourceChunkIds") or [],
        ),
        topics_covered=raw.get("topics_covered") or raw.get("topicsCovered") or [],
        source_chunk_ids=raw.get("source_chunk_ids") or raw.get("sourceChunkIds"),
        topic_not_found=raw.get("topic_not_found") or raw.get("topicNotFound") or [],
        topic_suggestions=raw.get("topic_suggestions") or raw.get("topicSuggestions") or [],
        status=raw.get("status", ""),
    )





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
            "exam_id": request.exam_id,
            "answers": request.answers,
            "exam_questions": [eq.model_dump(by_alias=True) for eq in request.exam_questions] if request.exam_questions else None,
        }
    )

    elapsed = (time.monotonic() - t0) * 1000
    logger.info(
        "Chat complete | session=%s | intent=%s | %dms",
        request.session_id,
        result["intent"],
        int(elapsed),
    )
    raw_exam = result.get("exam")
    exam = _build_exam_from_raw(raw_exam) if raw_exam else None

    raw_exercise = result.get("exercise")
    exercise = _build_exercise_from_raw(raw_exercise) if raw_exercise else None

    # ── Persist messages to DB (fire-and-forget) ───────────────────────────
    try:
        user_msg_id = str(uuid.uuid4())
        assistant_msg_id = str(uuid.uuid4())
        await _save_chat_message({
            "id": user_msg_id,
            "session_id": request.session_id,
            "role": "user",
            "content": request.message,
        })
        await _save_chat_message({
            "id": assistant_msg_id,
            "session_id": request.session_id,
            "role": "assistant",
            "content": result["response"],
            "metadata_json": json.dumps({"intent": result["intent"]}),
        })
    except Exception:
        logger.exception(
            "Failed to persist chat messages for session %s — chat response unaffected",
            request.session_id,
        )

    return ApiResponse(
        data=ChatResponse(
            response=result["response"],
            intent=result["intent"],
            trace_id=result["trace_id"],
            exam=exam,
            exercise=exercise,
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
        raise HTTPException(
            status_code=404, detail=f"Session '{session_id}' not found after rename"
        )
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


@router.get(
    "/sessions/{session_id}/evaluations",
    response_model=ApiResponse[list[ExamEvaluationSummary]],
)
async def get_session_evaluations_endpoint(
    session_id: str,
) -> ApiResponse[list[ExamEvaluationSummary]]:
    """Return all exam evaluations for a session, grouped by exam and newest first."""
    from src.memory.schema import get_session_evaluations as _get_session_evaluations

    groups = await _get_session_evaluations(session_id)
    summaries = [
        ExamEvaluationSummary(
            examId=g["exam_id"],
            createdAt=g["created_at"],
            averageScore=g.get("averageScore"),
            results=[EvaluationResult(**r) for r in g["results"]],
        )
        for g in groups
    ]
    return ApiResponse(data=summaries, error=None, trace_id=str(uuid.uuid4()))


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ApiResponse[ChatHistoryResponse],
)
async def get_session_messages_endpoint(
    session_id: str,
    limit: int = 10,
    before_id: str | None = None,
) -> ApiResponse[ChatHistoryResponse]:
    """Return paginated chat messages for a session, newest-first.

    - ``limit``: number of messages to return (1-50, default 10).
    - ``before_id``: cursor for pagination — returns messages older than this id.
    """
    logger.info(
        "Chat messages requested | session=%s | limit=%d | before_id=%s",
        session_id,
        limit,
        before_id,
    )
    detail = await _get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    limit = min(max(1, limit), 50)
    rows = await _get_chat_messages(session_id, limit=limit + 1, before_id=before_id)
    has_more = len(rows) > limit
    rows = rows[:limit]  # trim the extra row used for has_more probe

    total = await _get_chat_message_count(session_id)
    oldest_id = rows[-1]["id"] if rows else None

    messages = [
        ChatMessageRecord(
            id=row["id"],
            sessionId=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row.get("metadata_json") or "{}"),
            createdAt=row["created_at"],
        )
        for row in rows
    ]

    return ApiResponse(
        data=ChatHistoryResponse(
            messages=messages,
            hasMore=has_more,
            oldestId=oldest_id,
            total=total,
        ),
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

    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

    for file in files:
        # Reject oversized files before temp write (A-4)
        if file.size and file.size > MAX_FILE_SIZE:
            logger.warning(
                "File '%s' rejected: size %d exceeds 20 MB limit",
                file.filename,
                file.size,
            )
            results.append(
                IngestResult(
                    sessionId=effective_session_id,
                    status="error",
                    classification="",
                    topicsDetected=[],
                    chunksCreated=0,
                )
            )
            await file.close()
            continue

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
                await _update_session_status(effective_session_id, "active")
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
        effective_session_id,
        len(files),
        ok,
        fail,
        int(elapsed),
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

    # ── Resolve student_id: request override → session lookup → session_id fallback ──
    student_id = request.student_id
    if not student_id:
        session_detail = await _get_session(request.session_id)
        student_id = session_detail["student_id"] if session_detail else request.session_id
    logger.info("Resolved student_id=%s for session=%s", student_id, request.session_id)

    # Load student profile for weak-topic prioritization
    student_profile = None
    if student_id:
        from src.tools.get_student_summary import get_student_summary as _summary_tool

        profile = await _summary_tool.ainvoke({"student_id": student_id})
        if profile is not None:
            student_profile = profile
            logger.info(
                "Loaded student profile: weak_topics=%s, session_count=%d",
                profile.get("weak_topics", []),
                profile.get("session_count", 0),
            )

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
            "student_profile": student_profile,
        },
    )

    exam = _build_exam_from_raw(result, topic=request.topic, difficulty=request.preferences.difficulty)
    try:
        await _insert_generated_exam(exam.id, request.session_id, exam.topic, exam.difficulty)
    except Exception:
        logger.exception("Failed to insert generated exam %s into database", exam.id)

    return ApiResponse(
        data=exam,
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
            # eq.type is QuestionTypeEnum. Value can be extracted via eq.type.value if it is an Enum,
            # or just as string. Let's make sure we handle both by getting it as a string or .value
            q_type = eq.type.value if hasattr(eq.type, "value") else str(eq.type)
            question_map[eq.id] = {
                "question": eq.prompt,
                "base_answer": eq.base_answer or "",
                "topic": eq.topic,
                "difficulty": str(eq.difficulty),
                "source_chunk_ids": eq.source_chunk_ids or [],
                "type": q_type,
                "options": eq.options or [],
            }

    # Convert dict[str, str] answers to list[dict] format expected by evaluator
    answers_list: list[dict] = []
    answers_by_id: dict[str, dict] = {}
    for question_id, student_answer in request.answers.items():
        if question_id in question_map:
            qdata = question_map[question_id]
            entry = {
                "question_id": question_id,
                "question": qdata["question"],
                "base_answer": qdata["base_answer"],
                "student_answer": student_answer,
                "source_chunk_ids": qdata["source_chunk_ids"],
                "topic": qdata["topic"],
                "difficulty": qdata["difficulty"],
                "type": qdata.get("type", "open"),
                "options": qdata.get("options", []),
            }
        else:
            if request.exam_questions:
                logger.warning(
                    "question_id '%s' not found in exam_questions, using empty placeholders",
                    question_id,
                )
            entry = {
                "question_id": question_id,
                "question": "",
                "base_answer": "",
                "student_answer": student_answer,
                "source_chunk_ids": [],
                "topic": "",
                "difficulty": "medium",
                "type": "open",
                "options": [],
            }
        answers_list.append(entry)
        answers_by_id[question_id] = entry

    # ── Resolve student_id: request override → session lookup → session_id fallback ──
    student_id = request.student_id
    if not student_id:
        session_detail = await _get_session(request.session_id)
        student_id = session_detail["student_id"] if session_detail else request.session_id
    logger.info("Resolved student_id=%s for session=%s", student_id, request.session_id)

    # Ensure the student row exists so FK constraints on topic_scores pass
    await _ensure_student_exists(student_id)

    try:
        results = _evaluate_tool.invoke(
            {
                "session_id": request.session_id,
                "exam_id": request.exam_id,
                "answers": answers_list,
                "student_id": student_id,
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
                questionText=answers_by_id.get(r.get("question_id", ""), {}).get("question", ""),
                userAnswer=answers_by_id.get(r.get("question_id", ""), {}).get("student_answer", ""),
                sourceChunks=r.get("source_chunks", []),
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
