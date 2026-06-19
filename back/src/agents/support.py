"""Support Agent — Reactive agent for student profile and progress queries.

4-node StateGraph with conditional routing:
  START → fetch_student_profile
    ├── query → fetch_session_history → compute_progress_summary → generate_response
    └── update → compute_progress_summary → generate_response → END

``query`` flow: read full profile + history → compute weak topics → respond.
``update`` flow: read profile → compute weak topics → respond (no history needed).
"""

from __future__ import annotations

import logging
from typing import Literal

try:
    from langfuse import observe
except ImportError:
    def observe(name: str | None = None):  # noqa: D103
        def decorator(fn):  # noqa: D103
            return fn
        return decorator

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ── Graph state schema ────────────────────────────────────────────────────────


class SupportState(TypedDict):
    """State for the 4-node Support Agent StateGraph."""

    session_id: str
    student_id: str
    query_type: Literal["query", "update"]
    profile_data: dict | None
    session_history: list[dict]
    topic_scores: list[dict]
    weak_topics: list[str]
    preferences: dict | None
    response: str
    status: str


# ═══════════════════════════════════════════════════════════════════════════════
# Node: fetch_student_profile
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_student_profile(state: SupportState) -> dict:
    """Retrieve student profile and topic scores from SQLite.

    Calls ``get_student_profile`` and ``get_topic_scores`` from the
    memory module. Auto-creates the student row if it does not exist.
    Returns partial state with ``profile_data``, ``topic_scores``,
    ``preferences``, and ``status``.
    """
    import asyncio

    from src.memory.schema import get_student_profile, get_topic_scores

    student_id: str = state.get("student_id", "")

    try:
        # Run async DB calls synchronously inside graph node
        async def _fetch():
            profile = await get_student_profile(student_id)
            scores = await get_topic_scores(student_id)
            return profile, scores

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _fetch())
                    profile, scores = future.result()
            else:
                profile, scores = loop.run_until_complete(_fetch())
        except RuntimeError:
            profile, scores = asyncio.run(_fetch())

        if profile is None:
            logger.warning("Student %s not found in DB", student_id)
            # Auto-create on first access (SUP-01)
            from src.memory.schema import upsert_student_profile

            async def _create():
                await upsert_student_profile(student_id, {
                    "difficulty": "medium",
                    "question_types": ["mcq"],
                    "question_count": 5,
                    "include_topics": [],
                    "exclude_topics": [],
                })
            try:
                asyncio.run(_create())
            except RuntimeError:
                pass
            profile = {"id": student_id, "preferences": {}, "session_count": 0}
            scores = []

        prefs = profile.get("preferences", {}) if profile else {}

        return {
            "profile_data": profile,
            "topic_scores": scores,
            "preferences": prefs,
            "status": "fetched",
        }

    except Exception as exc:
        logger.exception("Failed to fetch student profile for %s", student_id)
        return {
            "profile_data": None,
            "topic_scores": [],
            "preferences": None,
            "status": "error",
            "response": f"Error al recuperar el perfil: {exc}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: fetch_session_history
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_session_history(state: SupportState) -> dict:
    """Retrieve recent session history for the student.

    Only called in the ``query`` flow. Returns partial state with
    ``session_history`` populated.
    """
    import asyncio

    from src.memory.schema import get_recent_sessions

    student_id: str = state.get("student_id", "")

    try:
        async def _fetch():
            return await get_recent_sessions(student_id)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _fetch())
                    history = future.result()
            else:
                history = loop.run_until_complete(_fetch())
        except RuntimeError:
            history = asyncio.run(_fetch())

        return {"session_history": history}

    except Exception as exc:
        logger.exception("Failed to fetch session history for %s", student_id)
        return {"session_history": [], "errors": [f"History fetch error: {exc}"]}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: compute_progress_summary
# ═══════════════════════════════════════════════════════════════════════════════


def compute_progress_summary(state: SupportState) -> dict:
    """Compute progress summary: identify weak topics from topic_scores.

    Uses ``compute_weak_topics`` (threshold=6.0, limit=3) from the
    memory module. Also computes an average score across all topics.
    Returns partial state with ``weak_topics`` populated.
    """
    import asyncio

    from src.memory.schema import compute_weak_topics

    student_id: str = state.get("student_id", "")
    topic_scores: list[dict] = state.get("topic_scores", [])

    try:
        async def _compute():
            return await compute_weak_topics(student_id)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _compute())
                    weak = future.result()
            else:
                weak = loop.run_until_complete(_compute())
        except RuntimeError:
            weak = asyncio.run(_compute())

        return {"weak_topics": weak}

    except Exception as exc:  # noqa: F841
        logger.exception("Failed to compute weak topics for %s", student_id)
        # Fallback: compute from in-memory topic_scores
        weak = sorted(
            [t["topic"] for t in topic_scores if t.get("score", 0) < 6.0],
            key=lambda t: next((s["score"] for s in topic_scores if s["topic"] == t), 10),
        )[:3]
        return {"weak_topics": weak}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: generate_response
# ═══════════════════════════════════════════════════════════════════════════════


def generate_response(state: SupportState) -> dict:
    """Generate a natural-language response about student progress.

    On ``query`` type: reports weak topics, session count, and recommendations.
    On ``update`` type: confirms profile update and reports new weak topics.
    """
    query_type: str = state.get("query_type", "query")
    profile: dict | None = state.get("profile_data")
    weak_topics: list[str] = state.get("weak_topics", [])
    topic_scores: list[dict] = state.get("topic_scores", [])
    session_history: list[dict] = state.get("session_history", [])
    prefs: dict | None = state.get("preferences")

    session_count = 0
    if profile:
        session_count = profile.get("session_count", 0)

    if query_type == "update":
        # Confirmation message for profile update
        diff = prefs.get("difficulty", "medium") if prefs else "medium"
        response = (
            f"Perfil actualizado correctamente. "
            f"Preferencias guardadas: dificultad {diff}. "
        )
        if weak_topics:
            response += (
                f"Temas débiles detectados: {', '.join(weak_topics)}. "
                f"Se priorizarán en la próxima generación de exámenes."
            )
        else:
            response += "No se detectaron temas débiles."
        return {"response": response, "status": "done"}

    # Query flow: full progress report
    if not profile:
        return {
            "response": "No se encontró el perfil del estudiante.",
            "status": "done",
        }

    avg_score = 0.0
    if topic_scores:
        avg_score = sum(t.get("score", 0.0) for t in topic_scores) / len(topic_scores)

    parts = [
        f"Resumen de progreso para el estudiante {state.get('student_id', '')}:",
    ]

    if session_count > 0:
        parts.append(f"- Sesiones completadas: {session_count}")

    if topic_scores:
        parts.append(f"- Promedio general: {avg_score:.1f}/10")
        parts.append("- Puntajes por tema:")
        for ts in topic_scores:
            parts.append(f"  • {ts['topic']}: {ts['score']:.1f}/10")

    if weak_topics:
        parts.append(
            f"- Temas que necesitan refuerzo: {', '.join(weak_topics)}"
        )
        parts.append(
            "Recomendación: enfocar el estudio en estos temas antes "
            "del próximo examen."
        )
    else:
        parts.append("- Todos los temas están por encima del umbral. ¡Buen trabajo!")

    if session_history:
        parts.append(f"- Últimas {len(session_history)} sesiones registradas.")

    return {"response": "\n".join(parts), "status": "done"}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional routing helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _route_after_fetch(state: SupportState) -> str:
    """Route after fetch_student_profile based on query_type.

    ``query`` → fetch_session_history (full read path).
    ``update`` → compute_progress_summary (skip history, go straight to summary).
    """
    if state.get("status") == "error":
        return "generate_response"
    if state.get("query_type") == "update":
        return "compute_progress_summary"
    return "fetch_session_history"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════


@observe(name="support_agent")
def build_support_agent() -> StateGraph:
    """Build the 4-node Support Agent StateGraph with conditional routing.

    Topology:
      START → fetch_student_profile
        ├── query → fetch_session_history → compute_progress_summary
        │                                         ↓
        └── update → ────────────────────→ generate_response → END
    """
    builder = StateGraph(SupportState)

    builder.add_node("fetch_student_profile", fetch_student_profile)
    builder.add_node("fetch_session_history", fetch_session_history)
    builder.add_node("compute_progress_summary", compute_progress_summary)
    builder.add_node("generate_response", generate_response)

    # Entry
    builder.add_edge(START, "fetch_student_profile")

    # Conditional: query → session_history → summary; update → summary
    builder.add_conditional_edges(
        "fetch_student_profile",
        _route_after_fetch,
        {
            "fetch_session_history": "fetch_session_history",
            "compute_progress_summary": "compute_progress_summary",
            "generate_response": "generate_response",
        },
    )

    # fetch_session_history → compute_progress_summary
    builder.add_edge("fetch_session_history", "compute_progress_summary")

    # compute_progress_summary → generate_response
    builder.add_edge("compute_progress_summary", "generate_response")

    # generate_response → END
    builder.add_edge("generate_response", END)

    return builder
