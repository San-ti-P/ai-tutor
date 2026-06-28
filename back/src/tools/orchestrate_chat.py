"""Public tool boundary for the Orchestrator agent graph.

The API router invokes this tool instead of importing agent internals
directly, preserving clean-architecture dependency direction:
API → tools → agents.
"""

from __future__ import annotations

import logging
import uuid

from langchain_core.tools import tool
from langfuse import observe, propagate_attributes

logger = logging.getLogger(__name__)


@tool
@observe(name="orchestrate_chat", as_type="tool")
async def orchestrate_chat(
    messages: list[dict],
    thread_id: str | None = None,
    student_id: str | None = None,
) -> dict:
    """Invoke the Orchestrator agent graph for a chat interaction.

    This is the single public entry point for chat requests. The API router
    calls this tool instead of compiling the orchestrator graph directly,
    keeping agent internals behind the tools boundary.

    Args:
        messages: List of LangChain-format message dicts with ``role`` and
            ``content`` keys (e.g. ``{"role": "user", "content": "Hola"}``).
            The tool extracts the last user message to drive the orchestrator.
        thread_id: Optional thread ID for LangGraph checkpoint persistence.
            Defaults to a generated UUID4 when not provided.
        student_id: Optional explicit student identity. When provided, it
            bypasses the sessions table lookup in ``load_profile``.

    Returns:
        A dict with keys: ``response`` (str), ``intent`` (str),
        ``status`` (str), ``trace_id`` (str).
    """
    from src.agents.orchestrator import get_orchestrator_graph

    # Extract the last user message from the messages list
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    session_id = thread_id or str(uuid.uuid4())
    logger.info("[orchestrate_chat] START | session=%s", session_id)

    graph = await get_orchestrator_graph()

    # Use a unique thread_id per request to prevent LangGraph from restoring
    # stale checkpoint state (results, plan, etc.) from previous requests on
    # the same session. The session_id is preserved in initial_state so all
    # tool calls (profile loading, ChromaDB queries) still target the right session.
    run_thread_id = f"{session_id}:{uuid.uuid4()}"

    initial_state: dict = {
        "session_id": session_id,
        "user_message": user_message,
        "intent": "general_chat",
        "confidence": 0.0,
        "plan": [],
        "current_step": 0,
        "results": [],
        "errors": [],
        "response": "",
        "status": "pending",
        "iteration_count": 0,
        "student_profile": None,
        "student_id": student_id,
        "session_context": None,
        "messages_history": [],
    }

    # ── Observability wiring ────────────────────────────────────────────────
    from src.observability import flush_traces, get_tracer

    tracer = get_tracer()

    config = {"configurable": {"thread_id": run_thread_id}}

    try:
        # Per Langfuse docs: CallbackHandler must be created INSIDE
        # propagate_attributes to inherit trace-level session/user context.
        with propagate_attributes(session_id=session_id):
            langfuse_handler = tracer.get_callback_handler(
                session_id=session_id,
                user_id=None,
            )
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]
                config["metadata"] = {"langfuse_session_id": session_id}

            final_state = await graph.ainvoke(initial_state, config=config)
    finally:
        flush_traces()

    # Extract structured exam data for UI rendering (Epic 7 US-7.3)
    exam = None
    if final_state.get("intent") == "generate_exam":
        for r in final_state.get("results", []):
            if r.get("tool") == "generate_exam" and isinstance(r.get("result"), dict):
                exam = r["result"]
                break

    return {
        "response": final_state["response"],
        "intent": final_state["intent"],
        "status": final_state.get("status", "complete"),
        "trace_id": str(uuid.uuid4()),
        "exam": exam,
    }
