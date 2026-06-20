"""Langfuse CallbackHandler factory — internal module.

Produces LangChain-compatible ``CallbackHandler`` instances that
auto-trace every LLM call and nested tool invocation when passed
via the graph ``config`` dict.

Langfuse >=4.0 uses ``TraceContext`` instead of constructor-level
``session_id`` / ``user_id``.  Those are set on traces via
``langfuse_context.update_current_trace()`` inside ``@observe``-decorated
functions instead.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_callback_handler(
    *,
    session_id: str,
    user_id: str | None = None,
    client: object | None = None,
) -> Any | None:
    """Return a Langfuse ``CallbackHandler`` for auto-tracing LLM/tool calls.

    When *client* is ``None`` (or the global client is unavailable) the
    function returns ``None`` — consumers should simply not add the
    handler to the callbacks list.

    Args:
        session_id: Thread/session identifier (stored as metadata).
        user_id: Optional user identifier (stored as metadata).
        client: Optional pre-obtained Langfuse client (injected for testability).
    """
    if client is None:
        from src.observability._client import _get_langfuse_client

        client = _get_langfuse_client()

    if client is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # In Langfuse >=4.0, CallbackHandler() requires no positional args
        # when the global Langfuse client is already initialized.
        # session_id and user_id are set on the root trace via @observe.
        handler = CallbackHandler()
        # Store metadata on the handler for downstream use
        handler._session_id = session_id  # type: ignore[attr-defined]
        handler._user_id = user_id or "unknown"  # type: ignore[attr-defined]
        return handler
    except Exception:
        logger.warning(
            "Failed to create Langfuse CallbackHandler — skipping tracing for session %s",
            session_id,
            exc_info=True,
        )
        return None
