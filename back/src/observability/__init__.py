"""Observability module — Langfuse tracing for agents, tools, RAG.

Public API::

    from src.observability import ObservabilityManager, get_tracer, flush_traces

The module-level singleton is created lazily on first ``get_tracer()`` call
and reused thereafter.  Consumers never instantiate ``ObservabilityManager``
directly — use ``get_tracer()``.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = logging.getLogger(__name__)

_manager: ObservabilityManager | None = None


# ---------------------------------------------------------------------------
# ObservabilityManager
# ---------------------------------------------------------------------------


class ObservabilityManager:
    """Central observability coordinator.

    Lazily initialises a Langfuse client via ``_client._get_langfuse_client()``
    on first use.  Exposes a boolean ``enabled`` flag that gates all tracing
    operations.  When disabled, every method returns a safe no-op value.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._client: Langfuse | None = None

    # -- properties -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a Langfuse client was successfully created."""
        self._ensure_init()
        return self._client is not None

    # -- tracing API ----------------------------------------------------------

    def create_trace(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any | None:
        """Create a top-level Langfuse trace.

        Returns a trace-span object (or no-op sentinel when disabled).  The
        span serves as the root for an entire execution.

        Uses the Langfuse v4 API (``start_observation``) which implicitly
        creates a trace when the first span is started.  Trace-level
        attributes (session_id, user_id, trace_name) are set via
        ``propagate_attributes()`` so child spans and the CallbackHandler
        inherit them.
        """
        self._ensure_init()
        if self._client is None:
            return None
        try:
            from langfuse import propagate_attributes

            meta = dict(metadata or {})
            meta.setdefault("session_id", session_id)
            if user_id:
                meta.setdefault("user_id", user_id)

            # propagate_attributes sets trace-level baggage so child spans
            # (via @observe or CallbackHandler) inherit session/user context.
            # Environment stays in metadata so it appears in start_observation
            # AND is filterable in the Langfuse dashboard.
            with propagate_attributes(
                session_id=session_id,
                user_id=user_id or "unknown",
                trace_name=name,
                metadata=meta,
            ):
                trace = self._client.start_observation(
                    name=name,
                    as_type="span",
                    metadata=meta,
                )

            return trace
        except Exception:
            logger.warning("Failed to create Langfuse trace '%s'", name, exc_info=True)
            return None

    def get_callback_handler(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> Any | None:
        """Return a LangChain CallbackHandler for automatic LLM/tool tracing."""
        from src.observability._callbacks import get_callback_handler

        self._ensure_init()
        return get_callback_handler(
            session_id=session_id,
            user_id=user_id,
            client=self._client,
        )

    def compute_aggregate_metrics(self, trace: Any) -> dict[str, Any]:
        """Return execution-level aggregates for *trace*."""
        from src.observability._metrics import compute_aggregate_metrics

        return compute_aggregate_metrics(trace)

    def flush(self) -> None:
        """Flush queued spans to Langfuse (non-blocking)."""
        self._ensure_init()
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        """Flush and shut down the Langfuse client."""
        self._ensure_init()
        if self._client is None:
            return
        try:
            self._client.flush()
            self._client.shutdown()
        except Exception:
            logger.debug("Langfuse shutdown failed", exc_info=True)

    # -- internal -------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        from src.observability._client import _get_langfuse_client

        self._client = _get_langfuse_client()


# ---------------------------------------------------------------------------
# Public module-level API
# ---------------------------------------------------------------------------


def get_tracer() -> ObservabilityManager:
    """Return the module-level singleton ``ObservabilityManager``."""
    global _manager
    if _manager is None:
        _manager = ObservabilityManager()
    return _manager


def flush_traces() -> None:
    """Flush all pending Langfuse spans on the singleton tracer."""
    get_tracer().flush()
