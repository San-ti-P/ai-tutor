"""Lazy Langfuse singleton — internal module.

Returns a configured Langfuse client on first call, or None if
required environment keys are missing or init fails.  Errors are
logged at WARNING level and never propagate to callers.
"""

from __future__ import annotations

import logging

from src.config import settings

logger = logging.getLogger(__name__)

_langfuse_client: object | None = None
_init_attempted: bool = False


def _get_langfuse_client() -> object | None:
    """Return the lazy-singleton Langfuse client, or None if unavailable.

    Reads ``settings.langfuse_public_key``, ``_secret_key``, and
    ``_host``.  If any key is empty the client is never created and
    ``None`` is returned silently (observability is considered
    disabled).

    On the first successful init the client is cached and reused.
    If init fails the exception is logged at WARNING and ``None`` is
    returned — callers must handle the ``None`` case gracefully.
    """
    global _langfuse_client, _init_attempted

    if _langfuse_client is not None:
        return _langfuse_client

    if _init_attempted:
        return None

    _init_attempted = True

    if (
        not settings.langfuse_public_key
        or not settings.langfuse_secret_key
        or not settings.langfuse_host
    ):
        logger.warning(
            "Langfuse keys not configured — observability disabled. "
            "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env"
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse client initialised (host=%s)", settings.langfuse_host)
        return _langfuse_client
    except Exception:
        logger.warning("Langfuse init failed — observability disabled", exc_info=True)
        return None


def _reset_langfuse_client() -> None:
    """Reset the singleton (used in tests)."""
    global _langfuse_client, _init_attempted
    _langfuse_client = None
    _init_attempted = False
