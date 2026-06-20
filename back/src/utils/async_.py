"""Async utility for running coroutines from synchronous code.

Provides ``run_async_in_sync`` — the single source of truth for the
event-loop-detection pattern duplicated 5 times across support.py
and evaluator.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any


def run_async_in_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously, handling all event-loop states.

    Three branches:
    1. No event loop exists (RuntimeError) → ``asyncio.run(coro)``
    2. Loop exists but not running → ``loop.run_until_complete(coro)``
    3. Loop is running (e.g. inside async test or FastAPI) →
       ``ThreadPoolExecutor`` with ``asyncio.run`` in a separate thread

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.

    Raises:
        Any exception raised by the coroutine propagates upward.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running loop — need a new event loop in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        # Loop exists but not running
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop at all — create one
        return asyncio.run(coro)
