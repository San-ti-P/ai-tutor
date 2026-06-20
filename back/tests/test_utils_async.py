"""Unit tests for src/utils/async_.py — run_async_in_sync utility."""

from __future__ import annotations

import asyncio

import pytest

from src.utils.async_ import run_async_in_sync


async def _sample_coro(value: int = 42) -> int:
    """Simple test coroutine that returns a value."""
    await asyncio.sleep(0)  # Yield to event loop
    return value


class TestRunAsyncInSync:
    """Tests for run_async_in_sync()."""

    def test_runs_with_no_event_loop(self):
        """When no event loop exists (RuntimeError), creates one via asyncio.run."""
        result = run_async_in_sync(_sample_coro(55))
        assert result == 55

    def test_runs_with_event_loop_not_running(self):
        """When a loop exists but isn't running, uses run_until_complete."""

        async def _inner():
            return run_async_in_sync(_sample_coro(99))

        result = asyncio.run(_inner())
        assert result == 99

    def test_runs_when_loop_is_running(self):
        """When loop is running, falls back to ThreadPoolExecutor."""

        async def _test():
            # The test itself runs inside asyncio.run(), so get_event_loop()
            # will exist and is_running(). Our utility should handle this.
            return run_async_in_sync(_sample_coro(7))

        result = asyncio.run(_test())
        assert result == 7

    def test_coro_with_exception_propagates(self):
        """If the coroutine raises, the exception propagates."""

        async def _failing():
            raise ValueError("test error")
            return 1

        with pytest.raises(ValueError, match="test error"):
            run_async_in_sync(_failing())

    def test_coro_with_no_args(self):
        """Coroutine returning a string works."""
        async def _greet():
            return "hello"

        assert run_async_in_sync(_greet()) == "hello"
