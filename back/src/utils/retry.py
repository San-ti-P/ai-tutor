"""Retry-decision utilities for agent graphs.

Pure functions that determine whether an agent should retry a failed
operation based on validation state. No I/O, no agent-specific logic.
"""

from __future__ import annotations


def should_retry(validation_errors: list[str], retry_count: int, status: str) -> str:
    """Return 'retry' if validation errors exist AND retry_count < 3, else 'done'.

    Does NOT retry on terminal statuses ('error', 'no_material').

    Args:
        validation_errors: List of validation error strings.
        retry_count: Current retry attempt number.
        status: Overall agent status string.

    Returns:
        ``'retry'`` if regeneration should be attempted, ``'done'`` otherwise.
    """
    if status in ("error", "no_material"):
        return "done"
    if validation_errors and retry_count < 3:
        return "retry"
    return "done"
