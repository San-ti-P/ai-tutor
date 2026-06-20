"""Unit tests for src.utils.retry — retry-decision utilities."""

from __future__ import annotations

from src.utils.retry import should_retry


class TestShouldRetry:
    """Tests for the pure should_retry utility."""

    def test_errors_and_under_limit_returns_retry(self) -> None:
        """Errors present + retry_count < 3 → 'retry'."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=1,
            status="pending",
        )
        assert result == "retry"

    def test_at_limit_returns_done(self) -> None:
        """retry_count == 3 → 'done' even with errors."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=3,
            status="pending",
        )
        assert result == "done"

    def test_over_limit_returns_done(self) -> None:
        """retry_count > 3 → 'done'."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=4,
            status="pending",
        )
        assert result == "done"

    def test_no_errors_returns_done(self) -> None:
        """Empty errors → 'done'."""
        result = should_retry(
            validation_errors=[],
            retry_count=0,
            status="pending",
        )
        assert result == "done"

    def test_status_error_returns_done(self) -> None:
        """Terminal status 'error' → 'done' regardless of other state."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=0,
            status="error",
        )
        assert result == "done"

    def test_status_no_material_returns_done(self) -> None:
        """Terminal status 'no_material' → 'done'."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=0,
            status="no_material",
        )
        assert result == "done"

    def test_retry_count_boundary(self) -> None:
        """Exactly at the boundary (retry_count == 2) with errors → 'retry'."""
        result = should_retry(
            validation_errors=["claim missing"],
            retry_count=2,
            status="pending",
        )
        assert result == "retry"
