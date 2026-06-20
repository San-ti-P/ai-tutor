"""Aggregate-metrics computation over a Langfuse trace tree — internal."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_aggregate_metrics(trace: Any) -> dict[str, Any]:
    """Walk *trace* span tree and compute execution-level aggregates.

    Returns a dict with:
        - ``total_steps``: count of non-root spans
        - ``total_tokens``: sum of prompt + completion tokens
        - ``total_cost``: estimated cost (USD, from Langfuse usage)
        - ``total_latency_ms``: cumulative latency across all spans
        - ``tool_success_rate``: fraction of tool spans with status != "error"
        - ``avg_score``: mean of numeric scores in ``total_scores`` (0.0 if none)

    All values are safe — if *trace* is ``None`` or has no observations,
    every aggregate returns 0 / 0.0.
    """
    try:
        observations = getattr(trace, "observations", []) or []
    except Exception:
        return _empty_metrics()

    if not observations:
        return _empty_metrics()

    total_tokens = 0
    total_cost = 0.0
    total_latency_ms = 0.0
    tool_calls = 0
    tool_errors = 0
    scores: list[float] = []

    for obs in observations:
        try:
            # Tokens from generation spans
            usage = getattr(obs, "usage", None)
            if usage:
                total_tokens += (getattr(usage, "input", 0) or 0) + (
                    getattr(usage, "output", 0) or 0
                )

            # Cost from generation spans
            computed_cost = getattr(obs, "calculated_total_cost", 0.0) or 0.0
            total_cost += computed_cost

            # Latency
            latency = getattr(obs, "latency", 0.0) or 0.0
            total_latency_ms += latency

            # Tool success rate
            obs_type = getattr(obs, "type", "")
            if obs_type == "span" and getattr(obs, "name", "").startswith("tool_"):
                tool_calls += 1
                if getattr(obs, "status_message", ""):
                    tool_errors += 1

            # Scores
            obs_scores = getattr(obs, "scores", []) or []
            for s in obs_scores:
                val = getattr(s, "value", None)
                if isinstance(val, (int, float)):
                    scores.append(float(val))

        except Exception:
            # Individual observation parsing should never fail the whole loop
            continue

    tool_success_rate = (
        (tool_calls - tool_errors) / tool_calls if tool_calls > 0 else 1.0
    )

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_steps": len(observations),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "total_latency_ms": round(total_latency_ms, 2),
        "tool_success_rate": round(tool_success_rate, 4),
        "avg_score": round(avg_score, 2),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_steps": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_latency_ms": 0.0,
        "tool_success_rate": 0.0,
        "avg_score": 0.0,
    }
