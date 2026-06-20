"""Shared pure-function utilities for the ai-tutor codebase.

Conventions:
- Pure functions only — no I/O (no file reads, no network, no DB queries).
- No agent-specific logic — utilities are reusable across all agents.
- No side effects — deterministic output for given input.
- Must have unit tests — every exported function must be covered.
"""
