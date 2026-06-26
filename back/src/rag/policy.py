"""Shared RAG-only policy — canonical system prompt and no-material helper.

All RAG-dependent agents (query_material, evaluator, exam_generator,
exercise_generator) import from this module to enforce consistent
grounding in ingested material and uniform no-material messaging.

This module MUST NOT import from agent implementations — agents
consume the policy, not the reverse.
"""

RAG_ONLY_SYSTEM_PROMPT: str = (
    "Sos un tutor académico. Respondé usando ÚNICAMENTE la información "
    "proporcionada en los fragmentos del material. Si no hay fragmentos "
    "o la información no alcanza, no inventes respuestas."
)


def no_material_message() -> str:
    """Return the canonical "no material available" user-facing message.

    Used by query_material, evaluator, and orchestrator when ChromaDB
    returns zero chunks for a session's collection.
    """
    return (
        "No encontré material cargado para responder esa pregunta. "
        "Subí apuntes o documentos primero."
    )
