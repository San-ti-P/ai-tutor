"""Build hierarchical topic tree from unified list (TXR-06).

- ≤5 topics: deterministic prefix grouping (no LLM cost).
- ≥5 topics: LLM call to organize into nested hierarchy using same
  schema-in-prompt pattern as extract.py.
- Deterministic fallback on LLM failure.
"""

from __future__ import annotations

import json
import logging
import re

from src.llm import get_llm
from src.config import settings

logger = logging.getLogger("tutor.topic_extraction.tree")

# ── Regex for extracting JSON from LLM output ─────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}")


async def build_topic_tree(topics: list[str]) -> dict:
    """Build a nested dict hierarchy from a flat topic list.

    Args:
        topics: Unified topic list from ``unify_topics()``.

    Returns:
        Nested dict like ``{"Agentes": {"Tipos": {}, "Entorno": {}}}``.
        Always a plain ``dict``, serializable with ``json.dumps``.
    """
    if not topics:
        return {}

    if len(topics) <= 5:
        # Deterministic: group by first word, each leaf is empty dict
        tree: dict = {}
        for topic in sorted(topics):
            parts = topic.split()
            first = parts[0]
            tree.setdefault(first, {})
        return {k: {} for k in sorted(tree)}

    # ≥5 topics: LLM call
    try:
        llm = get_llm(temperature=settings.topic_extraction_temperature)

        schema_desc = '{"Categoría Principal": {"Subtema": {}, ...}, ...}'
        prompt = (
            "Organizá estos temas académicos en un árbol jerárquico. "
            "IMPORTANTE: Creá entre 3 y 7 categorías principales (raíces). "
            "No pongas todo bajo una sola categoría. Agrupá por áreas conceptuales distintas. "
            "Cada categoría principal debe contener subtemas anidados.\n\n"
            "Devolvé ÚNICAMENTE un objeto JSON donde cada clave es una categoría "
            "de nivel superior y cada valor es un objeto anidado de subtemas "
            "(objetos vacíos para las hojas).\n\n"
            f"Esquema: {schema_desc}\n\n"
            "Temas:\n" + "\n".join(f"- {t}" for t in topics)
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "Sos un experto en taxonomía académica. Organizás temas en jerarquías. "
                    "Siempre creás entre 3 y 7 categorías raíz, nunca una sola. "
                    "Devolvé ÚNICAMENTE JSON válido que coincida con este patrón: "
                    f"{schema_desc}\n"
                    "Ningún otro texto. Solo JSON válido."
                )
            ),
            HumanMessage(content=prompt),
        ]

        response = await llm.ainvoke(messages)
        response_text: str = response.content if hasattr(response, "content") else str(response)

        match = _JSON_RE.search(response_text)
        if match:
            tree = json.loads(match.group(0))
            if isinstance(tree, dict):
                logger.debug("LLM-built topic tree: %d top-level categories", len(tree))
                return tree

        raise ValueError(f"LLM response not a valid dict: {response_text[:200]!r}")

    except Exception as exc:
        logger.warning("Topic tree LLM call failed, using flat fallback: %s", exc)
        # Deterministic fallback: flat dict
        return {t: {} for t in sorted(topics)}
