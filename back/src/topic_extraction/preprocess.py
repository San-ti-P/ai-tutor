"""Spanish NLP preprocessing for topic extraction.

Provides stopword removal, Snowball Spanish stemming, and Jaccard
similarity — all used by the unification stage (unify.py).  NLP is
applied to topic strings only, never to LLM input (TXR-01).

Lazy-loads NLTK data to avoid import-time downloads.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("tutor.topic_extraction.preprocess")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_stopwords: set[str] | None = None
_stemmer: Any | None = None


def _ensure_nltk() -> None:
    """Lazy-load NLTK stopwords and stemmer (first call only)."""
    global _stopwords, _stemmer

    if _stopwords is not None:
        return

    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import SnowballStemmer

    _stopwords = set(nltk_stopwords.words("spanish"))
    _stemmer = SnowballStemmer("spanish")
    logger.debug("NLTK Spanish stopwords and stemmer loaded (%d stopwords)", len(_stopwords))


# ── Public API ────────────────────────────────────────────────────────────────


def remove_stopwords(text: str) -> list[str]:
    """Tokenize *text* and remove Spanish stopwords.

    Splits on whitespace, lowercases, and filters out tokens that match
    the NLTK Spanish stopword list.

    Args:
        text: Raw topic string (e.g. ``"Agentes inteligentes y su entorno"``).

    Returns:
        List of lowercase content-word tokens.  Empty list if *text* is
        empty or consists entirely of stopwords.
    """
    _ensure_nltk()
    assert _stopwords is not None

    if not text or not text.strip():
        return []

    tokens = text.lower().split()
    return [t for t in tokens if t not in _stopwords]


def stem_topic(topic: str) -> set[str]:
    """Stem a topic string to a set of keyword stems.

    Lowercases → tokenizes on whitespace → removes Spanish stopwords →
    applies Snowball Spanish stemmer → returns unique stems.

    Args:
        topic: A single topic string.

    Returns:
        Set of stemmed keywords.  Empty set if *topic* is empty or
        consists only of stopwords.

    Example:
        >>> stem_topic("Agentes y su entorno")
        {'agnt', 'entorn'}
    """
    _ensure_nltk()
    assert _stemmer is not None

    tokens = remove_stopwords(topic)
    return {_stemmer.stem(t) for t in tokens}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets of stems.

    .. math::

        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    By convention, two empty sets have similarity 1.0.

    Args:
        set_a: First set of stemmed keywords.
        set_b: Second set of stemmed keywords.

    Returns:
        Float in [0.0, 1.0].
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
