"""Pure text-splitting utilities for academic content processing.

Functions:
- split_sentences: break text on sentence boundaries (.!?)
- split_into_claims: granular claims via sentences + semicolon splits
- parse_file_to_text: read file content via markitdown (PDF/TXT)

Note: ``parse_file_to_text`` does I/O (file reading via markitdown). This is
an intentional exception to the pure-function rule because markitdown parsing
is duplicated across the codebase and centralizing it avoids copy-paste bugs.
"""

from __future__ import annotations

import re
from pathlib import Path

# Regex matching sentence boundaries: . ! ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Matches lowercase word breaks: lowercase-letter-hyphen-newline-lowercase-letter.
# Only merges when BOTH sides are lowercase Latin (incl. Spanish accents).
# Preserves legitimate hyphens (uppercase, mixed-case, numbers).
_DEHYPHENATE_RE = re.compile(r"([a-záéíóúñ])-\n([a-záéíóúñ])")


def split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries (. ! ? followed by whitespace).

    Args:
        text: Raw text to split into sentences.

    Returns:
        List of sentence strings (whitespace-stripped).
        Empty text returns an empty list.
    """
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip())]


def split_into_claims(text: str, min_length: int = 10) -> list[str]:
    """Split text into atomic claims for embedding comparison.

    First splits on sentence boundaries, then further splits long sentences
    on semicolons for more granular claim extraction. Short fragments
    (below ``min_length``) are filtered out.

    Args:
        text: Raw text to split into claims.
        min_length: Minimum character length for a claim to be kept.

    Returns:
        List of claim strings. Empty text returns an empty list.
    """
    if not text or not text.strip():
        return []

    sentences = split_sentences(text)
    claims: list[str] = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Further split on semicolons for long compound sentences
        parts = [p.strip() for p in s.split(";") if p.strip()]
        claims.extend(parts)

    return [c for c in claims if len(c) > min_length]


def dehyphenate_text(raw_text: str) -> str:
    """Merge lowercase words hyphenated across line breaks.

    Academic PDFs parsed by markitdown contain mid-word breaks
    (e.g. ``"am-\\nbiente"``) that produce incoherent vectors.
    This function joins only lowercase-to-lowercase splits;
    uppercase or mixed-case hyphen patterns are left unchanged
    (legitimate hyphens in names, acronyms, etc.).

    Uses pure stdlib ``re`` — no dependencies.

    Args:
        raw_text: Raw text from markitdown or another PDF parser.

    Returns:
        Text with line-break hyphenation artifacts removed.
    """
    return _DEHYPHENATE_RE.sub(r"\1\2", raw_text)


def parse_file_to_text(file_path: str) -> str:
    """Parse a PDF or TXT file and return its text content via markitdown.

    Centralized markitdown parsing — the single source of truth for
    converting ingested files to raw text.

    Args:
        file_path: Path to a PDF or TXT file on disk.

    Returns:
        The extracted text content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import markitdown

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    md = markitdown.MarkItDown()
    result = md.convert(str(p))
    return result.text_content
