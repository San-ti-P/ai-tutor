# Epic 15 — Semantic Chunking for RAG Pipeline

**Status:** Done
**Depends on:** Epic 02 (Ingestor/RAG)

> **Goal**: Fix broken RAG chunks caused by markitdown hyphenation artifacts and character-count splitting. Replace misleading "semantic chunks" claim with real paragraph-aware splitting plus dehyphenation pre-processing.

---

## 1. Problem

`chunk_text()` in `back/src/rag/__init__.py` used `RecursiveCharacterTextSplitter` with `separators=["\n\n", "\n", ". ", " ", ""]` and `chunk_size=512`. The docstring claimed "semantic chunks" — it wasn't semantic. It split on character count, picking the nearest syntactic boundary before the limit.

**Root cause**: markitdown parses PDFs into markdown but produces **zero markdown headings**. Real-world test on `apunteAgentes_IA2007.pdf` (53,580 chars):

- 0 headings (`#`, `##`, `###`)
- 53 hyphenated line breaks: `"am-\nbiente"` → word split mid-syllable
- 51 soft-hyphen breaks: `"char-\nchar"`
- Only 19 paragraph breaks (`\n\n`)
- 123 chunks produced — many incoherent (mid-word splits, table row fragments)

`MarkdownHeaderTextSplitter` and `SemanticChunker` were evaluated and **rejected** — no headers exist to split on, and `SemanticChunker` adds latency + a new dependency for marginal gain.

---

## 2. Solution

### Two-phase fix: dehyphenate first, then chunk on paragraphs.

```
Raw markitdown text
  → dehyphenate_text()          [merge "am-\nbiente" → "ambiente"]
  → chunk_text()                [split on \n\n, sentences, then chars]
  → embed_and_store()           [unchanged]
```

### 2.1 Dehyphenation — `back/src/utils/text.py` (NEW)

```python
def dehyphenate_text(raw_text: str) -> str:
    """Merge lowercase words hyphenated across line breaks."""
    return _DEHYPHENATE_RE.sub(r"\1\2", raw_text)

_DEHYPHENATE_RE = re.compile(r"([a-záéíóúñ])-\n([a-záéíóúñ])")
```

Pure stdlib `re`. Only merges `lowercase-\nlowercase` — preserves legitimate hyphens (uppercase, mixed-case, numbers). Spanish accents included.

### 2.2 Paragraph-aware separators — `back/src/rag/__init__.py`

```
Before: ["\n\n", "\n", ". ", " ", ""]
After:  ["\n\n", ". ", "! ", "? ", "\n", " ", ""]
```

Paragraph boundaries (`\n\n`) first, then sentences (`. `, `! `, `? `), then raw newlines, then character fallback. `chunk_size` increased 512→800.

### 2.3 Integration — `back/src/agents/ingestor.py`

```python
clean_text = dehyphenate_text(state["raw_text"])
chunks = chunk_text(clean_text)
```

One line wiring. Dehyphenation happens before chunking so no hyphen artifacts reach the vector store.

---

## 3. Why-Not Docs

| Approach | Verdict | Why |
|----------|---------|-----|
| `MarkdownHeaderTextSplitter` | ❌ | Zero headers in markitdown output — nothing to split on |
| `ExperimentalMarkdownSyntaxTextSplitter` | ❌ | Same — needs headers |
| `SemanticChunker` (langchain_experimental) | ❌ | New dependency, embeds every sentence, adds latency; dehyphenation alone fixes most quality |

---

## 4. Files Changed

| File | Action | Lines |
|------|--------|-------|
| `back/src/utils/text.py` | NEW — `dehyphenate_text()` | +25 |
| `back/src/rag/__init__.py` | MODIFY — separators, docstring | ~12 |
| `back/src/config.py` | MODIFY — `chunk_size: 800` | 1 |
| `back/src/agents/ingestor.py` | MODIFY — wire dehyphenation | +6 |
| `back/tests/test_utils_text.py` | MODIFY — 7 new + 13 preserved tests | +41 |
| `back/tests/test_rag.py` | MODIFY — max-size assertions | ~6 |
| **Total** | | **~78 insertions, 14 deletions** |

---

## 5. Verification

- **575/575** full test suite passes (0 failures)
- **13/13** spec scenarios compliant
- **50/50** affected-file tests pass
- Ruff clean on changed files
- Zero regressions — embedding `"passage:"` / `"query:"` prefixes preserved

---

## 6. Acceptance Criteria

- [x] `dehyphenate_text()` merges `"am-\nbiente"` → `"ambiente"` (7 edge-case scenarios)
- [x] `dehyphenate_text()` preserves `"Dr.-\nSmith"` (uppercase hyphen)
- [x] `dehyphenate_text()` is no-op on non-Latin text
- [x] `chunk_text()` splits on paragraph boundaries before newlines
- [x] `chunk_size` = 800 bytes
- [x] `chunk_and_embed()` calls `dehyphenate_text()` before `chunk_text()`
- [x] Existing `TestSplitSentences`, `TestSplitIntoClaims`, `TestParseFileToText` preserved
- [x] No embedding prefix changes (out of scope)

---

## 7. SDD Artifacts

All artifacts in Engram under project `ai-tutor`:

| Artifact | Topic Key |
|----------|-----------|
| Exploration | `sdd/semantic-chunking-markdown/explore` |
| Proposal | `sdd/semantic-chunking-markdown/proposal` |
| Spec | `sdd/semantic-chunking-markdown/spec` |
| Tasks | `sdd/semantic-chunking-markdown/tasks` |
| Apply Progress | `sdd/semantic-chunking-markdown/apply-progress` |
| Verify Report | `sdd/semantic-chunking-markdown/verify-report` |
| Archive | `sdd/semantic-chunking-markdown/archive-report` |
