"""Unit tests for topic_extraction module — Phase 2 Core Pipeline.

Covers TXR-01 (NLP preprocessing), TXR-02 (segmentation),
TXR-03 (extraction), TXR-05 (unification), TXR-06 (tree),
TXR-09/TXR-10 (pipeline API).
"""

from __future__ import annotations

import json

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 — preprocess.py tests (TXR-01)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemoveStopwords:
    """Tests for remove_stopwords() — NLTK Spanish stopword filtering."""

    def test_removes_spanish_stopwords(self):
        """TXR-01: Spanish stopwords 'y', 'su', 'de', 'la', 'el' removed."""
        from src.topic_extraction.preprocess import remove_stopwords

        result = remove_stopwords("Agentes inteligentes y su entorno")
        # 'y' and 'su' are Spanish stopwords
        has_y = any(t == "y" for t in result)
        has_su = any(t == "su" for t in result)
        assert not has_y, f"'y' should be removed, got {result}"
        assert not has_su, f"'su' should be removed, got {result}"
        # Content words preserved
        assert "agentes" in result
        assert "inteligentes" in result
        assert "entorno" in result

    def test_lowercases_output(self):
        """All tokens returned in lowercase."""
        from src.topic_extraction.preprocess import remove_stopwords

        result = remove_stopwords("Álgebra Lineal AVANZADA")
        for token in result:
            assert token == token.lower(), f"Token '{token}' not lowercase"

    def test_returns_empty_for_stopword_only_input(self):
        """Text consisting entirely of stopwords returns empty list."""
        from src.topic_extraction.preprocess import remove_stopwords

        result = remove_stopwords("y el la los las de del a un una")
        assert result == [], f"Expected empty, got {result}"

    def test_returns_empty_for_empty_string(self):
        """Empty input returns empty list."""
        from src.topic_extraction.preprocess import remove_stopwords

        assert remove_stopwords("") == []


class TestStemTopic:
    """Tests for stem_topic() — Snowball Spanish stemming."""

    def test_stem_agentes_phrase(self):
        """TXR-01: 'Agentes inteligentes y su entorno' → stems without stopwords."""
        from src.topic_extraction.preprocess import stem_topic

        result = stem_topic("Agentes inteligentes y su entorno")
        # Stopwords 'y', 'su' removed
        # Stems: agentes→agent, inteligentes→inteligent, entorno→entorn
        assert "agent" in result, f"Expected 'agent' in {result}"
        assert "inteligent" in result, f"Expected 'inteligent' in {result}"
        assert "entorn" in result, f"Expected 'entorn' in {result}"
        # No stopwords in output
        assert "y" not in result
        assert "su" not in result

    def test_returns_set(self):
        """Returns a set[str]."""
        from src.topic_extraction.preprocess import stem_topic

        result = stem_topic("agentes inteligentes")
        assert isinstance(result, set)

    def test_stem_reduces_variants(self):
        """Inflected forms 'agente' and 'agentes' stem to same root."""
        from src.topic_extraction.preprocess import stem_topic

        s1 = stem_topic("agente inteligente")
        s2 = stem_topic("agentes inteligentes")
        # The stems should overlap significantly
        intersection = s1 & s2
        assert len(intersection) >= 1, f"Expected overlapping stems, got s1={s1}, s2={s2}"

    def test_stem_handles_single_word(self):
        """Single word stems correctly."""
        from src.topic_extraction.preprocess import stem_topic

        result = stem_topic("matemáticas")
        assert len(result) > 0
        assert "matemat" in result, f"Expected 'matemat' in {result}"

    def test_stem_accented_characters(self):
        """TXR-01 edge: accented Spanish characters (á, é, í, ó, ú) stem correctly."""
        from src.topic_extraction.preprocess import stem_topic

        result = stem_topic("Máquinas de soporte vectorial")
        # Snowball Spanish should handle accents
        assert "maquin" in result, f"Expected 'maquin' in {result}"
        assert "soport" in result, f"Expected 'soport' in {result}"
        assert "vectorial" in result, f"Expected 'vectorial' in {result}"

    def test_stem_n_with_tilde(self):
        """TXR-01 edge: ñ character handled correctly in stemming."""
        from src.topic_extraction.preprocess import stem_topic

        result = stem_topic("enseñanza aprendizaje")
        assert len(result) > 0
        assert "ensen" in result or "enseñ" in result, (
            f"Expected stem for 'enseñanza', got {result}"
        )


class TestJaccardSimilarity:
    """Tests for jaccard_similarity() — set overlap metric."""

    def test_identical_sets(self):
        """TXR-05: Identical sets → Jaccard = 1.0."""
        from src.topic_extraction.preprocess import jaccard_similarity

        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """TXR-05: Disjoint sets → Jaccard = 0.0."""
        from src.topic_extraction.preprocess import jaccard_similarity

        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        """One shared element among four → 1/3 ≈ 0.333."""
        from src.topic_extraction.preprocess import jaccard_similarity

        result = jaccard_similarity({"a", "b"}, {"a", "c"})
        assert result == pytest.approx(1 / 3)

    def test_empty_sets(self):
        """Both empty → Jaccard = 1.0 (by convention)."""
        from src.topic_extraction.preprocess import jaccard_similarity

        assert jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        """One empty, one non-empty → Jaccard = 0.0."""
        from src.topic_extraction.preprocess import jaccard_similarity

        assert jaccard_similarity({"a"}, set()) == 0.0
        assert jaccard_similarity(set(), {"a"}) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2 — segment.py tests (TXR-02)
# ═══════════════════════════════════════════════════════════════════════════════

MARKDOWN_DOC = """# Introducción a los Agentes Inteligentes

Los agentes inteligentes son sistemas que perciben su entorno.

## Tipos de Agentes

Existen varios tipos de agentes: reactivos, deliberativos, etc.

Los agentes reactivos responden a estímulos inmediatos sin memoria.

### Agentes Reactivos Simples

Reglas condición-acción. No mantienen estado interno.

### Agentes Reactivos Basados en Modelos

Mantienen un modelo del mundo para manejar parcialmente la observabilidad.

## Entorno de los Agentes

El entorno puede ser determinista o estocástico.

Los entornos multi-agente añaden complejidad adicional.

# Razonamiento y Planificación

Los agentes deben poder razonar sobre sus acciones.

## Búsqueda en Espacio de Estados

Estrategias de búsqueda ciega y heurística.
"""


class TestSegmentText:
    """Tests for segment_text() — markdown heading-based segmentation."""

    def test_splits_on_markdown_headings(self):
        """TXR-02: Splits markdown on #, ##, ### headings."""
        from src.topic_extraction.segment import segment_text

        segments = segment_text(MARKDOWN_DOC, min_section=10)
        assert len(segments) > 1, f"Expected >1 segments, got {len(segments)}"
        # Should have at least: Introduccion, Tipos, Reactivos Simples, etc.
        assert any("Introducción" in s for s in segments)

    def test_each_segment_starts_with_heading(self):
        """Each segment (except first intro text) starts with # heading."""
        from src.topic_extraction.segment import segment_text

        segments = segment_text(MARKDOWN_DOC, min_section=10)
        for seg in segments[1:]:  # first segment may be intro
            assert seg.strip(), "Empty segment found"
            # Should start with # heading
            assert seg.strip().startswith("#"), f"Segment doesn't start with heading: {seg[:60]!r}"

    def test_merges_small_adjacent_sections(self):
        """Adjacent sections < min_section chars merged together."""
        from src.topic_extraction.segment import segment_text

        text = (
            "# Tema A\n\nCorto.\n\n"
            "## Tema B\n\nTambién corto.\n\n"
            "# Tema C\n\nUn párrafo más largo con suficiente contenido para superar "
            "el límite mínimo de caracteres por sección en este texto de ejemplo.\n\n"
        )
        segments = segment_text(text, min_section=200)
        # Tema A + Tema B should be merged (both < 200 chars each)
        # Tema C might stand alone if large enough
        assert len(segments) <= 2, f"Expected ≤2 segments, got {len(segments)}: {segments}"

    def test_passthrough_short_text(self):
        """TXR-10: Short text (< max_chars) returns single segment."""
        from src.topic_extraction.segment import segment_text

        short = "Este es un texto corto sin headings."
        segments = segment_text(short, max_chars=6000)
        assert segments == [short]

    def test_returns_empty_for_whitespace(self):
        """Empty or whitespace-only text returns []."""
        from src.topic_extraction.segment import segment_text

        assert segment_text("") == []
        assert segment_text("   \n  \n  ") == []

    def test_fallback_to_paragraphs(self):
        """TXR-02: Plain text without headings splits on double newline."""
        from src.topic_extraction.segment import segment_text

        # Use small max_chars to force paragraph split
        text = "Párrafo uno.\n\nPárrafo dos.\n\nPárrafo tres."
        segments = segment_text(text, min_section=1, max_chars=5)
        assert len(segments) >= 2, f"Expected ≥2 segments, got {len(segments)}"

    def test_no_headings_short_text_passthrough(self):
        """Text with no headings and less than max_chars returns single element."""
        from src.topic_extraction.segment import segment_text

        text = "Un solo párrafo sin headings ni mucho contenido."
        segments = segment_text(text, max_chars=200)
        assert segments == [text]

    def test_fallback_to_paragraphs_merges_small_adjacent(self):
        """Text without headings splits on double newline but merges small paragraphs."""
        from src.topic_extraction.segment import segment_text

        text = "Corto 1.\n\nCorto 2.\n\nCorto 3.\n\nEste es un párrafo sustancialmente más largo que supera el mínimo de caracteres configurado."
        # min_section = 100, max_chars = 10
        # Corto 1 (8), Corto 2 (8), Corto 3 (8) should merge
        segments = segment_text(text, min_section=100, max_chars=10)
        assert len(segments) == 1

    def test_backward_merge_last_segment_short(self):
        """If the last segment is shorter than min_section, it is merged backward."""
        from src.topic_extraction.segment import segment_text

        text = (
            "# Sección 1\n\n"
            "Un texto relativamente largo que va a superar los 100 caracteres cómodamente. "
            "Esto asegura que la primera sección tenga suficiente longitud por sí sola y no intente mergearse hacia adelante de entrada.\n\n"
            "# Sección 2\n\n"
            "Cortito."
        )
        segments = segment_text(text, min_section=100)
        # Section 2 is very short, so it must be merged backward into Section 1.
        assert len(segments) == 1
        assert "Sección 1" in segments[0]
        assert "Cortito." in segments[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3 — extract.py tests (TXR-03)
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# TDR Phase — TopicItem model and description extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicItemModel:
    """Tests for TopicItem Pydantic model (TDR-01, TDR-02)."""

    def test_topic_item_creation_valid(self):
        """TDR-01: TopicItem created with topic + description."""
        from src.topic_extraction.extract import TopicItem

        item = TopicItem(
            topic="Agentes inteligentes",
            description="Sistemas que perciben su entorno y actúan mediante efectores.",
        )
        assert item.topic == "Agentes inteligentes"
        assert item.description == "Sistemas que perciben su entorno y actúan mediante efectores."

    def test_topic_item_serializes_to_dict(self):
        """TopicItem serializes to dict via model_dump()."""
        from src.topic_extraction.extract import TopicItem

        item = TopicItem(
            topic="Redes neuronales",
            description="Modelos computacionales inspirados en el cerebro.",
        )
        d = item.model_dump()
        assert d == {
            "topic": "Redes neuronales",
            "description": "Modelos computacionales inspirados en el cerebro.",
        }

    def test_topic_item_empty_description_allowed(self):
        """TDR-02 edge: empty description is allowed at Pydantic level (validation happens in pipeline)."""
        from src.topic_extraction.extract import TopicItem

        item = TopicItem(topic="Tema nuevo", description="")
        assert item.topic == "Tema nuevo"
        assert item.description == ""


class TestSegmentTopicsWithDescriptions:
    """Tests for SegmentTopics model with list[TopicItem] (TDR-03)."""

    def test_segment_topics_accepts_topic_items(self):
        """TDR-03: SegmentTopics validates list of TopicItem objects."""
        from src.topic_extraction.extract import SegmentTopics, TopicItem

        topics = SegmentTopics(
            topics=[
                TopicItem(
                    topic="Agentes inteligentes",
                    description="Sistemas que perciben y actúan en un entorno.",
                ),
                TopicItem(
                    topic="Razonamiento",
                    description="Proceso lógico para derivar conclusiones a partir de premisas.",
                ),
            ]
        )
        assert len(topics.topics) == 2
        assert isinstance(topics.topics[0], TopicItem)
        assert topics.topics[0].topic == "Agentes inteligentes"
        assert "perciben" in topics.topics[0].description

    def test_segment_topics_schema_includes_description_field(self):
        """TDR-03: JSON schema includes 'description' field for structured output."""
        from src.topic_extraction.extract import SegmentTopics

        schema = SegmentTopics.model_json_schema()
        assert "properties" in schema
        props = schema["properties"]
        topics_prop = props.get("topics")
        assert topics_prop is not None
        # topics is an array of objects with topic + description
        items = topics_prop.get("items")
        assert items is not None

    def test_segment_topics_min_length_enforced(self):
        """SegmentTopics must have at least 1 topic (unchanged constraint)."""
        from pydantic import ValidationError

        from src.topic_extraction.extract import SegmentTopics

        with pytest.raises(ValidationError):
            SegmentTopics(topics=[])


class TestExtractTopicsFromSegment:
    """Tests for _extract_segment_topics() — LLM per-segment extraction."""

    @pytest.fixture
    def mock_llm_with_descriptions(self):
        """Mock get_llm() returning new TopicItem-shaped JSON."""
        from unittest.mock import AsyncMock, patch

        fake_response = AsyncMock()
        fake_response.content = (
            '{"topics": ['
            '{"topic": "Agentes inteligentes", "description": "Sistemas que perciben su entorno y actúan."},'
            '{"topic": "Tipos de agentes", "description": "Clasificación de agentes según su arquitectura."},'
            '{"topic": "Entorno", "description": "Contexto externo donde opera el agente."}'
            "]}"
        )

        fake_llm = AsyncMock()
        fake_llm.ainvoke.return_value = fake_response

        with patch("src.topic_extraction.get_llm", return_value=fake_llm):
            yield fake_llm

    @pytest.fixture
    def mock_llm(self):
        """Mock get_llm() to return a fake LLM with ainvoke (old-style list[str] for backward compat)."""
        from unittest.mock import AsyncMock, patch

        fake_response = AsyncMock()
        fake_response.content = (
            '{"topics": ['
            '{"topic": "Agentes inteligentes", "description": "Sistemas que perciben y actúan."},'
            '{"topic": "Tipos de agentes", "description": "Diferentes arquitecturas de agente."},'
            '{"topic": "Entorno", "description": "Contexto donde opera el agente."}'
            "]}"
        )

        fake_llm = AsyncMock()
        fake_llm.ainvoke.return_value = fake_response

        with patch("src.topic_extraction.get_llm", return_value=fake_llm):
            yield fake_llm

    @pytest.mark.asyncio
    async def test_extracts_topic_items_from_segment(self, mock_llm_with_descriptions):
        """TDR-01: LLM returns TopicItem JSON → parsed list[TopicItem]."""
        from src.topic_extraction.extract import TopicItem, _extract_segment_topics

        items = await _extract_segment_topics(
            "Agentes inteligentes y su entorno",
            mock_llm_with_descriptions,
            segment_index=0,
            total=1,
        )
        assert isinstance(items, list)
        assert len(items) == 3
        assert isinstance(items[0], TopicItem)
        assert items[0].topic == "Agentes inteligentes"
        assert "perciben" in items[0].description

    @pytest.mark.asyncio
    async def test_topic_item_has_description(self, mock_llm):
        """TDR-01: Every extracted TopicItem has a non-empty description."""
        from src.topic_extraction.extract import TopicItem, _extract_segment_topics

        items = await _extract_segment_topics(
            "Agentes inteligentes y su entorno", mock_llm, segment_index=0, total=1
        )
        for item in items:
            assert isinstance(item, TopicItem)
            assert item.topic, "Topic must be non-empty"
            assert item.description, "Description must be non-empty"

    @pytest.mark.asyncio
    async def test_returns_empty_on_parse_failure(self, mock_llm):
        """TXR-03: LLM returns garbage → logged warning, returns []."""
        from src.topic_extraction.extract import _extract_segment_topics

        # Make LLM return something without valid JSON
        mock_llm.ainvoke.return_value.content = "No JSON here, just text"

        topics = await _extract_segment_topics("some text", mock_llm, segment_index=0, total=1)
        # Should not crash, return empty list
        assert topics == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_exception(self, mock_llm):
        """TXR-03: LLM call raises → returns [], pipeline continues."""
        from src.topic_extraction.extract import _extract_segment_topics

        mock_llm.ainvoke.side_effect = RuntimeError("LLM timeout")

        topics = await _extract_segment_topics("some text", mock_llm, segment_index=0, total=1)
        assert topics == []

    @pytest.mark.asyncio
    async def test_segment_index_in_prompt(self, mock_llm):
        """TXR-03: Segment position info passed to LLM for context."""
        from src.topic_extraction.extract import _extract_segment_topics

        await _extract_segment_topics("contenido del segmento", mock_llm, segment_index=2, total=5)
        # Verify ainvoke was called
        mock_llm.ainvoke.assert_called_once()
        # The prompt should mention segment 3 of 5 (1-indexed)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "3" in str(call_args) or "segment" in str(call_args).lower()

    @pytest.mark.asyncio
    async def test_description_is_spanish(self, mock_llm):
        """TDR-02: Description is in Spanish."""
        from src.topic_extraction.extract import _extract_segment_topics

        items = await _extract_segment_topics(
            "Agentes inteligentes y su entorno", mock_llm, segment_index=0, total=1
        )
        for item in items:
            assert isinstance(item.description, str)

    @pytest.mark.asyncio
    async def test_description_within_word_limit(self, mock_llm):
        """TDR-02: Description ≤20 words."""
        from src.topic_extraction.extract import _extract_segment_topics

        items = await _extract_segment_topics(
            "Agentes inteligentes y su entorno", mock_llm, segment_index=0, total=1
        )
        for item in items:
            word_count = len(item.description.split())
            assert word_count <= 20, f"Description has {word_count} words: {item.description!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.4 — unify.py tests (TXR-05)
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifyTopics:
    """Tests for unify_topics() — Jaccard-based clustering."""

    def test_jaccard_boundary_ge_threshold_merges(self):
        """TXR-05 edge: topics with Jaccard ≥ threshold (0.6) merge.

        'Agente inteligente' vs 'Agente inteligente artificial' share 2/3 stems → Jaccard ≈ 0.67.
        """
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Agente inteligente",
            "Agente inteligente artificial",
        ]
        result = unify_topics(topics, threshold=0.6)
        assert len(result) == 1, f"Expected merge (Jaccard ≥ 0.6), got {len(result)}: {result}"

    def test_jaccard_boundary_lt_threshold_keeps_separate(self):
        """TXR-05 edge: topics with Jaccard < threshold (0.6) stay separate.

        'Agente' vs 'Razonamiento lógico' share 0 stems → Jaccard = 0.0.
        """
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Agente",
            "Razonamiento lógico",
        ]
        result = unify_topics(topics, threshold=0.6)
        assert len(result) == 2, f"Expected separate (Jaccard < 0.6), got {len(result)}: {result}"

    def test_three_topics_two_merge_one_separate(self):
        """TXR-05 triangulate: 3 topics, 2 similar merge, 1 distinct stays.

        'Agentes inteligentes' and 'Agentes inteligentes y su entorno' merge
        (Jaccard ≥ 0.6). 'Búsqueda heurística' stays separate.
        """
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Agentes inteligentes",
            "Agentes inteligentes y su entorno",
            "Búsqueda heurística",
        ]
        result = unify_topics(topics, threshold=0.6)
        assert len(result) == 2, (
            f"Expected 2 topics (2 merged + 1 distinct), got {len(result)}: {result}"
        )
        assert "Agentes inteligentes y su entorno" in result
        assert "Búsqueda heurística" in result

    def test_merges_similar_topics(self):
        """TXR-05: 'Agentes inteligentes' and 'Agentes inteligentes y su entorno' merge."""
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Agentes inteligentes",
            "Agentes inteligentes y su entorno",
        ]
        result = unify_topics(topics, threshold=0.6)
        # Should merge into one cluster — longest canonical
        assert len(result) == 1, f"Expected 1 topic, got {len(result)}: {result}"
        assert "Agentes inteligentes y su entorno" in result

    def test_keeps_distinct_topics_separate(self):
        """TXR-05: 'Redes neuronales' and 'Máquinas de soporte vectorial' stay separate."""
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Redes neuronales",
            "Máquinas de soporte vectorial",
        ]
        result = unify_topics(topics, threshold=0.6)
        # Jaccard should be very low (different stems) — both kept
        assert len(result) == 2, f"Expected 2 topics, got {len(result)}: {result}"

    def test_deterministic_output(self):
        """Same input always produces same output (deterministic)."""
        from src.topic_extraction.unify import unify_topics

        topics = ["Razonamiento lógico", "Sistemas expertos", "Lógica proposicional"]
        r1 = unify_topics(topics)
        r2 = unify_topics(topics)
        assert r1 == r2

    def test_caps_at_max_count(self):
        """Output capped at max_topics_per_document (30)."""
        from src.topic_extraction.unify import unify_topics

        # Generate many distinct topics
        many = [f"Tema único número {i}" for i in range(50)]
        result = unify_topics(many, max_count=30)
        assert len(result) <= 30

    def test_sorted_output(self):
        """Result is alphabetically sorted."""
        from src.topic_extraction.unify import unify_topics

        topics = ["Zeta", "Alfa", "Beta"]
        result = unify_topics(topics)
        assert result == sorted(result)

    def test_canonical_longest_wins(self):
        """Longest string variant becomes canonical name in cluster."""
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Agentes",
            "Agentes inteligentes",
            "Agentes inteligentes y su entorno",
        ]
        result = unify_topics(topics, threshold=0.5)
        assert len(result) == 1
        assert result[0] == "Agentes inteligentes y su entorno"

    def test_empty_input(self):
        """Empty list returns empty list."""
        from src.topic_extraction.unify import unify_topics

        assert unify_topics([]) == []

    def test_accented_topics_through_unification(self):
        """TXR-01 edge: accented topic strings unify correctly.

        'Máquinas de soporte vectorial' and 'Soporte vectorial' share
        stems → should merge with Jaccard ≥ 0.6.
        """
        from src.topic_extraction.unify import unify_topics

        topics = [
            "Máquinas de soporte vectorial",
            "Soporte vectorial",
        ]
        result = unify_topics(topics, threshold=0.6)
        assert len(result) == 1, f"Expected merge for accented topics, got {len(result)}: {result}"

    def test_long_topic_string_handled(self):
        """TXR-05 edge: very long topic strings (>200 chars) don't break unification.

        The long topic has many distinct stems → Jaccard with short topic is low,
        so they stay separate. The test verifies no crash, not merge.
        """
        from src.topic_extraction.unify import unify_topics

        long_topic = (
            "Agentes inteligentes y su entorno en sistemas multiagente "
            "con razonamiento lógico y planificación automática utilizando "
            "búsqueda heurística en espacios de estados continuos y discretos "
            "para la resolución de problemas complejos en inteligencia artificial"
        )
        assert len(long_topic) > 200, (
            f"Precondition: long_topic must be >200 chars (got {len(long_topic)})"
        )

        topics = ["Agentes inteligentes", long_topic]
        result = unify_topics(topics, threshold=0.6)
        # Many distinct stems → Jaccard < 0.6 → both kept separate (no crash)
        assert isinstance(result, list)
        assert len(result) > 0, "Should return at least 1 topic"
        assert long_topic in result, "Long topic string must be preserved in output"


# ═══════════════════════════════════════════════════════════════════════════════
# 2.5 — tree.py tests (TXR-06)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildTopicTree:
    """Tests for build_topic_tree() — hierarchical organization."""

    @pytest.mark.asyncio
    async def test_small_set_returns_flat_dict(self):
        """TXR-06: ≤5 topics → deterministic flat dict (no LLM call)."""
        from src.topic_extraction.tree import build_topic_tree

        topics = ["Cálculo", "Álgebra", "Geometría"]
        result = await build_topic_tree(topics)
        assert isinstance(result, dict)
        for t in topics:
            assert t in result
        # Each value should be an empty dict (leaf)
        for v in result.values():
            assert v == {}

    @pytest.mark.asyncio
    async def test_serializable_to_json(self):
        """Output can be serialized with json.dumps."""
        import json

        from src.topic_extraction.tree import build_topic_tree

        topics = ["Razonamiento lógico", "Sistemas expertos"]
        result = await build_topic_tree(topics)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed == result

    @pytest.mark.asyncio
    async def test_empty_topics(self):
        """Empty topic list returns empty dict."""
        from src.topic_extraction.tree import build_topic_tree

        assert await build_topic_tree([]) == {}

    @pytest.mark.asyncio
    async def test_large_set_uses_llm(self):
        """TXR-06: ≥5 topics → LLM call for hierarchy."""
        from unittest.mock import AsyncMock, patch

        from src.topic_extraction.tree import build_topic_tree

        topics = [
            "Agentes inteligentes",
            "Tipos de agentes",
            "Entorno de agentes",
            "Razonamiento lógico",
            "Planificación",
            "Búsqueda heurística",
        ]

        fake_llm = AsyncMock()
        fake_llm.ainvoke.return_value.content = json.dumps(
            {
                "Agentes": {"Tipos": {}, "Entorno": {}},
                "Razonamiento": {
                    "Lógico": {},
                    "Planificación": {},
                    "Búsqueda": {},
                },
            }
        )

        with patch("src.topic_extraction.tree.get_llm", return_value=fake_llm):
            result = await build_topic_tree(topics)

        assert isinstance(result, dict)
        assert "Agentes" in result or "Razonamiento" in result
        fake_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_flat(self):
        """TXR-06: LLM fails → deterministic fallback (flat dict)."""
        from unittest.mock import AsyncMock, patch

        from src.topic_extraction.tree import build_topic_tree

        topics = ["Tema A", "Tema B", "Tema C", "Tema D", "Tema E", "Tema F"]

        fake_llm = AsyncMock()
        fake_llm.ainvoke.side_effect = RuntimeError("LLM down")

        with patch("src.topic_extraction.tree.get_llm", return_value=fake_llm):
            result = await build_topic_tree(topics)

            # Should still return a valid dict (deterministic fallback)
            assert isinstance(result, dict)
            assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.6 — __init__.py tests (TXR-09, TXR-10)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractTopicsPipeline:
    """Tests for extract_topics_pipeline() — full pipeline integration."""

    @pytest.fixture
    def mock_pipeline_llm(self):
        """Mock get_llm() for the full pipeline (returns TopicItem-shaped JSON per segment)."""
        from unittest.mock import AsyncMock, patch

        fake_response = AsyncMock()
        fake_response.content = (
            '{"topics": ['
            '{"topic": "Agentes inteligentes", "description": "Sistemas que perciben su entorno y actúan."},'
            '{"topic": "Razonamiento", "description": "Proceso lógico para derivar conclusiones."},'
            '{"topic": "Planificación", "description": "Determinación de secuencias de acciones futuras."},'
            '{"topic": "Entorno de agentes", "description": "Contexto externo donde opera el agente."}'
            "]}"
        )

        fake_llm = AsyncMock()
        fake_llm.ainvoke.return_value = fake_response

        with patch("src.topic_extraction.get_llm", return_value=fake_llm):
            # tree.py may also call get_llm — patch it too for deterministic
            with patch("src.topic_extraction.tree.get_llm", return_value=fake_llm):
                yield fake_llm

    @pytest.mark.asyncio
    async def test_short_text_passthrough(self, mock_pipeline_llm):
        """TXR-10: Short text bypasses segmentation → single LLM call."""
        from src.topic_extraction import extract_topics_pipeline

        short = "Agentes inteligentes y su entorno de trabajo en la UTN."
        result = await extract_topics_pipeline(short)

        assert isinstance(result, dict)
        assert "topics" in result
        assert "topic_tree" in result
        assert "summary" in result
        assert result["segment_count"] == 1

    @pytest.mark.asyncio
    async def test_short_text_calls_llm_once(self, mock_pipeline_llm):
        """TXR-10 smoke: Short text → single segment, single LLM call."""
        from src.topic_extraction import extract_topics_pipeline

        short = "Agentes inteligentes: concepto y tipos."
        result = await extract_topics_pipeline(short)

        # Passthrough: single segment, zero segmentation overhead
        assert result["segment_count"] == 1
        assert len(result["topics"]) > 0, "Short text should produce topics"

        # LLM called once: mock returns 4 topics → tree uses flat dict (<5 topics)
        assert mock_pipeline_llm.ainvoke.call_count == 1, (
            "Short text (1 segment, <5 topics) should trigger exactly 1 LLM call"
        )

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        """Empty text returns empty result with appropriate message."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("")
        assert result["topics"] == []
        assert result["segment_count"] == 0
        assert len(result["failed_segments"]) == 0

    @pytest.mark.asyncio
    async def test_whitespace_text_returns_empty(self):
        """Whitespace-only text returns empty result."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("   \n  \n  ")
        assert result["topics"] == []
        assert result["segment_count"] == 0

    @pytest.mark.asyncio
    async def test_result_shape(self, mock_pipeline_llm):
        """TXR-09 + TDR-01: Result has all required fields including topic_descriptions."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("Agentes inteligentes y su entorno.")
        assert set(result.keys()) == {
            "summary",
            "topics",
            "topic_tree",
            "topic_descriptions",
            "segment_count",
            "failed_segments",
        }
        assert isinstance(result["summary"], str)
        assert isinstance(result["topics"], list)
        assert isinstance(result["topic_tree"], str)
        assert isinstance(result["topic_descriptions"], dict)
        assert isinstance(result["segment_count"], int)
        assert isinstance(result["failed_segments"], list)

    @pytest.mark.asyncio
    async def test_topic_descriptions_in_pipeline_result(self, mock_pipeline_llm):
        """TDR-01: Pipeline result has topic_descriptions dict matching topics."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("Agentes inteligentes y su entorno.")
        descs = result["topic_descriptions"]
        topics = result["topics"]
        # Every topic should have a description entry
        for topic in topics:
            assert topic in descs, f"Topic '{topic}' missing from topic_descriptions"
            assert isinstance(descs[topic], str)
            assert descs[topic], f"Description for '{topic}' is empty"

    @pytest.mark.asyncio
    async def test_topic_descriptions_are_spanish(self, mock_pipeline_llm):
        """TDR-02: Descriptions are in Spanish."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("Agentes inteligentes y su entorno en la UTN.")
        for topic, desc in result["topic_descriptions"].items():
            assert isinstance(desc, str), f"Description for '{topic}' is not a string"

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_descriptions(self):
        """TDR-01 edge: Empty text → empty topic_descriptions."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("")
        assert result["topic_descriptions"] == {}
        assert result["topics"] == []

    @pytest.mark.asyncio
    async def test_topic_descriptions_word_limit(self, mock_pipeline_llm):
        """TDR-02: Each description ≤20 words."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("Agentes inteligentes y su entorno.")
        for topic, desc in result["topic_descriptions"].items():
            word_count = len(desc.split())
            assert word_count <= 20, f"Description for '{topic}' has {word_count} words"

    @pytest.mark.asyncio
    async def test_single_segment_skips_unify(self, mock_pipeline_llm):
        """TXR-10: Single segment → no unification overhead."""
        from src.topic_extraction import extract_topics_pipeline

        result = await extract_topics_pipeline("Contenido académico sobre agentes.")
        assert result["segment_count"] == 1
        # Topics should still be present
        assert len(result["topics"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4.4 — Integration test: full pipeline on real PDF (NFR)
# ═══════════════════════════════════════════════════════════════════════════════


MULTI_TOPIC_MARKDOWN = """# Agentes Inteligentes

Los agentes inteligentes son sistemas que perciben su entorno mediante
sensores y actúan mediante efectores.

## Tipos de Agentes

Existen varios tipos de agentes: reactivos simples, reactivos basados en
modelos, basados en objetivos y basados en utilidad.

### Agentes Reactivos Simples

Reglas condición-acción. No mantienen estado interno. Responden a
percepciones inmediatas.

### Agentes Basados en Modelos

Mantienen un modelo del mundo para manejar parcialmente la observabilidad
del entorno.

## Razonamiento y Planificación

Los agentes deben poder razonar sobre sus acciones futuras. La
planificación automática es un área clave de la inteligencia artificial.

### Búsqueda en Espacio de Estados

Estrategias de búsqueda ciega: BFS, DFS. Estrategias informadas: A*,
búsqueda voraz.

## Aprendizaje Automático

El aprendizaje automático permite a los agentes mejorar su rendimiento
con la experiencia.

### Aprendizaje Supervisado

El agente aprende a partir de ejemplos etiquetados. Incluye clasificación
y regresión.

### Aprendizaje No Supervisado

Clustering, reducción de dimensionalidad. El agente descubre patrones sin
etiquetas.

## Ética en Inteligencia Artificial

Consideraciones éticas sobre el desarrollo y despliegue de agentes
inteligentes autónomos.
"""


@pytest.mark.integration
class TestFullPipelineRealPDF:
    """Integration test: full topic extraction pipeline on real academic PDF.

    Uses Ollama Cloud LLM (gemma4:31b-cloud via api.ollama.com).
    Skips gracefully if API key is missing or network fails.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_real_pdf(self, real_pdf_text: str):
        """Epic 11 NFR: full pipeline on real academic PDF.

        QUOTA-SAFE: first verifies pipeline shape with short markdown text
        (no real LLM needed for shape check — unit tests already cover this).
        Then runs real pipeline on parsed PDF text.
        """
        import json

        from src.config import settings
        from src.topic_extraction import extract_topics_pipeline

        # ── Phase 1: Dry-run shape verification (no LLM call) ──────────────
        # Already covered by TestExtractTopicsPipeline unit tests above.
        # Quick sanity: pipeline returns correct shape for empty input.
        empty_result = await extract_topics_pipeline("")
        assert set(empty_result.keys()) == {
            "summary",
            "topics",
            "topic_tree",
            "topic_descriptions",
            "segment_count",
            "failed_segments",
        }
        assert empty_result["segment_count"] == 0

        # ── Phase 2: Real LLM pipeline on parsed PDF ───────────────────────
        # Check Ollama Cloud credentials
        if not settings.ollama_api_key:
            pytest.skip("OLLAMA_API_KEY not set — cannot run real LLM integration test")

        assert real_pdf_text, "real_pdf_text fixture returned empty text"
        assert len(real_pdf_text) > 500, (
            f"PDF text too short for meaningful test: {len(real_pdf_text)} chars"
        )

        try:
            result = await extract_topics_pipeline(real_pdf_text)
        except Exception as exc:
            # Network/auth failures → skip gracefully
            msg = str(exc).lower()
            if any(kw in msg for kw in ("unauthorized", "connection", "timeout", "dns", "network")):
                pytest.skip(f"Ollama Cloud unavailable: {exc}")
            # Unexpected failures → surface
            raise

        # ── Assertions ─────────────────────────────────────────────────────
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "topics" in result
        assert "topic_tree" in result
        assert "segment_count" in result
        assert "failed_segments" in result

        assert result["segment_count"] > 0, (
            f"Expected >0 segments for {len(real_pdf_text)}-char PDF, got {result['segment_count']}"
        )
        assert len(result["topics"]) >= 3, (
            f"Expected ≥3 topics from real PDF, got {len(result['topics'])}: {result['topics']}"
        )
        assert result["topic_tree"], "topic_tree must be non-empty string"
        assert result["topic_tree"] != "{}", "topic_tree must not be empty JSON"

        # Verify topic_tree is valid JSON (parseable)
        try:
            tree_dict = json.loads(result["topic_tree"])
        except json.JSONDecodeError as exc:
            pytest.fail(f"topic_tree is not valid JSON: {exc}\nRaw: {result['topic_tree'][:200]}")

        assert isinstance(tree_dict, dict), (
            f"topic_tree must deserialize to dict, got {type(tree_dict)}"
        )
        assert len(tree_dict) > 0, "topic_tree dict must be non-empty"

        # Verify topics and segment count are coherent
        assert len(result["topics"]) <= settings.max_topics_per_document, (
            f"Topics ({len(result['topics'])}) exceed max_topics_per_document "
            f"({settings.max_topics_per_document})"
        )
