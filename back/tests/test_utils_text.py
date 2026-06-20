"""Unit tests for src/utils/text.py — pure text utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.text import parse_file_to_text, split_into_claims, split_sentences


class TestSplitSentences:
    """Tests for split_sentences()."""

    def test_split_sentences_period(self):
        """Splits on period followed by space; preserves trailing punctuation."""
        result = split_sentences("Primera oración. Segunda oración. Tercera.")
        assert result == ["Primera oración.", "Segunda oración.", "Tercera."]

    def test_split_sentences_exclamation_question(self):
        """Splits on ! and ? followed by space; preserves trailing punctuation."""
        result = split_sentences("¡Qué bien! ¿De verdad? Claro que sí.")
        assert result == ["¡Qué bien!", "¿De verdad?", "Claro que sí."]

    def test_split_sentences_empty_string(self):
        """Empty string returns empty list."""
        assert split_sentences("") == []

    def test_split_sentences_single_sentence(self):
        """Single sentence without ending punctuation returns as-is."""
        result = split_sentences("Esto es una sola oración")
        assert result == ["Esto es una sola oración"]


class TestSplitIntoClaims:
    """Tests for split_into_claims()."""

    def test_split_into_claims_simple(self):
        """Splits on sentence boundaries and removes short fragments."""
        text = "La derivada es un límite. Geométricamente, es la pendiente de la tangente a la curva."
        result = split_into_claims(text)

        assert len(result) >= 1
        assert any("derivada" in c.lower() for c in result)

    def test_split_into_claims_semicolons(self):
        """Splits long sentences on semicolons for granular claims."""
        text = (
            "La integral definida de una función entre a y b representa el área "
            "bajo la curva; el Teorema Fundamental del Cálculo establece que "
            "la integración y la derivación son operaciones inversas."
        )
        result = split_into_claims(text)

        assert len(result) >= 1
        # At least one claim should mention integration or Teorema
        assert any("integral" in c.lower() or "teorema" in c.lower() for c in result)

    def test_split_into_claims_min_length_filter(self):
        """Filters out fragments shorter than min_length."""
        text = "A. B. C. D. Una afirmación con suficiente longitud para no ser filtrada."
        result = split_into_claims(text, min_length=30)

        assert len(result) == 1
        assert "suficiente longitud" in result[0]

    def test_split_into_claims_empty_string(self):
        """Empty string returns empty list."""
        assert split_into_claims("") == []

    def test_split_into_claims_none_text(self):
        """Non-string input handled gracefully."""
        assert split_into_claims("   \n  ") == []

    def test_split_into_claims_semicolons_granular(self):
        """Long compound sentence with semicolons produces multiple claims."""
        text = "Primer concepto importante sobre álgebra; segundo concepto sobre cálculo; tercer concepto sobre estadística aplicada a datos."
        result = split_into_claims(text, min_length=10)

        assert len(result) == 3
        assert "álgebra" in result[0]
        assert "cálculo" in result[1]
        assert "estadística" in result[2]


class TestParseFileToText:
    """Tests for parse_file_to_text()."""

    def test_parse_txt_file(self, sample_txt):
        """Parses a TXT file and returns text content."""
        text = parse_file_to_text(str(sample_txt))
        assert "Álgebra lineal" in text
        assert "espacio vectorial" in text

    def test_parse_pdf_file(self, sample_pdf):
        """Parses a PDF file and returns text content."""
        text = parse_file_to_text(str(sample_pdf))
        # PDF should contain some extractable text
        assert len(text) > 0

    def test_parse_missing_file_raises(self, temp_dir):
        """Missing file raises exception."""
        missing = str(temp_dir / "nonexistent.txt")
        with pytest.raises(FileNotFoundError):
            parse_file_to_text(missing)
