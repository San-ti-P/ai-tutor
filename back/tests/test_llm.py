"""Unit tests for src/llm.py — LLM factory functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from src.config import settings


class _FakeSchema(BaseModel):
    """Schema used for testing get_structured_llm."""
    name: str = Field(...)
    value: int = Field(...)


class TestGetLLM:
    """Tests for get_llm() factory."""

    def test_get_llm_ollama(self):
        """Returns ChatOllama when settings.llm_provider == 'ollama'."""
        from src.llm import get_llm

        # Save original provider and restore after
        original = settings.llm_provider
        settings.llm_provider = "ollama"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                mock_ollama.assert_called_once_with(
                    model=settings.ollama_model_name,
                    temperature=0,
                )
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original

    def test_get_llm_groq(self):
        """Returns ChatGroq when settings.llm_provider == 'groq'."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "groq"

        try:
            with patch("langchain_groq.ChatGroq") as mock_groq:
                result = get_llm()
                mock_groq.assert_called_once_with(
                    model=settings.groq_model_name,
                    temperature=0,
                )
                assert result is mock_groq.return_value
        finally:
            settings.llm_provider = original

    def test_get_llm_returns_base_chat_model(self):
        """Returned object has invoke and with_structured_output methods."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "ollama"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                mock_ollama.return_value.invoke = MagicMock()
                mock_ollama.return_value.with_structured_output = MagicMock()

                llm = get_llm()
                assert hasattr(llm, "invoke")
                assert hasattr(llm, "with_structured_output")
        finally:
            settings.llm_provider = original


class TestGetStructuredLLM:
    """Tests for get_structured_llm() factory."""

    def test_get_structured_llm_calls_with_structured_output(self):
        """Returns a model with with_structured_output applied."""
        from src.llm import get_structured_llm

        original = settings.llm_provider
        settings.llm_provider = "ollama"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                mock_structured = mock_ollama.return_value.with_structured_output.return_value

                result = get_structured_llm(_FakeSchema)
                mock_ollama.return_value.with_structured_output.assert_called_once_with(_FakeSchema)
                assert result is mock_structured
        finally:
            settings.llm_provider = original

    def test_get_structured_llm_with_different_schema(self):
        """Works with any Pydantic schema."""

        class _OtherSchema(BaseModel):
            score: float
            text: str

        from src.llm import get_structured_llm

        original = settings.llm_provider
        settings.llm_provider = "groq"

        try:
            with patch("langchain_groq.ChatGroq") as mock_groq:
                mock_structured = mock_groq.return_value.with_structured_output.return_value

                result = get_structured_llm(_OtherSchema)
                mock_groq.return_value.with_structured_output.assert_called_once_with(_OtherSchema)
                assert result is mock_structured
        finally:
            settings.llm_provider = original
