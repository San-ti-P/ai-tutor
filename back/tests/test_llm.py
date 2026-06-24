"""Unit tests for src/llm.py — LLM factory functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
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
                    base_url=settings.ollama_base_url,
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


class TestGetLLMOpenCodeGo:
    """Tests for get_llm() with opencode-go provider."""

    def test_get_llm_opencode_go(self):
        """Returns ChatOpenAI with OpenCode Go base_url when provider == 'opencode-go'."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "opencode-go"

        try:
            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                result = get_llm()
                mock_openai.assert_called_once_with(
                    model=settings.opencode_go_model_name,
                    base_url=settings.opencode_go_base_url,
                    api_key=settings.opencode_go_api_key,
                    temperature=0,
                )
                assert result is mock_openai.return_value
        finally:
            settings.llm_provider = original


class TestGetLLMOpenAI:
    """Tests for get_llm() with openai provider."""

    def test_get_llm_openai_default(self):
        """Returns ChatOpenAI with default OpenAI endpoint (no base_url override)."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "openai"

        try:
            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                result = get_llm()
                mock_openai.assert_called_once_with(
                    model=settings.openai_model_name,
                    temperature=0,
                )
                assert result is mock_openai.return_value
        finally:
            settings.llm_provider = original

    def test_get_llm_openai_custom_base_url(self):
        """Returns ChatOpenAI with custom base_url when openai_base_url is set."""
        from src.llm import get_llm

        original_provider = settings.llm_provider
        original_base_url = settings.openai_base_url
        settings.llm_provider = "openai"
        settings.openai_base_url = "https://custom.api.example.com/v1"

        try:
            with patch("langchain_openai.ChatOpenAI") as mock_openai:
                result = get_llm()
                mock_openai.assert_called_once_with(
                    model=settings.openai_model_name,
                    temperature=0,
                    base_url="https://custom.api.example.com/v1",
                )
                assert result is mock_openai.return_value
        finally:
            settings.llm_provider = original_provider
            settings.openai_base_url = original_base_url


class TestGetLLMOllamaCloud:
    """Tests for get_llm() with Ollama (cloud mode with API key)."""

    def test_get_llm_ollama_with_api_key(self):
        """Passes Authorization header when ollama_api_key is set."""
        from src.llm import get_llm

        original_provider = settings.llm_provider
        original_api_key = settings.ollama_api_key
        settings.llm_provider = "ollama"
        settings.ollama_api_key = "test-cloud-key"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                mock_ollama.assert_called_once_with(
                    model=settings.ollama_model_name,
                    base_url=settings.ollama_base_url,
                    temperature=0,
                    client_kwargs={
                        "headers": {"Authorization": "Bearer test-cloud-key"}
                    },
                )
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original_provider
            settings.ollama_api_key = original_api_key

    def test_get_llm_ollama_cloud_base_url(self):
        """Uses custom base_url for Ollama Cloud endpoint."""
        from src.llm import get_llm

        original_provider = settings.llm_provider
        original_base_url = settings.ollama_base_url
        original_api_key = settings.ollama_api_key
        settings.llm_provider = "ollama"
        settings.ollama_base_url = "https://api.ollama.com"
        settings.ollama_api_key = "cloud-key-123"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                mock_ollama.assert_called_once_with(
                    model=settings.ollama_model_name,
                    base_url="https://api.ollama.com",
                    temperature=0,
                    client_kwargs={
                        "headers": {"Authorization": "Bearer cloud-key-123"}
                    },
                )
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original_provider
            settings.ollama_base_url = original_base_url
            settings.ollama_api_key = original_api_key


class TestGetLLMUnknownProvider:
    """Tests for get_llm() with unknown provider."""

    def test_get_llm_unknown_provider_raises(self):
        """Raises ValueError for unknown llm_provider value."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "nonexistent"

        try:
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm()
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
