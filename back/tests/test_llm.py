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
        original_temp = settings.ollama_temperature
        settings.llm_provider = "ollama"
        settings.ollama_temperature = 0

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                expected_kwargs: dict = {
                    "model": settings.ollama_model_name,
                    "base_url": settings.ollama_base_url,
                    "temperature": 0,
                }
                if settings.ollama_api_key:
                    expected_kwargs["client_kwargs"] = {
                        "headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}
                    }
                mock_ollama.assert_called_once_with(**expected_kwargs)
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original
            settings.ollama_temperature = original_temp

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
        original_temp = settings.ollama_temperature
        settings.llm_provider = "ollama"
        settings.ollama_api_key = "test-cloud-key"
        settings.ollama_temperature = 0

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                mock_ollama.assert_called_once_with(
                    model=settings.ollama_model_name,
                    base_url=settings.ollama_base_url,
                    temperature=0,
                    client_kwargs={"headers": {"Authorization": "Bearer test-cloud-key"}},
                )
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original_provider
            settings.ollama_api_key = original_api_key
            settings.ollama_temperature = original_temp

    def test_get_llm_ollama_cloud_base_url(self):
        """Uses custom base_url for Ollama Cloud endpoint."""
        from src.llm import get_llm

        original_provider = settings.llm_provider
        original_base_url = settings.ollama_base_url
        original_api_key = settings.ollama_api_key
        original_temp = settings.ollama_temperature
        settings.llm_provider = "ollama"
        settings.ollama_base_url = "https://api.ollama.com"
        settings.ollama_api_key = "cloud-key-123"
        settings.ollama_temperature = 0

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_llm()
                mock_ollama.assert_called_once_with(
                    model=settings.ollama_model_name,
                    base_url="https://api.ollama.com",
                    temperature=0,
                    client_kwargs={"headers": {"Authorization": "Bearer cloud-key-123"}},
                )
                assert result is mock_ollama.return_value
        finally:
            settings.llm_provider = original_provider
            settings.ollama_base_url = original_base_url
            settings.ollama_api_key = original_api_key
            settings.ollama_temperature = original_temp


class TestGetLLMOpenCodeGoAnthropic:
    """Tests for get_llm() with opencode-go-anthropic provider."""

    def test_get_llm_opencode_go_anthropic(self):
        """Returns ChatAnthropic with Anthropic-compatible endpoint."""
        from src.llm import get_llm

        original = settings.llm_provider
        settings.llm_provider = "opencode-go-anthropic"

        try:
            with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
                result = get_llm()
                mock_anthropic.assert_called_once_with(
                    model=settings.opencode_go_anthropic_model_name,
                    base_url=settings.opencode_go_anthropic_base_url,
                    api_key=settings.opencode_go_api_key,
                    temperature=0,
                )
                assert result is mock_anthropic.return_value
        finally:
            settings.llm_provider = original


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

    def test_get_structured_llm_ollama_json_mode(self):
        """For Ollama: creates ChatOllama with format='json' in a chain, not with_structured_output."""
        from src.llm import get_structured_llm

        original = settings.llm_provider
        settings.llm_provider = "ollama"

        try:
            with patch("langchain_ollama.ChatOllama") as mock_ollama:
                result = get_structured_llm(_FakeSchema)
                # Chain returned, not the raw LLM
                mock_ollama.return_value.with_structured_output.assert_not_called()
                mock_ollama.assert_called_once()
                _, kwargs = mock_ollama.call_args
                assert kwargs["format"] == "json"
                assert kwargs["temperature"] == 0
                assert hasattr(result, "invoke")
                assert not hasattr(result, "with_structured_output")
        finally:
            settings.llm_provider = original

    def test_get_structured_llm_with_different_schema(self):
        """Works with any Pydantic schema — uses schema-in-prompt chain for all providers."""
        from langchain_core.runnables import Runnable

        class _OtherSchema(BaseModel):
            score: float
            text: str

        from src.llm import get_structured_llm

        original = settings.llm_provider
        settings.llm_provider = "groq"

        try:
            with patch("langchain_groq.ChatGroq") as mock_groq:
                mock_groq.return_value.invoke = MagicMock()
                mock_groq.return_value.invoke.return_value.content = '{"score": 8.5, "text": "ok"}'

                result = get_structured_llm(_OtherSchema)
                assert isinstance(result, Runnable)
                # NEVER uses with_structured_output
                mock_groq.return_value.with_structured_output.assert_not_called()
        finally:
            settings.llm_provider = original


# ==============================================================================
# Phase 5.4: Markdown fence stripping in _ollama_json_mode_chain._parse (Epic 13, LT-1)
# ==============================================================================


class TestMarkdownFenceStripping:
    """LT-1: _parse strips ```json fences before JSON scanning."""

    def _get_parse_fn(self):
        """Build a minimal _parse function matching the Ollama chain pattern."""
        import json

        from pydantic import BaseModel

        from src.llm import _CLEAN_FENCES

        class _TestSchema(BaseModel):
            score: float
            feedback: str

        def _parse(text: str) -> BaseModel:
            text = _CLEAN_FENCES.sub("", text).strip()
            for end in range(len(text) - 1, -1, -1):
                if text[end] != "}":
                    continue
                depth = 1
                start = end - 1
                while start >= 0 and depth > 0:
                    if text[start] == "}":
                        depth += 1
                    elif text[start] == "{":
                        depth -= 1
                    start -= 1
                if depth == 0:
                    candidate = text[start + 1 : end + 1]
                    try:
                        return _TestSchema.model_validate(json.loads(candidate))
                    except Exception:
                        continue
            raise ValueError(f"No valid JSON in: {text[:200]!r}")

        return _parse

    def test_strips_markdown_fences_json_tag(self):
        """LLM output wrapped in ```json ... ``` is correctly parsed."""
        parse = self._get_parse_fn()
        raw = '```json\n{"score": 8.5, "feedback": "Buen trabajo"}\n```'
        result = parse(raw)
        assert result.score == 8.5
        assert result.feedback == "Buen trabajo"

    def test_strips_markdown_fences_no_lang_tag(self):
        """LLM output wrapped in ``` ... ``` (no json tag) is correctly parsed."""
        parse = self._get_parse_fn()
        raw = '```\n{"score": 7.0, "feedback": "Correcto"}\n```'
        result = parse(raw)
        assert result.score == 7.0

    def test_strips_fences_with_extra_whitespace(self):
        """Fences with trailing whitespace are handled."""
        parse = self._get_parse_fn()
        raw = '```json  \n{"score": 9.0, "feedback": "Excelente"}  \n```  '
        result = parse(raw)
        assert result.score == 9.0

    def test_no_fences_still_parses(self):
        """Plain JSON without fences still parses correctly."""
        parse = self._get_parse_fn()
        raw = '{"score": 6.0, "feedback": "Aceptable"}'
        result = parse(raw)
        assert result.score == 6.0

    def test_nested_json_with_fences(self):
        """Fences around nested JSON (containing braces inside values) works."""
        parse = self._get_parse_fn()
        raw = (
            '```json\n'
            '{"score": 7.5, "feedback": "La derivada f\'(x) = lim(h->0) [f(x+h)-f(x)]/h es clave"}\n'
            '```'
        )
        result = parse(raw)
        assert result.score == 7.5
        assert "derivada" in result.feedback
