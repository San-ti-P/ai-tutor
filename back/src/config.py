"""Application settings loaded from environment variables."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    opencode_go_api_key: str = ""
    ollama_api_key: str = ""

    # ── Langfuse ──────────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_environment: str = ""  # Let SDK read LANGFUSE_TRACING_ENVIRONMENT
    langfuse_release: str = ""  # LANGFUSE_RELEASE

    # ── OCR (Mathpix) ─────────────────────────────────────────────────────
    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

    # ── Storage ───────────────────────────────────────────────────────────
    chroma_persist_directory: str = "./data/chroma"
    sqlite_db_path: str = "./data/tutor.db"

    # ── Server ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Agent / Guardrails ────────────────────────────────────────────────
    max_iterations_per_task: int = 15
    ocr_confidence_threshold: float = 0.85
    classification_confidence_threshold: float = 0.75
    anti_hallucination_threshold: float = 0.40
    judge_sample_rate: float = 0.10
    judge_disagreement_threshold: float = 2.0

    # ── RAG / Embedding ───────────────────────────────────────────────────
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5

    # ── Topic Extraction ───────────────────────────────────────────────────
    topic_segment_size: int = 6000
    """Max chars per segment sent to LLM for topic extraction."""

    topic_similarity_threshold: float = 0.6
    """Jaccard similarity threshold for merging near-duplicate topics."""

    max_topics_per_document: int = 30
    """Maximum number of unified topics returned per document."""

    topic_min_section_chars: int = 200
    """Merge adjacent markdown sections below this character count."""

    # ── LLM Provider ──────────────────────────────────────────────────────
    # "ollama" | "groq" | "opencode-go" | "opencode-go-anthropic" | "openai"
    llm_provider: str = "opencode-go"

    # -- Ollama (local or cloud) --
    ollama_model_name: str = "gemma4:e4b-it-q8_0"  # local 4B model, fast for dev
    ollama_base_url: str = "http://localhost:11434"
    # Set ollama_api_key for Ollama Cloud (https://ollama.com → Settings → Keys)

    # -- Groq --
    groq_model_name: str = "llama-3.1-8b-instant"  # cheapest, tool-use capable

    # -- OpenCode Go — OpenAI-compatible ($10/mo subscription) --
    # Models: deepseek-v4-pro, deepseek-v4-flash, kimi-k2.7-code, kimi-k2.6,
    #         glm-5.2, mimo-v2.5, mimo-v2.5-pro
    opencode_go_model_name: str = "deepseek-v4-pro"
    opencode_go_base_url: str = "https://opencode.ai/zen/go/v1"

    # -- OpenCode Go — Anthropic-compatible ($10/mo subscription) --
    # Models: minimax-m3, minimax-m2.7, qwen3.7-max, qwen3.7-plus, qwen3.6-plus
    opencode_go_anthropic_model_name: str = "minimax-m3"
    opencode_go_anthropic_base_url: str = "https://opencode.ai/zen/go"
    # ChatAnthropic appends /v1/messages — base_url must NOT include /v1

    # -- OpenAI (direct) --
    openai_model_name: str = "gpt-4o"
    openai_base_url: str = ""  # empty = default OpenAI endpoint

    # ──────────────────────────────────────────────────────────────────────

    def _build_ollama_kwargs(self) -> tuple[type, dict[str, Any]]:
        """Return (ChatOllama, kwargs) for Ollama (local or cloud)."""
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": self.ollama_model_name,
            "base_url": self.ollama_base_url,
            "temperature": 0,
        }
        if self.ollama_api_key:
            kwargs["client_kwargs"] = {
                "headers": {"Authorization": f"Bearer {self.ollama_api_key}"}
            }
        return ChatOllama, kwargs

    @staticmethod
    def _build_groq_kwargs(groq_model_name: str) -> tuple[type, dict[str, Any]]:
        """Return (ChatGroq, kwargs) for Groq."""
        from langchain_groq import ChatGroq

        return ChatGroq, {"model": groq_model_name, "temperature": 0}

    def _build_opencode_go_kwargs(self) -> tuple[type, dict[str, Any]]:
        """Return (ChatOpenAI, kwargs) for OpenCode Go (OpenAI-compatible)."""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI, {
            "model": self.opencode_go_model_name,
            "base_url": self.opencode_go_base_url,
            "api_key": self.opencode_go_api_key,
            "temperature": 0,
        }

    def _build_opencode_go_anthropic_kwargs(self) -> tuple[type, dict[str, Any]]:
        """Return (ChatAnthropic, kwargs) for OpenCode Go Anthropic-compatible models.

        Used for MiniMax M3, Qwen3.7, and other models that use the Anthropic
        Messages API format (``/v1/messages`` endpoint).
        """
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic, {
            "model": self.opencode_go_anthropic_model_name,
            "base_url": self.opencode_go_anthropic_base_url,
            "api_key": self.opencode_go_api_key,
            "temperature": 0,
        }

    def _build_openai_kwargs(self) -> tuple[type, dict[str, Any]]:
        """Return (ChatOpenAI, kwargs) for OpenAI (direct or custom base URL)."""
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": self.openai_model_name,
            "temperature": 0,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        if self.openai_api_key:
            kwargs["api_key"] = self.openai_api_key
        return ChatOpenAI, kwargs

    @property
    def llm_kwargs(self) -> tuple[type, dict[str, Any]]:
        """Return (model_class, kwargs) for the configured LLM provider.

        Supported providers:
        - ``ollama``: local Ollama or Ollama Cloud (set ollama_api_key + base_url)
        - ``groq``: Groq Cloud (fast, cheap inference)
        - ``opencode-go``: OpenCode Go — OpenAI-compatible endpoint
        - ``opencode-go-anthropic``: OpenCode Go — Anthropic-compatible endpoint
        - ``openai``: OpenAI direct or any OpenAI-compatible API
        """
        provider = self.llm_provider

        if provider == "ollama":
            return self._build_ollama_kwargs()
        if provider == "groq":
            return self._build_groq_kwargs(self.groq_model_name)
        if provider == "opencode-go":
            return self._build_opencode_go_kwargs()
        if provider == "opencode-go-anthropic":
            return self._build_opencode_go_anthropic_kwargs()
        if provider == "openai":
            return self._build_openai_kwargs()

        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            f"Valid options: ollama, groq, opencode-go, opencode-go-anthropic, openai"
        )

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
