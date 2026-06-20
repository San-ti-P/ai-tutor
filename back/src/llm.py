"""LLM factory — single source of truth for language model instantiation.

Every agent that needs a language model MUST import from here instead of
calling ``settings.llm_kwargs`` directly. This ensures provider/model
changes happen in exactly one place.

Functions:
- get_llm(): returns a configured BaseChatModel (Ollama or Groq)
- get_structured_llm(schema): returns a model with structured output
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.config import settings


def get_llm(
    callbacks: list[Any] | None = None,
) -> BaseChatModel:
    """Return a configured LLM instance for the current provider.

    Reads ``settings.llm_kwargs`` and instantiates the appropriate
    chat model class. Supports both Ollama and Groq providers.

    Args:
        callbacks: Optional list of LangChain callbacks (e.g. Langfuse
            CallbackHandler) injected into the LLM config.

    Returns:
        A configured BaseChatModel ready for .invoke() calls.
    """
    llm_cls, llm_kwargs = settings.llm_kwargs
    # Merge callbacks into kwargs if provided
    if callbacks:
        llm_kwargs = {**llm_kwargs, "callbacks": callbacks}
    return llm_cls(**llm_kwargs)


def get_structured_llm(
    schema: type[BaseModel],
    callbacks: list[Any] | None = None,
) -> BaseChatModel:
    """Return an LLM configured with structured output for a Pydantic schema.

    Convenience wrapper around ``get_llm().with_structured_output(schema)``.

    Args:
        schema: A Pydantic BaseModel subclass defining the output structure.
        callbacks: Optional list of LangChain callbacks injected into the LLM.

    Returns:
        A BaseChatModel with .with_structured_output(schema) applied.
    """
    return get_llm(callbacks=callbacks).with_structured_output(schema)
