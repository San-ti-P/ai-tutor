"""LLM factory — single source of truth for language model instantiation.

Every agent that needs a language model MUST import from here instead of
calling ``settings.llm_kwargs`` directly. This ensures provider/model
changes happen in exactly one place.

Functions:
- get_llm(): returns a configured BaseChatModel (Ollama, Groq, etc.)
- get_structured_llm(schema): returns a model with structured output
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.config import settings


def get_llm(
    callbacks: list[Any] | None = None,
) -> BaseChatModel:
    """Return a configured LLM instance for the current provider.

    Reads ``settings.llm_kwargs`` and instantiates the appropriate
    chat model class. Supports all configured providers.

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


def _ollama_json_mode_chain(
    schema: type[BaseModel],
    callbacks: list[Any] | None = None,
) -> Runnable:
    """Build a chain for Ollama using ``format="json"`` + schema in prompt.

    Ollama's native structured output (``format={json_schema}``) is not
    reliably enforced by all models (e.g. Gemma may use wrong field names).
    JSON mode with the schema embedded in the system prompt is more
    portable, matching Ollama's own documented examples.
    """
    import json
    import re

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_ollama import ChatOllama

    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    llm_kwargs: dict[str, Any] = {
        "model": settings.ollama_model_name,
        "base_url": settings.ollama_base_url,
        "temperature": 0,
        "format": "json",
    }
    if settings.ollama_api_key:
        llm_kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}
        }
    if callbacks:
        llm_kwargs["callbacks"] = callbacks

    ollama = ChatOllama(**llm_kwargs)

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "Respond with JSON matching this exact schema:\n{schema}\n\n"
            "No other text. Just valid JSON.",
        ),
        ("user", "{input}"),
    ])

    def _parse(text: str) -> BaseModel:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            raise ValueError(f"No JSON found in response: {text[:300]!r}")
        return schema.model_validate(json.loads(json_match.group(0)))

    chain = (
        RunnableLambda(lambda x: {"input": x, "schema": schema_json})
        | prompt_template
        | ollama
        | StrOutputParser()
        | _parse
    )
    return chain


def get_structured_llm(
    schema: type[BaseModel],
    callbacks: list[Any] | None = None,
) -> Runnable:
    """Return an LLM configured with structured output for a Pydantic schema.

    For Ollama: uses ``format="json"`` with schema in system prompt.
    For other providers: uses native ``with_structured_output(schema)``.

    Args:
        schema: A Pydantic BaseModel subclass defining the output structure.
        callbacks: Optional list of LangChain callbacks injected into the LLM.

    Returns:
        A Runnable that takes a prompt string and returns a validated
        instance of *schema*.
    """
    if settings.llm_provider == "ollama":
        return _ollama_json_mode_chain(schema, callbacks)
    return get_llm(callbacks=callbacks).with_structured_output(schema)
