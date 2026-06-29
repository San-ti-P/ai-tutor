"""LLM factory — single source of truth for language model instantiation.

Every agent that needs a language model MUST import from here instead of
calling ``settings.llm_kwargs`` directly. This ensures provider/model
changes happen in exactly one place.

Functions:
- get_llm(): returns a configured BaseChatModel (Ollama, Groq, etc.)
- get_structured_llm(schema): returns a model with JSON-in-prompt
  structured output. NEVER uses ``with_structured_output`` — Ollama
  and Groq models lack reliable native support. Schema is injected
  into the prompt; response is JSON-parsed and validated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level LLM cache keyed by provider + model fingerprint.
# Avoids rebuilding HTTP connection pools on every call under concurrent load.
_llm_cache: dict[str, BaseChatModel] = {}


def _clear_llm_cache() -> None:
    """Clear the LLM instance cache. Exposed for test isolation."""
    _llm_cache.clear()

# Regex to strip markdown ```json ... ``` fences from LLM output.
_CLEAN_FENCES = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def get_llm(
    callbacks: list[Any] | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Return a configured LLM instance for the current provider.

    Reads ``settings.llm_kwargs`` and instantiates the appropriate
    chat model class. Supports all configured providers.

    E2E modes (checked in priority order):
    1. ``E2E_RECORD_MODE=true`` + ``E2E_LIVE_LLM=true``:
       real LLM wrapped in RecordingLLM — saves responses to seed file.
    2. ``E2E_TEST_MODE=true`` + ``E2E_LIVE_LLM`` not true:
       MockLLM — replays pre-recorded seeds.
    3. ``E2E_LIVE_LLM=true``:
       real LLM passthrough (no recording).
    4. Otherwise: real LLM (normal operation).

    Args:
        callbacks: Optional list of LangChain callbacks (e.g. Langfuse
            CallbackHandler) injected into the LLM config.
        temperature: Override the default temperature for this call.
            When None, uses the provider-configured default.

    Returns:
        A configured BaseChatModel ready for .invoke() calls.
    """
    import os

    e2e_record = os.getenv("E2E_RECORD_MODE", "").lower() == "true"
    e2e_test = os.getenv("E2E_TEST_MODE", "").lower() == "true"
    e2e_live = os.getenv("E2E_LIVE_LLM", "").lower() == "true"

    # Priority 1: Record mode — real LLM + response capture
    if e2e_record and e2e_live:
        llm_cls, llm_kwargs = settings.llm_kwargs
        if temperature is not None:
            llm_kwargs = {**llm_kwargs, "temperature": temperature}
        if callbacks:
            llm_kwargs = {**llm_kwargs, "callbacks": callbacks}
        real_llm = llm_cls(**llm_kwargs)

        from src.llm_test import get_recording_llm

        return get_recording_llm(real_llm)  # type: ignore[return-value]

    # Priority 2: Mock mode — replay from seeds
    if e2e_test and not e2e_live:
        from src.llm_test import get_mock_llm

        return get_mock_llm()  # type: ignore[return-value]

    # Priority 3 & 4: Live LLM or normal operation
    llm_cls, llm_kwargs = settings.llm_kwargs
    if temperature is not None:
        llm_kwargs = {**llm_kwargs, "temperature": temperature}
    # Build a cache key from provider, model, AND temperature fingerprint
    cls_name = getattr(llm_cls, "__name__", str(llm_cls))
    raw_key = f"{settings.llm_provider}:{cls_name}:{llm_kwargs.get('model', '')}:t{llm_kwargs.get('temperature', 'default')}"
    cache_key = hashlib.sha256(raw_key.encode()).hexdigest()[:16]

    if cache_key not in _llm_cache:
        if callbacks:
            llm_kwargs = {**llm_kwargs, "callbacks": callbacks}
        _llm_cache[cache_key] = llm_cls(**llm_kwargs)
    else:
        # Re-inject callbacks if provided (they may change per invocation)
        cached = _llm_cache[cache_key]
        if callbacks:
            cached.callbacks = callbacks

    return _llm_cache[cache_key]


def _parse_json_for_schema(schema: type[BaseModel]) -> Callable[[str], BaseModel]:
    """Return a callable that parses JSON from LLM text output.

    Strips markdown fences, finds the last complete JSON object, and
    validates against *schema*. Returns a BaseModel instance.
    Raises ValueError if no valid JSON is found.
    """
    import json

    def _parse(text: str) -> BaseModel:
        text = _CLEAN_FENCES.sub("", text).strip()

        # Find the last complete JSON object by scanning from the end
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
                    return schema.model_validate(json.loads(candidate))
                except Exception:
                    continue
        raise ValueError(f"No valid JSON object found in response: {text[:300]!r}")

    return _parse


def _schema_in_prompt_chain(
    schema: type[BaseModel],
    callbacks: list[Any] | None = None,
    temperature: float = 0.0,
) -> Runnable[Any, Any]:
    """Build a chain that appends the JSON schema to the user prompt.

    For Ollama: uses ``format="json"`` on a dedicated ChatOllama instance.
    For OpenAI-compatible (OpenCode Go): uses standard completion via ``get_llm()``.

    The chain: schema JSON injected → LLM → StrOutputParser → JSON parse → model_validate.
    """
    import json

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    llm: BaseChatModel
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        llm_kwargs: dict[str, Any] = {
            "model": settings.ollama_model_name,
            "base_url": settings.ollama_base_url,
            "temperature": temperature,
            "format": "json",
        }
        if settings.ollama_api_key:
            llm_kwargs["client_kwargs"] = {
                "headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}
            }
        if callbacks:
            llm_kwargs["callbacks"] = callbacks
        llm = ChatOllama(**llm_kwargs)
    else:
        llm = get_llm(callbacks=callbacks, temperature=temperature)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Sos un asistente servicial. Seguí cuidadosamente las instrucciones y "
                "ejemplos del usuario, luego respondé con JSON válido que coincida con "
                "el esquema de abajo. Las claves del JSON deben estar en inglés (según "
                "el esquema), pero los valores de texto deben estar en español. No "
                "generes ningún texto fuera del objeto JSON.",
            ),
            ("user", "{input}\n\nDevolvé JSON válido que coincida con este esquema:\n{schema}"),
        ]
    )

    chain: Runnable[Any, Any] = (
        RunnableLambda(lambda x: {"input": x, "schema": schema_json})
        | prompt_template
        | llm
        | StrOutputParser()
        | _parse_json_for_schema(schema)
    )
    return chain


def get_structured_llm(
    schema: type[BaseModel],
    callbacks: list[Any] | None = None,
    temperature: float = 0.0,
) -> Runnable[Any, Any]:
    """Return an LLM configured with structured output for a Pydantic schema.

    Uses schema-in-prompt approach (appends JSON schema to user prompt,
    parses response) for ALL providers. Native ``with_structured_output``
    is NOT used because Ollama and Groq models lack reliable support.

    Args:
        schema: A Pydantic BaseModel subclass defining the output structure.
        callbacks: Optional list of LangChain callbacks injected into the LLM.
        temperature: Sampling temperature (default 0.0 for deterministic output).

    Returns:
        A Runnable that takes a prompt string and returns a validated
        instance of *schema*.
    """
    return _schema_in_prompt_chain(schema, callbacks, temperature)
