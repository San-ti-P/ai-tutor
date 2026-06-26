"""Isolated test: Ollama + Langfuse token tracking.

Mirrors exactly how our system wires Langfuse:
  1. propagate_attributes(session_id=...)
  2. CallbackHandler() created INSIDE
  3. LLM invoked with config={"callbacks": [handler]} + RunnableConfig
  4. Check if token usage appears in Langfuse dashboard

Run from back/:
    uv run python scripts/verify_langfuse_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Test environment ──
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "test-isolated"

# ── Langfuse client (mirrors _client.py) ──
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
)

# ── Ollama model (mirrors config.py defaults) ──
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma4:e4b-it-q8_0",
    base_url="http://localhost:11434",
    temperature=0,
)

print("Ollama model: gemma4:e4b-it-q8_0 @ http://localhost:11434")
print(f"Langfuse env: {os.environ['LANGFUSE_TRACING_ENVIRONMENT']}")
print()

# ── Pattern 1: CallbackHandler inside propagate (our fixed pattern) ──
print("=== Pattern 1: CallbackHandler INSIDE propagate_attributes ===")
with propagate_attributes(
    session_id="isolated-inside-prop",
    trace_name="isolated-inside-prop-trace",
):
    handler = CallbackHandler()
    config: RunnableConfig = {
        "callbacks": [handler],
        "metadata": {"langfuse_session_id": "isolated-inside-prop"},
    }
    result = model.invoke("What is 2+2? Answer in one word.", config=config)
    print(f"  Response: {result.content}")
    print(
        f"  usage_metadata: {result.usage_metadata if hasattr(result, 'usage_metadata') else 'N/A'}"
    )
    print(f"  response_metadata: {result.response_metadata}")
langfuse.flush()
print("  Flushed.\n")

# ── Pattern 2: @observe decorator (how our tools are traced) ──
print("=== Pattern 2: @observe + CallbackHandler inside ===")
from langfuse import observe


@observe(name="isolated-observe-test", as_type="tool")
def tool_with_llm(session_id: str) -> str:
    with propagate_attributes(session_id=session_id):
        handler = CallbackHandler()
        config: RunnableConfig = {
            "callbacks": [handler],
            "metadata": {"langfuse_session_id": session_id},
        }
        result = model.invoke("Say 'hello' in one word.", config=config)
        return result.content


result = tool_with_llm("isolated-observe-sess")
print(f"  Response: {result}")
langfuse.flush()
print("  Flushed.\n")

# ── Pattern 3: start_as_current_observation + generation update ──
print("=== Pattern 3: start_as_current_observation + gen update ===")
with langfuse.start_as_current_observation(
    name="isolated-direct-root",
    as_type="span",
) as root:
    with propagate_attributes(session_id="isolated-direct-sess"):
        # Create generation INSIDE propagate
        gen = root.start_observation(
            name="ollama-call",
            as_type="generation",
            model="gemma4:e4b-it-q8_0",
            input={"query": "What is 3+3?"},
            model_parameters={"temperature": 0},
        )
        result = model.invoke("What is 3+3? Answer in one word.")
        gen.update(
            output=result.content,
            usage_details=result.usage_metadata if hasattr(result, "usage_metadata") else None,
        )
        gen.end()
        print(f"  Response: {result.content}")
        print(
            f"  usage_metadata: {result.usage_metadata if hasattr(result, 'usage_metadata') else 'N/A'}"
        )
langfuse.flush()
print("  Flushed.\n")

print("Done. Check https://us.cloud.langfuse.com for traces:")
print("  - 'isolated-inside-prop-trace'")
print("  - 'isolated-observe-test'")
print("  - 'isolated-direct-root'")
print()
print("Look for GENERATION spans with usageDetails (token counts).")
