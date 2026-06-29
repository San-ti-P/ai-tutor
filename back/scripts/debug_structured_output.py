"""Debug: capture the full messages with_structured_output sends to Ollama."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from typing import Any, cast

# Intercept the LLM call to see raw messages
from langchain_core.callbacks import BaseCallbackHandler

from src.agents.exam_generator import ExamGeneration
from src.llm import get_llm


class MessageCapture(BaseCallbackHandler):
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any) -> None:
        print("=== MESSAGES SENT TO LLM ===")
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i} ({type(msg).__name__}) ---")
            print(msg.content[:2000] if hasattr(msg, "content") else str(msg)[:2000])
        print("\n=== END MESSAGES ===")


# Create model with callback capture
llm = get_llm()
structured_llm = llm.with_structured_output(ExamGeneration)

# Invoke with a realistic prompt
prompt = """Generá un examen académico basado en el siguiente material.

[CHUNK:abc] Un agente inteligente es reactivo, proactivo y tiene habilidad social.

PREFERENCIAS:
Total questions: 2
MCQ questions: 1
Open-answer questions: 1
Difficulty: medium

REQUISITOS:
- Cada pregunta DEBE basarse en hechos textuales de los chunks.
- Para MCQs: 3-5 opciones, exactamente una correcta.
- Para open-answer: prompts que requieran explicación, incluir base_answer.
"""

result = cast(ExamGeneration, structured_llm.invoke(prompt, config={"callbacks": [MessageCapture()]}))

print("\n=== PARSED RESULT ===")
print(f"mcq_questions: {len(result.mcq_questions)}")
for mq in result.mcq_questions:
    print(f"  MCQ: {mq.stem[:80]}...")
print(f"open_questions: {len(result.open_questions)}")
for oq in result.open_questions:
    print(f"  OPEN: {oq.prompt[:80]}...")
print(f"metadata: {result.metadata}")
