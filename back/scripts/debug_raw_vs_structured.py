"""Quick test: raw LLM vs with_structured_output on the same prompt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from dotenv import load_dotenv
load_dotenv(".env")

from langchain_ollama import ChatOllama

model = ChatOllama(model="gemma4:e4b-it-q8_0", base_url="http://localhost:11434", temperature=0)

prompt = """Genera un examen academico basado en el siguiente material.

[CHUNK:abc] Un agente inteligente es reactivo, proactivo y tiene habilidad social.

PREFERENCIAS:
Total questions: 2
MCQ questions: 1
Open-answer questions: 1
Difficulty: medium

Responde SOLO con JSON valido con esta estructura exacta:
{"mcq_questions": [{"stem": "...", "options": ["..."], "correct_option_index": 0, "source_chunk_ids": ["abc"], "difficulty": "medium", "topic": "agentes"}], "open_questions": [{"prompt": "...", "base_answer": "...", "key_points": ["..."], "source_chunk_ids": ["abc"], "difficulty": "medium", "topic": "agentes"}], "metadata": {}}
"""

print("=== RAW LLM (no with_structured_output) ===")
result = model.invoke(prompt)
print(f"Response ({len(result.content)} chars):")
print(result.content[:1500])
print()

# Now test with structured output
from src.llm import get_structured_llm
from src.agents.exam_generator import ExamGeneration

print("=== with_structured_output ===")
m = get_structured_llm(ExamGeneration)
result2 = m.invoke(prompt)
print(f"mcq: {len(result2.mcq_questions)}, open: {len(result2.open_questions)}")
print(f"metadata: {result2.metadata}")
