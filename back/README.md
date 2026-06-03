# AI Tutor — Backend

Multi-agent LLM system for adaptive university exam preparation.

## Quick Start

```bash
# Install uv if not present
pip install uv

# Create venv and install deps
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Copy env and fill in keys
cp .env.example .env

# Run dev server
uvicorn src.main:app --reload
```

## Project Structure

```
src/
├── agents/        # LangGraph agents (orchestrator, ingestor, generators, evaluator, support)
├── tools/         # Tool definitions consumed by agents
├── rag/           # ChromaDB, chunking, embeddings, retrieval, thematic index
├── memory/        # SQLite schema + student profile CRUD
├── api/           # FastAPI router + Pydantic schemas
└── observability/ # Langfuse tracing setup
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src
```

## Lint & Format

```bash
ruff check src/ tests/
ruff format src/ tests/
```
