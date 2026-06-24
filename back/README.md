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

## LLM Provider Configuration

Set `LLM_PROVIDER` in `.env`. Supported: `ollama`, `groq`, `opencode-go`, `openai`.

### Ollama (local)
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=gemma4:e4b-it-q8_0
# Uses http://localhost:11434 by default
```

### Ollama Cloud
For cloud-hosted models (e.g. `gemma4:31b-cloud`). Get API key at [ollama.com](https://ollama.com) → Settings → Keys.

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=gemma4:31b-cloud
OLLAMA_BASE_URL=https://api.ollama.com
OLLAMA_API_KEY=<your-key>
```

### Groq
Fast, free tier available. Sign up at [console.groq.com](https://console.groq.com).

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=<your-key>
GROQ_MODEL_NAME=llama-3.1-8b-instant
```

### OpenCode Go
OpenAI-compatible endpoint, $10/mo subscription. Subscribe at [opencode.ai/auth](https://opencode.ai/auth).

```bash
LLM_PROVIDER=opencode-go
OPENCODE_GO_API_KEY=<your-key>
OPENCODE_GO_MODEL_NAME=deepseek-v4-pro
```

Available Go models: `deepseek-v4-pro`, `deepseek-v4-flash`, `kimi-k2.7-code`, `kimi-k2.6`, `qwen3.7-max`, `qwen3.7-plus`, `glm-5.2`, `minimax-m3`, `mimo-v2.5`. See [opencode.ai/docs/go](https://opencode.ai/docs/go) for full list.

### OpenAI (direct)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o
# OPENAI_BASE_URL=https://custom.api.com/v1  # For OpenAI-compatible proxies
```

## Project Structure

```
src/
├── agents/        # LangGraph agents (orchestrator, ingestor, generators, evaluator, support)
├── tools/         # Tool definitions consumed by agents
├── rag/           # ChromaDB, chunking, embeddings, retrieval, thematic index
├── memory/        # SQLite schema + student profile CRUD
├── api/           # FastAPI router + Pydantic schemas
├── llm.py         # LLM factory (get_llm, get_structured_llm)
├── config.py      # Settings from .env via pydantic-settings
└── observability/ # Langfuse tracing setup
```

## Testing

```bash
# Unit tests (no external services needed)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Integration tests (needs real LLM + embeddings + PDFs)
pytest tests/ -v -m integration
```

## Lint & Format

```bash
ruff check src/ tests/
ruff format src/ tests/
```
