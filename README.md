# AI Tutor — Tutor Académico Personal

Multi-agent LLM system for adaptive university exam preparation. 6 specialized agents coordinate via LangGraph to ingest academic material, generate personalized exams/exercises, evaluate answers, and track student progress across sessions.

**Stack**: Python 3.12+ · LangGraph · LangChain · FastAPI · ChromaDB · SQLite · Langfuse · Next.js 15 · React 19 · Tailwind 4
**Course**: IA 2026 — UTN Santa Fe (CIDISI)

## Quick Start

```bash
# Backend
cd back
cp .env.example .env   # Edit .env with your LLM provider keys (see below)
uv sync
uv run uvicorn src.main:app --reload

# Frontend
cd front
npm install
npm run dev
```

## LLM Providers

The system supports 4 providers. Switch by changing `LLM_PROVIDER` in `back/.env`.

| Provider | Model Examples | Cost | Setup |
|----------|---------------|------|-------|
| `ollama` | `gemma4:e4b-it-q8_0` (local), `gemma4:31b-cloud` (cloud) | Free (local) / Pay-per-use (cloud) | Install [Ollama](https://ollama.com) for local; add `OLLAMA_API_KEY` for cloud |
| `groq` | `llama-3.1-8b-instant` | Free tier | Sign up at [console.groq.com](https://console.groq.com) |
| `opencode-go` | `deepseek-v4-pro`, `kimi-k2.7-code`, `qwen3.7-max` | $10/mo subscription | Subscribe at [opencode.ai/auth](https://opencode.ai/auth) |
| `openai` | `gpt-4o` | Pay-per-use | [platform.openai.com](https://platform.openai.com) API key |

### Provider config examples

**Ollama (local)** — no API key needed:
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=gemma4:e4b-it-q8_0
```

**Ollama Cloud** — for cloud-hosted models like `gemma4:31b-cloud`:
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=gemma4:31b-cloud
OLLAMA_BASE_URL=https://api.ollama.com
OLLAMA_API_KEY=<your-ollama-cloud-key>
```

**Groq** — fast, free-tier friendly:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=<your-groq-key>
GROQ_MODEL_NAME=llama-3.1-8b-instant
```

**OpenCode Go** — OpenAI-compatible endpoint, $10/mo:
```bash
LLM_PROVIDER=opencode-go
OPENCODE_GO_API_KEY=<your-opencode-go-key>
OPENCODE_GO_MODEL_NAME=deepseek-v4-pro
```

**OpenAI** — direct or custom base URL:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o
# OPENAI_BASE_URL=https://custom.api.com/v1  # For OpenAI-compatible proxies
```

## Architecture

See [`AGENTS.md`](AGENTS.md) for full architecture docs, conventions, and guardrails.

```
ai-tutor/
├── back/                  # FastAPI + LangGraph + ChromaDB
│   └── src/
│       ├── agents/        # One file per agent
│       ├── tools/         # Tool definitions
│       ├── rag/           # ChromaDB, embeddings, retrieval
│       ├── memory/        # SQLite + student profiles
│       ├── api/           # FastAPI routes
│       └── observability/ # Langfuse tracing
├── front/                 # Next.js 15 App Router
├── epics/                 # Implementation breakdown per agent
└── init_PRD.md            # Product requirements
```

## Deliveries

| Date | Milestone |
|------|-----------|
| 08/06 | Concept |
| 22/06 | MVP |
| 29/06 | Complete + Defense |
