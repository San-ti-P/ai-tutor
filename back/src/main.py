"""FastAPI application entry point with lifespan, CORS, and router."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env into os.environ so Langfuse SDK global singleton finds keys
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.api.router import router  # noqa: E402 (dotenv must load first)
from src.memory.schema import init_db  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from src.observability import flush_traces, get_tracer

    await init_db()
    get_tracer()  # eager init — triggers lazy Langfuse client creation

    # Bootstrap NLTK Spanish stopwords — guard for air-gapped deploys
    try:
        import nltk

        nltk.download("stopwords", quiet=True)
    except Exception:
        pass  # Air-gapped deploy: stopwords downloaded manually

    yield
    flush_traces()
    from src.agents.orchestrator import close_orchestrator_graph

    await close_orchestrator_graph()


app = FastAPI(title="AI Tutor API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
