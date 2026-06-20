"""FastAPI application entry point with lifespan, CORS, and router."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import router
from src.memory.schema import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from src.observability import get_tracer, flush_traces

    await init_db()
    get_tracer()  # eager init — triggers lazy Langfuse client creation
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

app.include_router(router)
