"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    mathpix_app_id: str = ""
    mathpix_app_key: str = ""

    chroma_persist_directory: str = "./data/chroma"
    sqlite_db_path: str = "./data/tutor.db"

    host: str = "0.0.0.0"
    port: int = 8000

    max_iterations_per_task: int = 15
    ocr_confidence_threshold: float = 0.85

    # RAG / Embedding
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
