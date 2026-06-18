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
    classification_confidence_threshold: float = 0.60
    anti_hallucination_threshold: float = 0.55
    judge_sample_rate: float = 0.10
    judge_disagreement_threshold: float = 2.0

    # RAG / Embedding
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5

    # LLM
    llm_provider: str = "ollama"  # "ollama" or "groq"
    groq_model_name: str = "llama-3.1-8b-instant"  # cheapest, tool-use capable
    ollama_model_name: str = "gemma4:e4b-it-q8_0"  # local 4B model, fast for dev
    ollama_base_url: str = "http://localhost:11434"

    @property
    def llm_kwargs(self) -> dict:
        """Return (model_class, kwargs) for the configured LLM provider."""
        if self.llm_provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq, {"model": self.groq_model_name, "temperature": 0}
        # default: ollama
        from langchain_ollama import ChatOllama

        return ChatOllama, {"model": self.ollama_model_name, "temperature": 0}

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
