"""
Configuration settings for the RAG system.
Uses environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "RAG Chat System"
    DEBUG: bool = True

    # Qdrant Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "pdf_documents"
    QDRANT_VECTOR_SIZE: int = 1024  # BGE-M3 dense vector dimension

    # Embedding Model (BGE-M3)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cuda"  # or "cpu"

    # Reranker Model
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_K: int = 5
    RERANKER_DEVICE: str = "cuda"

    # LLM Inference (vLLM / Ollama)
    LLM_API_BASE: str = "http://localhost:8000/v1"  # vLLM default
    LLM_MODEL_NAME: str = "meta-llama/Llama-3-8B-Instruct"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.7

    # Chunking Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Search Parameters
    SEARCH_TOP_K: int = 10

    # File Storage
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
