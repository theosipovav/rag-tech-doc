"""
RAG PDF Chat System - Application Package
"""
from app.config import settings
from app.rag_pipeline import get_rag_pipeline, RAGPipeline

__version__ = "1.0.0"
__all__ = ["settings", "get_rag_pipeline", "RAGPipeline"]
