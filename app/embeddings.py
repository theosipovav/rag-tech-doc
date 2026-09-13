"""
Embedding service using BGE-M3 model.
Provides synchronous and asynchronous embedding generation.
"""
import numpy as np
from typing import List, Optional
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Service for generating embeddings using BGE-M3 model."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.model: Optional[SentenceTransformer] = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the embedding model into memory."""
        if self._loaded:
            logger.info("Embedding model already loaded")
            return

        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            self._loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self._loaded:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")
        
        if not texts:
            return []

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if not self._loaded:
            return settings.QDRANT_VECTOR_SIZE
        return self.model.get_sentence_embedding_dimension()


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
