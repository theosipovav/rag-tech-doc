"""
Reranking service using BGE Reranker model.
Improves search results by re-ranking retrieved documents.
"""
from typing import List, Tuple, Optional
from loguru import logger

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    logger.warning("FlagEmbedding not installed. Reranking will be disabled.")

from app.config import settings


class RerankerService:
    """Service for reranking search results using BGE Reranker."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device or settings.RERANKER_DEVICE
        self.reranker = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the reranker model into memory."""
        if self._loaded:
            logger.info("Reranker model already loaded")
            return

        if not FLAG_EMBEDDING_AVAILABLE:
            logger.warning("FlagEmbedding not available, skipping reranker loading")
            return

        try:
            logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
            self.reranker = FlagReranker(
                self.model_name,
                use_fp16=True  # Use FP16 for faster inference on GPU
            )
            self._loaded = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, score) tuples sorted by score descending
        """
        if not self._loaded:
            logger.warning("Reranker not loaded, returning original order")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        top_k = top_k or settings.RERANKER_TOP_K

        try:
            logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")
            
            # Prepare pairs for reranking
            pairs = [[query, doc] for doc in documents]
            
            # Get scores
            scores = self.reranker.compute_score(pairs, normalize=True)
            
            # Handle both single score and list of scores
            if isinstance(scores, float):
                scores = [scores]
            
            # Create index-score pairs
            indexed_scores = list(enumerate(scores))
            
            # Sort by score descending
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            return indexed_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original order on error
            return [(i, 0.0) for i in range(len(documents))]

    def rerank_with_texts(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[dict]:
        """
        Rerank documents and return with texts.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of dicts with index, score, and text
        """
        ranked_indices = self.rerank(query, documents, top_k)
        
        results = []
        for idx, score in ranked_indices:
            results.append({
                "index": idx,
                "score": float(score),
                "text": documents[idx]
            })
        
        return results


# Singleton instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create the reranker service singleton."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
