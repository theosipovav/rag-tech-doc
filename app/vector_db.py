"""
Vector database service using Qdrant.
Handles storage and retrieval of document embeddings.
"""
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchParams,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not installed. Vector search will be disabled.")

from app.config import settings
from app.schemas import ChunkMetadata


class VectorDBService:
    """Service for managing vector storage and search in Qdrant."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None
    ):
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        self.client: Optional[QdrantClient] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Qdrant server."""
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available")
            return False

        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                prefer_grpc=False  # Use HTTP for easier debugging
            )
            self._connected = True
            logger.info("Connected to Qdrant successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._connected and self.client is not None

    def create_collection(self) -> bool:
        """Create the collection if it doesn't exist."""
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return False

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    on_disk_payload=True  # Store payload on disk for efficiency
                )
                
                # Create indexes for metadata fields
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_file",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="page_number",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def upsert_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert or update documents in the vector store.
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return False

        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            logger.error("Texts, embeddings, and metadatas must have same length")
            return False

        try:
            points = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                point_id = str(uuid.uuid4())
                
                payload = {
                    "text": text,
                    "source_file": metadata.get("source_file", "unknown"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index", i),
                    "created_at": metadata.get("created_at", datetime.utcnow().isoformat())
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                )

            logger.info(f"Upserting {len(points)} documents to Qdrant")
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Upsert completed: {operation_info}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        min_score: float = 0.0,
        filter_by_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score
            filter_by_file: Optional filename to filter by
            
        Returns:
            List of search results with text, metadata, and score
        """
        if not self.is_connected:
            logger.error("Not connected to Qdrant")
            return []

        top_k = top_k or settings.SEARCH_TOP_K

        try:
            # Build filter if needed
            search_filter = None
            if filter_by_file:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=filter_by_file)
                        )
                    ]
                )

            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=min_score,
                query_filter=search_filter,
                search_params=SearchParams(
                    hnsw_ef=128,
                    exact=False
                )
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "source_file": result.payload.get("source_file", ""),
                    "page_number": result.payload.get("page_number"),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "score": float(result.score)
                })

            logger.debug(f"Search returned {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_collection(self) -> bool:
        """Delete the collection (for testing/reset)."""
        if not self.is_connected:
            return False

        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.is_connected:
            return {"error": "Not connected"}

        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Singleton instance
_vector_db_service: Optional[VectorDBService] = None


def get_vector_db_service() -> VectorDBService:
    """Get or create the vector DB service singleton."""
    global _vector_db_service
    if _vector_db_service is None:
        _vector_db_service = VectorDBService()
    return _vector_db_service
