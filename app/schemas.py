"""
Schemas for request/response validation using Pydantic.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChunkMetadata(BaseModel):
    """Metadata for a text chunk."""
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkResponse(BaseModel):
    """Response containing chunk information."""
    id: str
    text: str
    metadata: ChunkMetadata
    score: Optional[float] = None


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""
    texts: List[str] = Field(..., description="List of texts to embed")


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""
    embeddings: List[List[float]]
    model: str
    usage: dict


class SearchRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, description="Number of results to return")
    min_score: float = Field(default=0.0, description="Minimum similarity score")


class SearchResponse(BaseModel):
    """Response from semantic search."""
    query: str
    results: List[ChunkResponse]
    total_found: int


class RerankRequest(BaseModel):
    """Request for reranking documents."""
    query: str
    documents: List[str]
    top_k: int = Field(default=5)


class RerankResponse(BaseModel):
    """Response from reranking."""
    query: str
    results: List[dict]  # {index, score, text}


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    message: str = Field(..., description="User's question")
    conversation_history: List[ChatMessage] = Field(
        default=[], 
        description="Previous conversation messages"
    )
    top_k: int = Field(default=5, description="Number of context chunks to use")
    include_sources: bool = Field(default=True, description="Include source citations")


class ChatResponse(BaseModel):
    """Response from chat completion."""
    answer: str
    sources: List[ChunkResponse] = []
    conversation_id: Optional[str] = None
    processing_time_ms: float


class UploadResponse(BaseModel):
    """Response from file upload and indexing."""
    filename: str
    status: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    qdrant_connected: bool
    embedding_model_loaded: bool
    reranker_model_loaded: bool
    llm_available: bool
