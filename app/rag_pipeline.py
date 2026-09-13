"""
RAG Pipeline - Orchestrates the retrieval-augmented generation workflow.
Combines all services for end-to-end query processing.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.config import settings
from app.schemas import ChatMessage, ChunkResponse, ChunkMetadata
from app.embeddings import get_embedding_service
from app.vector_db import get_vector_db_service
from app.reranker import get_reranker_service
from app.llm import get_llm_service
from app.pdf_parser import PDFParser
from app.chunking import get_chunking_service


class RAGPipeline:
    """Orchestrates the RAG pipeline for document indexing and querying."""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_db = get_vector_db_service()
        self.reranker_service = get_reranker_service()
        self.llm_service = get_llm_service()
        self.chunking_service = get_chunking_service()
        self.pdf_parser = PDFParser(extract_tables=True)
        
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all services."""
        logger.info("Initializing RAG pipeline...")
        
        # Initialize embedding model
        try:
            self.embedding_service.load_model()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
        
        # Connect to vector DB
        if not self.vector_db.connect():
            logger.error("Failed to connect to vector DB")
            return False
        
        # Create collection if needed
        if not self.vector_db.create_collection():
            logger.error("Failed to create vector DB collection")
            return False
        
        # Load reranker (optional)
        try:
            self.reranker_service.load_model()
        except Exception as e:
            logger.warning(f"Reranker not loaded: {e}")
        
        # Initialize LLM (optional)
        try:
            self.llm_service.initialize()
        except Exception as e:
            logger.warning(f"LLM not initialized: {e}")
        
        self._initialized = True
        logger.info("RAG pipeline initialized successfully")
        return True

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._initialized

    def index_document(self, file_path: str) -> Tuple[bool, int]:
        """
        Index a single PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (success, number of chunks indexed)
        """
        logger.info(f"Starting document indexing: {file_path}")
        start_time = time.time()
        
        try:
            # Step 1: Parse PDF
            logger.info("Step 1: Parsing PDF...")
            parsed_data = self.pdf_parser.parse_pdf(file_path)
            pages = self.pdf_parser.get_pages_with_metadata(parsed_data)
            logger.info(f"Parsed {len(pages)} pages")
            
            # Step 2: Chunk documents
            logger.info("Step 2: Chunking documents...")
            chunks = self.chunking_service.chunk_documents(pages)
            logger.info(f"Created {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No chunks created")
                return False, 0
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            embeddings = self.embedding_service.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Store in vector DB
            logger.info("Step 4: Storing in vector DB...")
            success = self.vector_db.upsert_documents(texts, embeddings, metadatas)
            
            if success:
                elapsed = time.time() - start_time
                logger.info(
                    f"Document indexed successfully: {len(chunks)} chunks "
                    f"in {elapsed:.2f}s"
                )
                return True, len(chunks)
            else:
                logger.error("Failed to store documents in vector DB")
                return False, 0
                
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return False, 0

    def index_documents(self, file_paths: List[str]) -> Tuple[bool, int]:
        """
        Index multiple PDF documents.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Tuple of (success, total chunks indexed)
        """
        total_chunks = 0
        successful = 0
        
        for file_path in file_paths:
            success, chunks = self.index_document(file_path)
            if success:
                successful += 1
                total_chunks += chunks
        
        logger.info(
            f"Indexed {successful}/{len(file_paths)} documents, "
            f"{total_chunks} total chunks"
        )
        
        return successful > 0, total_chunks

    def query(
        self,
        question: str,
        top_k: int = None,
        use_reranking: bool = True,
        min_score: float = 0.0
    ) -> Tuple[str, List[ChunkResponse]]:
        """
        Process a user query using RAG.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            use_reranking: Whether to use reranking
            min_score: Minimum similarity score
            
        Returns:
            Tuple of (answer, source chunks)
        """
        if not self._initialized:
            raise RuntimeError("RAG pipeline not initialized")
        
        top_k = top_k or settings.RERANKER_TOP_K if use_reranking else settings.SEARCH_TOP_K
        start_time = time.time()
        
        # Step 1: Embed the query
        logger.info(f"Embedding query: {question[:50]}...")
        query_embedding = self.embedding_service.embed_query(question)
        
        # Step 2: Search vector DB
        logger.info("Searching vector DB...")
        search_results = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=settings.SEARCH_TOP_K * 2 if use_reranking else top_k,
            min_score=min_score
        )
        
        if not search_results:
            logger.warning("No relevant documents found")
            return "I couldn't find relevant information to answer your question.", []
        
        # Step 3: Rerank results (optional)
        if use_reranking and self.reranker_service.is_loaded:
            logger.info("Reranking results...")
            documents = [result["text"] for result in search_results]
            ranked = self.reranker_service.rerank_with_texts(
                query=question,
                documents=documents,
                top_k=top_k
            )
            
            # Get top ranked results
            search_results = [search_results[r["index"]] for r in ranked]
            logger.info(f"Reranked to {len(search_results)} results")
        
        # Step 4: Format results
        source_chunks = []
        for result in search_results:
            metadata = ChunkMetadata(
                source_file=result.get("source_file", ""),
                page_number=result.get("page_number"),
                chunk_index=result.get("chunk_index", 0),
                created_at=datetime.utcnow()
            )
            
            chunk_response = ChunkResponse(
                id=result.get("id", ""),
                text=result.get("text", ""),
                metadata=metadata,
                score=result.get("score", 0.0)
            )
            source_chunks.append(chunk_response)
        
        # Step 5: Generate answer with LLM
        if self.llm_service.is_available:
            logger.info("Generating answer with LLM...")
            context_dicts = [
                {
                    "text": chunk.text,
                    "source_file": chunk.metadata.source_file,
                    "page_number": chunk.metadata.page_number
                }
                for chunk in source_chunks
            ]
            
            answer = self.llm_service.generate_rag_response(
                query=question,
                context_chunks=context_dicts
            )
        else:
            # Fallback: Return concatenated context
            logger.info("LLM not available, returning context")
            answer = "Based on the following relevant excerpts:\n\n"
            for i, chunk in enumerate(source_chunks, 1):
                answer += f"[{i}] {chunk.text[:200]}...\n\n"
            answer += "\nNote: LLM is not available for generating a synthesized answer."
        
        elapsed = time.time() - start_time
        logger.info(f"Query processed in {elapsed:.2f}s")
        
        return answer, source_chunks

    def chat(
        self,
        message: str,
        conversation_history: List[ChatMessage] = None,
        top_k: int = None
    ) -> Tuple[str, List[ChunkResponse], float]:
        """
        Process a chat message with conversation history.
        
        Args:
            message: User's message
            conversation_history: Previous conversation messages
            top_k: Number of context chunks
            
        Returns:
            Tuple of (answer, sources, processing_time_ms)
        """
        start_time = time.time()
        
        # For now, treat each message as a new query
        # In a more advanced implementation, we could use conversation history
        # to improve retrieval
        answer, sources = self.query(message, top_k=top_k)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return answer, sources, processing_time_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "initialized": self._initialized,
            "embedding_model": self.embedding_service.is_loaded,
            "vector_db_connected": self.vector_db.is_connected,
            "reranker_loaded": self.reranker_service.is_loaded,
            "llm_available": self.llm_service.is_available
        }
        
        if self.vector_db.is_connected:
            stats["vector_db_stats"] = self.vector_db.get_stats()
        
        return stats


# Singleton instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
