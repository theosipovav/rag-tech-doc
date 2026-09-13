"""
FastAPI Backend for RAG Chat System.
Provides REST API endpoints for document upload, indexing, and chat.
"""
import os
import time
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    ChunkResponse,
    ChunkMetadata
)
from app.rag_pipeline import get_rag_pipeline


# Configure logging
logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG" if settings.DEBUG else "INFO"
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-based Chat System with PDF document understanding",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
rag_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global rag_pipeline
    
    logger.info("Starting up RAG Chat System...")
    
    # Ensure data directories exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize RAG pipeline
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline.initialize():
        logger.error("Failed to initialize RAG pipeline")
    else:
        logger.info("RAG pipeline initialized successfully")
    
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Chat System...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health and component status."""
    stats = rag_pipeline.get_stats() if rag_pipeline else {}
    
    return HealthResponse(
        status="healthy" if rag_pipeline and rag_pipeline.is_initialized else "unhealthy",
        qdrant_connected=stats.get("vector_db_connected", False),
        embedding_model_loaded=stats.get("embedding_model", False),
        reranker_model_loaded=stats.get("reranker_loaded", False),
        llm_available=stats.get("llm_available", False)
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and index")
):
    """
    Upload and index a PDF document.
    
    The document will be processed through the offline pipeline:
    1. PDF parsing
    2. Chunking
    3. Embedding generation
    4. Vector storage
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Save uploaded file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
        file_path = Path(settings.UPLOAD_DIR) / safe_filename
        
        logger.info(f"Saving uploaded file: {safe_filename}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index the document
        logger.info(f"Indexing document: {file_path}")
        success, chunks_indexed = rag_pipeline.index_document(str(file_path))
        
        if success:
            return UploadResponse(
                filename=safe_filename,
                status="success",
                chunks_indexed=chunks_indexed,
                message=f"Successfully indexed {chunks_indexed} chunks"
            )
        else:
            # Clean up failed upload
            if file_path.exists():
                file_path.unlink()
            
            raise HTTPException(
                status_code=500,
                detail="Failed to index document"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/upload/batch", tags=["Documents"])
async def upload_documents_batch(
    files: List[UploadFile] = File(..., description="PDF files to upload and index"),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and index multiple PDF documents in batch.
    
    This endpoint processes files in the background to avoid timeout.
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Validate files
    pdf_files = []
    for file in files:
        if file.filename.lower().endswith('.pdf'):
            pdf_files.append(file)
        else:
            logger.warning(f"Skipping non-PDF file: {file.filename}")
    
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No valid PDF files provided")
    
    saved_paths = []
    try:
        # Save all files
        for file in pdf_files:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
            file_path = Path(settings.UPLOAD_DIR) / safe_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_paths.append(str(file_path))
        
        total_chunks = 0
        successful = 0
        
        # Index each document
        for file_path in saved_paths:
            success, chunks = rag_pipeline.index_document(file_path)
            if success:
                successful += 1
                total_chunks += chunks
        
        return {
            "status": "success",
            "files_processed": len(pdf_files),
            "files_indexed": successful,
            "total_chunks": total_chunks,
            "message": f"Indexed {successful}/{len(pdf_files)} files ({total_chunks} chunks)"
        }
        
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing files: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the RAG system.
    
    Ask questions about the indexed documents. The system will:
    1. Search for relevant context
    2. Rerank results
    3. Generate an answer using LLM
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        start_time = time.time()
        
        # Process the query
        answer, sources, processing_time_ms = rag_pipeline.chat(
            message=request.message,
            conversation_history=request.conversation_history,
            top_k=request.top_k
        )
        
        # Format response
        response = ChatResponse(
            answer=answer,
            sources=sources if request.include_sources else [],
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"Chat response generated in {processing_time_ms:.2f}ms "
            f"with {len(sources)} sources"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Semantic search in indexed documents.
    
    Returns relevant text chunks without LLM generation.
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Embed query
        query_embedding = rag_pipeline.embedding_service.embed_query(request.query)
        
        # Search vector DB
        results = rag_pipeline.vector_db.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = ChunkMetadata(
                source_file=result.get("source_file", ""),
                page_number=result.get("page_number"),
                chunk_index=result.get("chunk_index", 0)
            )
            
            chunk_response = ChunkResponse(
                id=result.get("id", ""),
                text=result.get("text", ""),
                metadata=metadata,
                score=result.get("score", 0.0)
            )
            formatted_results.append(chunk_response)
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_found=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get system statistics."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return rag_pipeline.get_stats()


@app.delete("/reset", tags=["System"])
async def reset_database():
    """
    Reset the vector database (delete all indexed documents).
    
    Use with caution - this will remove all indexed content.
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        success = rag_pipeline.vector_db.delete_collection()
        if success:
            # Recreate collection
            rag_pipeline.vector_db.create_collection()
            return {"status": "success", "message": "Database reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset database")
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
