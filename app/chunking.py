"""
Chunking service using LlamaIndex SentenceSplitter.
Splits documents into semantic chunks for embedding.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

# Check availability at module level
try:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import TextNode
    _llamaindex_available = True
except ImportError as e:
    _llamaindex_available = False
    logger.warning(f"LlamaIndex not installed. Using fallback chunking: {e}")

from app.config import settings


class ChunkingService:
    """Service for splitting text into semantic chunks."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = None
        
        if _llamaindex_available:
            try:
                self.splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    paragraph_splitter="\n\n",
                    sentence_splitter="."
                )
                logger.info(
                    f"SentenceSplitter initialized: "
                    f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize SentenceSplitter: {e}")

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        if _llamaindex_available and self.splitter:
            return self._chunk_with_llamaindex(text, metadata)
        else:
            return self._chunk_fallback(text, metadata)

    def _chunk_with_llamaindex(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use LlamaIndex SentenceSplitter for chunking."""
        try:
            node = TextNode(text=text, metadata=metadata)
            nodes = self.splitter.get_nodes_from_documents([node])
            
            chunks = []
            for i, node in enumerate(nodes):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                
                chunks.append({
                    "text": node.text,
                    "metadata": chunk_metadata
                })
            
            logger.debug(f"Split text into {len(chunks)} chunks using LlamaIndex")
            return chunks
            
        except Exception as e:
            logger.warning(f"LlamaIndex chunking failed: {e}, using fallback")
            return self._chunk_fallback(text, metadata)

    def _chunk_fallback(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback chunking using simple text splitting."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_index
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                    chunk_index += 1
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
        
        logger.debug(f"Split text into {len(chunks)} chunks using fallback method")
        return chunks

    def chunk_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts with 'text' and optional 'metadata'
            
        Returns:
            Flattened list of all chunks
        """
        all_chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks


# Singleton instance
_chunking_service: Optional[ChunkingService] = None


def get_chunking_service(
    chunk_size: int = None,
    chunk_overlap: int = None
) -> ChunkingService:
    """Get or create the chunking service singleton."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _chunking_service
