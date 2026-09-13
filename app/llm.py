"""
LLM service for generating responses using vLLM or Ollama.
Supports OpenAI-compatible API endpoints.
"""
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
from loguru import logger

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI client not installed. LLM inference will be disabled.")

from app.config import settings
from app.schemas import ChatMessage


class LLMService:
    """Service for LLM inference using vLLM or Ollama."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "not-needed"
    ):
        self.api_base = api_base or settings.LLM_API_BASE
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.api_key = api_key
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        self._available = False

    def initialize(self) -> bool:
        """Initialize the LLM client."""
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI client not available")
            return False

        try:
            logger.info(f"Initializing LLM client: {self.api_base}")
            
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
            
            self.async_client = AsyncOpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
            
            # Test connection
            self._check_connection()
            self._available = True
            logger.info(f"LLM client initialized successfully: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._available

    def _check_connection(self) -> bool:
        """Check if LLM server is reachable."""
        try:
            models = self.client.models.list()
            logger.debug(f"Available models: {[m.id for m in models]}")
            return True
        except Exception as e:
            logger.warning(f"LLM server connection check failed: {e}")
            return False

    def generate_response(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self._available:
            raise RuntimeError("LLM service not available")

        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE

        # Build message list with system prompt
        openai_messages = []
        
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        try:
            logger.debug(f"Generating response with {len(messages)} messages")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            answer = response.choices[0].message.content
            logger.debug(f"Generated response: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_streaming(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Chunks of generated text
        """
        if not self._available:
            raise RuntimeError("LLM service not available")

        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = temperature or settings.LLM_TEMPERATURE

        # Build message list with system prompt
        openai_messages = []
        
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        try:
            stream = self.async_client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in await stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[ChatMessage] = None
    ) -> str:
        """
        Generate a RAG-based response with context.
        
        Args:
            query: User's question
            context_chunks: List of context chunks with text and metadata
            conversation_history: Optional conversation history
            
        Returns:
            Generated response with citations
        """
        # Build context string
        context_text = self._format_context(context_chunks)
        
        # Build system prompt
        system_prompt = self._build_system_prompt()
        
        # Build user prompt with context
        user_prompt = f"""Context information from documents:
{context_text}

Based on the context above, answer the following question. If the answer cannot be found in the context, say so.
Question: {query}
Answer:"""

        # Build messages
        messages = []
        if conversation_history:
            messages = conversation_history
        messages.append(ChatMessage(role="user", content=user_prompt))

        return self.generate_response(
            messages=messages,
            system_prompt=system_prompt
        )

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_file", "Unknown")
            page = chunk.get("page_number", "")
            page_info = f" (page {page})" if page else ""
            
            formatted.append(
                f"[Source {i}]{page_info}:\n{chunk.get('text', '')}\n"
            )
        
        return "\n".join(formatted)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for RAG."""
        return """You are a helpful assistant that answers questions based on the provided context.
- Always base your answers on the given context documents
- Cite your sources when possible (e.g., "According to document X...")
- If the context doesn't contain enough information, acknowledge this
- Be concise but thorough
- Use markdown formatting for better readability"""


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
