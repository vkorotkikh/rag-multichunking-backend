"""
Embedding utilities for generating document and query embeddings.
"""
import asyncio
from typing import List, Optional, Union
from openai import OpenAI, AsyncOpenAI
from ..config.settings import get_settings
from ..models.document import DocumentChunk

# TODO: Add support for other embedding providers

class OpenAIEmbeddingService:
    """Service for generating embeddings using OpenAI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()

        self.api_key = api_key or settings.openai.api_key
        self.model = model or settings.openai.embedding_model
        
        # Initialize both sync and async clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Is there anything else that I need to do here?
        # adding chunk size and batch size for openai embedding
        self.chunk_size = settings.openai.chunk_size
        self.batch_size = settings.openai.batch_size
        
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text synchronously.

        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.replace("\n", " ")  # OpenAI recommends replacing newlines
            )

class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        
        self.api_key = api_key or settings.openai.api_key
        self.model = model or settings.openai.embedding_model
        
        # Initialize both sync and async clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text synchronously.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.replace("\n", " ")  # OpenAI recommends replacing newlines
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches synchronously.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Clean texts
                cleaned_batch = [text.replace("\n", " ") for text in batch]
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error generating batch embeddings: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches asynchronously.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[str]) -> List[List[float]]:
            async with semaphore:
                try:
                    # Clean texts
                    cleaned_batch = [text.replace("\n", " ") for text in batch]
                    
                    response = await self.async_client.embeddings.create(
                        model=self.model,
                        input=cleaned_batch
                    )
                    
                    return [data.embedding for data in response.data]
                    
                except Exception as e:
                    print(f"Error generating batch embeddings: {e}")
                    return [[] for _ in batch]
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        embeddings = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)
        
        return embeddings
    
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Add embeddings to document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings_batch_async(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create a copy of the chunk with embedding
            embedded_chunk = chunk.model_copy()
            embedded_chunk.embedding = embedding
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def set_embedding_service(service: EmbeddingService) -> None:
    """Set global embedding service instance."""
    global _embedding_service
    _embedding_service = service


# Convenience functions
async def embed_text(text: str) -> List[float]:
    """Generate embedding for text."""
    service = get_embedding_service()
    return await service.generate_embedding_async(text)


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    service = get_embedding_service()
    return await service.generate_embeddings_batch_async(texts)


async def embed_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Add embeddings to document chunks."""
    service = get_embedding_service()
    return await service.embed_chunks(chunks)


