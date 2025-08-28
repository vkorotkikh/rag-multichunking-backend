"""
Embedding utilities for generating document and query embeddings.

Supports multiple embedding providers:
- OpenAI (text-embedding-3-large, text-embedding-3-small, etc.)
- SentenceTransformers (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- HuggingFace Transformers (any text embedding model)
- BGE (BAAI/bge-large-en-v1.5, etc.)
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import warnings

# Optional imports with fallbacks
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI not available. Install with: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("SentenceTransformers not available. Install with: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Install with: pip install transformers")

from ..config.settings import get_settings
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Base class for all embedding services."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.model = None
        self.dimension = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        pass
        
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text synchronously."""
        pass
        
    @abstractmethod
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        pass
        
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts synchronously."""
        pass
        
    @abstractmethod
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        pass
        
    async def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to document chunks."""
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.generate_embeddings_batch_async(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.model_copy()
            embedded_chunk.embedding = embedding
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
        
    def get_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self.dimension


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Service for generating embeddings using OpenAI."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
            
        settings = get_settings()
        openai_config = settings.openai
        
        if openai_config is None:
            # Use defaults if no OpenAI config
            model_name = model or "text-embedding-3-large"
            self.api_key = api_key
            self.chunk_size = 8191
            self.batch_size = 100
        else:
            model_name = model or openai_config.embedding_model
            self.api_key = api_key or openai_config.api_key
            self.chunk_size = openai_config.chunk_size
            self.batch_size = openai_config.batch_size
            
        super().__init__(model_name)
        
        # Set dimension based on model
        self.dimension = self._get_openai_dimension(model_name)
        
        # Initialize clients
        self.client = None
        self.async_client = None
        
    async def initialize(self) -> None:
        """Initialize OpenAI clients."""
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
    def _get_openai_dimension(self, model: str) -> int:
        """Get embedding dimension for OpenAI models."""
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, 1536)  # Default to ada-002 dimension
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text synchronously."""
        if not self.client:
            asyncio.run(self.initialize())
            
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text.replace("\n", " ")  # OpenAI recommends replacing newlines
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return []

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        if not self.async_client:
            await self.initialize()
            
        try:
            response = await self.async_client.embeddings.create(
                model=self.model_name,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {e}")
            return []
            
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches synchronously."""
        if not texts:
            return []
            
        if not self.client:
            asyncio.run(self.initialize())
            
        batch_size = batch_size or self.batch_size
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Clean texts
                cleaned_batch = [text.replace("\n", " ") for text in batch]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error generating OpenAI batch embeddings: {e}")
                embeddings.extend([[] for _ in batch])
        
        return embeddings
        
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches asynchronously."""
        if not texts:
            return []
            
        if not self.async_client:
            await self.initialize()
            
        batch_size = batch_size or self.batch_size
        max_concurrent = max_concurrent or 5
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch: List[str]) -> List[List[float]]:
            async with semaphore:
                try:
                    cleaned_batch = [text.replace("\n", " ") for text in batch]
                    
                    response = await self.async_client.embeddings.create(
                        model=self.model_name,
                        input=cleaned_batch
                    )
                    
                    return [data.embedding for data in response.data]
                    
                except Exception as e:
                    logger.error(f"Error generating OpenAI batch embeddings: {e}")
                    return [[] for _ in batch]
        
        # Process all batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results
        embeddings = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)
        
        return embeddings


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Service for generating embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
            
        settings = get_settings()
        model_name = model_name or settings.embedding.model_name
        super().__init__(model_name)
        
        self.device = device or settings.embedding.device
        self.normalize_embeddings = settings.embedding.normalize_embeddings
        self.batch_size = settings.embedding.batch_size
        self.max_length = settings.embedding.max_length
        
    async def initialize(self) -> None:
        """Initialize SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized SentenceTransformer model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model {self.model_name}: {e}")
            raise
            
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text synchronously."""
        if not self.model:
            asyncio.run(self.initialize())
            
        try:
            embedding = self.model.encode(
                text, 
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating SentenceTransformer embedding: {e}")
            return []
            
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        # SentenceTransformers doesn't have async API, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding, text)
        
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts synchronously."""
        if not texts:
            return []
            
        if not self.model:
            asyncio.run(self.initialize())
            
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating SentenceTransformer batch embeddings: {e}")
            return [[] for _ in texts]
            
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        # SentenceTransformers doesn't have async API, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings_batch, texts, batch_size)


class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """Service for generating embeddings using HuggingFace Transformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError("Transformers and PyTorch required. Install with: pip install transformers torch")
            
        settings = get_settings()
        model_name = model_name or settings.embedding.model_name
        super().__init__(model_name)
        
        self.device = device or settings.embedding.device
        self.normalize_embeddings = settings.embedding.normalize_embeddings
        self.batch_size = settings.embedding.batch_size
        self.max_length = settings.embedding.max_length
        self.trust_remote_code = settings.embedding.trust_remote_code
        
        self.tokenizer = None
        
    async def initialize(self) -> None:
        """Initialize HuggingFace model and tokenizer."""
        try:
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=self.trust_remote_code
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=self.trust_remote_code
            )
            
            # Move to device
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            
            # Estimate dimension (run a test encoding)
            test_encoding = self._encode_text("test")
            self.dimension = len(test_encoding)
            
            logger.info(f"Initialized HuggingFace model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model {self.model_name}: {e}")
            raise
            
    def _encode_text(self, text: str) -> List[float]:
        """Encode single text using HuggingFace model."""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        
        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean pooling of last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
            
        return embeddings.cpu().numpy().tolist()
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text synchronously."""
        if not self.model:
            asyncio.run(self.initialize())
            
        try:
            return self._encode_text(text)
        except Exception as e:
            logger.error(f"Error generating HuggingFace embedding: {e}")
            return []
            
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding for a single text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding, text)
        
    def generate_embeddings_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts synchronously."""
        if not texts:
            return []
            
        if not self.model:
            asyncio.run(self.initialize())
            
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self._encode_text(text)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error generating HuggingFace embedding: {e}")
                    batch_embeddings.append([])
                    
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
        
    async def generate_embeddings_batch_async(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings_batch, texts, batch_size)


class BGEEmbeddingService(SentenceTransformerEmbeddingService):
    """Service for generating embeddings using BGE models via SentenceTransformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, **kwargs):
        # Default to a popular BGE model if not specified
        if model_name is None:
            model_name = "BAAI/bge-large-en-v1.5"
        super().__init__(model_name, device, **kwargs)
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with BGE query instruction for better retrieval."""
        if not self.model:
            asyncio.run(self.initialize())
            
        try:
            # Add BGE instruction for queries (optional enhancement)
            if len(text) < 100:  # Assume short text is a query
                text = f"Represent this sentence for searching relevant passages: {text}"
                
            embedding = self.model.encode(
                text, 
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating BGE embedding: {e}")
            return []


class EmbeddingServiceFactory:
    """Factory for creating embedding services based on configuration."""
    
    @staticmethod
    def create_service(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingService:
        """Create an embedding service based on provider."""
        settings = get_settings()
        provider = provider or settings.embedding.provider
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")
            return OpenAIEmbeddingService(model=model_name, **kwargs)
            
        elif provider == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
            return SentenceTransformerEmbeddingService(model_name=model_name, **kwargs)
            
        elif provider == "huggingface":
            if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
                raise ImportError("Transformers and PyTorch required. Install with: pip install transformers torch")
            return HuggingFaceEmbeddingService(model_name=model_name, **kwargs)
            
        elif provider == "bge":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("SentenceTransformers not available for BGE. Install with: pip install sentence-transformers")
            return BGEEmbeddingService(model_name=model_name, **kwargs)
            
        else:
            raise ValueError(f"Unknown embedding provider: {provider}. Supported: openai, sentence-transformers, huggingface, bge")


# Legacy class name for backwards compatibility
EmbeddingService = OpenAIEmbeddingService

# Global embedding service instance
_embedding_service: Optional[BaseEmbeddingService] = None


def get_embedding_service() -> BaseEmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingServiceFactory.create_service()
        # Schedule initialization
        try:
            asyncio.create_task(_embedding_service.initialize())
        except RuntimeError:
            # If no event loop is running, initialize will happen on first use
            pass
    return _embedding_service


def set_embedding_service(service: BaseEmbeddingService) -> None:
    """Set global embedding service instance."""
    global _embedding_service
    _embedding_service = service


def create_embedding_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingService:
    """Create a new embedding service with specified parameters."""
    return EmbeddingServiceFactory.create_service(provider, model_name, **kwargs)


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