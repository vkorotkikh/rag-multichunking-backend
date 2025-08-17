"""
RAG Multi-Chunking Backend

A comprehensive RAG application backend supporting multiple chunking strategies,
vector stores, and reranking capabilities.
"""

from .rag_pipeline import RAGPipeline, create_rag_pipeline
from .models.document import (
    Document, DocumentChunk, QueryRequest, QueryResponse,
    IndexRequest, IndexResponse, SearchResult,
    ChunkingStrategy, VectorStore
)
from .chunkers import (
    BaseChunker, FixedChunker, ParagraphChunker, 
    RecursiveChunker, SemanticChunker, create_chunker
)
from .vector_stores import (
    BaseVectorStore, PineconeVectorStore, WeaviateVectorStore,
    create_vector_store, create_vector_store_from_settings
)
from .rerankers import (
    BaseReranker, CohereReranker, BM25Reranker,
    RerankerType, create_reranker, create_reranker_from_settings
)
from .utils.embeddings import EmbeddingService, get_embedding_service
from .config.settings import Settings, get_settings, update_settings

__version__ = "1.0.0"

__all__ = [
    # Core pipeline
    "RAGPipeline",
    "create_rag_pipeline",
    
    # Models
    "Document",
    "DocumentChunk", 
    "QueryRequest",
    "QueryResponse",
    "IndexRequest",
    "IndexResponse",
    "SearchResult",
    "ChunkingStrategy",
    "VectorStore",
    
    # Chunkers
    "BaseChunker",
    "FixedChunker",
    "ParagraphChunker",
    "RecursiveChunker", 
    "SemanticChunker",
    "create_chunker",
    
    # Vector Stores
    "BaseVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "create_vector_store",
    "create_vector_store_from_settings",
    
    # Rerankers
    "BaseReranker",
    "CohereReranker",
    "BM25Reranker",
    "RerankerType",
    "create_reranker",
    "create_reranker_from_settings",
    
    # Utils
    "EmbeddingService",
    "get_embedding_service",
    
    # Config
    "Settings",
    "get_settings",
    "update_settings",
]


