"""
Utility modules for RAG multi-chunking backend.
"""

from .embeddings import (
    EmbeddingService,
    get_embedding_service,
    set_embedding_service,
    embed_text,
    embed_texts,
    embed_chunks
)

__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "set_embedding_service",
    "embed_text",
    "embed_texts", 
    "embed_chunks"
]


