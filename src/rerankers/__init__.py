"""
Rerankers module for different reranking strategies.
"""
from enum import Enum
from typing import Dict, Type, Optional
from .base import BaseReranker
from .cohere_reranker import CohereReranker, CohereRerankerV2, CohereMultilingualReranker
from .bm25_reranker import BM25Reranker, BM25PlusReranker


class RerankerType(str, Enum):
    """Available reranker types."""
    COHERE = "cohere"
    COHERE_V2 = "cohere_v2"
    COHERE_MULTILINGUAL = "cohere_multilingual"
    BM25 = "bm25"
    BM25_PLUS = "bm25_plus"


class RerankerFactory:
    """Factory class for creating reranker instances."""
    
    _rerankers: Dict[RerankerType, Type[BaseReranker]] = {
        RerankerType.COHERE: CohereReranker,
        RerankerType.COHERE_V2: CohereRerankerV2,
        RerankerType.COHERE_MULTILINGUAL: CohereMultilingualReranker,
        RerankerType.BM25: BM25Reranker,
        RerankerType.BM25_PLUS: BM25PlusReranker,
    }
    
    @classmethod
    def create_reranker(
        cls,
        reranker_type: RerankerType,
        top_k: int = 10,
        relevance_threshold: float = 0.0,
        **kwargs
    ) -> BaseReranker:
        """
        Create a reranker instance based on the specified type.
        
        Args:
            reranker_type: The reranker type to create
            top_k: Number of top results to return
            relevance_threshold: Minimum relevance threshold
            **kwargs: Additional arguments specific to the reranker
            
        Returns:
            Reranker instance
            
        Raises:
            ValueError: If reranker type is not supported
        """
        if reranker_type not in cls._rerankers:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")
        
        reranker_class = cls._rerankers[reranker_type]
        
        # Handle different constructor signatures
        if reranker_type in [RerankerType.COHERE, RerankerType.COHERE_V2, RerankerType.COHERE_MULTILINGUAL]:
            return reranker_class(
                top_k=top_k,
                relevance_threshold=relevance_threshold,
                **kwargs
            )
        else:  # BM25 variants
            return reranker_class(
                top_k=top_k,
                relevance_threshold=relevance_threshold,
                **kwargs
            )
    
    @classmethod
    def create_from_settings(cls, reranker_type: Optional[RerankerType] = None) -> BaseReranker:
        """
        Create a reranker instance from application settings.
        
        Args:
            reranker_type: Override the reranker type
            
        Returns:
            Configured reranker instance
        """
        from ..config.settings import get_settings
        settings = get_settings()
        
        actual_type = reranker_type or RerankerType.COHERE
        
        return cls.create_reranker(
            reranker_type=actual_type,
            top_k=settings.reranker.top_k,
            relevance_threshold=settings.reranker.relevance_threshold
        )
    
    @classmethod
    def get_available_rerankers(cls) -> list[RerankerType]:
        """Get list of available reranker types."""
        return list(cls._rerankers.keys())


# Convenience functions
def create_reranker(reranker_type: RerankerType, **kwargs) -> BaseReranker:
    """Create a reranker instance."""
    return RerankerFactory.create_reranker(reranker_type, **kwargs)


def create_reranker_from_settings(reranker_type: Optional[RerankerType] = None) -> BaseReranker:
    """Create a reranker instance from settings."""
    return RerankerFactory.create_from_settings(reranker_type)


def get_available_rerankers() -> list[RerankerType]:
    """Get available reranker types."""
    return RerankerFactory.get_available_rerankers()


__all__ = [
    "BaseReranker",
    "CohereReranker",
    "CohereRerankerV2", 
    "CohereMultilingualReranker",
    "BM25Reranker",
    "BM25PlusReranker",
    "RerankerType",
    "RerankerFactory",
    "create_reranker",
    "create_reranker_from_settings",
    "get_available_rerankers"
]


