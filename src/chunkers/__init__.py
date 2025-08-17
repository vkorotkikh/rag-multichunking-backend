"""
Chunkers module for different chunking strategies.
"""
from typing import Dict, Type
from .base import BaseChunker
from .fixed import FixedChunker
from .paragraph import ParagraphChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from ..models.document import ChunkingStrategy


class ChunkerFactory:
    """Factory class for creating chunkers."""
    
    _chunkers: Dict[ChunkingStrategy, Type[BaseChunker]] = {
        ChunkingStrategy.FIXED: FixedChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
    }
    
    @classmethod
    def create_chunker(
        self,
        strategy: ChunkingStrategy,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> BaseChunker:
        """
        Create a chunker instance based on the specified strategy.
        
        Args:
            strategy: The chunking strategy to use
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            **kwargs: Additional arguments specific to the chunker
            
        Returns:
            Chunker instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy not in self._chunkers:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
        chunker_class = self._chunkers[strategy]
        return chunker_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> list[ChunkingStrategy]:
        """Get list of available chunking strategies."""
        return list(cls._chunkers.keys())


# Convenience functions
def create_chunker(strategy: ChunkingStrategy, **kwargs) -> BaseChunker:
    """Create a chunker instance."""
    return ChunkerFactory.create_chunker(strategy, **kwargs)


def get_available_strategies() -> list[ChunkingStrategy]:
    """Get available chunking strategies."""
    return ChunkerFactory.get_available_strategies()


__all__ = [
    "BaseChunker",
    "FixedChunker", 
    "ParagraphChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "ChunkerFactory",
    "create_chunker",
    "get_available_strategies"
]


