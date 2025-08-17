"""
Base chunker interface and common utilities.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.document import Document, DocumentChunk


class BaseChunker(ABC):
    """Base class for all chunking strategies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        pass
    
    def _create_chunk(
        self, 
        content: str, 
        document: Document, 
        chunk_index: int,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None
    ) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata inheritance."""
        return DocumentChunk(
            content=content,
            metadata=document.metadata.model_copy(),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char
        )
    
    def _get_text_length(self, text: str) -> int:
        """Get text length (can be overridden for token-based chunking)."""
        return len(text)
    
    def _split_with_overlap(self, text: str, separators: List[str]) -> List[str]:
        """Split text with overlap using given separators."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Find the best split point using separators
            best_split = end
            for separator in separators:
                # Look for separator near the end position
                sep_pos = text.rfind(separator, start, end)
                if sep_pos != -1 and sep_pos > start:
                    best_split = sep_pos + len(separator)
                    break
            
            chunks.append(text[start:best_split])
            
            # Calculate next start position with overlap
            next_start = max(start + 1, best_split - self.chunk_overlap)
            start = next_start
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]


