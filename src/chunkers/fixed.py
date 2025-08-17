"""
Fixed-size chunker implementation.
"""
from typing import List
from .base import BaseChunker
from ..models.document import Document, DocumentChunk


class FixedChunker(BaseChunker):
    """
    Chunks documents into fixed-size pieces based on character count.
    
    This is the simplest chunking strategy that splits text into chunks
    of a fixed character length with optional overlap.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document into fixed-size pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        text = document.content
        chunks = []
        chunk_index = 0
        
        start = 0
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Extract chunk content
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunk = self._create_chunk(
                    content=chunk_content,
                    document=document,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next position with overlap
            if end >= len(text):
                break
                
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks


