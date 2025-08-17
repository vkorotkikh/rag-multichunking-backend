"""
Paragraph-based chunker implementation.
"""
import re
from typing import List
from .base import BaseChunker
from ..models.document import Document, DocumentChunk


class ParagraphChunker(BaseChunker):
    """
    Chunks documents based on paragraph boundaries.
    
    This chunker respects natural paragraph breaks and combines
    paragraphs to reach the target chunk size while maintaining
    semantic coherence.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        # Pattern to split on paragraph boundaries
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document based on paragraph boundaries.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        text = document.content
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        if not paragraphs:
            return []
        
        chunks = []
        chunk_index = 0
        current_chunk = ""
        current_start = 0
        paragraph_start = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                paragraph_start += len(paragraphs[i]) + 2  # +2 for \n\n
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = paragraph_start
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = self._create_chunk(
                        content=current_chunk,
                        document=document,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with current paragraph
                current_chunk = paragraph
                current_start = paragraph_start
            
            paragraph_start += len(paragraphs[i]) + 2  # +2 for \n\n
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                content=current_chunk,
                document=document,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        # Apply overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, document)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = self.paragraph_pattern.split(text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _apply_overlap(self, chunks: List[DocumentChunk], document: Document) -> List[DocumentChunk]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap text from previous chunk
            overlap_text = prev_chunk.content[-self.chunk_overlap:] if len(prev_chunk.content) > self.chunk_overlap else prev_chunk.content
            
            # Create new chunk with overlap
            new_content = overlap_text + "\n\n" + current_chunk.content
            
            new_chunk = self._create_chunk(
                content=new_content,
                document=document,
                chunk_index=current_chunk.chunk_index,
                start_char=current_chunk.start_char,
                end_char=current_chunk.end_char
            )
            
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks


