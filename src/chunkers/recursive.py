"""
Recursive chunker implementation.
"""
from typing import List, Optional
from .base import BaseChunker
from ..models.document import Document, DocumentChunk


class RecursiveChunker(BaseChunker):
    """
    Recursively chunks documents using a hierarchy of separators.
    
    This chunker attempts to split text at natural boundaries by trying
    different separators in order of preference (paragraphs, sentences, words, characters).
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation endings
            "? ",    # Question endings
            "; ",    # Semicolon breaks
            ", ",    # Comma breaks
            " ",     # Word breaks
            ""       # Character breaks (fallback)
        ]
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Recursively chunk document using hierarchical separators.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        text = document.content
        if not text.strip():
            return []
        
        # Get initial splits
        text_splits = self._split_text(text, self.separators)
        
        chunks = []
        chunk_index = 0
        char_position = 0
        
        for split in text_splits:
            if not split.strip():
                char_position += len(split)
                continue
                
            chunk = self._create_chunk(
                content=split.strip(),
                document=document,
                chunk_index=chunk_index,
                start_char=char_position,
                end_char=char_position + len(split)
            )
            chunks.append(chunk)
            chunk_index += 1
            char_position += len(split)
        
        return chunks
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        
        Args:
            text: Text to split
            separators: List of separators to try in order
            
        Returns:
            List of text chunks
        """
        final_chunks = []
        
        # Start with the full text
        splits = [text]
        
        for separator in separators:
            new_splits = []
            
            for split in splits:
                # If chunk is already small enough, keep it
                if len(split) <= self.chunk_size:
                    new_splits.append(split)
                else:
                    # Try to split with current separator
                    if separator == "":
                        # Character-level splitting (fallback)
                        new_splits.extend(self._split_by_characters(split))
                    else:
                        # Split by separator
                        sub_splits = self._split_by_separator(split, separator)
                        new_splits.extend(sub_splits)
            
            splits = new_splits
            
            # If all chunks are small enough, we're done
            if all(len(split) <= self.chunk_size for split in splits):
                break
        
        # Merge small adjacent chunks and apply overlap
        final_chunks = self._merge_small_chunks(splits)
        final_chunks = self._apply_overlap_to_chunks(final_chunks)
        
        return final_chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a specific separator."""
        if separator not in text:
            return [text]
        
        parts = text.split(separator)
        
        # Keep separator with the parts (except the last one)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            else:
                result.append(part)
        
        return [part for part in result if part.strip()]
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by characters as a fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small adjacent chunks to better utilize chunk size."""
        if not chunks:
            return []
        
        merged = []
        current_chunk = ""
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Check if we can merge with current chunk
            potential_chunk = current_chunk + (" " if current_chunk else "") + chunk
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = chunk
        
        # Add final chunk
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def _apply_overlap_to_chunks(self, chunks: List[str]) -> List[str]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap from previous chunk
            overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
            overlap_text = prev_chunk[overlap_start:]
            
            # Create overlapped chunk
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped.append(overlapped_chunk)
        
        return overlapped

