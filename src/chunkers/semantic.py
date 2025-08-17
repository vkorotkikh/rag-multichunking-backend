"""
Semantic chunker implementation using sentence embeddings.

This module implements semantic chunking, which groups text based on semantic similarity
rather than just character count or structural boundaries. It uses pre-trained sentence
embedding models to understand the meaning of sentences and groups similar ones together.

The semantic chunking process involves:
1. Splitting text into individual sentences
2. Converting each sentence into a high-dimensional vector (embedding) using SentenceTransformer
3. Calculating similarity between sentence embeddings using cosine similarity
4. Grouping sentences with high similarity scores together
5. Ensuring chunks respect size constraints while maintaining semantic coherence

This approach produces more coherent chunks that preserve topical boundaries and
improve retrieval quality in RAG systems.
"""
import re
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseChunker
from ..models.document import Document, DocumentChunk


class SemanticChunker(BaseChunker):
    """
    Semantically-aware chunker that groups sentences by similarity.
    
    This chunker uses sentence embeddings to group semantically similar
    sentences together, creating more coherent chunks that respect
    semantic boundaries rather than just text length.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        min_sentences_per_chunk: int = 2
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self._model: Optional[SentenceTransformer] = None
        
        # Sentence splitting pattern
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document using semantic similarity.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        text = document.content
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # If we have very few sentences, use simpler chunking
        if len(sentences) < self.min_sentences_per_chunk * 2:
            return self._simple_chunk_fallback(document, sentences)
        
        # Generate embeddings for sentences
        embeddings = self.model.encode(sentences)
        
        # Group sentences semantically
        sentence_groups = self._group_sentences_semantically(sentences, embeddings)
        
        # Create chunks from sentence groups
        chunks = self._create_chunks_from_groups(sentence_groups, document, text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Basic sentence splitting (can be improved with spaCy or NLTK)
        sentences = self.sentence_pattern.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_sentences_semantically(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[List[int]]:
        """
        Group sentences based on semantic similarity.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings
            
        Returns:
            List of sentence groups (indices)
        """
        n_sentences = len(sentences)
        if n_sentences == 0:
            return []
        
        # Initialize groups
        groups = []
        used_sentences = set()
        
        for i in range(n_sentences):
            if i in used_sentences:
                continue
            
            # Start new group with current sentence
            current_group = [i]
            used_sentences.add(i)
            current_length = len(sentences[i])
            
            # Find similar sentences to add to this group
            for j in range(i + 1, n_sentences):
                if j in used_sentences:
                    continue
                
                # Calculate similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                
                # Check if we should add this sentence to the current group
                potential_length = current_length + len(sentences[j])
                
                if (similarity >= self.similarity_threshold and 
                    potential_length <= self.chunk_size and
                    len(current_group) >= self.min_sentences_per_chunk):
                    
                    current_group.append(j)
                    used_sentences.add(j)
                    current_length = potential_length
            
            # Add group if it meets minimum requirements
            if len(current_group) >= self.min_sentences_per_chunk or current_length >= self.chunk_size // 2:
                groups.append(current_group)
            else:
                # If group is too small, try to merge with previous group
                if groups and len(groups[-1]) < self.chunk_size // len(sentences[groups[-1][0]]):
                    groups[-1].extend(current_group)
                else:
                    groups.append(current_group)
        
        return groups
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _create_chunks_from_groups(
        self, 
        sentence_groups: List[List[int]], 
        document: Document,
        original_text: str
    ) -> List[DocumentChunk]:
        """Create document chunks from sentence groups."""
        chunks = []
        sentences = self._split_into_sentences(original_text)
        
        for chunk_index, group in enumerate(sentence_groups):
            if not group:
                continue
            
            # Combine sentences in the group
            group_sentences = [sentences[i] for i in sorted(group)]
            chunk_content = " ".join(group_sentences)
            
            # Calculate character positions (approximate)
            start_char = self._find_text_position(original_text, group_sentences[0])
            end_char = start_char + len(chunk_content)
            
            chunk = self._create_chunk(
                content=chunk_content,
                document=document,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char
            )
            chunks.append(chunk)
        
        # Apply overlap if needed
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_semantic_overlap(chunks, document)
        
        return chunks
    
    def _find_text_position(self, text: str, sentence: str) -> int:
        """Find the position of a sentence in the original text."""
        try:
            return text.index(sentence)
        except ValueError:
            return 0  # Fallback if exact match not found
    
    def _apply_semantic_overlap(self, chunks: List[DocumentChunk], document: Document) -> List[DocumentChunk]:
        """Apply overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Get overlap text from previous chunk
            overlap_words = prev_chunk.content.split()[-self.chunk_overlap//10:]  # Rough word-based overlap
            overlap_text = " ".join(overlap_words) if overlap_words else ""
            
            # Create new chunk with overlap
            new_content = overlap_text + " " + current_chunk.content if overlap_text else current_chunk.content
            
            new_chunk = self._create_chunk(
                content=new_content,
                document=document,
                chunk_index=current_chunk.chunk_index,
                start_char=current_chunk.start_char,
                end_char=current_chunk.end_char
            )
            
            overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _simple_chunk_fallback(self, document: Document, sentences: List[str]) -> List[DocumentChunk]:
        """Fallback to simple chunking when we have too few sentences."""
        text = " ".join(sentences)
        chunks = []
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
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
            
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks

