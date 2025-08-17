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
    Semantically-aware chunker that groups sentences by similarity using sentence embeddings.
    
    This chunker leverages pre-trained transformer-based sentence embedding models to understand
    the semantic meaning of text and create chunks that respect topical boundaries rather than
    arbitrary character limits.
    
    ## How Semantic Chunking Works:
    
    1. **Sentence Segmentation**: Text is first split into individual sentences using regex patterns
    2. **Embedding Generation**: Each sentence is converted to a high-dimensional vector (embedding)
       using a pre-trained SentenceTransformer model
    3. **Similarity Calculation**: Cosine similarity is computed between all sentence pairs
    4. **Semantic Grouping**: Sentences with similarity above a threshold are grouped together
    5. **Size Constraints**: Groups are formed while respecting maximum chunk size limits
    6. **Chunk Creation**: Final chunks are created from semantically coherent sentence groups
    
    ## SentenceTransformer Models:
    
    This implementation uses SentenceTransformer, a Python framework for sentence, text and image
    embeddings. These models are fine-tuned versions of transformer models (BERT, RoBERTa, etc.)
    specifically optimized for creating meaningful sentence-level representations.
    
    ### Supported Models:
    
    **Lightweight Models (Fast, Lower Quality):**
    - `all-MiniLM-L6-v2` (default): 384-dim, 22M parameters, balanced speed/quality
    - `all-MiniLM-L12-v2`: 384-dim, 33M parameters, better quality
    
    **Standard Models (Balanced):**
    - `all-mpnet-base-v2`: 768-dim, 109M parameters, high quality general-purpose
    - `sentence-transformers/paraphrase-mpnet-base-v2`: Similar to above
    
    **Multilingual Models:**
    - `paraphrase-multilingual-MiniLM-L12-v2`: 384-dim, supports 50+ languages
    - `paraphrase-multilingual-mpnet-base-v2`: 768-dim, higher quality multilingual
    
    **Specialized Models:**
    - `msmarco-distilbert-base-v4`: Optimized for search/retrieval tasks
    - `sentence-transformers/multi-qa-mpnet-base-dot-v1`: Question-answering optimized
    
    ## Embedding Models Explained:
    
    **What are Embedding Models?**
    Sentence embedding models are neural networks that convert text into dense numerical vectors
    (embeddings) that capture semantic meaning. Similar sentences produce similar vectors, enabling
    mathematical operations for similarity comparison.
    
    **Purpose in Semantic Chunking:**
    - **Semantic Understanding**: Unlike character-based chunking, embeddings understand meaning
    - **Topic Coherence**: Groups sentences discussing the same topic together
    - **Context Preservation**: Maintains logical flow within chunks
    - **Quality Retrieval**: Better chunk boundaries improve RAG system performance
    
    **How They're Used:**
    1. Each sentence becomes a vector (e.g., 384 or 768 dimensions)
    2. Cosine similarity measures how "close" two vectors are (0=different, 1=identical)
    3. Sentences with high similarity (>threshold) are grouped into the same chunk
    4. This preserves semantic coherence while respecting size constraints
    
    ## Parameters:
    
    Args:
        chunk_size (int): Maximum character length per chunk. Default: 1000
        chunk_overlap (int): Character overlap between consecutive chunks. Default: 200
        model_name (str): SentenceTransformer model identifier. Default: "all-MiniLM-L6-v2"
        similarity_threshold (float): Minimum similarity score to group sentences. Range: 0.0-1.0. Default: 0.7
        min_sentences_per_chunk (int): Minimum sentences required per chunk. Default: 2
    
    ## Performance Considerations:
    
    - **Speed**: Semantic chunking is slower than fixed/paragraph chunking due to embedding computation
    - **Memory**: Larger models (768-dim) use more memory than smaller ones (384-dim)
    - **Quality**: Higher-dimensional models generally produce better semantic understanding
    - **Caching**: Models are loaded once and reused for efficiency
    
    ## Example Usage:
    
    ```python
    # Fast, lightweight chunking
    chunker = SemanticChunker(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        chunk_size=800
    )
    
    # High-quality chunking
    chunker = SemanticChunker(
        model_name="all-mpnet-base-v2", 
        similarity_threshold=0.8,
        chunk_size=1200
    )
    
    # Multilingual support
    chunker = SemanticChunker(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold=0.7
    )
    ```
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        min_sentences_per_chunk: int = 2
    ):
        """
        Initialize the semantic chunker with embedding model and similarity parameters.
        
        Args:
            chunk_size: Maximum character length per chunk (default: 1000)
            chunk_overlap: Character overlap between consecutive chunks (default: 200)
            model_name: SentenceTransformer model identifier (default: "all-MiniLM-L6-v2")
                Common options:
                - "all-MiniLM-L6-v2": Fast, 384-dim (22M params) - recommended for speed
                - "all-mpnet-base-v2": High quality, 768-dim (109M params) - recommended for quality
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual, 384-dim
            similarity_threshold: Minimum cosine similarity to group sentences (0.0-1.0, default: 0.7)
                Lower values (0.5-0.6) = more diverse groupings
                Higher values (0.8-0.9) = stricter semantic similarity
            min_sentences_per_chunk: Minimum sentences required per chunk (default: 2)
        
        Note:
            The embedding model is loaded lazily on first use to avoid startup overhead.
            First-time model loading will download weights if not cached locally.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self._model: Optional[SentenceTransformer] = None
        
        # Enhanced sentence splitting pattern - handles various punctuation
        # Matches sentence endings followed by whitespace, excluding common abbreviations
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy loading of the SentenceTransformer embedding model.
        
        This property ensures the model is only loaded when actually needed, improving
        initialization time. The model is cached after first load for efficiency.
        
        Returns:
            SentenceTransformer: The loaded embedding model instance
            
        Note:
            First access will download the model if not cached (~20-500MB depending on model).
            Subsequent accesses reuse the cached model instance.
        """
        if self._model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk document using semantic similarity between sentences.
        
        This method implements the complete semantic chunking pipeline:
        
        1. **Text Preprocessing**: Split document into individual sentences
        2. **Embedding Generation**: Convert each sentence to vector representation using SentenceTransformer
        3. **Similarity Analysis**: Calculate cosine similarity between all sentence pairs
        4. **Semantic Grouping**: Group sentences with similarity above threshold
        5. **Size Management**: Ensure groups respect chunk size constraints
        6. **Chunk Assembly**: Create final chunks from semantically coherent groups
        
        The process preserves semantic boundaries while maintaining practical size limits,
        resulting in chunks that are more topically coherent than simple character-based splitting.
        
        Args:
            document: Document to chunk containing content and metadata
            
        Returns:
            List[DocumentChunk]: Semantically coherent chunks with inherited metadata
            
        Raises:
            Exception: If embedding model fails to load or process text
            
        Example:
            ```python
            doc = Document(content="AI is transforming healthcare. Machine learning helps doctors...")
            chunker = SemanticChunker(similarity_threshold=0.8)
            chunks = chunker.chunk(doc)
            # Results in chunks grouped by topic (AI/healthcare vs other topics)
            ```
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
        """
        Split text into individual sentences using regex-based segmentation.
        
        This method implements sentence boundary detection using regular expressions.
        While not as sophisticated as spaCy or NLTK's sentence tokenizers, it provides
        a good balance of speed and accuracy for most text types.
        
        The current pattern looks for sentence-ending punctuation (., !, ?) followed
        by whitespace and a capital letter, which handles most common cases while
        avoiding splits on abbreviations and decimals.
        
        For production use with complex text, consider alternatives:
        - spaCy: `nlp.pipe([text])[0].sents` (more accurate, handles edge cases)
        - NLTK: `sent_tokenize(text)` (good accuracy, handles abbreviations)
        - Custom rules: Domain-specific patterns for specialized text
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List[str]: Individual sentences with minimal length filtering applied
            
        Note:
            - Sentences shorter than 10 characters are filtered out
            - Leading/trailing whitespace is stripped from each sentence
            - Empty strings after processing are excluded
        """
        # Enhanced sentence splitting - looks for sentence endings followed by space and capital letter
        # This avoids splitting on abbreviations like "Dr. Smith" or decimals like "3.14"
        sentences = self.sentence_pattern.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter very short sentences that are likely fragments or noise
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _group_sentences_semantically(
        self, 
        sentences: List[str], 
        embeddings: np.ndarray
    ) -> List[List[int]]:
        """
        Group sentences based on semantic similarity using cosine similarity.
        
        This method implements a greedy clustering algorithm that groups sentences with
        high semantic similarity together while respecting size constraints.
        
        Algorithm:
        1. Iterate through each sentence as a potential group starter
        2. For each starter, find other sentences with similarity > threshold
        3. Add similar sentences to the group if size constraints allow
        4. Mark grouped sentences as used to avoid duplication
        5. Handle edge cases for very small or large groups
        
        The cosine similarity is calculated using sklearn's optimized implementation,
        which measures the cosine of the angle between embedding vectors:
        - Score of 1.0 = identical semantic meaning
        - Score of 0.0 = completely unrelated
        - Score of -1.0 = opposite meaning (rare in practice)
        
        Args:
            sentences: List of original sentence strings
            embeddings: NumPy array of sentence embeddings (shape: [n_sentences, embedding_dim])
                       Each row is a dense vector representation of the corresponding sentence
            
        Returns:
            List[List[int]]: Groups of sentence indices, where each inner list contains
                           indices of sentences that should be grouped together
                           
        Example:
            Input sentences: ["AI helps doctors", "Machine learning in medicine", "Weather is nice"]
            Embeddings: [[0.1, 0.2, ...], [0.15, 0.18, ...], [-0.5, 0.8, ...]]
            Output: [[0, 1], [2]]  # First two sentences grouped (medical topic), third separate
        """
        n_sentences = len(sentences)
        if n_sentences == 0:
            return []
        
        # Calculate similarity matrix using sklearn's optimized cosine similarity
        # This computes all pairwise similarities efficiently in vectorized operations
        similarity_matrix = cosine_similarity(embeddings)
        
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
                
                # Get similarity from precomputed matrix (more efficient)
                similarity = similarity_matrix[i, j]
                
                # Check if we should add this sentence to the current group
                potential_length = current_length + len(sentences[j])
                
                if (similarity >= self.similarity_threshold and 
                    potential_length <= self.chunk_size):
                    
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
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded embedding model.
        
        Returns:
            dict: Model information including name, dimensions, and parameters
        """
        if self._model is None:
            return {
                "model_name": self.model_name,
                "status": "not_loaded",
                "embedding_dimension": "unknown"
            }
        
        return {
            "model_name": self.model_name,
            "status": "loaded",
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(self._model, 'max_seq_length', 'unknown'),
            "similarity_threshold": self.similarity_threshold,
            "min_sentences_per_chunk": self.min_sentences_per_chunk
        }
    
    @staticmethod
    def get_recommended_models() -> dict:
        """
        Get information about recommended SentenceTransformer models for different use cases.
        
        Returns:
            dict: Recommended models categorized by use case with performance characteristics
        """
        return {
            "lightweight": {
                "all-MiniLM-L6-v2": {
                    "dimensions": 384,
                    "parameters": "22M",
                    "use_case": "Fast inference, good balance of speed/quality",
                    "languages": "English",
                    "speed": "⭐⭐⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐"
                },
                "all-MiniLM-L12-v2": {
                    "dimensions": 384,
                    "parameters": "33M", 
                    "use_case": "Better quality than L6, still fast",
                    "languages": "English",
                    "speed": "⭐⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐"
                }
            },
            "high_quality": {
                "all-mpnet-base-v2": {
                    "dimensions": 768,
                    "parameters": "109M",
                    "use_case": "High quality general-purpose embeddings",
                    "languages": "English",
                    "speed": "⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐⭐"
                },
                "sentence-transformers/paraphrase-mpnet-base-v2": {
                    "dimensions": 768,
                    "parameters": "109M",
                    "use_case": "Excellent for paraphrase detection and similarity",
                    "languages": "English", 
                    "speed": "⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐⭐"
                }
            },
            "multilingual": {
                "paraphrase-multilingual-MiniLM-L12-v2": {
                    "dimensions": 384,
                    "parameters": "118M",
                    "use_case": "50+ languages, good speed",
                    "languages": "Multilingual (50+)",
                    "speed": "⭐⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐"
                },
                "paraphrase-multilingual-mpnet-base-v2": {
                    "dimensions": 768,
                    "parameters": "278M",
                    "use_case": "High quality multilingual embeddings",
                    "languages": "Multilingual (50+)",
                    "speed": "⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐⭐"
                }
            },
            "specialized": {
                "msmarco-distilbert-base-v4": {
                    "dimensions": 768,
                    "parameters": "66M",
                    "use_case": "Optimized for search and retrieval tasks",
                    "languages": "English",
                    "speed": "⭐⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐⭐ (for search)"
                },
                "multi-qa-mpnet-base-dot-v1": {
                    "dimensions": 768,
                    "parameters": "109M",
                    "use_case": "Question-answering and QA retrieval",
                    "languages": "English",
                    "speed": "⭐⭐⭐",
                    "quality": "⭐⭐⭐⭐⭐ (for QA)"
                }
            }
        }
    
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

