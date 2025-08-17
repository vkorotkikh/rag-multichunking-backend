"""
Basic tests for RAG Multi-Chunking Backend components.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.document import Document, DocumentMetadata, ChunkingStrategy
from chunkers import create_chunker
from chunkers.fixed import FixedChunker
from chunkers.paragraph import ParagraphChunker
from chunkers.recursive import RecursiveChunker


class TestDocumentModels:
    """Test document models and validation."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = Document(
            content="This is a test document.",
            metadata=DocumentMetadata(
                source="test.txt",
                title="Test Document"
            )
        )
        
        assert doc.content == "This is a test document."
        assert doc.metadata.source == "test.txt"
        assert doc.metadata.title == "Test Document"
    
    def test_chunking_strategy_enum(self):
        """Test chunking strategy enum values."""
        assert ChunkingStrategy.FIXED == "fixed"
        assert ChunkingStrategy.PARAGRAPH == "paragraph"
        assert ChunkingStrategy.SEMANTIC == "semantic"
        assert ChunkingStrategy.RECURSIVE == "recursive"


class TestChunkers:
    """Test chunking implementations."""
    
    def setup_method(self):
        """Set up test data."""
        self.test_content = """
        This is the first paragraph of our test document. It contains some basic information
        about testing and should be split appropriately by our chunking algorithms.
        
        This is the second paragraph. It discusses different aspects of the document
        and provides more context for our chunking tests.
        
        The third paragraph concludes our test document. It wraps up the discussion
        and provides a natural ending point for our content.
        """
        
        self.test_document = Document(
            content=self.test_content.strip(),
            metadata=DocumentMetadata(source="test.txt")
        )
    
    def test_fixed_chunker(self):
        """Test fixed chunker functionality."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(self.test_document)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk.chunk_index, int) for chunk in chunks)
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # Size + some tolerance
    
    def test_paragraph_chunker(self):
        """Test paragraph chunker functionality.""" 
        chunker = ParagraphChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(self.test_document)
        
        assert len(chunks) > 0
        assert all(chunk.content.strip() for chunk in chunks)  # No empty chunks
    
    def test_recursive_chunker(self):
        """Test recursive chunker functionality."""
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.chunk(self.test_document)
        
        assert len(chunks) > 0
        assert all(chunk.metadata.source == "test.txt" for chunk in chunks)
    
    def test_chunker_factory(self):
        """Test chunker factory functionality."""
        fixed_chunker = create_chunker(ChunkingStrategy.FIXED, chunk_size=200)
        assert isinstance(fixed_chunker, FixedChunker)
        
        paragraph_chunker = create_chunker(ChunkingStrategy.PARAGRAPH)
        assert isinstance(paragraph_chunker, ParagraphChunker)
        
        recursive_chunker = create_chunker(ChunkingStrategy.RECURSIVE)
        assert isinstance(recursive_chunker, RecursiveChunker)
    
    def test_empty_document(self):
        """Test chunking with empty document."""
        empty_doc = Document(content="", metadata=DocumentMetadata())
        chunker = FixedChunker()
        chunks = chunker.chunk(empty_doc)
        
        assert len(chunks) == 0
    
    def test_chunk_metadata_inheritance(self):
        """Test that chunks inherit document metadata."""
        doc = Document(
            content="Test content for metadata inheritance.",
            metadata=DocumentMetadata(
                source="metadata_test.txt",
                title="Metadata Test",
                tags=["test", "metadata"]
            )
        )
        
        chunker = FixedChunker(chunk_size=50)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.source == "metadata_test.txt"
            assert chunk.metadata.title == "Metadata Test"
            assert chunk.metadata.tags == ["test", "metadata"]


class TestConfiguration:
    """Test configuration and settings."""
    
    def test_settings_import(self):
        """Test that settings can be imported."""
        from config.settings import Settings, get_settings
        
        # This should not raise an error
        settings_class = Settings
        assert settings_class is not None


class TestUtilities:
    """Test utility functions."""
    
    def test_embeddings_import(self):
        """Test that embedding utilities can be imported."""
        from utils.embeddings import EmbeddingService
        
        # This should not raise an error
        service_class = EmbeddingService
        assert service_class is not None


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])


