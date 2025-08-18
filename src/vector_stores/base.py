"""
Base vector store interface and common utilities.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..models.document import DocumentChunk, SearchResult


class BaseVectorStore(ABC):
    """Base class for all vector store implementations."""
    
    def __init__(self, embedding_dimension: int = 3072):
        self.embedding_dimension = embedding_dimension
    
    @abstractmethod
    async def create_index(self, index_name: str, **kwargs) -> bool:
        """
        Create a new index in the vector store.
        
        Args:
            index_name: Name of the index to create
            **kwargs: Additional index configuration
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index
            
        Returns:
            True if index exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete an index from the vector store.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def upsert_chunks(
        self, 
        chunks: List[DocumentChunk], 
        index_name: str,
        batch_size: int = 100
    ) -> bool:
        """
        Insert or update document chunks in the vector store.
        
        Args:
            chunks: List of document chunks to upsert
            index_name: Name of the index
            batch_size: Number of chunks to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            index_name: Name of the index to search
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str, index_name: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            index_name: Name of the index
            
        Returns:
            Document chunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete_chunks(self, chunk_ids: List[str], index_name: str) -> bool:
        """
        Delete specific chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            index_name: Name of the index
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about an index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary containing index statistics
        """
        pass
    
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        if chunk.id:
            return chunk.id
        
        # Generate ID based on content hash and metadata
        import hashlib
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        source = chunk.metadata.source or "unknown"
        return f"{source}_{chunk.chunk_index}_{content_hash[:8]}"
    
    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Prepare metadata for storage in vector store."""
        metadata = {
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,  # Store content in metadata for retrieval
        }
        
        # Add document metadata
        if chunk.metadata:
            if chunk.metadata.source:
                metadata["source"] = chunk.metadata.source
            if chunk.metadata.title:
                metadata["title"] = chunk.metadata.title
            if chunk.metadata.author:
                metadata["author"] = chunk.metadata.author
            if chunk.metadata.file_type:
                metadata["file_type"] = chunk.metadata.file_type
            if chunk.metadata.page_number is not None:
                metadata["page_number"] = chunk.metadata.page_number
            if chunk.metadata.section:
                metadata["section"] = chunk.metadata.section
            if chunk.metadata.tags:
                metadata["tags"] = chunk.metadata.tags
            if chunk.start_char is not None:
                metadata["start_char"] = chunk.start_char
            if chunk.end_char is not None:
                metadata["end_char"] = chunk.end_char
        
        return metadata
    
    def _chunk_from_result(self, result_data: Dict[str, Any], score: float) -> SearchResult:
        """Convert vector store result to SearchResult."""
        metadata = result_data.get("metadata", {})
        
        # Reconstruct DocumentChunk
        chunk = DocumentChunk(
            id=result_data.get("id"),
            content=metadata.get("content", ""),
            chunk_index=metadata.get("chunk_index", 0),
            start_char=metadata.get("start_char"),
            end_char=metadata.get("end_char"),
            embedding=result_data.get("values")
        )
        
        # Reconstruct metadata
        from ..models.document import DocumentMetadata
        chunk.metadata = DocumentMetadata(
            source=metadata.get("source"),
            title=metadata.get("title"),
            author=metadata.get("author"),
            file_type=metadata.get("file_type"),
            page_number=metadata.get("page_number"),
            section=metadata.get("section"),
            tags=metadata.get("tags", [])
        )
        
        return SearchResult(chunk=chunk, score=score)


