"""
Pinecone vector store implementation.
"""
import asyncio
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from .base import BaseVectorStore
from ..models.document import DocumentChunk, SearchResult
from ..config.settings import get_settings


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None):
        super().__init__()
        settings = get_settings()
        
        self.api_key = api_key or settings.pinecone.api_key
        self.environment = environment or settings.pinecone.environment
        self.embedding_dimension = settings.pinecone.dimension
        self.metric = settings.pinecone.metric
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self._indexes = {}  # Cache for index connections
    
    def _get_index(self, index_name: str):
        """Get or create index connection."""
        if index_name not in self._indexes:
            self._indexes[index_name] = self.pc.Index(index_name)
        return self._indexes[index_name]
    
    async def create_index(self, index_name: str, **kwargs) -> bool:
        """Create a new Pinecone index."""
        try:
            # Check if index already exists
            if await self.index_exists(index_name):
                return True
            
            # Create index with serverless spec
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=kwargs.get("cloud", "aws"),
                    region=kwargs.get("region", "us-east-1")
                )
            )
            
            # Wait for index to be ready
            await asyncio.sleep(2)
            while not await self.index_exists(index_name):
                await asyncio.sleep(1)
            
            return True
        except Exception as e:
            print(f"Error creating Pinecone index {index_name}: {e}")
            return False
    
    async def index_exists(self, index_name: str) -> bool:
        """Check if a Pinecone index exists."""
        try:
            indexes = self.pc.list_indexes()
            return any(index.name == index_name for index in indexes)
        except Exception:
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """Delete a Pinecone index."""
        try:
            if await self.index_exists(index_name):
                self.pc.delete_index(index_name)
                # Remove from cache
                if index_name in self._indexes:
                    del self._indexes[index_name]
            return True
        except Exception as e:
            print(f"Error deleting Pinecone index {index_name}: {e}")
            return False
    
    async def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        index_name: str,
        batch_size: int = 100
    ) -> bool:
        """Upsert document chunks to Pinecone."""
        try:
            if not chunks:
                return True
            
            index = self._get_index(index_name)
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vectors = []
                
                for chunk in batch:
                    if not chunk.embedding:
                        print(f"Warning: Chunk {chunk.id} has no embedding, skipping")
                        continue
                    
                    chunk_id = self._generate_chunk_id(chunk)
                    metadata = self._prepare_metadata(chunk)
                    
                    vectors.append({
                        "id": chunk_id,
                        "values": chunk.embedding,
                        "metadata": metadata
                    })
                
                if vectors:
                    # Upsert batch
                    index.upsert(vectors=vectors)
            
            return True
        except Exception as e:
            print(f"Error upserting chunks to Pinecone: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in Pinecone."""
        try:
            index = self._get_index(index_name)
            
            # Convert filters to Pinecone format
            pinecone_filter = self._convert_filters(filters) if filters else None
            
            # Perform search
            response = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Convert results
            results = []
            for match in response.matches:
                result_data = {
                    "id": match.id,
                    "values": match.values if hasattr(match, 'values') else None,
                    "metadata": match.metadata
                }
                search_result = self._chunk_from_result(result_data, match.score)
                results.append(search_result)
            
            return results
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []
    
    async def get_chunk_by_id(self, chunk_id: str, index_name: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID from Pinecone."""
        try:
            index = self._get_index(index_name)
            
            response = index.fetch(ids=[chunk_id])
            
            if chunk_id in response.vectors:
                vector_data = response.vectors[chunk_id]
                result_data = {
                    "id": chunk_id,
                    "values": vector_data.values,
                    "metadata": vector_data.metadata
                }
                search_result = self._chunk_from_result(result_data, 1.0)
                return search_result.chunk
            
            return None
        except Exception as e:
            print(f"Error fetching chunk {chunk_id} from Pinecone: {e}")
            return None
    
    async def delete_chunks(self, chunk_ids: List[str], index_name: str) -> bool:
        """Delete specific chunks from Pinecone."""
        try:
            if not chunk_ids:
                return True
            
            index = self._get_index(index_name)
            
            # Delete in batches (Pinecone has limits)
            batch_size = 1000
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i + batch_size]
                index.delete(ids=batch)
            
            return True
        except Exception as e:
            print(f"Error deleting chunks from Pinecone: {e}")
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about a Pinecone index."""
        try:
            index = self._get_index(index_name)
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            print(f"Error getting Pinecone index stats: {e}")
            return {}
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filters to Pinecone filter format."""
        pinecone_filter = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # For list values, use $in operator
                pinecone_filter[key] = {"$in": value}
            elif isinstance(value, dict):
                # For range queries
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    pinecone_filter[key] = {f"${k}": v for k, v in value.items()}
                else:
                    pinecone_filter[key] = value
            else:
                # Direct equality
                pinecone_filter[key] = value
        
        return pinecone_filter


