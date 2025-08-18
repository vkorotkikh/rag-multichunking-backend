"""
Weaviate vector store implementation.
"""
import asyncio
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from .base import BaseVectorStore
from ..models.document import DocumentChunk, SearchResult
from ..config.settings import get_settings


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__()
        settings = get_settings()
        
        self.url = url or settings.weaviate.url
        self.api_key = api_key or settings.weaviate.api_key
        self.class_name = settings.weaviate.class_name
        
        # Initialize Weaviate client
        auth = Auth.api_key(self.api_key) if self.api_key else None
        self.client = weaviate.connect_to_custom(
            http_host=self.url,
            http_port=443,
            http_secure=True,
            auth_credentials=auth
        )
    
    async def create_index(self, index_name: str = None, **kwargs) -> bool:
        """Create a new Weaviate class (collection)."""
        try:
            class_name = index_name or self.class_name
            
            # Check if class already exists
            if await self.index_exists(class_name):
                return True
            
            # Define class properties
            properties = [
                Property(name="content", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="author", data_type=DataType.TEXT),
                Property(name="file_type", data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="section", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="start_char", data_type=DataType.INT),
                Property(name="end_char", data_type=DataType.INT),
            ]
            
            # Create class with vector configuration
            collection = self.client.collections.create(
                name=class_name,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),  # We'll provide our own vectors
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=kwargs.get("distance_metric", "cosine")
                )
            )
            
            return True
        except Exception as e:
            print(f"Error creating Weaviate class {class_name}: {e}")
            return False
    
    async def index_exists(self, index_name: str = None) -> bool:
        """Check if a Weaviate class exists."""
        try:
            class_name = index_name or self.class_name
            return self.client.collections.exists(class_name)
        except Exception:
            return False
    
    async def delete_index(self, index_name: str = None) -> bool:
        """Delete a Weaviate class."""
        try:
            class_name = index_name or self.class_name
            if await self.index_exists(class_name):
                self.client.collections.delete(class_name)
            return True
        except Exception as e:
            print(f"Error deleting Weaviate class {class_name}: {e}")
            return False
    
    async def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        index_name: str = None,
        batch_size: int = 100
    ) -> bool:
        """Upsert document chunks to Weaviate."""
        try:
            if not chunks:
                return True
            
            class_name = index_name or self.class_name
            collection = self.client.collections.get(class_name)
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                objects_to_insert = []
                
                for chunk in batch:
                    if not chunk.embedding:
                        print(f"Warning: Chunk {chunk.id} has no embedding, skipping")
                        continue
                    
                    chunk_id = self._generate_chunk_id(chunk)
                    
                    # Prepare properties
                    properties = {
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                    }
                    
                    # Add metadata properties
                    if chunk.metadata:
                        if chunk.metadata.source:
                            properties["source"] = chunk.metadata.source
                        if chunk.metadata.title:
                            properties["title"] = chunk.metadata.title
                        if chunk.metadata.author:
                            properties["author"] = chunk.metadata.author
                        if chunk.metadata.file_type:
                            properties["file_type"] = chunk.metadata.file_type
                        if chunk.metadata.page_number is not None:
                            properties["page_number"] = chunk.metadata.page_number
                        if chunk.metadata.section:
                            properties["section"] = chunk.metadata.section
                        if chunk.metadata.tags:
                            properties["tags"] = chunk.metadata.tags
                        if chunk.start_char is not None:
                            properties["start_char"] = chunk.start_char
                        if chunk.end_char is not None:
                            properties["end_char"] = chunk.end_char
                    
                    objects_to_insert.append({
                        "uuid": chunk_id,
                        "properties": properties,
                        "vector": chunk.embedding
                    })
                
                if objects_to_insert:
                    # Insert batch
                    with collection.batch.dynamic() as batch:
                        for obj in objects_to_insert:
                            batch.add_object(
                                properties=obj["properties"],
                                uuid=obj["uuid"],
                                vector=obj["vector"]
                            )
            
            return True
        except Exception as e:
            print(f"Error upserting chunks to Weaviate: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        index_name: str = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents in Weaviate."""
        try:
            class_name = index_name or self.class_name
            collection = self.client.collections.get(class_name)
            
            # Build query
            query_builder = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance"]
            )
            
            # Add filters if provided
            if filters:
                weaviate_filter = self._convert_filters(filters)
                if weaviate_filter:
                    query_builder = query_builder.where(weaviate_filter)
            
            # Execute query
            response = query_builder
            
            # Convert results
            results = []
            for obj in response.objects:
                # Calculate similarity score from distance
                distance = obj.metadata.distance if obj.metadata else 0
                score = 1.0 - distance  # Convert distance to similarity
                
                # Create chunk from object
                chunk = DocumentChunk(
                    id=str(obj.uuid),
                    content=obj.properties.get("content", ""),
                    chunk_index=obj.properties.get("chunk_index", 0),
                    start_char=obj.properties.get("start_char"),
                    end_char=obj.properties.get("end_char"),
                    embedding=obj.vector if hasattr(obj, 'vector') else None
                )
                
                # Set metadata
                from ..models.document import DocumentMetadata
                chunk.metadata = DocumentMetadata(
                    source=obj.properties.get("source"),
                    title=obj.properties.get("title"),
                    author=obj.properties.get("author"),
                    file_type=obj.properties.get("file_type"),
                    page_number=obj.properties.get("page_number"),
                    section=obj.properties.get("section"),
                    tags=obj.properties.get("tags", [])
                )
                
                search_result = SearchResult(chunk=chunk, score=score)
                results.append(search_result)
            
            return results
        except Exception as e:
            print(f"Error searching Weaviate: {e}")
            return []
    
    async def get_chunk_by_id(self, chunk_id: str, index_name: str = None) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID from Weaviate."""
        try:
            class_name = index_name or self.class_name
            collection = self.client.collections.get(class_name)
            
            obj = collection.query.fetch_object_by_id(chunk_id)
            
            if obj:
                chunk = DocumentChunk(
                    id=str(obj.uuid),
                    content=obj.properties.get("content", ""),
                    chunk_index=obj.properties.get("chunk_index", 0),
                    start_char=obj.properties.get("start_char"),
                    end_char=obj.properties.get("end_char"),
                    embedding=obj.vector if hasattr(obj, 'vector') else None
                )
                
                from ..models.document import DocumentMetadata
                chunk.metadata = DocumentMetadata(
                    source=obj.properties.get("source"),
                    title=obj.properties.get("title"),
                    author=obj.properties.get("author"),
                    file_type=obj.properties.get("file_type"),
                    page_number=obj.properties.get("page_number"),
                    section=obj.properties.get("section"),
                    tags=obj.properties.get("tags", [])
                )
                
                return chunk
            
            return None
        except Exception as e:
            print(f"Error fetching chunk {chunk_id} from Weaviate: {e}")
            return None
    
    async def delete_chunks(self, chunk_ids: List[str], index_name: str = None) -> bool:
        """Delete specific chunks from Weaviate."""
        try:
            if not chunk_ids:
                return True
            
            class_name = index_name or self.class_name
            collection = self.client.collections.get(class_name)
            
            # Delete chunks
            for chunk_id in chunk_ids:
                collection.data.delete_by_id(chunk_id)
            
            return True
        except Exception as e:
            print(f"Error deleting chunks from Weaviate: {e}")
            return False
    
    async def get_index_stats(self, index_name: str = None) -> Dict[str, Any]:
        """Get statistics about a Weaviate class."""
        try:
            class_name = index_name or self.class_name
            collection = self.client.collections.get(class_name)
            
            # Get object count
            aggregate_result = collection.aggregate.over_all(total_count=True)
            
            return {
                "total_object_count": aggregate_result.total_count,
                "class_name": class_name,
            }
        except Exception as e:
            print(f"Error getting Weaviate class stats: {e}")
            return {}
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Convert generic filters to Weaviate filter format."""
        try:
            filter_conditions = []
            
            for key, value in filters.items():
                if isinstance(value, list):
                    # For list values, use ContainsAny
                    filter_conditions.append(
                        Filter.by_property(key).contains_any(value)
                    )
                elif isinstance(value, dict):
                    # For range queries
                    if "gte" in value:
                        filter_conditions.append(
                            Filter.by_property(key).greater_or_equal(value["gte"])
                        )
                    if "lte" in value:
                        filter_conditions.append(
                            Filter.by_property(key).less_or_equal(value["lte"])
                        )
                    if "gt" in value:
                        filter_conditions.append(
                            Filter.by_property(key).greater_than(value["gt"])
                        )
                    if "lt" in value:
                        filter_conditions.append(
                            Filter.by_property(key).less_than(value["lt"])
                        )
                else:
                    # Direct equality
                    filter_conditions.append(
                        Filter.by_property(key).equal(value)
                    )
            
            # Combine filters with AND
            if len(filter_conditions) == 1:
                return filter_conditions[0]
            elif len(filter_conditions) > 1:
                result = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    result = result & condition
                return result
            
            return None
        except Exception as e:
            print(f"Error converting filters: {e}")
            return None
    
    def __del__(self):
        """Cleanup Weaviate client connection."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception:
            pass


