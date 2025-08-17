"""
Vector stores module for different vector database implementations.
"""
from typing import Dict, Type, Optional
from .base import BaseVectorStore
from .pinecone_store import PineconeVectorStore
from .weaviate_store import WeaviateVectorStore
from ..models.document import VectorStore
from ..config.settings import get_settings


class VectorStoreFactory:
    """Factory class for creating vector store instances."""
    
    _stores: Dict[VectorStore, Type[BaseVectorStore]] = {
        VectorStore.PINECONE: PineconeVectorStore,
        VectorStore.WEAVIATE: WeaviateVectorStore,
    }
    
    @classmethod
    def create_vector_store(
        cls,
        store_type: VectorStore,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store instance based on the specified type.
        
        Args:
            store_type: The vector store type to create
            **kwargs: Additional arguments specific to the vector store
            
        Returns:
            Vector store instance
            
        Raises:
            ValueError: If store type is not supported
        """
        if store_type not in cls._stores:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        store_class = cls._stores[store_type]
        return store_class(**kwargs)
    
    @classmethod
    def create_from_settings(cls, store_type: Optional[VectorStore] = None) -> BaseVectorStore:
        """
        Create a vector store instance from application settings.
        
        Args:
            store_type: Override the vector store type from settings
            
        Returns:
            Configured vector store instance
        """
        settings = get_settings()
        actual_store_type = store_type or VectorStore(settings.vector_store)
        
        if actual_store_type == VectorStore.PINECONE:
            if not settings.pinecone:
                raise ValueError("Pinecone configuration not found in settings")
            return PineconeVectorStore(
                api_key=settings.pinecone.api_key,
                environment=settings.pinecone.environment
            )
        elif actual_store_type == VectorStore.WEAVIATE:
            if not settings.weaviate:
                raise ValueError("Weaviate configuration not found in settings")
            return WeaviateVectorStore(
                url=settings.weaviate.url,
                api_key=settings.weaviate.api_key
            )
        else:
            raise ValueError(f"Unsupported vector store type: {actual_store_type}")
    
    @classmethod
    def get_available_stores(cls) -> list[VectorStore]:
        """Get list of available vector store types."""
        return list(cls._stores.keys())


# Convenience functions
def create_vector_store(store_type: VectorStore, **kwargs) -> BaseVectorStore:
    """Create a vector store instance."""
    return VectorStoreFactory.create_vector_store(store_type, **kwargs)


def create_vector_store_from_settings(store_type: Optional[VectorStore] = None) -> BaseVectorStore:
    """Create a vector store instance from settings."""
    return VectorStoreFactory.create_from_settings(store_type)


def get_available_stores() -> list[VectorStore]:
    """Get available vector store types."""
    return VectorStoreFactory.get_available_stores()


__all__ = [
    "BaseVectorStore",
    "PineconeVectorStore",
    "WeaviateVectorStore",
    "VectorStoreFactory",
    "create_vector_store",
    "create_vector_store_from_settings",
    "get_available_stores"
]


