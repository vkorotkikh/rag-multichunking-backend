"""
Configuration settings for the RAG multichunking backend.
"""
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="o3-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")


class PineconeConfig(BaseModel):
    """Pinecone configuration."""
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(..., description="Pinecone index name")
    dimension: int = Field(default=3072, description="Vector dimension")
    metric: str = Field(default="cosine", description="Distance metric")


class WeaviateConfig(BaseModel):
    """Weaviate configuration."""
    url: str = Field(..., description="Weaviate instance URL")
    api_key: Optional[str] = Field(None, description="Weaviate API key (if cloud)")
    class_name: str = Field(default="Document", description="Weaviate class name")


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    chunk_size: int = Field(default=1000, ge=100, description="Default chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Chunk overlap")
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        description="Text separators for recursive chunking"
    )


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    model_name: str = Field(default="rerank-english-v3.0", description="Cohere reranker model")
    top_k: int = Field(default=10, ge=1, description="Number of documents to rerank")
    relevance_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance threshold")


class Settings(BaseSettings):
    """Main application settings."""
    
    # API configurations
    openai: OpenAIConfig
    pinecone: Optional[PineconeConfig] = None
    weaviate: Optional[WeaviateConfig] = None
    
    # Processing configurations
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    # Vector store selection
    vector_store: Literal["pinecone", "weaviate"] = Field(default="pinecone")
    
    # Chunking strategy
    chunking_strategy: Literal["fixed", "paragraph", "semantic", "recursive"] = Field(default="recursive")
    
    # Application settings
    log_level: str = Field(default="INFO")
    max_concurrent_requests: int = Field(default=10, ge=1)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global settings
    if settings is None:
        settings = Settings()
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    current_dict = settings.model_dump() if settings else {}
    current_dict.update(kwargs)
    settings = Settings(**current_dict)
    return settings

