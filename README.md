# RAG Multi-Chunking Backend

A comprehensive, production-ready RAG (Retrieval-Augmented Generation) backend system that supports multiple chunking strategies, vector stores, and reranking capabilities. Built with modern Python frameworks including LangChain, Pydantic, and OpenAI's latest models.

## 🌟 Features

### Chunking Strategies
- **Fixed Chunking**: Simple character-based chunking with configurable size and overlap
- **Paragraph Chunking**: Respects natural paragraph boundaries for better semantic coherence
- **Semantic Chunking**: Uses sentence embeddings to group semantically similar content
- **Recursive Chunking**: Hierarchical splitting using multiple separators for optimal boundaries

### Vector Stores
- **Pinecone**: Managed cloud vector database with excellent performance
- **Weaviate**: Open-source vector database with graph capabilities
- Unified interface supporting easy switching between providers

### Reranking
- **Cohere Rerank**: Neural reranking using Cohere's specialized models
- **BM25**: Traditional keyword-based reranking for baseline comparison
- **BM25+**: Enhanced BM25 variant with improved long document handling

### AI Integration
- **OpenAI o3**: Latest OpenAI model for response generation
- **Text Embedding 3**: Advanced embeddings for high-quality vector representations
- Configurable temperature and token limits

## 🏗️ Architecture

```
rag-multichunking-backend/
├── src/
│   ├── chunkers/           # Document chunking strategies
│   ├── vector_stores/      # Vector database integrations
│   ├── rerankers/         # Result reranking implementations
│   ├── models/            # Pydantic data models
│   ├── config/            # Configuration management
│   ├── utils/             # Utility functions
│   └── rag_pipeline.py    # Main orchestrator
├── tests/                 # Test suite
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── example_usage.py       # Usage examples
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd rag-multichunking-backend

# Install dependencies
pip install -r requirements.txt

# For semantic chunking (optional)
python -m spacy download en_core_web_sm
```

### 2. Configuration

Copy the environment template and configure your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` with your actual API keys:

```env
# Required
OPENAI__API_KEY=your_openai_api_key
PINECONE__API_KEY=your_pinecone_api_key
PINECONE__ENVIRONMENT=your_pinecone_environment

# Optional (for reranking)
COHERE_API_KEY=your_cohere_api_key
```

### 3. Basic Usage

```python
import asyncio
from src import create_rag_pipeline, Document, IndexRequest

async def main():
    # Create RAG pipeline
    pipeline = create_rag_pipeline(
        chunking_strategy="recursive",
        vector_store_type="pinecone",
        use_reranking=True
    )
    
    # Index documents
    documents = [
        Document(content="Your document content here...")
    ]
    
    index_request = IndexRequest(documents=documents)
    await pipeline.index_documents(index_request)
    
    # Query and generate response
    result = await pipeline.query_and_generate(
        query="What is this document about?",
        top_k=5
    )
    
    print(result['generated_response'])

asyncio.run(main())
```

## 📋 Detailed Usage

### Chunking Strategies

```python
from src import ChunkingStrategy, create_chunker

# Fixed-size chunking
chunker = create_chunker(
    strategy=ChunkingStrategy.FIXED,
    chunk_size=1000,
    chunk_overlap=200
)

# Semantic chunking
chunker = create_chunker(
    strategy=ChunkingStrategy.SEMANTIC,
    similarity_threshold=0.7,
    model_name="all-MiniLM-L6-v2"
)

# Process document
chunks = chunker.chunk(document)
```

### Vector Store Configuration

```python
from src import VectorStore, create_vector_store_from_settings

# Using Pinecone
pipeline = create_rag_pipeline(vector_store_type=VectorStore.PINECONE)

# Using Weaviate  
pipeline = create_rag_pipeline(vector_store_type=VectorStore.WEAVIATE)
```

### Reranking Options

```python
from src import RerankerType

# Cohere neural reranking
pipeline = create_rag_pipeline(
    reranker_type=RerankerType.COHERE,
    use_reranking=True
)

# BM25 keyword reranking
pipeline = create_rag_pipeline(
    reranker_type=RerankerType.BM25,
    use_reranking=True
)
```

### Advanced Querying

```python
# Query with filters
result = await pipeline.query_and_generate(
    query="What is machine learning?",
    top_k=10,
    filters={
        "source": "ai_textbook.pdf",
        "tags": ["machine learning", "AI"]
    },
    system_prompt="You are an AI expert. Answer concisely."
)

# Custom query processing
query_request = QueryRequest(
    query="Explain neural networks",
    top_k=5,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    use_reranking=True,
    rerank_top_k=20
)

query_response = await pipeline.query(query_request)
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI__API_KEY` | OpenAI API key | Required |
| `OPENAI__MODEL` | OpenAI model name | `o3-mini` |
| `PINECONE__API_KEY` | Pinecone API key | Required for Pinecone |
| `WEAVIATE__URL` | Weaviate instance URL | Required for Weaviate |
| `CHUNKING__CHUNK_SIZE` | Default chunk size | `1000` |
| `CHUNKING__CHUNK_OVERLAP` | Chunk overlap | `200` |
| `VECTOR_STORE` | Default vector store | `pinecone` |

### Programmatic Configuration

```python
from src import get_settings, update_settings

# Get current settings
settings = get_settings()

# Update settings
update_settings(
    chunking_strategy="semantic",
    vector_store="weaviate",
    reranker_top_k=15
)
```

## 📊 Performance Considerations

### Chunking Strategy Performance

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| Fixed | ⭐⭐⭐⭐⭐ | ⭐⭐ | Large documents, speed priority |
| Paragraph | ⭐⭐⭐⭐ | ⭐⭐⭐ | Well-formatted text |
| Recursive | ⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose, balanced |
| Semantic | ⭐⭐ | ⭐⭐⭐⭐⭐ | High-quality requirements |

### Vector Store Comparison

| Feature | Pinecone | Weaviate |
|---------|----------|----------|
| Cloud Managed | ✅ | ✅ Self-hosted option |
| Filtering | ✅ Advanced | ✅ Graph-based |
| Scalability | ✅ Excellent | ✅ Good |
| Setup Complexity | Low | Medium |

## 🧪 Testing

Run the example script to test your setup:

```bash
python example_usage.py
```

The example demonstrates:
- Document indexing with different chunking strategies
- Query processing and response generation
- Reranking comparisons
- Metadata filtering

## 🔍 Monitoring and Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
LOG_LEVEL=DEBUG
```

### Index Statistics

```python
# Get index information
index_info = await pipeline.get_index_info()
print(f"Vector count: {index_info['stats']['total_vector_count']}")
```

### Performance Metrics

```python
# All responses include timing information
result = await pipeline.query_and_generate(query="test")
print(f"Processing time: {result['total_processing_time']:.2f}s")
```

## 🛠️ Development

### Project Structure

- `src/chunkers/`: Implements different text chunking strategies
- `src/vector_stores/`: Vector database integrations
- `src/rerankers/`: Result reranking implementations  
- `src/models/`: Pydantic models for type safety
- `src/config/`: Configuration management
- `src/utils/`: Shared utilities (embeddings, etc.)

### Adding New Components

#### New Chunking Strategy

```python
from src.chunkers.base import BaseChunker

class CustomChunker(BaseChunker):
    def chunk(self, document: Document) -> List[DocumentChunk]:
        # Implement your chunking logic
        pass
```

#### New Vector Store

```python
from src.vector_stores.base import BaseVectorStore

class CustomVectorStore(BaseVectorStore):
    async def search(self, query_embedding, index_name, top_k):
        # Implement search logic
        pass
```

## 📚 API Reference

### Core Classes

- `RAGPipeline`: Main orchestrator class
- `Document`: Input document model
- `QueryRequest`/`QueryResponse`: Query interface
- `IndexRequest`/`IndexResponse`: Indexing interface

### Factory Functions

- `create_rag_pipeline()`: Creates configured pipeline
- `create_chunker()`: Creates chunker instance
- `create_vector_store()`: Creates vector store instance
- `create_reranker()`: Creates reranker instance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Q: Getting API key errors?**
A: Ensure your `.env` file is properly configured with valid API keys.

**Q: Semantic chunking is slow?**
A: Semantic chunking requires downloading sentence transformer models. This is normal on first run.

**Q: Vector store connection fails?**
A: Check your vector store configuration and ensure the service is accessible.

### Getting Help

- Check the `example_usage.py` for working examples
- Review the error logs for specific issues
- Ensure all required dependencies are installed

## 🔮 Future Enhancements

- Support for additional vector stores (Qdrant, Chroma)
- Multi-modal document support (images, tables)
- Streaming response generation
- Async batch processing improvements
- Advanced caching mechanisms

