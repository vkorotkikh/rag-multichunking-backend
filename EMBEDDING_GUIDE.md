# Local Embedding Support Guide

This guide shows how to use the expanded embedding support in the RAG Multi-Chunking Backend, which now supports multiple local embedding providers beyond just OpenAI.

## Supported Embedding Providers

### 1. OpenAI Embeddings (Remote)
- **Models**: `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002`
- **Requires**: OpenAI API key
- **Dimensions**: 1536-3072 depending on model
- **Best for**: High-quality embeddings, production use

### 2. SentenceTransformers (Local)
- **Models**: Any model from [SentenceTransformers Hub](https://www.sbert.net/docs/pretrained_models.html)
- **Popular models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `all-distilroberta-v1`
- **Requires**: `sentence-transformers` package
- **Best for**: Privacy, offline use, cost-effective

### 3. BGE Embeddings (Local)
- **Models**: BGE models from BAAI (e.g., `BAAI/bge-large-en-v1.5`)
- **Requires**: `sentence-transformers` package
- **Best for**: High-quality local embeddings, retrieval tasks

### 4. HuggingFace Transformers (Local)
- **Models**: Any text embedding model on HuggingFace Hub
- **Requires**: `transformers`, `torch` packages
- **Best for**: Custom models, specific domains

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# Embedding Configuration
EMBEDDING__PROVIDER=sentence-transformers  # openai, sentence-transformers, huggingface, bge
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2     # Model name for local providers
EMBEDDING__DEVICE=cpu                       # cpu or cuda
EMBEDDING__NORMALIZE_EMBEDDINGS=true       # Normalize embeddings
EMBEDDING__BATCH_SIZE=32                    # Batch size for processing
EMBEDDING__MAX_LENGTH=512                   # Max sequence length
EMBEDDING__TRUST_REMOTE_CODE=false         # Allow remote code for HF models

# OpenAI Configuration (if using OpenAI)
OPENAI__API_KEY=your_openai_api_key_here
OPENAI__EMBEDDING_MODEL=text-embedding-3-large
OPENAI__CHUNK_SIZE=8191
OPENAI__BATCH_SIZE=100
```

### Popular Model Recommendations

#### For General Use (SentenceTransformers)
```bash
EMBEDDING__PROVIDER=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2     # Fast, 384 dims
# OR
EMBEDDING__MODEL_NAME=all-mpnet-base-v2     # Higher quality, 768 dims
```

#### For High-Quality Retrieval (BGE)
```bash
EMBEDDING__PROVIDER=bge
EMBEDDING__MODEL_NAME=BAAI/bge-large-en-v1.5  # 1024 dims, excellent retrieval
# OR
EMBEDDING__MODEL_NAME=BAAI/bge-small-en-v1.5  # 384 dims, faster
```

#### For GPU Usage
```bash
EMBEDDING__DEVICE=cuda                      # Use GPU for faster processing
EMBEDDING__BATCH_SIZE=64                    # Increase batch size for GPU
```

## Usage Examples

### 1. Basic Usage (Global Service)

```python
import asyncio
from src.utils.embeddings import embed_text, embed_texts

async def basic_example():
    # Single text embedding
    text = "This is a sample document."
    embedding = await embed_text(text)
    print(f"Embedding dimension: {len(embedding)}")
    
    # Multiple texts
    texts = ["First document", "Second document", "Third document"]
    embeddings = await embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings")

asyncio.run(basic_example())
```

### 2. Custom Service Creation

```python
from src.utils.embeddings import create_embedding_service

# Create SentenceTransformers service
st_service = create_embedding_service(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

# Create BGE service
bge_service = create_embedding_service(
    provider="bge",
    model_name="BAAI/bge-large-en-v1.5",
    device="cuda"  # Use GPU if available
)

# Initialize and use
await st_service.initialize()
embedding = await st_service.generate_embedding_async("Test text")
```

### 3. Document Chunks Integration

```python
from src.utils.embeddings import embed_chunks
from src.models.document import DocumentChunk

# Create document chunks
chunks = [
    DocumentChunk(content="First chunk content", chunk_id="1"),
    DocumentChunk(content="Second chunk content", chunk_id="2"),
    DocumentChunk(content="Third chunk content", chunk_id="3")
]

# Add embeddings to chunks
embedded_chunks = await embed_chunks(chunks)

# Each chunk now has an embedding
for chunk in embedded_chunks:
    print(f"Chunk {chunk.chunk_id}: embedding dimension {len(chunk.embedding)}")
```

### 4. Switching Between Providers

```python
from src.utils.embeddings import set_embedding_service, create_embedding_service

# Start with local model
local_service = create_embedding_service(
    provider="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)
set_embedding_service(local_service)

# Process some documents with local model
embeddings1 = await embed_texts(["Text 1", "Text 2"])

# Switch to BGE for better quality
bge_service = create_embedding_service(
    provider="bge",
    model_name="BAAI/bge-large-en-v1.5"
)
set_embedding_service(bge_service)

# Process more documents with BGE
embeddings2 = await embed_texts(["Text 3", "Text 4"])
```

## Performance Comparison

| Provider | Model | Dimensions | Speed | Quality | GPU Support |
|----------|-------|------------|-------|---------|-------------|
| OpenAI | text-embedding-3-large | 3072 | Medium | Excellent | N/A |
| SentenceTransformers | all-MiniLM-L6-v2 | 384 | Fast | Good | Yes |
| SentenceTransformers | all-mpnet-base-v2 | 768 | Medium | Very Good | Yes |
| BGE | bge-large-en-v1.5 | 1024 | Medium | Excellent | Yes |
| BGE | bge-small-en-v1.5 | 384 | Fast | Very Good | Yes |

## Installation Requirements

### For SentenceTransformers and BGE
```bash
pip install sentence-transformers torch
```

### For HuggingFace Transformers
```bash
pip install transformers torch
```

### For OpenAI
```bash
pip install openai
```

### For GPU Support (NVIDIA)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Best Practices

### 1. Model Selection
- **Development/Testing**: Use `all-MiniLM-L6-v2` for fast prototyping
- **Production (Local)**: Use `BAAI/bge-large-en-v1.5` for best quality
- **Production (Remote)**: Use OpenAI `text-embedding-3-large` for ultimate quality

### 2. Hardware Optimization
- **CPU**: Use smaller models like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`
- **GPU**: Use larger models and increase batch sizes
- **Memory**: Monitor model size vs available RAM

### 3. Batch Processing
```python
# Good: Process in batches
embeddings = await service.generate_embeddings_batch_async(
    texts, 
    batch_size=32
)

# Avoid: Processing one by one
embeddings = []
for text in texts:
    embedding = await service.generate_embedding_async(text)
    embeddings.append(embedding)
```

### 4. Caching for Development
```python
# Cache embeddings during development
import pickle

# Save embeddings
with open('embeddings_cache.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Load embeddings
with open('embeddings_cache.pkl', 'rb') as f:
    embeddings = pickle.load(f)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure required packages are installed
2. **CUDA Errors**: Check GPU compatibility and drivers
3. **Memory Issues**: Reduce batch size or use smaller models
4. **Model Download Failures**: Check internet connection and HuggingFace access

### Error Handling

The embedding services include built-in error handling:

```python
# Services return empty embeddings on error rather than crashing
embedding = await service.generate_embedding_async("text")
if not embedding:
    print("Embedding generation failed")
```

### Logging

Enable logging to debug issues:

```python
import logging
logging.basicConfig(level=logging.INFO)

# This will show model loading and embedding generation info
service = create_embedding_service("sentence-transformers", "all-MiniLM-L6-v2")
await service.initialize()
```

## Migration from OpenAI-only

If you're migrating from the previous OpenAI-only implementation:

1. **No code changes needed** - The global service functions remain the same
2. **Update configuration** - Set `EMBEDDING__PROVIDER` to your preferred local provider
3. **Install dependencies** - Install `sentence-transformers` and `torch`
4. **Test thoroughly** - Different models may produce different embedding dimensions

The implementation maintains backward compatibility while adding new capabilities.

## Advanced Usage

### Custom Models

You can use any compatible model:

```python
# Custom SentenceTransformers model
service = create_embedding_service(
    provider="sentence-transformers",
    model_name="your-custom-model-name"
)

# Custom HuggingFace model
service = create_embedding_service(
    provider="huggingface", 
    model_name="your-org/your-model",
    trust_remote_code=True  # If model requires custom code
)
```

### Mixed Providers

Use different providers for different purposes:

```python
# Fast local model for development
dev_service = create_embedding_service("sentence-transformers", "all-MiniLM-L6-v2")

# High-quality model for production
prod_service = create_embedding_service("bge", "BAAI/bge-large-en-v1.5")

# Use appropriate service based on context
if is_production:
    embeddings = await prod_service.generate_embeddings_batch_async(texts)
else:
    embeddings = await dev_service.generate_embeddings_batch_async(texts)
```

This expanded embedding support provides flexibility to choose the best embedding solution for your specific needs, whether that's cost-effectiveness, privacy, performance, or quality.

