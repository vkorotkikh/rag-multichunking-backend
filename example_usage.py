"""
Example usage of the RAG Multi-Chunking Backend.

This script demonstrates how to use the various components of the RAG pipeline
including different chunking strategies, vector stores, and reranking.
"""

import asyncio
import os
from typing import List

# Add src to Python path for imports
import sys
sys.path.append('src')

from src import (
    RAGPipeline, create_rag_pipeline,
    Document, DocumentMetadata, IndexRequest, QueryRequest,
    ChunkingStrategy, VectorStore, RerankerType
)


async def main():
    """Main example function demonstrating RAG pipeline usage."""
    print("🚀 RAG Multi-Chunking Backend Example")
    print("=" * 50)
    
    # Sample documents to index
    sample_documents = [
        Document(
            content="""
            Artificial Intelligence (AI) is a broad field of computer science that aims to create 
            systems capable of performing tasks that typically require human intelligence. These 
            tasks include learning, reasoning, problem-solving, perception, and language understanding.
            
            Machine Learning is a subset of AI that focuses on algorithms that can learn and improve 
            from experience without being explicitly programmed. Deep Learning, in turn, is a subset 
            of machine learning that uses neural networks with multiple layers to model and understand 
            complex patterns in data.
            
            Recent advances in AI have led to significant breakthroughs in various domains including 
            natural language processing, computer vision, and autonomous systems. Large Language Models 
            (LLMs) like GPT and BERT have revolutionized how we interact with AI systems.
            """,
            metadata=DocumentMetadata(
                source="ai_overview.txt",
                title="Introduction to Artificial Intelligence",
                author="AI Research Team",
                tags=["AI", "machine learning", "deep learning"]
            )
        ),
        Document(
            content="""
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of large 
            language models with external knowledge retrieval. In a RAG system, when a user asks 
            a question, the system first retrieves relevant documents or passages from a knowledge 
            base, then uses this retrieved information along with the original query to generate 
            a more accurate and informative response.
            
            The RAG process typically involves several steps: document preprocessing and chunking, 
            embedding generation, vector storage, similarity search, and response generation. 
            Different chunking strategies can significantly impact the quality of retrieval, 
            including fixed-size chunking, paragraph-based chunking, and semantic chunking.
            
            Vector databases like Pinecone and Weaviate are commonly used to store and retrieve 
            document embeddings efficiently. Reranking techniques can further improve the relevance 
            of retrieved documents before generating the final response.
            """,
            metadata=DocumentMetadata(
                source="rag_guide.txt",
                title="Understanding RAG Systems",
                author="RAG Specialist",
                tags=["RAG", "retrieval", "generation", "NLP"]
            )
        ),
        Document(
            content="""
            Vector databases are specialized databases designed to store, index, and query 
            high-dimensional vectors efficiently. They are essential components in modern AI 
            applications, particularly for similarity search, recommendation systems, and 
            retrieval-augmented generation.
            
            Popular vector databases include Pinecone, Weaviate, Qdrant, and Chroma. Each has 
            its own strengths: Pinecone offers managed cloud services with excellent performance, 
            Weaviate provides powerful graph capabilities, Qdrant focuses on efficiency and 
            flexibility, while Chroma is designed for simplicity and ease of use.
            
            When choosing a vector database, consider factors such as scalability, query performance, 
            filtering capabilities, cloud vs. self-hosted options, and integration with your 
            existing technology stack.
            """,
            metadata=DocumentMetadata(
                source="vector_db_comparison.txt",
                title="Vector Database Comparison",
                author="Database Expert",
                tags=["vector database", "Pinecone", "Weaviate", "similarity search"]
            )
        )
    ]
    
    # Example 1: Basic RAG Pipeline with Recursive Chunking and Pinecone
    print("\n📚 Example 1: Basic RAG Pipeline")
    print("-" * 30)
    
    pipeline = create_rag_pipeline(
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        vector_store_type=VectorStore.PINECONE,
        reranker_type=RerankerType.COHERE,
        use_reranking=True
    )
    
    # Index documents
    print("Indexing documents...")
    index_request = IndexRequest(
        documents=sample_documents,
        chunking_strategy=ChunkingStrategy.RECURSIVE,
        vector_store=VectorStore.PINECONE,
        batch_size=50
    )
    
    index_response = await pipeline.index_documents(index_request)
    print(f"✅ Indexed {index_response.indexed_documents} documents")
    print(f"📄 Generated {index_response.total_chunks} chunks")
    print(f"⏱️  Processing time: {index_response.processing_time:.2f}s")
    
    if index_response.errors:
        print(f"❌ Errors: {index_response.errors}")
    
    # Query the indexed documents
    print("\n🔍 Querying the knowledge base...")
    
    queries = [
        "What is artificial intelligence?",
        "How does RAG work?",
        "What are the differences between vector databases?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        result = await pipeline.query_and_generate(
            query=query,
            top_k=3,
            use_reranking=True
        )
        
        print(f"Generated Response:\n{result['generated_response']}")
        print(f"Processing time: {result['total_processing_time']:.2f}s")
        print(f"Found {result['search_results'].total_results} relevant chunks")
    
    # Example 2: Compare Different Chunking Strategies
    print("\n\n🔬 Example 2: Comparing Chunking Strategies")
    print("-" * 45)
    
    chunking_strategies = [
        ChunkingStrategy.FIXED,
        ChunkingStrategy.PARAGRAPH,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.RECURSIVE
    ]
    
    for strategy in chunking_strategies:
        print(f"\n📋 Testing {strategy.value.title()} Chunking:")
        
        # Create pipeline with different chunking strategy
        test_pipeline = create_rag_pipeline(
            chunking_strategy=strategy,
            vector_store_type=VectorStore.PINECONE,
            use_reranking=False  # Disable reranking for comparison
        )
        
        # Index with current strategy
        test_index_request = IndexRequest(
            documents=sample_documents[:1],  # Use just one document for quick test
            chunking_strategy=strategy,
            vector_store=VectorStore.PINECONE
        )
        
        test_index_response = await test_pipeline.index_documents(test_index_request)
        print(f"  Chunks generated: {test_index_response.total_chunks}")
        print(f"  Processing time: {test_index_response.processing_time:.2f}s")
    
    # Example 3: Using Different Rerankers
    print("\n\n🎯 Example 3: Comparing Reranking Methods")
    print("-" * 40)
    
    reranker_types = [
        (RerankerType.BM25, "BM25 (Keyword-based)"),
        (RerankerType.COHERE, "Cohere (Neural)")
    ]
    
    test_query = "What is machine learning in AI?"
    
    for reranker_type, description in reranker_types:
        print(f"\n🔄 Testing {description}:")
        
        try:
            rerank_pipeline = create_rag_pipeline(
                chunking_strategy=ChunkingStrategy.RECURSIVE,
                vector_store_type=VectorStore.PINECONE,
                reranker_type=reranker_type,
                use_reranking=True
            )
            
            result = await rerank_pipeline.query_and_generate(
                query=test_query,
                top_k=3,
                use_reranking=True
            )
            
            print(f"  Reranking used: {result['search_results'].reranking_used}")
            print(f"  Results: {result['search_results'].total_results}")
            print(f"  Processing time: {result['total_processing_time']:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error with {description}: {e}")
    
    # Example 4: Advanced Filtering and Metadata Search
    print("\n\n🔍 Example 4: Advanced Filtering")
    print("-" * 30)
    
    # Query with metadata filters
    filtered_result = await pipeline.query_and_generate(
        query="artificial intelligence",
        top_k=5,
        filters={
            "tags": ["AI", "machine learning"],  # Filter by tags
            "author": "AI Research Team"  # Filter by author
        }
    )
    
    print("Filtered search results:")
    print(f"Found {filtered_result['search_results'].total_results} results")
    print(f"Response: {filtered_result['generated_response'][:200]}...")
    
    # Get index information
    print("\n\n📊 Index Information:")
    print("-" * 20)
    
    index_info = await pipeline.get_index_info()
    print(f"Index name: {index_info.get('index_name')}")
    print(f"Vector store: {index_info.get('vector_store')}")
    print(f"Chunking strategy: {index_info.get('chunking_strategy')}")
    print(f"Reranking enabled: {index_info.get('reranking_enabled')}")
    
    if 'stats' in index_info:
        stats = index_info['stats']
        print(f"Vector count: {stats.get('total_vector_count', 'N/A')}")
    
    print("\n✅ Example completed successfully!")


def setup_environment():
    """Setup environment variables for the example."""
    # Set default values if not provided
    env_vars = {
        'OPENAI__API_KEY': 'your-openai-api-key',
        'PINECONE__API_KEY': 'your-pinecone-api-key',
        'PINECONE__ENVIRONMENT': 'your-pinecone-environment',
        'PINECONE__INDEX_NAME': 'rag-test-index',
        'COHERE_API_KEY': 'your-cohere-api-key',
        'VECTOR_STORE': 'pinecone',
        'CHUNKING_STRATEGY': 'recursive'
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
            print(f"⚠️  Using default value for {key}. Set this in your .env file.")


if __name__ == "__main__":
    print("Setting up environment...")
    setup_environment()
    
    print("\n⚠️  IMPORTANT SETUP NOTES:")
    print("1. Copy env_example.txt to .env and fill in your API keys")
    print("2. Make sure you have valid API keys for:")
    print("   - OpenAI (for embeddings and generation)")
    print("   - Pinecone or Weaviate (for vector storage)")
    print("   - Cohere (optional, for reranking)")
    print("3. Install dependencies: pip install -r requirements.txt")
    print()
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with valid API keys")
        print("2. Installed all required dependencies")
        print("3. Created the necessary vector store indexes")


