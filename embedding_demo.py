"""
Demonstration of different embedding providers in the RAG Multi-Chunking Backend.

This script shows how to use:
- OpenAI embeddings
- SentenceTransformers embeddings  
- HuggingFace Transformers embeddings
- BGE embeddings

Run with: python embedding_demo.py
"""
import asyncio
import os
from typing import List
import time

# Set up basic environment for demo
os.environ["OPENAI__API_KEY"] = "your-key-here"  # Replace with actual key if testing OpenAI
os.environ["EMBEDDING__PROVIDER"] = "sentence-transformers"  # Start with local models
os.environ["EMBEDDING__MODEL_NAME"] = "all-MiniLM-L6-v2"
os.environ["EMBEDDING__DEVICE"] = "cpu"
os.environ["EMBEDDING__NORMALIZE_EMBEDDINGS"] = "true"
os.environ["EMBEDDING__BATCH_SIZE"] = "16"

from src.utils.embeddings import (
    EmbeddingServiceFactory,
    OpenAIEmbeddingService,
    SentenceTransformerEmbeddingService, 
    HuggingFaceEmbeddingService,
    BGEEmbeddingService,
    create_embedding_service,
    embed_text,
    embed_texts
)


async def test_embedding_service(service_name: str, service, test_texts: List[str]):
    """Test an embedding service with sample texts."""
    print(f"\n{'='*50}")
    print(f"Testing {service_name}")
    print(f"{'='*50}")
    
    try:
        # Initialize the service
        await service.initialize()
        print(f"✓ Initialized {service_name}")
        print(f"  Model: {service.model_name}")
        print(f"  Dimension: {service.get_dimension()}")
        
        # Test single embedding
        start_time = time.time()
        single_embedding = await service.generate_embedding_async(test_texts[0])
        single_time = time.time() - start_time
        
        print(f"✓ Single embedding generated in {single_time:.3f}s")
        print(f"  Length: {len(single_embedding)}")
        print(f"  Sample values: {single_embedding[:5]}")
        
        # Test batch embedding
        start_time = time.time()
        batch_embeddings = await service.generate_embeddings_batch_async(test_texts)
        batch_time = time.time() - start_time
        
        print(f"✓ Batch embeddings ({len(test_texts)} texts) generated in {batch_time:.3f}s")
        print(f"  Average time per text: {batch_time/len(test_texts):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {service_name}: {e}")
        return False


async def demo_embedding_providers():
    """Demonstrate different embedding providers."""
    print("RAG Multi-Chunking Backend - Embedding Providers Demo")
    print("="*60)
    
    # Sample texts for testing
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Vector embeddings represent text as numerical vectors in high-dimensional space.",
        "Semantic similarity can be computed using cosine similarity between embeddings."
    ]
    
    print(f"Testing with {len(test_texts)} sample texts...")
    
    # Test different providers
    providers_to_test = []
    
    # 1. SentenceTransformers (should work with requirements.txt)
    try:
        st_service = SentenceTransformerEmbeddingService(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
        providers_to_test.append(("SentenceTransformers (all-MiniLM-L6-v2)", st_service))
    except ImportError as e:
        print(f"⚠ SentenceTransformers not available: {e}")
    
    # 2. BGE embeddings (also uses SentenceTransformers)
    try:
        bge_service = BGEEmbeddingService(
            model_name="BAAI/bge-small-en-v1.5",  # Using smaller model for demo
            device="cpu"
        )
        providers_to_test.append(("BGE (BAAI/bge-small-en-v1.5)", bge_service))
    except ImportError as e:
        print(f"⚠ BGE embeddings not available: {e}")
    
    # 3. HuggingFace Transformers
    try:
        hf_service = HuggingFaceEmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        providers_to_test.append(("HuggingFace Transformers", hf_service))
    except ImportError as e:
        print(f"⚠ HuggingFace Transformers not available: {e}")
    
    # 4. OpenAI (only if API key is available)
    if os.getenv("OPENAI__API_KEY") and os.getenv("OPENAI__API_KEY") != "your-key-here":
        try:
            openai_service = OpenAIEmbeddingService()
            providers_to_test.append(("OpenAI (text-embedding-3-large)", openai_service))
        except ImportError as e:
            print(f"⚠ OpenAI not available: {e}")
    else:
        print("⚠ OpenAI embeddings skipped (no API key provided)")
    
    # Test each provider
    results = {}
    for provider_name, service in providers_to_test:
        success = await test_embedding_service(provider_name, service, test_texts)
        results[provider_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_providers = [name for name, success in results.items() if success]
    failed_providers = [name for name, success in results.items() if not success]
    
    print(f"✓ Successful providers ({len(successful_providers)}):")
    for provider in successful_providers:
        print(f"  - {provider}")
    
    if failed_providers:
        print(f"\n✗ Failed providers ({len(failed_providers)}):")
        for provider in failed_providers:
            print(f"  - {provider}")
    
    print(f"\nTotal providers tested: {len(results)}")


async def demo_factory_pattern():
    """Demonstrate using the factory pattern."""
    print(f"\n{'='*60}")
    print("FACTORY PATTERN DEMO")
    print(f"{'='*60}")
    
    # Demo different ways to create services
    configs = [
        {
            "provider": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu"
        },
        {
            "provider": "bge", 
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "cpu"
        }
    ]
    
    for config in configs:
        try:
            print(f"\nCreating service with config: {config}")
            service = EmbeddingServiceFactory.create_service(**config)
            await service.initialize()
            
            # Test with a simple text
            test_text = "This is a test sentence."
            embedding = await service.generate_embedding_async(test_text)
            
            print(f"✓ Created {config['provider']} service")
            print(f"  Model: {service.model_name}")
            print(f"  Dimension: {service.get_dimension()}")
            print(f"  Embedding length: {len(embedding)}")
            
        except Exception as e:
            print(f"✗ Failed to create {config['provider']} service: {e}")


async def demo_convenience_functions():
    """Demonstrate convenience functions."""
    print(f"\n{'='*60}")
    print("CONVENIENCE FUNCTIONS DEMO")
    print(f"{'='*60}")
    
    # Set environment to use a local model
    os.environ["EMBEDDING__PROVIDER"] = "sentence-transformers"
    os.environ["EMBEDDING__MODEL_NAME"] = "all-MiniLM-L6-v2"
    
    try:
        # Test convenience functions
        test_text = "Testing convenience functions for embeddings."
        test_texts = [
            "First test sentence.",
            "Second test sentence.", 
            "Third test sentence."
        ]
        
        print("Testing embed_text()...")
        embedding = await embed_text(test_text)
        print(f"✓ Single embedding: length {len(embedding)}")
        
        print("Testing embed_texts()...")
        embeddings = await embed_texts(test_texts)
        print(f"✓ Batch embeddings: {len(embeddings)} embeddings generated")
        
    except Exception as e:
        print(f"✗ Error testing convenience functions: {e}")


async def main():
    """Main demo function."""
    try:
        await demo_embedding_providers()
        await demo_factory_pattern()
        await demo_convenience_functions()
        
        print(f"\n{'='*60}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("\nTo use different embedding providers in your application:")
        print("1. Set EMBEDDING__PROVIDER in your .env file")
        print("2. Set EMBEDDING__MODEL_NAME for the specific model")
        print("3. Configure other settings like device, batch_size, etc.")
        print("4. Use the convenience functions or create services directly")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

