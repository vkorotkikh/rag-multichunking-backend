"""
Simple test for local embedding functionality.
Tests the new embedding implementations without full project dependencies.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic SentenceTransformers functionality
def test_sentence_transformers():
    """Test if SentenceTransformers is working."""
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ SentenceTransformers import successful")
        
        # Test model loading
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✓ Embedding generated: shape {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"✗ SentenceTransformers test failed: {e}")
        return False


def test_torch():
    """Test if PyTorch is working."""
    try:
        import torch
        print("✓ PyTorch import successful")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False


async def test_embedding_classes():
    """Test our embedding classes directly."""
    print("\nTesting embedding classes...")
    
    # Mock settings
    class MockSettings:
        class MockEmbedding:
            provider = "sentence-transformers"
            model_name = "all-MiniLM-L6-v2"
            device = "cpu"
            normalize_embeddings = True
            batch_size = 16
            max_length = 512
            trust_remote_code = False
            
        embedding = MockEmbedding()
    
    # Temporarily replace settings
    import src.config.settings as settings_module
    original_get_settings = settings_module.get_settings
    settings_module.get_settings = lambda: MockSettings()
    
    try:
        from src.utils.embeddings import SentenceTransformerEmbeddingService
        
        # Test SentenceTransformer service
        service = SentenceTransformerEmbeddingService()
        await service.initialize()
        print("✓ SentenceTransformerEmbeddingService initialized")
        
        # Test single embedding
        test_text = "Testing embedding generation."
        embedding = await service.generate_embedding_async(test_text)
        print(f"✓ Single embedding: length {len(embedding)}")
        
        # Test batch embedding
        test_texts = ["First text.", "Second text.", "Third text."]
        embeddings = await service.generate_embeddings_batch_async(test_texts)
        print(f"✓ Batch embeddings: {len(embeddings)} embeddings generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original settings
        settings_module.get_settings = original_get_settings


def main():
    """Run all tests."""
    print("Testing Local Embedding Implementation")
    print("=" * 50)
    
    results = []
    
    # Test dependencies
    print("1. Testing dependencies...")
    results.append(test_torch())
    results.append(test_sentence_transformers())
    
    # Test our implementation
    print("\n2. Testing implementation...")
    try:
        result = asyncio.run(test_embedding_classes())
        results.append(result)
    except Exception as e:
        print(f"✗ Implementation test failed: {e}")
        results.append(False)
    
    # Summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("🎉 All tests passed! Local embeddings are working correctly.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

