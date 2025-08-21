"""
Semantic Chunking Demonstration - Advanced RAG Multi-Chunking Backend

This demo showcases the semantic chunking capabilities with detailed explanations
of how sentence embeddings work and different model options.
"""

import asyncio
import sys
import os

# Add src to Python path for imports
sys.path.append('src')

from src.chunkers.semantic import SemanticChunker
from src.models.document import Document, DocumentMetadata


def print_separator(title: str):
    """Print a formatted separator with title."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_model_info():
    """Display comprehensive information about available embedding models."""
    print_separator("📚 SENTENCE TRANSFORMER MODELS - COMPREHENSIVE GUIDE")
    
    print("""
🧠 WHAT ARE EMBEDDING MODELS?

Embedding models are neural networks that convert text into dense numerical vectors (embeddings).
These vectors capture semantic meaning - similar text produces similar vectors, enabling
mathematical comparison of meaning rather than just word matching.

Key Concepts:
• Vector Dimension: Number of features in each embedding (384, 768, 1024...)
• Cosine Similarity: Measures angle between vectors (0=unrelated, 1=identical)
• Semantic Space: High-dimensional space where related concepts cluster together
• Pre-training: Models learned from massive text corpora to understand language

🎯 HOW SEMANTIC CHUNKING WORKS:

1. Sentence Segmentation: Split text into individual sentences
2. Embedding Generation: Convert each sentence → vector representation
3. Similarity Calculation: Compare all sentence pairs using cosine similarity
4. Semantic Grouping: Group sentences with similarity > threshold
5. Size Management: Respect chunk size while preserving semantic coherence
    """)
    
    # Get model recommendations
    models = SemanticChunker.get_recommended_models()
    
    for category, model_dict in models.items():
        print(f"\n🏷️  {category.upper().replace('_', ' ')} MODELS:")
        print("-" * 50)
        
        for model_name, info in model_dict.items():
            print(f"\n📦 {model_name}")
            print(f"   Dimensions: {info['dimensions']}")
            print(f"   Parameters: {info['parameters']}")
            print(f"   Languages: {info['languages']}")
            print(f"   Speed: {info['speed']}")
            print(f"   Quality: {info['quality']}")
            print(f"   Use Case: {info['use_case']}")


async def demo_model_comparison():
    """Demonstrate different embedding models with the same text."""
    print_separator("🔬 MODEL COMPARISON DEMO")
    
    # Sample text with different topics for clear semantic boundaries
    sample_text = """
    Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis.
    Machine learning algorithms can analyze medical images with incredible accuracy.
    Deep learning models are being trained on vast datasets of patient records.
    
    The weather today is particularly sunny with clear blue skies.
    Temperature is expected to reach 75 degrees Fahrenheit this afternoon.
    There's a gentle breeze coming from the southwest direction.
    
    Python is a popular programming language for data science applications.
    Libraries like NumPy and Pandas make data manipulation straightforward.
    Jupyter notebooks provide an interactive environment for analysis.
    """
    
    doc = Document(
        content=sample_text.strip(),
        metadata=DocumentMetadata(source="demo_text.txt", title="Multi-topic Demo")
    )
    
    # Test different models
    models_to_test = [
        {
            "name": "all-MiniLM-L6-v2",
            "description": "⚡ Fast & Lightweight (384-dim, 22M params)",
            "threshold": 0.7
        },
        {
            "name": "all-mpnet-base-v2", 
            "description": "🎯 High Quality (768-dim, 109M params)",
            "threshold": 0.8
        }
    ]
    
    for model_config in models_to_test:
        print(f"\n🧠 Testing: {model_config['name']}")
        print(f"Description: {model_config['description']}")
        print(f"Similarity Threshold: {model_config['threshold']}")
        
        try:
            chunker = SemanticChunker(
                model_name=model_config['name'],
                similarity_threshold=model_config['threshold'],
                chunk_size=500,
                min_sentences_per_chunk=2
            )
            
            # Get model info
            model_info = chunker.get_model_info()
            print(f"Status: {model_info['status']}")
            
            # Perform chunking
            chunks = chunker.chunk(doc)
            
            if model_info['status'] == 'loaded':
                print(f"Embedding Dimension: {model_info['embedding_dimension']}")
            
            print(f"Generated {len(chunks)} semantic chunks:")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\n  📄 Chunk {i} ({len(chunk.content)} chars):")
                print(f"     {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")
            
        except Exception as e:
            print(f"❌ Error with {model_config['name']}: {e}")
            print("   This might be due to missing dependencies or network issues.")


async def demo_similarity_thresholds():
    """Demonstrate how similarity thresholds affect chunking."""
    print_separator("🎚️  SIMILARITY THRESHOLD IMPACT")
    
    text = """
    Climate change is causing rising sea levels worldwide. Global warming affects weather patterns.
    Environmental protection requires immediate action from governments. Carbon emissions must be reduced.
    
    The smartphone market is highly competitive today. Mobile technology advances rapidly each year.
    Tech companies invest billions in research and development. Innovation drives consumer adoption.
    
    Cooking requires patience and practice to master. Culinary arts combine creativity with technique.
    Professional chefs train for years to perfect their skills. Restaurant kitchens demand precision.
    """
    
    doc = Document(content=text.strip())
    
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\n🎯 Similarity Threshold: {threshold}")
        print(f"Interpretation: {'Loose grouping (diverse topics together)' if threshold < 0.6 else 'Moderate grouping (related topics)' if threshold < 0.8 else 'Strict grouping (very similar topics only)'}")
        
        chunker = SemanticChunker(
            similarity_threshold=threshold,
            chunk_size=600,
            min_sentences_per_chunk=2
        )
        
        chunks = chunker.chunk(doc)
        print(f"Result: {len(chunks)} chunks created")
        
        for i, chunk in enumerate(chunks, 1):
            sentences = chunk.content.split('. ')
            print(f"  Chunk {i}: {len(sentences)} sentences")


async def demo_multilingual_support():
    """Demonstrate multilingual semantic chunking."""
    print_separator("🌍 MULTILINGUAL SEMANTIC CHUNKING")
    
    multilingual_text = """
    Artificial intelligence is transforming many industries today.
    Machine learning enables computers to learn from data automatically.
    
    L'intelligence artificielle transforme de nombreuses industries aujourd'hui.
    L'apprentissage automatique permet aux ordinateurs d'apprendre automatiquement.
    
    La inteligencia artificial está transformando muchas industrias hoy.
    El aprendizaje automático permite a las computadoras aprender automáticamente.
    
    Natural language processing helps computers understand human language.
    Deep learning models achieve remarkable results in text analysis.
    """
    
    doc = Document(content=multilingual_text.strip())
    
    print("🔍 Testing multilingual model: paraphrase-multilingual-MiniLM-L12-v2")
    print("This model supports 50+ languages and should group similar concepts")
    print("across languages based on semantic meaning, not just language.")
    
    try:
        chunker = SemanticChunker(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            similarity_threshold=0.6,  # Lower threshold for cross-language similarity
            chunk_size=800
        )
        
        chunks = chunker.chunk(doc)
        
        print(f"\nResult: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n📄 Chunk {i}:")
            lines = chunk.content.split('\n')
            languages_in_chunk = []
            for line in lines:
                if line.strip():
                    if any(word in line.lower() for word in ['intelligence', 'machine', 'learning']):
                        if 'artificielle' in line:
                            languages_in_chunk.append('French')
                        elif 'inteligencia' in line:
                            languages_in_chunk.append('Spanish')
                        else:
                            languages_in_chunk.append('English')
            
            print(f"   Languages detected: {set(languages_in_chunk)}")
            print(f"   Content preview: {chunk.content[:150]}...")
            
    except Exception as e:
        print(f"❌ Error with multilingual model: {e}")
        print("   Multilingual models may require additional setup or dependencies.")


async def demo_performance_analysis():
    """Analyze performance characteristics of semantic chunking."""
    print_separator("📊 PERFORMANCE ANALYSIS")
    
    import time
    
    # Generate test text of varying sizes
    base_text = """
    Quantum computing represents a paradigm shift in computational power. Unlike classical computers 
    that use bits, quantum computers use quantum bits or qubits. These qubits can exist in 
    superposition, allowing quantum computers to process vast amounts of data simultaneously.
    
    Machine learning algorithms are becoming increasingly sophisticated. Deep neural networks 
    can now recognize patterns in data that were previously impossible to detect. This has 
    led to breakthroughs in image recognition, natural language processing, and predictive analytics.
    
    Blockchain technology provides a decentralized approach to data storage and verification.
    Each block contains a cryptographic hash of the previous block, creating an immutable chain.
    This technology has applications beyond cryptocurrency, including supply chain management.
    """
    
    text_sizes = {
        "Small": base_text,
        "Medium": base_text * 3,
        "Large": base_text * 10
    }
    
    models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    
    print("⏱️  Performance Comparison:")
    print("Model                    | Text Size | Chunks | Time (s) | Quality")
    print("-" * 65)
    
    for model_name in models:
        for size_name, text in text_sizes.items():
            doc = Document(content=text)
            
            chunker = SemanticChunker(
                model_name=model_name,
                similarity_threshold=0.75,
                chunk_size=800
            )
            
            start_time = time.time()
            chunks = chunker.chunk(doc)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Simple quality metric: variation in chunk sizes (lower is better)
            if chunks:
                chunk_sizes = [len(chunk.content) for chunk in chunks]
                size_variation = max(chunk_sizes) - min(chunk_sizes)
                quality = "High" if size_variation < 300 else "Medium" if size_variation < 600 else "Low"
            else:
                quality = "N/A"
            
            print(f"{model_name:24} | {size_name:9} | {len(chunks):6} | {processing_time:8.2f} | {quality}")


async def main():
    """Run the complete semantic chunking demonstration."""
    print("🚀 SEMANTIC CHUNKING DEMONSTRATION")
    print("Advanced RAG Multi-Chunking Backend")
    
    # Display model information
    print_model_info()
    
    # Run demonstrations
    await demo_model_comparison()
    await demo_similarity_thresholds() 
    await demo_multilingual_support()
    await demo_performance_analysis()
    
    print_separator("✅ DEMONSTRATION COMPLETE")
    print("""
🎓 KEY TAKEAWAYS:

1. **Model Selection**: Choose based on your speed vs. quality requirements
   - MiniLM models: Fast, good for real-time applications
   - MPNet models: High quality, better for accuracy-critical use cases

2. **Similarity Threshold**: Adjust based on your content diversity
   - Lower (0.5-0.6): Groups diverse but related content
   - Higher (0.8-0.9): Only groups very similar content

3. **Embedding Dimensions**: Higher dimensions generally mean better quality
   - 384-dim: Fast, lower memory usage
   - 768-dim: Better semantic understanding

4. **Use Cases**: Semantic chunking is ideal for:
   - Academic papers with clear topic sections
   - Technical documentation with different concepts
   - Multi-topic articles requiring coherent grouping

5. **Performance**: Consider text size and model choice for your use case
   - Small texts: Any model works well
   - Large texts: Consider MiniLM for speed

🔧 NEXT STEPS:
- Experiment with different models for your specific content type
- Adjust similarity thresholds based on your chunking requirements
- Consider hybrid approaches combining semantic with other chunking strategies
    """)


if __name__ == "__main__":
    print("📋 Prerequisites:")
    print("- Ensure 'sentence-transformers' is installed: pip install sentence-transformers")
    print("- Ensure 'scikit-learn' is installed: pip install scikit-learn")
    print("- First run will download models (~20-500MB depending on model)")
    print("- Stable internet connection recommended for model downloads\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        print("\nPossible solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check internet connection for model downloads")
        print("3. Ensure sufficient disk space for model files")

