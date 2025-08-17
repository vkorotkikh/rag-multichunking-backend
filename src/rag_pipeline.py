"""
Main RAG pipeline orchestrating chunking, embedding, vector storage, and reranking.
"""
import time
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from .models.document import (
    Document, DocumentChunk, QueryRequest, QueryResponse, 
    IndexRequest, IndexResponse, SearchResult, ChunkingStrategy, VectorStore
)
from .chunkers import ChunkerFactory, create_chunker
from .vector_stores import VectorStoreFactory, create_vector_store_from_settings
from .rerankers import RerankerFactory, RerankerType, create_reranker_from_settings
from .utils.embeddings import EmbeddingService, get_embedding_service
from .config.settings import get_settings


class RAGPipeline:
    """
    Main RAG pipeline for document processing and retrieval.
    
    This class orchestrates the entire RAG workflow:
    1. Document chunking using various strategies
    2. Embedding generation
    3. Vector storage and indexing
    4. Query processing and retrieval
    5. Reranking of results
    6. Response generation using OpenAI o3
    """
    
    def __init__(
        self,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        vector_store_type: VectorStore = VectorStore.PINECONE,
        reranker_type: RerankerType = RerankerType.COHERE,
        use_reranking: bool = True
    ):
        self.settings = get_settings()
        self.chunking_strategy = chunking_strategy
        self.vector_store_type = vector_store_type
        self.reranker_type = reranker_type
        self.use_reranking = use_reranking
        
        # Initialize components
        self.embedding_service = get_embedding_service()
        self.vector_store = create_vector_store_from_settings(vector_store_type)
        
        if use_reranking:
            self.reranker = create_reranker_from_settings(reranker_type)
        else:
            self.reranker = None
        
        # Initialize OpenAI client for response generation
        self.openai_client = AsyncOpenAI(api_key=self.settings.openai.api_key)
        
        # Default index name
        self.default_index = "rag_documents"
    
    async def index_documents(self, request: IndexRequest) -> IndexResponse:
        """
        Index documents into the vector store.
        
        Args:
            request: Index request containing documents and configuration
            
        Returns:
            Index response with processing results
        """
        start_time = time.time()
        
        try:
            # Create chunker
            chunker = create_chunker(
                strategy=request.chunking_strategy,
                chunk_size=self.settings.chunking.chunk_size,
                chunk_overlap=self.settings.chunking.chunk_overlap
            )
            
            # Chunk all documents
            all_chunks = []
            for document in request.documents:
                chunks = chunker.chunk(document)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return IndexResponse(
                    indexed_documents=0,
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    chunking_strategy=request.chunking_strategy,
                    vector_store=request.vector_store,
                    errors=["No chunks generated from documents"]
                )
            
            # Generate embeddings
            embedded_chunks = await self.embedding_service.embed_chunks(all_chunks)
            
            # Filter out chunks without embeddings
            valid_chunks = [chunk for chunk in embedded_chunks if chunk.embedding]
            
            if not valid_chunks:
                return IndexResponse(
                    indexed_documents=len(request.documents),
                    total_chunks=0,
                    processing_time=time.time() - start_time,
                    chunking_strategy=request.chunking_strategy,
                    vector_store=request.vector_store,
                    errors=["No valid embeddings generated"]
                )
            
            # Ensure index exists
            index_name = self.default_index
            if not await self.vector_store.index_exists(index_name):
                await self.vector_store.create_index(index_name)
            
            # Upsert chunks to vector store
            success = await self.vector_store.upsert_chunks(
                chunks=valid_chunks,
                index_name=index_name,
                batch_size=request.batch_size
            )
            
            errors = [] if success else ["Failed to upsert chunks to vector store"]
            
            return IndexResponse(
                indexed_documents=len(request.documents),
                total_chunks=len(valid_chunks),
                processing_time=time.time() - start_time,
                chunking_strategy=request.chunking_strategy,
                vector_store=request.vector_store,
                errors=errors
            )
            
        except Exception as e:
            return IndexResponse(
                indexed_documents=0,
                total_chunks=0,
                processing_time=time.time() - start_time,
                chunking_strategy=request.chunking_strategy,
                vector_store=request.vector_store,
                errors=[f"Indexing failed: {str(e)}"]
            )
    
    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query and return relevant results.
        
        Args:
            request: Query request with search parameters
            
        Returns:
            Query response with search results
        """
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding_async(request.query)
            
            if not query_embedding:
                return QueryResponse(
                    query=request.query,
                    results=[],
                    total_results=0,
                    processing_time=time.time() - start_time,
                    chunking_strategy=request.chunking_strategy,
                    vector_store=request.vector_store,
                    reranking_used=False
                )
            
            # Search vector store
            search_k = request.rerank_top_k if request.use_reranking else request.top_k
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                index_name=self.default_index,
                top_k=search_k,
                filters=request.filters
            )
            
            # Apply reranking if requested
            final_results = search_results
            reranking_used = False
            
            if request.use_reranking and self.reranker and search_results:
                try:
                    reranked_results = await self.reranker.rerank(
                        query=request.query,
                        search_results=search_results,
                        top_k=request.top_k
                    )
                    final_results = reranked_results
                    reranking_used = True
                except Exception as e:
                    print(f"Reranking failed, using original results: {e}")
                    final_results = search_results[:request.top_k]
            else:
                final_results = search_results[:request.top_k]
            
            return QueryResponse(
                query=request.query,
                results=final_results,
                total_results=len(final_results),
                processing_time=time.time() - start_time,
                chunking_strategy=request.chunking_strategy,
                vector_store=request.vector_store,
                reranking_used=reranking_used
            )
            
        except Exception as e:
            return QueryResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_time=time.time() - start_time,
                chunking_strategy=request.chunking_strategy,
                vector_store=request.vector_store,
                reranking_used=False
            )
    
    async def generate_response(
        self, 
        query: str, 
        search_results: List[SearchResult],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using OpenAI o3 based on query and search results.
        
        Args:
            query: User query
            search_results: Relevant search results
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response
        """
        try:
            # Prepare context from search results
            context_pieces = []
            for i, result in enumerate(search_results, 1):
                score_info = f"[Score: {result.rerank_score or result.score:.3f}]"
                context_pieces.append(f"[{i}] {score_info} {result.chunk.content}")
            
            context = "\n\n".join(context_pieces)
            
            # Default system prompt
            default_system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively. 

Guidelines:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, say so
- Cite relevant parts of the context when appropriate
- Be concise but thorough
- If asked about sources, refer to the numbered context pieces"""

            system_message = system_prompt or default_system_prompt
            
            # Prepare user message
            user_message = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""

            # Generate response using OpenAI o3
            response = await self.openai_client.chat.completions.create(
                model=self.settings.openai.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.settings.openai.temperature,
                max_tokens=max_tokens or self.settings.openai.max_tokens
            )
            
            return response.choices[0].message.content or "I couldn't generate a response."
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    
    async def query_and_generate(
        self, 
        query: str,
        top_k: int = 10,
        use_reranking: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG workflow: query, retrieve, and generate response.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            use_reranking: Whether to use reranking (overrides instance setting)
            filters: Optional metadata filters
            system_prompt: Optional system prompt for response generation
            
        Returns:
            Dictionary containing query results and generated response
        """
        # Create query request
        request = QueryRequest(
            query=query,
            top_k=top_k,
            chunking_strategy=self.chunking_strategy,
            vector_store=self.vector_store_type,
            use_reranking=use_reranking if use_reranking is not None else self.use_reranking,
            filters=filters or {}
        )
        
        # Get search results
        query_response = await self.query(request)
        
        # Generate response if we have results
        generated_response = ""
        if query_response.results:
            generated_response = await self.generate_response(
                query=query,
                search_results=query_response.results,
                system_prompt=system_prompt
            )
        else:
            generated_response = "I couldn't find any relevant information to answer your question."
        
        return {
            "query": query,
            "search_results": query_response,
            "generated_response": generated_response,
            "total_processing_time": query_response.processing_time
        }
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        try:
            stats = await self.vector_store.get_index_stats(self.default_index)
            return {
                "index_name": self.default_index,
                "vector_store": self.vector_store_type.value,
                "chunking_strategy": self.chunking_strategy.value,
                "reranking_enabled": self.use_reranking,
                "reranker_type": self.reranker_type.value if self.use_reranking else None,
                "stats": stats
            }
        except Exception as e:
            return {
                "error": f"Failed to get index info: {str(e)}"
            }


# Convenience function for creating pipeline
def create_rag_pipeline(
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    vector_store_type: VectorStore = VectorStore.PINECONE,
    reranker_type: RerankerType = RerankerType.COHERE,
    use_reranking: bool = True
) -> RAGPipeline:
    """Create a RAG pipeline with specified configuration."""
    return RAGPipeline(
        chunking_strategy=chunking_strategy,
        vector_store_type=vector_store_type,
        reranker_type=reranker_type,
        use_reranking=use_reranking
    )


