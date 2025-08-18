"""
Cohere reranker implementation.
"""
import os
from typing import List, Optional
import cohere
from .base import BaseReranker
from ..models.document import SearchResult
from ..config.settings import get_settings


class CohereReranker(BaseReranker):
    """
    Cohere reranker implementation using Cohere's rerank API.
    
    This reranker uses Cohere's specialized reranking models to provide
    more accurate relevance scoring for search results.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "rerank-english-v3.0",
        top_k: int = 10,
        relevance_threshold: float = 0.0
    ):
        super().__init__(top_k, relevance_threshold)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("Cohere API key is required. Set COHERE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize Cohere client
        self.client = cohere.Client(api_key=self.api_key)
    
    async def rerank(
        self, 
        query: str, 
        search_results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using Cohere's rerank API.
        
        Args:
            query: The search query
            search_results: List of initial search results to rerank
            top_k: Number of top results to return
            
        Returns:
            List of reranked search results
        """
        if not search_results:
            return []
        
        try:
            # Prepare documents for reranking
            documents = [result.chunk.content for result in search_results]
            
            # Call Cohere rerank API
            rerank_response = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=documents,
                top_k=len(documents),  # Get scores for all documents
                return_documents=False  # We already have the documents
            )
            
            # Create reranked results
            reranked_results = []
            
            for rerank_result in rerank_response.results:
                original_index = rerank_result.index
                rerank_score = rerank_result.relevance_score
                
                # Get the original search result
                original_result = search_results[original_index]
                
                # Create new search result with rerank score
                reranked_result = SearchResult(
                    chunk=original_result.chunk,
                    score=original_result.score,
                    rerank_score=rerank_score
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            # Apply threshold filter
            filtered_results = self._apply_threshold_filter(reranked_results)
            
            # Limit results
            final_results = self._limit_results(filtered_results, top_k)
            
            return final_results
            
        except Exception as e:
            print(f"Error during Cohere reranking: {e}")
            # Fallback to original ranking if reranking fails
            return self._limit_results(search_results, top_k)


class CohereRerankerV2(CohereReranker):
    """Cohere reranker using the v2 model."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = 10,
        relevance_threshold: float = 0.0
    ):
        super().__init__(
            api_key=api_key,
            model_name="rerank-english-v2.0",
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )


class CohereMultilingualReranker(CohereReranker):
    """Cohere reranker with multilingual support."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = 10,
        relevance_threshold: float = 0.0
    ):
        super().__init__(
            api_key=api_key,
            model_name="rerank-multilingual-v3.0",
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )


