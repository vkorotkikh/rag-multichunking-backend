"""
Base reranker interface and common utilities.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.document import SearchResult


class BaseReranker(ABC):
    """Base class for all reranking implementations."""
    
    def __init__(self, top_k: int = 10, relevance_threshold: float = 0.0):
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
    
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        search_results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results based on relevance to the query.
        
        Args:
            query: The search query
            search_results: List of initial search results to rerank
            top_k: Number of top results to return (overrides instance setting)
            
        Returns:
            List of reranked search results
        """
        pass
    
    def _apply_threshold_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on relevance threshold."""
        if self.relevance_threshold <= 0:
            return results
        
        return [
            result for result in results 
            if (result.rerank_score or result.score) >= self.relevance_threshold
        ]
    
    def _limit_results(self, results: List[SearchResult], top_k: Optional[int] = None) -> List[SearchResult]:
        """Limit results to top_k."""
        limit = top_k or self.top_k
        return results[:limit] if limit > 0 else results


