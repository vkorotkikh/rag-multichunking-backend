"""
BM25 reranker implementation.
"""
import re
from typing import List, Optional, Set
from rank_bm25 import BM25Okapi
from .base import BaseReranker
from ..models.document import SearchResult


class BM25Reranker(BaseReranker):
    """
    BM25 reranker implementation using traditional keyword-based ranking.
    
    This reranker uses the BM25 algorithm to provide keyword-based relevance
    scoring as an alternative to neural reranking methods.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        relevance_threshold: float = 0.0,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        super().__init__(top_k, relevance_threshold)
        self.k1 = k1  # Controls term frequency normalization
        self.b = b    # Controls length normalization
        self.epsilon = epsilon  # Floor value for IDF
        
        # Preprocessing patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        self.stopwords = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self) -> Set[str]:
        """Get a basic set of English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'i', 'you', 'we', 'they',
            'this', 'these', 'those', 'but', 'or', 'not', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'have', 'had'
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 scoring.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase and extract words
        words = self.word_pattern.findall(text.lower())
        
        # Remove stopwords and very short words
        filtered_words = [
            word for word in words 
            if word not in self.stopwords and len(word) > 2
        ]
        
        return filtered_words
    
    async def rerank(
        self, 
        query: str, 
        search_results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using BM25 algorithm.
        
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
            # Preprocess query
            query_tokens = self._preprocess_text(query)
            
            if not query_tokens:
                # If no valid query tokens, return original ranking
                return self._limit_results(search_results, top_k)
            
            # Preprocess documents
            documents = []
            for result in search_results:
                doc_tokens = self._preprocess_text(result.chunk.content)
                documents.append(doc_tokens)
            
            # Initialize BM25
            bm25 = BM25Okapi(
                documents, 
                k1=self.k1, 
                b=self.b, 
                epsilon=self.epsilon
            )
            
            # Calculate BM25 scores
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Create reranked results
            reranked_results = []
            for i, (result, bm25_score) in enumerate(zip(search_results, bm25_scores)):
                # Normalize BM25 score to 0-1 range (approximately)
                normalized_score = min(1.0, max(0.0, bm25_score / 10.0))
                
                reranked_result = SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    rerank_score=normalized_score
                )
                reranked_results.append(reranked_result)
            
            # Sort by BM25 score (descending)
            reranked_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            # Apply threshold filter
            filtered_results = self._apply_threshold_filter(reranked_results)
            
            # Limit results
            final_results = self._limit_results(filtered_results, top_k)
            
            return final_results
            
        except Exception as e:
            print(f"Error during BM25 reranking: {e}")
            # Fallback to original ranking if reranking fails
            return self._limit_results(search_results, top_k)


class BM25PlusReranker(BM25Reranker):
    """
    BM25+ reranker variant with improved handling of long documents.
    """
    
    def __init__(
        self,
        top_k: int = 10,
        relevance_threshold: float = 0.0,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0
    ):
        super().__init__(top_k, relevance_threshold, k1, b)
        self.delta = delta  # Additional parameter for BM25+
    
    async def rerank(
        self, 
        query: str, 
        search_results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Rerank using BM25+ variant."""
        if not search_results:
            return []
        
        try:
            # Use the same preprocessing as base BM25
            query_tokens = self._preprocess_text(query)
            
            if not query_tokens:
                return self._limit_results(search_results, top_k)
            
            # Preprocess documents
            documents = []
            for result in search_results:
                doc_tokens = self._preprocess_text(result.chunk.content)
                documents.append(doc_tokens)
            
            # Manual BM25+ implementation
            scores = self._calculate_bm25_plus_scores(query_tokens, documents)
            
            # Create reranked results
            reranked_results = []
            for i, (result, score) in enumerate(zip(search_results, scores)):
                # Normalize score
                normalized_score = min(1.0, max(0.0, score / 10.0))
                
                reranked_result = SearchResult(
                    chunk=result.chunk,
                    score=result.score,
                    rerank_score=normalized_score
                )
                reranked_results.append(reranked_result)
            
            # Sort by score (descending)
            reranked_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            # Apply threshold filter and limit
            filtered_results = self._apply_threshold_filter(reranked_results)
            final_results = self._limit_results(filtered_results, top_k)
            
            return final_results
            
        except Exception as e:
            print(f"Error during BM25+ reranking: {e}")
            return self._limit_results(search_results, top_k)
    
    def _calculate_bm25_plus_scores(self, query_tokens: List[str], documents: List[List[str]]) -> List[float]:
        """Calculate BM25+ scores manually."""
        if not documents:
            return []
        
        # Calculate document frequencies
        doc_freqs = {}
        total_docs = len(documents)
        
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        
        # Calculate average document length
        total_length = sum(len(doc) for doc in documents)
        avg_doc_length = total_length / total_docs if total_docs > 0 else 0
        
        # Calculate scores for each document
        scores = []
        for doc in documents:
            score = 0.0
            doc_length = len(doc)
            
            # Count term frequencies in document
            term_freqs = {}
            for term in doc:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            
            for query_term in query_tokens:
                if query_term in term_freqs:
                    # Calculate IDF
                    df = doc_freqs.get(query_term, 0)
                    idf = max(self.epsilon, (total_docs - df + 0.5) / (df + 0.5))
                    
                    # Calculate term frequency component
                    tf = term_freqs[query_term]
                    
                    # BM25+ formula
                    tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length)))
                    
                    # Add delta for BM25+
                    score += idf * (tf_component + self.delta)
            
            scores.append(score)
        
        return scores


