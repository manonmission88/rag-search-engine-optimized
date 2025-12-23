import os

from .search_keyword import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies
from .enhance_query import (enhance_query,
                            reranking_method)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.hybrid_search_results = {}

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query, alpha=0.5, limit=5):
        """
        Performs a hybrid search combining BM25 and semantic search results.
        
        This method retrieves results from both BM25 (keyword-based) and semantic (embedding-based)
        search algorithms, normalizes their scores, and combines them using a weighted average.
        Documents are ranked by their hybrid score and the top results are returned.
        
        Args:
            query (str): The search query string to be used for both BM25 and semantic search.
            alpha (float, optional): Weight factor for combining search scores. Controls the balance
                between BM25 and semantic search results. Range: [0, 1] where:
                - 0.0: Pure semantic search
                - 0.5: Equal weighting (default)
                - 1.0: Pure BM25 search
                Defaults to 0.5.
            limit (int, optional): Maximum number of results to return. Defaults to 5.
        
        Returns:
            list: A list of tuples containing (document_id, result_dict) for the top-ranked documents.
                Each result_dict contains:
                - id (str): Document identifier
                - title (str): Document title
                - document (str): Document description/content
                - bm25_score (float): Normalized BM25 score [0, 1]
                - semantic_score (float): Normalized semantic search score [0, 1]
                - hybrid_score (float): Combined weighted score used for ranking
        """
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = [doc["score"] for doc in bm25_results]
        normalized_bm25 = normalize(bm25_scores)

        semantic_scores = [doc["score"] for doc in semantic_results]
        normalized_semantic = normalize(semantic_scores)

        combined_scores = {}
        for idx, doc in enumerate(bm25_results):
            doc_id = doc["document"]["id"]
            combined_scores[doc_id] = {
                "title": doc["document"]["title"],
                "document": doc["document"]["description"],
                "bm25_score": normalized_bm25[idx],
                "semantic_score": 0.0,
            }

        for idx, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "title": doc["title"],
                    "document": doc["document"],
                    "bm25_score": 0.0,
                    "semantic_score": normalized_semantic[idx],
                }
            else:
                combined_scores[doc_id]["semantic_score"] = max(
                    combined_scores[doc_id]["semantic_score"],
                    normalized_semantic[idx]
                )

        hybrid_results = []
        for doc_id, data in combined_scores.items():
            hybrid_score = alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]
            hybrid_results.append({
                "id": doc_id,
                "title": data["title"],
                "document": data["document"],
                "bm25_score": data["bm25_score"],
                "semantic_score": data["semantic_score"],
                "hybrid_score": hybrid_score,
            })

        sorted_results = sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True)
        return [(r["id"], r) for r in sorted_results[:limit]]
            
    def rrf_search(self, query, k=60,limit=10):
        """
        Perform a hybrid search using Reciprocal Rank Fusion (RRF) to combine BM25 and semantic search results.
        This method combines results from both BM25 (keyword-based) and semantic (vector-based) search
        by calculating RRF scores for each document. Documents are ranked based on their combined scores
        from both search methods.
        Args:
            query (str): The search query string.
            k (int, optional): The constant parameter used in RRF score calculation. Defaults to 60.
            limit (int, optional): The maximum number of results to return. Defaults to 10.
        Returns:
            list: A list of tuples containing (doc_id, result_dict) sorted by RRF score in descending order.
                  Each result_dict contains:
                    - title (str): The document title
                    - document (str): The document description or content
                    - bm25_rank (int or None): The rank from BM25 search, or None if not in BM25 results
                    - semantic_rank (int or None): The rank from semantic search, or None if not in semantic results
                    - rrf_score (float): The combined RRF score
        """
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        rrf_scores = {}
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc["document"]["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "title": doc["document"]["title"],
                    "document": doc["document"]["description"],
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "rrf_score": 0.0,
                }
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)
        
        for rank, doc in enumerate(semantic_results, 1):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {
                    "title": doc["title"],
                    "document": doc["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": 0.0,
                }
            else:
                rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return sorted_results[:limit]
        
def normalize(scores):
    min_score, max_score = min(scores), max(scores)
    diff = max_score - min_score
    n = len(scores)
    if diff == 0:
        scores = [1.0] * n
    else:
        for idx,score in enumerate(scores):
            scores[idx] = (score-min_score) / diff 
    return scores 
    
def normalize_command(scores):
    """Normalize a list of scores to the range [0, 1] and print them."""
    if not scores:
        print("No scores provided")
        return
    normalized_scores = normalize(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")

def weighted_search_command(query, alpha, limit):
    documents = load_movies()
    hs = HybridSearch(documents)
    results = hs.weighted_search(query, alpha, limit)
    final_result = []
    for _,data in results:
        data = {
            "title" : data['title'],
            "hybrid_score": data['hybrid_score'],
            "bm25_score": data['bm25_score'],
            "semantic_score": data['semantic_score'],
            "document": data['document'][:100]}
        final_result.append(data)
    return final_result 

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_command(query,k=60, enhance=None, rerank_method=None, limit=10):
    
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
    
    search_limit = limit * 5 if rerank_method else limit 
    results = searcher.rrf_search(query, k, search_limit)
    rr = False 
    if rerank_method:
        reranked_results = reranking_method(query, results, method=rerank_method, limit=limit)
        rr = True 
    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": rr,
        "results": results,
    }
