from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch


def evaluate_search(datasets, limit):
    """
    Evaluate search results against golden dataset.
    
    Args:
        datasets: List of test cases with 'query' and 'relevant_docs'
        limit: Number of results to evaluate (k for precision@k)
    
    Returns:
        List of evaluation results with query, precision, retrieved, and relevant docs
    """
    documents = load_movies()
    hs = HybridSearch(documents)
    
    results = []
    for data in datasets:
        query = data.get('query', "")
        relevant = data.get('relevant_docs', [])
        search_result = hs.rrf_search(query, 60, limit)
        retrieved_results = [doc['title'] for _, doc in search_result]
        relevant_in_results = sum(1 for title in retrieved_results if title in relevant)
        precision_score = relevant_in_results / limit
        
        results.append({
            'query': query,
            'precision': precision_score,
            'retrieved': retrieved_results,
            'relevant': relevant
        })
    
    return results
