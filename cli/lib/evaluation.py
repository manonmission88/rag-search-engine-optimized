from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from dotenv import load_dotenv
from openai import OpenAI
import os 
import json 

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = "llama-3.3-70b-versatile"
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
        recall_score = relevant_in_results / len(relevant)
        f1_score = 2 * ((precision_score * recall_score) / (precision_score+recall_score))
        
        results.append({
            'query': query,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'retrieved': retrieved_results,
            'relevant': relevant
        })
    
    return results


def llm_evaluate_results(query: str, results: list[tuple]) -> list[int]:
    """
    Use LLM to evaluate search result relevance.
    
    Args:
        query: The search query
        results: List of (doc_id, doc_data) tuples from search results
    
    Returns:
        List of relevance scores (0-3) for each result
    """
    if not api_key:
        print("Warning: GROQ_API_KEY not found. Skipping LLM evaluation.")
        return [0] * len(results)

    formatted_results = []
    for i, (_, doc_data) in enumerate(results, 1):
        formatted_results.append(f"{i}. {doc_data['title']}")

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    ranking_text = response.choices[0].message.content.strip()
    scores = json.loads(ranking_text)

    if len(scores) == len(results):
        return list(map(int, scores))

    raise ValueError(
        f"LLM response parsing error. Expected {len(results)} scores, got {len(scores)}. Response: {scores}"
    )