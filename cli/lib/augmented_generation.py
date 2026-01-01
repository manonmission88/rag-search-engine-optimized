import os
from dotenv import load_dotenv
from openai import OpenAI
from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

# Silence fork/parallellism warning from tokenizers when used before forking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing; set it in your environment or .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = "llama-3.3-70b-versatile"


def generate_answer(search_results, query, limit=5):
    context = ""

    for _, result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Hoopla is a streaming service for movies. You are a RAG agent that provides a human answer
to the user's query based on the documents that were retrieved during search. Provide a comprehensive
answer that addresses the user's query.

Query: {query}

Documents:
{context}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as exc:
        # Surface auth errors instead of crashing the CLI
        return f"Error from Groq API: {exc}"

    if not response.choices:
        return ""

    return (response.choices[0].message.content or "").strip()


def rag(query, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": [result for result in search_results[:limit]],
        "answer": answer,
    }


def rag_command(query):
    return rag(query)
