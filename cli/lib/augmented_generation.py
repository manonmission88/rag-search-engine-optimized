import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
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

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing; set it in your environment or .env")

gemini_client = genai.Client(api_key=gemini_api_key)
gemini_model = "gemini-2.5-flash"


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


def summarize(query, limit=DEFAULT_SEARCH_LIMIT):
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

    answer = summarize_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": [result for result in search_results[:limit]],
        "answer": answer,
    }


def summarize_answer(search_results, query, limit=3):
    context = ""

    for _, result in search_results[:limit]:
        context += f"{result['title']}: {result['document'][:200]}\n\n"

    prompt = f"""Hoopla is a streaming service. Summarize these movies for the query: {query}

Results:
{context}

Provide a concise 2-3 sentence answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as exc:
        return f"Error from Groq API: {exc}"  

    if not response.choices:
        return ""

    return (response.choices[0].message.content or "").strip()
    
def summarize_command(query):
    return summarize(query)




def citations(query, limit=DEFAULT_SEARCH_LIMIT):
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

    answer = citations_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": [result for result in search_results[:limit]],
        "answer": answer,
    }


def citations_answer(search_results, query, limit=3):
    context = ""

    for _, result in search_results[:limit]:
        context += f"{result['title']}: {result['document'][:200]}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{context}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
    except Exception as exc:
        return f"Error from Groq API: {exc}"

    if not response.choices:
        return ""

    return (response.choices[0].message.content or "").strip()
    
def citations_command(query):
    return citations(query)

def question(question_text, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        question_text, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "question": question_text,
            "search_results": [],
            "error": "No results found",
        }

    answer = question_answer(search_results, question_text, limit)

    return {
        "question": question_text,
        "search_results": [result for result in search_results[:limit]],
        "answer": answer,
    }


def question_answer(search_results, question_text, limit=5):
    context_parts = []
    for _, result in search_results[:limit]:
        context_parts.append(f"{result['title']}: {result['document']}")
    context = "\n\n".join(context_parts)

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question_text}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    try:
        response = gemini_client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )
    except Exception as exc:
        return f"Error from Gemini API: {exc}"

    answer_text = getattr(response, "text", "")
    return (answer_text or "").strip()


def question_command(question_text, limit=DEFAULT_SEARCH_LIMIT):
    return question(question_text, limit)