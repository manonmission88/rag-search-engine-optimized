import os
from time import sleep
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
import json
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = "llama-3.3-70b-versatile"


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query
    except Exception as e:
        print(f"[ERROR] Spell correction failed: {e}")
        logging.warning(f"Spell correction failed: {e}. Returning original query.")
        return query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
        
def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query
    except Exception as e:
        logging.warning(f"Query rewrite failed: {e}. Returning original query.")
        return query

def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
        
def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.

        Examples:

        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        Query: "{query}"
        """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        corrected = (response.choices[0].message.content or "").strip().strip('"')
        return corrected if corrected else query
    except Exception as e:
        logging.warning(f"Query rewrite failed: {e}. Returning original query.")
        return query
    
def llm_rerank_individual(query, documents, limit):
    scored_docs = []
    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:
        """
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)
        return scored_docs[:limit]

def llm_rerank_batch(query, documents, limit):
    
    # Format documents for the prompt
    doc_list_str = "\n".join([
        f"ID: {i}, Title: {doc.get('title', '')}, Content: {doc.get('document', '')}"
        for i, doc in enumerate(documents)
    ])
    
    prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]
"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    score_text = (response.choices[0].message.content or "").strip()
    ranked_ids = json.loads(score_text)
    
    # Sort documents by the new ranking
    ranked_docs = []
    for rank, doc_id in enumerate(ranked_ids):
        if doc_id < len(documents):
            ranked_docs.append({**documents[doc_id], "batch_rank": rank})
    return ranked_docs[:limit]

def llm_rerank_cross_encoder(query, documents, limit):
    """
    Rerank documents using a cross-encoder model.
    
    Args:
        query: Search query string
        documents: List of tuples (doc_id, doc_dict) from rrf_search
        limit: Number of results to return
    """
    pairs = []
    doc_list = []
    for item in documents:
        if isinstance(item, tuple):
            doc_id, doc = item
            doc_list.append((doc_id, doc))
        else:
            doc_list.append((None, item))
        
        # Get the document data
        doc_data = doc_list[-1][1]
        title = doc_data.get('title', '')
        content = doc_data.get('document', '')
        pairs.append([query, f"{title} - {content}"])
    
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    
    # Create list of (doc_id, doc_dict_with_score) and sort by score descending
    scored_docs = []
    for i, (doc_id, doc) in enumerate(doc_list):
        doc_with_score = {**doc, "cross_encoder_score": float(scores[i])}
        scored_docs.append((doc_id, doc_with_score))
    
    # Sort by cross-encoder score in descending order
    ranked_docs = sorted(scored_docs, key=lambda x: x[1]["cross_encoder_score"], reverse=True)
    
    return ranked_docs[:limit]

def reranking_method(query, documents = list[dict],method: str="batch", limit=5):
    match method:
        case "individual":
            return llm_rerank_individual(query, documents, limit)
        case "batch":
            return llm_rerank_batch(query, documents, limit)
        case "cross_encoder":
            return llm_rerank_cross_encoder(query, documents, limit)
        case _:
           return documents[:limit]
    