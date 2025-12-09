from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words, CACHE_DIR
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
import pickle

"""
Inverted Index Search Engine

This module implements an inverted index data structure for efficient keyword-based
search across movie documents. It includes text preprocessing, tokenization, and
document retrieval capabilities.

Key Components:
- InvertedIndex: Core data structure mapping terms to document IDs
- Tokenization: Text normalization with stopword removal and stemming
- Search: Query matching against indexed movie titles and descriptions

"""
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for word in set(tokens):
            self.index[word].add(doc_id)
        
    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            job_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, job_description)
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        # dumping our data into the disk (need to convert into binary format)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        
        
def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    if docs:
        print(f"First document for token 'merida' = {docs[0]}")
    else:
        # Helpful debug output when seed term is absent.
        print("No documents found for token 'merida'")
    
    
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = tokenize(query)
    for movie in movies:
        movie_tokens = tokenize(movie["title"])
        if tokenization_compare(query_tokens, movie_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text
    
def tokenization_compare(tokens1: list[str], tokens2: list[str]) -> bool:
    for query_token in tokens1:
        for title_token in tokens2:
            if query_token in title_token:
                return True
    return False
    
    
def filter_query(tokens: list[str]) -> list[str]:
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    filtered = []
    for word in tokens:
        if word in stop_words:
            continue
        filtered.append(stemmer.stem(word))
    return filtered


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, and stem tokens."""
    normalized = preprocess_text(text)
    raw_tokens = normalized.split()
    return filter_query(raw_tokens)