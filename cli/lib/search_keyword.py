from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words, CACHE_DIR
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os
import pickle
import math 

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
BM25_K1 = 1.5 
class InvertedIndex:
    """Inverted index data structure for efficient keyword-based document search.
    
    Maps terms to document IDs and maintains term frequency counts for BM25 scoring.
    Supports building, saving, and loading indices from disk.
    """
    
    def __init__(self):
        """Initialize an empty inverted index with default file paths."""
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        
    def __add_document(self, doc_id, text):
        """Add a document to the index by tokenizing and indexing its terms.
        
        Args:
            doc_id (int): Unique identifier for the document.
            text (str): Document text to tokenize and index.
        """
        tokens = tokenize(text)
        for word in set(tokens):
            self.index[word].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        
    def get_documents(self, term):
        """Retrieve all document IDs containing a given term.
        
        Args:
            term (str): The term to search for (case-insensitive).
            
        Returns:
            list[int]: Sorted list of document IDs containing the term.
        """
        doc_ids = self.index.get(term.lower(), set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term):
        """Get the term frequency (TF) of a term in a document.
        
        Args:
            doc_id (int or str): Document identifier (converted to int if string).
            term (str): Single term to count (multi-word terms raise an exception).
            
        Returns:
            int: Frequency count of the term in the document, or 0 if not found.
            
        Raises:
            Exception: If term tokenizes to more than one token.
        """
        # Convert doc_id to int if it's a string
        doc_id = int(doc_id)
        
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("Cannot be more than one")
        token = tokens[0]
        
        # Return 0 if doc doesn't exist or term not found
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(token, 0)
    
    def get_bm25_idf(self, term):
        """Calculate BM25 Inverse Document Frequency (IDF) score for a term.
        
        Args:
            term (str): Single term to calculate IDF for.
            
        Returns:
            float: BM25 IDF score.
            
        Raises:
            KeyError: If term tokenizes to more than one token.
        """
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise KeyError(" Token should not be more than one")
        total_doc_count = len(self.docmap)
        token = tokens[0]
        total_term_count = len(self.index[token])
        bm25_score = math.log((total_doc_count-total_term_count+0.5)/(total_term_count+0.5)+1)
        return bm25_score
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        """Calculate BM25 saturation-adjusted term frequency score.
        
        Args:
            doc_id (int or str): Document identifier.
            term (str): Term to score.
            k1 (float): BM25 saturation parameter (default: BM25_K1).
            
        Returns:
            float: Saturation-adjusted TF score.
        """
        tf_score = self.get_tf(doc_id, term)
        return (tf_score * (k1+1)) / (tf_score + k1)
        
    def build(self):
        """Build the inverted index from all loaded movies.
        
        Loads all movies, concatenates title and description for each,
        and adds them to both the index and document map.
        """
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            job_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, job_description)
    
    def save(self):
        """Save the index, document map, and term frequencies to disk as pickle files.
        
        Creates the cache directory if it doesn't exist. Pickles are written to:
        - cache/index.pkl
        - cache/docmap.pkl
        - cache/term_frequencies.pkl
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        # dumping our data into the disk (need to convert into binary format)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self):
        """Load the index, document map, and term frequencies from disk.
        
        Reads pickle files from cache directory. Index must have been
        previously built and saved.
        """
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
                    
                    
def build_command():
    """CLI command to build and save the inverted index from movies."""
    idx = InvertedIndex()
    idx.build()
    idx.save()
    
def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search the index for documents matching query tokens.
    
    Args:
        query (str): Search query string.
        limit (int): Maximum number of results to return (default: DEFAULT_SEARCH_LIMIT).
        
    Returns:
        list[dict]: List of movie documents matching the query, up to limit.
    """
    idx = InvertedIndex()
    movies = idx.load()
    seen, results = set(), []
    query_tokens = tokenize(query)
    for query in query_tokens:
        get_docs = idx.get_documents(query)
        for doc_id in get_docs:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            text = idx.docmap[doc_id]
            if len(results) >= limit:
                break
            results.append(text)
    return results

def tf_command(doc_id, term):
    """CLI command to get term frequency for a document.
    
    Args:
        doc_id (int or str): Document identifier.
        term (str): Term to count.
        
    Returns:
        int: Frequency count of term in document.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)     

def idf_command_score(term):
    """CLI command to get BM25 IDF score for a term.
    
    Args:
        term (str): Term to score.
        
    Returns:
        float: BM25 IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=None):
    """CLI command to get BM25 saturation-adjusted TF score.
    
    Args:
        doc_id (int or str): Document identifier.
        term (str): Term to score.
        k1 (float, optional): BM25 saturation parameter. Uses default if None.
        
    Returns:
        float: BM25 saturation-adjusted TF score.
    """
    idx = InvertedIndex()
    idx.load()
    if k1 is not None:
        saturated_tf_score = idx.get_bm25_tf(doc_id, term, k1)
    else:
        saturated_tf_score = idx.get_bm25_tf(doc_id, term)
    return saturated_tf_score

def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and removing punctuation.
    
    Args:
        text (str): Text to preprocess.
        
    Returns:
        str: Lowercased text with punctuation removed.
    """
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text
    
def tokenization_compare(tokens1: list[str], tokens2: list[str]) -> bool:
    """Check if any token from tokens1 is a substring of any token in tokens2.
    
    Args:
        tokens1 (list[str]): Query tokens.
        tokens2 (list[str]): Document tokens.
        
    Returns:
        bool: True if any query token is found in any document token, False otherwise.
    """
    for query_token in tokens1:
        for title_token in tokens2:
            if query_token in title_token:
                return True
    return False
    
    
def filter_query(tokens: list[str]) -> list[str]:
    """Filter tokens by removing stopwords and applying stemming.
    
    Args:
        tokens (list[str]): Raw tokens to filter.
        
    Returns:
        list[str]: Stemmed tokens with stopwords removed.
    """
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    filtered = []
    for word in tokens:
        if word in stop_words:
            continue
        filtered.append(stemmer.stem(word))
    return filtered


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, strip punctuation, remove stopwords, and stem.
    
    Applies full text normalization pipeline for consistent token generation.
    
    Args:
        text (str): Text to tokenize.
        
    Returns:
        list[str]: List of normalized, stemmed tokens.
    """
    normalized = preprocess_text(text)
    raw_tokens = normalized.split()
    return filter_query(raw_tokens)