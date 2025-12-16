from .search_utils import CACHE_DIR, load_movies
from sentence_transformers import SentenceTransformer
import numpy as np 
import os 
import re 
from collections import defaultdict
import json 

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 200
class SemanticSearch:
    def __init__(self):
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.embeddings = None 
        self.documents = None 
        self.document_map = {}
        self.movie_embeddings_path =  os.path.join(CACHE_DIR, "movie_embeddings.npy")
    
    def generate_embedding(self,text):
        if not text or not text.strip():
            raise ValueError("text should not be empty")
        return self.embedding_model.encode([text])[0]
    
    def build_embeddings(self,documents):
        self.documents = documents
        texts = []
        for doc in self.documents:
            if doc["id"] not in self.document_map:
                self.document_map[doc["id"]] = doc
            s = f"{doc['title']}: {doc['description']}"
            texts.append(s)
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        os.makedirs(os.path.dirname(self.movie_embeddings_path), exist_ok=True)
        np.save(self.movie_embeddings_path,self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self,documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
            
        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self,query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        top_results = []
        for id,doc_embed in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embed)
            top_results.append((similarity_score,self.documents[id]))
        top_results.sort(key=lambda x: x[0], reverse= True)
        final_results = []
        for score,doc in top_results:
            search_result = {
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            }
            final_results.append(search_result)
        return final_results[:limit]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
    
    def build_chunk_embeddings(self,documents):
        self.documents = documents
        texts = []
        for doc in self.documents:
            if doc["id"] not in self.document_map:
                self.document_map[doc["id"]] = doc
            s = f"{doc['title']}: {doc['description']}"
            texts.append(s)
        all_chunks = []
        metadata = []
        for doc in self.documents:
            if len(doc["description"]) == 0:
                continue 
            semantic_chunks = semantic_chunk(doc["description"],chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(semantic_chunks):
                all_chunks.append(chunk)
                data = {"movie_idx": doc["id"],
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(semantic_chunks)}
                metadata.append(data)
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        self.chunk_embeddings = embeddings
        os.makedirs(os.path.dirname(self.chunk_embeddings_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.chunk_metadata_path), exist_ok=True)
        np.save(self.chunk_embeddings_path,self.chunk_embeddings)
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": metadata, 
                   "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
    
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query, limit=10):
        """Search chunks using query embedding and aggregate by document.
        
        Args:
            query (str): Search query text.
            limit (int): Maximum number of results to return (default: 10).
        
        Returns:
            list[dict]: Top-matching documents with scores and metadata.
        """
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for global_idx, chunk_meta in enumerate(self.chunk_metadata):
            chunk_embedding = self.chunk_embeddings[global_idx] 
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            data = {
                "chunk_idx": chunk_meta["chunk_idx"],
                "movie_idx": chunk_meta["movie_idx"],
                "score": similarity_score
            }
            chunk_scores.append(data)
        
        movie_idx_to_score = {}
        for chunk_score in chunk_scores:
            movie_id = chunk_score["movie_idx"]
            if movie_id not in movie_idx_to_score:
                movie_idx_to_score[movie_id] = chunk_score["score"]
            else:
                movie_idx_to_score[movie_id] = max(
                    movie_idx_to_score[movie_id], 
                    chunk_score["score"]
                )
        sorted_scores = sorted(
            movie_idx_to_score.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:limit]
        
        final_results = []
        for movie_id, score in sorted_scores:
            doc = self.document_map[movie_id] 
            data = {
                "id": movie_id,
                "title": doc['title'],
                "document": doc['description'][:100], 
                "score": round(score, 2),
                "metadata": {}
            }
            final_results.append(data)
        return final_results
            
        
def chunk_text(text, chunk_size, overlap_value):
    words = text.split()
    chunks = []
    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap_value:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap_value
    return chunks
       
def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic.embedding_model}")
    print(f"Max sequence length: {semantic.embedding_model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    returning a value between -1 and 1, where 1 indicates identical direction,
    0 indicates orthogonal vectors, and -1 indicates opposite direction.
    
    Args:
        vec1 (array-like): First vector (1D numpy array or list).
        vec2 (array-like): Second vector (1D numpy array or list).
    
    Returns:
        float: Cosine similarity score between the two vectors.
               Returns 0.0 if either vector has zero magnitude (norm).
    
    Examples:
        >>> vec1 = np.array([1, 0, 0])
        >>> vec2 = np.array([1, 0, 0])
        >>> cosine_similarity(vec1, vec2)
        1.0
        
        >>> vec1 = np.array([1, 0, 0])
        >>> vec2 = np.array([0, 1, 0])
        >>> cosine_similarity(vec1, vec2)
        0.0
    """
    dot_product = np.dot(vec1,vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def search_command(query,limit):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_results = ss.search(query, limit)
    for idx, search_result in enumerate(search_results):
        print(f"{idx}. {search_result["title"]} (score: {search_result["score"]})")
        print(f" {search_result["description"][:100]}...")
        print()
        idx += 1

def chunk_command(text,chunk_size,overlap_value):
    chunk_texts = chunk_text(text,chunk_size, overlap_value)
    print(f"Chunking {len(text)} characters")
    for idx, description in enumerate(chunk_texts):
        print(f"{idx+1}. {description}\n")

def semantic_chunk_command(text,chunk_size,overlap):
    final_chunks = semantic_chunk(text,chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for idx,sent in enumerate(final_chunks):
        print(f"{idx+1}. {sent}")
        
def semantic_chunk(text, chunk_size, overlap):
    """Split text into semantic chunks based on sentences.
    
    Args:
        text (str): Text to chunk.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of sentences to overlap between chunks.
    
    Returns:
        list[str]: List of text chunks.
    """
    updated_text = text.strip()
    if len(updated_text)==0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    final_chunks = []
    n_sentences = len(sentences)
    punc = [".",'!', "?"]
    if n_sentences == 1:
        if not n_sentences[-1].endswith(punc):
            return [updated_text]
    i = 0
    while i < n_sentences:
        chunk_sentence = sentences[i : i + chunk_size]
        if final_chunks and len(chunk_sentence) <= overlap:
            break
        sentence = " ".join(chunk_sentence).strip()
        if len(sentence) > 0:
            final_chunks.append(sentence)
        i += chunk_size - overlap
    return final_chunks

def embed_chunks_command():
    chunk_ss = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = chunk_ss.load_or_create_chunk_embeddings(documents)
    return embeddings

def search_chunked_command(query, limit):
    documents = load_movies()
    chunk_ss = ChunkedSemanticSearch()
    embeddings = chunk_ss.load_or_create_chunk_embeddings(documents)
    search_results = chunk_ss.search_chunks(query,limit)
    for idx,data in enumerate(search_results):
        print(f"\n{idx}. {data["title"]} (score: {data["score"]:.4f})")
        print(f"   {data["document"]}...")

