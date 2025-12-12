from .search_utils import CACHE_DIR, load_movies
from sentence_transformers import SentenceTransformer
import numpy as np 
import os 

class SemanticSearch:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
    