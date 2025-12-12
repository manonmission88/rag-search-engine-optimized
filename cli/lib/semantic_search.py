from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic.embedding_model}")
    print(f"Max sequence length: {semantic.embedding_model.max_seq_length}")