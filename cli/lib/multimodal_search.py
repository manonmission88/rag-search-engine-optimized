import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from lib.search_utils import load_movies


class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        
        # Create texts by concatenating title and description for each document
        if self.documents:
            self.texts = [
                f"{doc['title']}: {doc['description']}" 
                for doc in self.documents
            ]
            
            # Generate embeddings for all texts
            self.text_embeddings = self.model.encode(
                self.texts, 
                show_progress_bar=True
            )
        else:
            self.texts = []
            self.text_embeddings = None
    
    def embed_image(self, image_path: str):
        """Generate an embedding for an image at the given path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embedding as a numpy array
        """
        img = Image.open(image_path)
        embedding = self.model.encode([img])[0]
        return embedding
    
    def search_with_image(self, image_path: str):
        """Search for similar documents using an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of top 5 matching documents with similarity scores
        """
        # Generate embedding for the image
        image_embedding = self.embed_image(image_path)
        
        # Calculate cosine similarity between image and all text embeddings
        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            # Reshape for cosine_similarity function
            sim = cosine_similarity(
                image_embedding.reshape(1, -1),
                text_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append({
                'id': self.documents[i]['id'],
                'title': self.documents[i]['title'],
                'description': self.documents[i]['description'],
                'similarity': sim
            })
        
        # Sort by similarity in descending order and return top 5
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]
    
    # def embed_text(self, text: str):
    #     """Generate an embedding for a text string.
        
    #     Args:
    #         text: Text to embed
            
    #     Returns:
    #         Text embedding as a numpy array
    #     """
    #     embedding = self.model.encode([text])[0]
    #     return embedding


def verify_image_embedding(image_path: str):
    """Verify image embedding generation by creating an embedding and printing its shape.
    
    Args:
        image_path: Path to the image file
    """
    searcher = MultiModalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    """Search for movies using an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of top matching movies with similarity scores
    """
    documents = load_movies()
    searcher = MultiModalSearch(documents=documents)
    results = searcher.search_with_image(image_path)
    return results 