from typing import List, Dict, Any
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class EmbeddingManager:
    """Manage embeddings for document chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype('float32'))
    
    def save_index(self, path: str) -> None:
        """Save the FAISS index and chunks to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save the chunks
        with open(f"{path}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load_index(self, path: str) -> None:
        """Load the FAISS index and chunks from disk."""
        # Load the FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load the chunks
        with open(f"{path}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)
