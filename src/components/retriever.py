import numpy as np
from typing import List, Dict, Any, Tuple
from .embeddings import EmbeddingManager

class Retriever:
    """Retrieve relevant document chunks based on a query."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The query text
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of tuples containing (chunk_text, similarity_score)
        """
        # Create embedding for the query
        query_embedding = self.embedding_manager.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search the index
        scores, indices = self.embedding_manager.index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Return the chunks and their scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.embedding_manager.chunks):
                results.append((self.embedding_manager.chunks[idx], scores[0][i]))
        
        return results
