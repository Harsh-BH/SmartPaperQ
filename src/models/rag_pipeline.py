import os
import time
import hashlib
from typing import Dict, Any, List, Tuple
import pickle
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import groq

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for paper Q&A"""
    
    def __init__(self):
        """Initialize the RAG pipeline with models and storage"""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create cache directories
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize document store
        self.documents = []
        self.embeddings = []
        self.document_id = None
        
        # Initialize Groq client if API key exists
        self.generator = GroqGenerator()
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            # Make sure we don't cut words
            if end < text_length:
                # Find the last space before the end
                while end > start and text[end] != ' ':
                    end -= 1
                if end == start:  # No space found, just use the original end
                    end = min(start + chunk_size, text_length)
            
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > 0 else end
        
        return chunks
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() or ""
        return text
    
    def _compute_document_id(self, text: str) -> str:
        """Compute unique ID for document based on content"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _save_to_cache(self, document_id: str, chunks: List[str], embeddings: np.ndarray) -> None:
        """Save document chunks and embeddings to cache"""
        cache_file = os.path.join(self.cache_dir, f"{document_id}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    
    def _load_from_cache(self, document_id: str) -> Tuple[List[str], np.ndarray]:
        """Load document chunks and embeddings from cache"""
        cache_file = os.path.join(self.cache_dir, f"{document_id}.pkl")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['chunks'], data['embeddings']
    
    def process_document(self, pdf_path: str, use_cache: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """Process a PDF document for RAG"""
        start_time = time.time()
        
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                return {"status": "error", "message": "Could not extract text from PDF"}
            
            # Generate document ID
            document_id = self._compute_document_id(text)
            self.document_id = document_id
            
            # Check cache
            cache_file = os.path.join(self.cache_dir, f"{document_id}.pkl")
            if use_cache and os.path.exists(cache_file):
                # Load from cache
                self.documents, self.embeddings = self._load_from_cache(document_id)
                processing_time = time.time() - start_time
                return {
                    "status": "success", 
                    "message": "Document loaded from cache", 
                    "processing_time": processing_time,
                    "chunk_count": len(self.documents)
                }
            
            # Chunk document - passing custom chunk size and overlap
            self.documents = self._chunk_document(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not self.documents:
                return {"status": "error", "message": "Document chunking failed"}
            
            # Compute embeddings
            self.embeddings = self.embedder.encode(self.documents)
            
            # Save to cache
            self._save_to_cache(document_id, self.documents, self.embeddings)
            
            processing_time = time.time() - start_time
            return {
                "status": "success", 
                "message": "Document processed successfully", 
                "processing_time": processing_time,
                "chunk_count": len(self.documents)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _retrieve_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve relevant document chunks for a query"""
        if not self.documents or not self.embeddings:
            return []
        
        # Compute query embedding
        query_embedding = self.embedder.encode(query)
        
        # Calculate similarity scores
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top chunks
        top_indices = np.argsort(-scores)[:top_k]
        
        # Return top chunks with scores
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG"""
        try:
            if not self.documents or not self.embeddings:
                return {
                    "status": "error", 
                    "message": "No document processed. Please upload and process a document first."
                }
            
            # Retrieve relevant chunks
            chunks = self._retrieve_chunks(question)
            if not chunks:
                return {
                    "status": "error", 
                    "message": "Could not retrieve relevant information from document."
                }
            
            # Format context for LLM
            context = "\n\n---\n\n".join([chunk for chunk, _ in chunks])
            
            # Generate answer
            answer = self.generator.generate_answer(question, context)
            
            return {
                "status": "success",
                "answer": answer,
                "sources": chunks
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GroqGenerator:
    """Generates answers using Groq API"""
    
    def __init__(self):
        # Get API key from environment variables
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = "llama3-70b-8192"  # Default model
        
        # Initialize client if API key exists
        if self.api_key:
            self.client = groq.Client(api_key=self.api_key)
        else:
            print("Warning: GROQ_API_KEY not found in environment variables")
            self.client = None
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer for a question given context"""
        if not self.client:
            return "Error: Groq API key not configured. Please add GROQ_API_KEY to your environment variables."
        
        # Create system prompt
        system_prompt = """You are a helpful research assistant. You answer questions about academic papers based on the provided context.
Answer the question using only the provided context. If the context doesn't contain the relevant information, say "I don't have enough information to answer this question."
Format your response using Markdown when appropriate (for headings, lists, emphasis, etc.)."""

        # Create user prompt with context
        user_prompt = f"""Context:
{context}

Question:
{question}"""

        # Generate answer using Groq API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
