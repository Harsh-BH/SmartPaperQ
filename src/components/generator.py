from typing import List, Dict, Any, Tuple
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class AnswerGenerator:
    """Generate answers based on retrieved document chunks using Groq API."""
    
    def __init__(self, model_name: str = "llama3-70b-8192"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model_name
    
    def generate_answer(self, query: str, chunks: List[Tuple[str, float]]) -> str:
        """
        Generate an answer based on the query and retrieved chunks.
        
        Args:
            query: The user's question
            chunks: List of (chunk_text, similarity_score) tuples
            
        Returns:
            Generated answer text
        """
        # Prepare context from chunks
        context = "\n\n".join([chunk[0] for chunk in chunks])
        
        # Prepare the prompt
        system_prompt = "You are SmartPaperQ, an AI research assistant. Answer questions based only on the provided research paper extracts. If you cannot answer from the given context, say so - do not make up information."
        
        user_prompt = f"""
        QUESTION: {query}
        
        RESEARCH PAPER EXTRACTS:
        {context}
        
        ANSWER:
        """
        
        # Generate the answer using Groq API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return completion.choices[0].message.content.strip()
