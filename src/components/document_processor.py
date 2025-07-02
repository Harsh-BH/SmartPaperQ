import os
from typing import List, Dict, Any
from pypdf import PdfReader

class DocumentProcessor:
    """Process PDF documents and extract text for further processing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    def process_document(self, pdf_path: str) -> List[str]:
        """Process a document and return chunks of text."""
        text = self.extract_text_from_pdf(pdf_path)
        return self.chunk_text(text)
