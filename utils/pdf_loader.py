import fitz  # PyMuPDF
import sys
import os
from pathlib import Path

def load_pdf(path: str) -> str:
    """
    Load and extract text from a PDF file.
    Uses PyMuPDF (fitz) for extraction with error handling and debugging.
    
    Args:
        path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    print(f"Opening PDF: {path}")
    
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return ""
        
    if os.path.getsize(path) == 0:
        print(f"Error: Empty file: {path}")
        return ""
    
    try:
        text_chunks = []
        
        # Open the document
        with fitz.open(path) as doc:
            # Get metadata
            metadata = doc.metadata
            if metadata:
                print(f"PDF Title: {metadata.get('title', 'Unknown')}")
                print(f"PDF Author: {metadata.get('author', 'Unknown')}")
            
            # Check if document is encrypted
            if doc.is_encrypted:
                print(f"Warning: Document is encrypted, trying empty password")
                success = doc.authenticate("")
                if not success:
                    print("Error: Could not decrypt document")
                    return ""
            
            # Process each page
            num_pages = len(doc)
            print(f"PDF has {num_pages} pages")
            
            if num_pages == 0:
                print("Warning: Document has no pages")
                return ""
                
            for i, page in enumerate(doc):
                # Get text from page
                text = page.get_text()
                if text.strip():
                    text_chunks.append(text)
                    
                # Progress report for large documents
                if (i+1) % 10 == 0 or i+1 == num_pages:
                    print(f"Processed {i+1}/{num_pages} pages")
                    
        # Combine all text
        result = "\n\n".join(text_chunks)
        print(f"Extracted {len(result)} characters of text")
        return result
        
    except fitz.FileDataError as e:
        print(f"Error: Invalid or corrupted PDF: {e}")
        return ""
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    # Test the function with a file if provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = load_pdf(pdf_path)
        print(f"\nExtracted text sample (first 500 chars):\n{text[:500]}...")
