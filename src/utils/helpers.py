import os
import hashlib
from typing import List, Dict, Any

def create_directory_if_not_exists(path: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_document_hash(content: str) -> str:
    """Generate a hash for a document."""
    return hashlib.md5(content.encode()).hexdigest()

def format_chunks_for_display(chunks: List[str], max_length: int = 200) -> List[str]:
    """Format chunks for display by truncating them."""
    formatted = []
    for chunk in chunks:
        if len(chunk) > max_length:
            formatted.append(chunk[:max_length] + "...")
        else:
            formatted.append(chunk)
    return formatted
