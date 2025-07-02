#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL

def inspect_vectorstore(path=VECTOR_STORE_PATH):
    """
    Inspect a FAISS vectorstore and print information about its contents
    """
    print(f"Inspecting vectorstore at: {path}")
    
    # Check if the vectorstore exists
    if not os.path.exists(path):
        print(f"Error: Vectorstore not found at {path}")
        return
    
    try:
        # Initialize embedding model
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load the vectorstore
        print("Loading vectorstore...")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        
        # Get all document IDs
        doc_ids = list(vectorstore.docstore._dict.keys())
        print(f"Found {len(doc_ids)} document chunks in vectorstore")
        
        # Extract and count unique paper IDs
        paper_ids = set()
        for doc_id in doc_ids:
            doc = vectorstore.docstore._dict[doc_id]
            paper_id = doc.metadata.get('id')
            if paper_id:
                paper_ids.add(paper_id)
        
        print(f"Found {len(paper_ids)} unique papers in vectorstore")
        
        # Print paper details
        print("\nPapers in vectorstore:")
        print("-" * 50)
        for i, paper_id in enumerate(paper_ids):
            # Find a document chunk for this paper to get metadata
            for doc_id in doc_ids:
                doc = vectorstore.docstore._dict[doc_id]
                if doc.metadata.get('id') == paper_id:
                    print(f"{i+1}. ID: {paper_id}")
                    print(f"   Title: {doc.metadata.get('title', 'Unknown')}")
                    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                    # Count chunks for this paper
                    chunk_count = sum(1 for d_id in doc_ids if 
                                    vectorstore.docstore._dict[d_id].metadata.get('id') == paper_id)
                    print(f"   Chunks: {chunk_count}")
                    print("-" * 50)
                    break
        
        # Test similarity search
        print("\nTesting similarity search...")
        results = vectorstore.similarity_search("abstract introduction", k=1)
        if results:
            print(f"Search successful, found result from paper: {results[0].metadata.get('title', 'Unknown')}")
        else:
            print("Search returned no results!")
            
    except Exception as e:
        print(f"Error inspecting vectorstore: {e}")

if __name__ == "__main__":
    # Use custom path if provided as argument
    if len(sys.argv) > 1:
        inspect_vectorstore(sys.argv[1])
    else:
        inspect_vectorstore()
