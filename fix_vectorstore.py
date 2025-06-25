#!/usr/bin/env python3
"""
Utility script to fix issues with the vectorstore and verify papers are properly indexed
"""

import os
import sys
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL
from utils.pdf_loader import load_pdf
import tempfile
import shutil

def verify_vectorstore():
    """
    Verify the vectorstore and try to fix common issues
    """
    print(f"Verifying vectorstore at: {VECTOR_STORE_PATH}")
    
    # Check if the vectorstore exists
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Error: Vectorstore not found at {VECTOR_STORE_PATH}")
        return False
    
    try:
        # Initialize embedding model
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Load the vectorstore
        print("Loading vectorstore...")
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Get all document IDs
        doc_ids = list(vectorstore.docstore._dict.keys())
        print(f"Found {len(doc_ids)} document chunks in vectorstore")
        
        # Check if there are any documents
        if not doc_ids:
            print("Vectorstore is empty! Let's create a new one.")
            return rebuild_vectorstore(embeddings)
        
        # Extract and count unique paper IDs
        paper_ids = set()
        paper_with_id = False
        
        for doc_id in doc_ids:
            doc = vectorstore.docstore._dict[doc_id]
            paper_id = doc.metadata.get('id')
            if paper_id:
                paper_with_id = True
                paper_ids.add(paper_id)
        
        print(f"Found {len(paper_ids)} unique papers in vectorstore")
        
        # Check if metadata is missing paper IDs
        if len(doc_ids) > 0 and not paper_with_id:
            print("Warning: Documents exist but no paper IDs found in metadata!")
            return fix_metadata(vectorstore, embeddings)
        
        # Test the search to verify the index is working
        print("\nTesting similarity search...")
        try:
            results = vectorstore.similarity_search("introduction", k=1)
            if results:
                print(f"Search successful, found result from paper: {results[0].metadata.get('title', 'Unknown')}")
                return True
            else:
                print("Search returned no results!")
                return rebuild_vectorstore(embeddings)
        except Exception as e:
            print(f"Error performing search: {e}")
            return rebuild_vectorstore(embeddings)
            
    except Exception as e:
        print(f"Error inspecting vectorstore: {e}")
        return False

def rebuild_vectorstore(embeddings):
    """
    Rebuild the vectorstore from scratch
    """
    print("\n=== Rebuilding vectorstore ===")
    
    # Backup the old vectorstore if it exists
    if os.path.exists(VECTOR_STORE_PATH):
        backup_path = f"{VECTOR_STORE_PATH}_backup_{int(time.time())}"
        print(f"Backing up existing vectorstore to {backup_path}")
        try:
            shutil.copytree(VECTOR_STORE_PATH, backup_path)
        except Exception as e:
            print(f"Warning: Couldn't backup vectorstore: {e}")
    
    # Find local PDF files
    papers_dir = Path("./papers")
    if papers_dir.exists():
        pdf_files = list(papers_dir.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            print("No PDF files found to rebuild index!")
            # Create empty vectorstore
            empty_doc = Document(page_content="init", metadata={"source": "init"})
            vs = FAISS.from_documents([empty_doc], embeddings)
            vs.save_local(VECTOR_STORE_PATH)
            print(f"Created empty vectorstore at {VECTOR_STORE_PATH}")
            return False
        
        # Process each PDF
        docs = []
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file}...")
                # Generate simple metadata
                metadata = {
                    "id": f"local-{pdf_file.stem}",
                    "title": pdf_file.stem,
                    "source": "local",
                    "file_path": str(pdf_file)
                }
                
                # Extract text
                text = load_pdf(str(pdf_file))
                if not text.strip():
                    print(f"Warning: No text extracted from {pdf_file}, skipping")
                    continue
                
                # Add as a document
                docs.append(Document(page_content=text, metadata=metadata))
                print(f"Added {pdf_file.name} to documents")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        if docs:
            # Create new vectorstore
            print(f"Creating new vectorstore with {len(docs)} documents")
            vs = FAISS.from_documents(docs, embeddings)
            vs.save_local(VECTOR_STORE_PATH)
            print(f"Successfully rebuilt vectorstore at {VECTOR_STORE_PATH}")
            return True
        else:
            print("No documents created during rebuild!")
            return False
    else:
        print(f"Papers directory {papers_dir} not found!")
        return False

def fix_metadata(vectorstore, embeddings):
    """
    Fix metadata in the vectorstore
    """
    print("\n=== Fixing vectorstore metadata ===")
    
    # Get all documents
    all_docs = []
    for doc_id in vectorstore.docstore._dict:
        doc = vectorstore.docstore._dict[doc_id]
        
        # Fix metadata
        if 'id' not in doc.metadata:
            # Generate an ID based on content hash
            import hashlib
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            doc.metadata['id'] = f"fixed-{content_hash}"
            
        if 'title' not in doc.metadata:
            # Try to extract title from content
            first_line = doc.page_content.strip().split('\n')[0]
            title = first_line[:50] + ('...' if len(first_line) > 50 else '')
            doc.metadata['title'] = title
            
        if 'source' not in doc.metadata:
            doc.metadata['source'] = 'unknown'
            
        all_docs.append(doc)
    
    # Create a new vectorstore with fixed metadata
    if all_docs:
        print(f"Creating new vectorstore with {len(all_docs)} fixed documents")
        
        # Backup the old vectorstore
        backup_path = f"{VECTOR_STORE_PATH}_backup_{int(time.time())}"
        try:
            shutil.copytree(VECTOR_STORE_PATH, backup_path)
            print(f"Backed up old vectorstore to {backup_path}")
        except Exception as e:
            print(f"Warning: Couldn't backup vectorstore: {e}")
        
        # Create new vectorstore
        new_vs = FAISS.from_documents(all_docs, embeddings)
        new_vs.save_local(VECTOR_STORE_PATH)
        print(f"Successfully fixed metadata and saved vectorstore to {VECTOR_STORE_PATH}")
        return True
    else:
        print("No documents to fix!")
        return False

if __name__ == "__main__":
    import time
    result = verify_vectorstore()
    if result:
        print("\nVectorstore verification successful!")
    else:
        print("\nVectorstore verification failed or required fixes.")
