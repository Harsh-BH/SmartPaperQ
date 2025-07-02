#!/usr/bin/env python3
"""
Fix the paper metadata issue in the vectorstore.
"""
import os
import sys
from pathlib import Path
import time
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.config import VECTOR_STORE_PATH, EMBEDDING_MODEL

def inspect_vectorstore():
    """
    Inspect the vectorstore to see why papers aren't showing up
    """
    print("\n==== Inspecting Vectorstore ====")
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Check if vectorstore exists
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"Vectorstore not found at {VECTOR_STORE_PATH}")
        return None
    
    # Load vectorstore
    print(f"Loading vectorstore from {VECTOR_STORE_PATH}")
    try:
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Get docs and check metadata
        docstore = vectorstore.docstore._dict
        doc_ids = list(docstore.keys())
        print(f"Found {len(doc_ids)} document chunks")
        
        if not doc_ids:
            print("Vectorstore is empty!")
            return None
        
        # Check document metadata
        docs_with_id = 0
        docs_with_title = 0
        docs_with_source = 0
        
        for i, doc_id in enumerate(doc_ids[:5]):  # Print metadata for up to 5 docs
            doc = docstore[doc_id]
            metadata = doc.metadata
            print(f"\nDoc {i+1} metadata: {metadata}")
            print(f"Doc {i+1} content: {doc.page_content[:50]}...")
            
            if 'id' in metadata:
                docs_with_id += 1
            if 'title' in metadata:
                docs_with_title += 1
            if 'source' in metadata:
                docs_with_source += 1
        
        print(f"\nMetadata Analysis:")
        print(f"Documents with 'id' field: {docs_with_id} out of {len(doc_ids)}")
        print(f"Documents with 'title' field: {docs_with_title} out of {len(doc_ids)}")
        print(f"Documents with 'source' field: {docs_with_source} out of {len(doc_ids)}")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def fix_vectorstore_metadata():
    """
    Fix the paper metadata in the vectorstore
    """
    print("\n==== Fixing Vectorstore Metadata ====")
    
    # First inspect
    vectorstore = inspect_vectorstore()
    if not vectorstore:
        print("Could not load vectorstore!")
        return False
    
    # Get all documents
    docs = []
    docstore = vectorstore.docstore._dict
    
    # Process all documents
    paper_counter = 0
    for doc_id in docstore:
        doc = docstore[doc_id]
        
        # Check if this document has proper metadata
        metadata = doc.metadata.copy()
        fixed = False
        
        # Fix missing paper ID
        if 'id' not in metadata or not metadata['id']:
            # Create a paper ID based on content hash
            content_hash = hashlib.md5(doc.page_content[:100].encode()).hexdigest()[:8]
            paper_id = f"paper-{content_hash}"
            metadata['id'] = paper_id
            paper_counter += 1
            fixed = True
            
        # Fix missing title
        if 'title' not in metadata or not metadata['title']:
            # Use first line of content as title
            first_line = doc.page_content.strip().split('\n')[0][:50]
            metadata['title'] = f"Document {paper_counter}: {first_line}..."
            fixed = True
            
        # Fix missing source
        if 'source' not in metadata or not metadata['source']:
            metadata['source'] = 'unknown'
            fixed = True
        
        # Only create new document if metadata was fixed
        if fixed:
            docs.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))
        else:
            docs.append(doc)
    
    # Create a new vectorstore with fixed documents
    print(f"Creating new vectorstore with {len(docs)} fixed documents...")
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Backup original vectorstore
    if os.path.exists(VECTOR_STORE_PATH):
        backup_path = f"{VECTOR_STORE_PATH}_backup_{int(time.time())}"
        print(f"Backing up existing vectorstore to {backup_path}")
        import shutil
        shutil.copytree(VECTOR_STORE_PATH, backup_path)
    
    # Create new vectorstore
    new_vectorstore = FAISS.from_documents(docs, embedding)
    new_vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"Saved fixed vectorstore to {VECTOR_STORE_PATH}")
    
    # Verify fix worked
    print("\n==== Verifying Fix ====")
    # Run inspection again to verify
    inspect_vectorstore()
    
    return True

if __name__ == "__main__":
    fix_vectorstore_metadata()
