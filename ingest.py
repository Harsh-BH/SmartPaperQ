import os
import argparse
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
from pathlib import Path
import hashlib
import shutil
import uuid
import traceback
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from utils.pdf_loader import load_pdf
from utils.config import OPENAI_API_KEY, ARXIV_API_URL, VECTOR_STORE_PATH, EMBEDDING_MODEL

# Configure embeddings based on environment
if OPENAI_API_KEY:
    embeddings = OpenAIEmbeddings()
else:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

class PaperIngestor:
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH):
        self.vector_store_path = vector_store_path
        self.papers_dir = Path("./papers")
        self.papers_dir.mkdir(exist_ok=True)
        
        # Directory for downloaded arxiv papers
        self.arxiv_dir = self.papers_dir / "arxiv"
        self.arxiv_dir.mkdir(exist_ok=True)
        
        # Initialize vector store if exists, otherwise create new one
        if os.path.exists(vector_store_path):
            self.vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings,
                allow_dangerous_deserialization=True  # Add this flag
            )
        else:
            self.vector_store = FAISS.from_documents(
                [Document(page_content="init", metadata={"source": "init"})],
                embeddings
            )
    
    def fetch_arxiv_papers(self, categories: List[str], max_results: int = 10) -> List[Dict[str, str]]:
        """
        Fetch papers from arXiv API for specified categories
        """
        papers = []
        
        # Fixed query structure for better results
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Get papers from last month for better results
        last_month = datetime.now() - timedelta(days=30)
        date_query = f"submittedDate:[{last_month.strftime('%Y%m%d')}000000 TO {datetime.now().strftime('%Y%m%d')}235959]"
        
        query = f"({category_query}) AND ({date_query})"
        print(f"arXiv query: {query}")
        
        # Query arXiv API
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=30)
            response.raise_for_status()  # Check for HTTP errors
            
            # Parse XML response
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Define namespace for parsing
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                # Extract papers
                entries = root.findall(".//atom:entry", ns)
                print(f"Found {len(entries)} entries in arXiv response")
                
                for entry in entries:
                    try:
                        id_element = entry.find("atom:id", ns)
                        if id_element is None:
                            print("Warning: Entry missing ID element, skipping")
                            continue
                            
                        paper_id = id_element.text.split("/")[-1]
                        
                        title_element = entry.find("atom:title", ns)
                        title = title_element.text.strip() if title_element is not None else "Unknown Title"
                        
                        summary_element = entry.find("atom:summary", ns)
                        summary = summary_element.text.strip() if summary_element is not None else ""
                        
                        authors = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
                        
                        # Get PDF link
                        pdf_link = None
                        for link in entry.findall("atom:link", ns):
                            if link.get("title") == "pdf":
                                pdf_link = link.get("href")
                                break
                                
                        if pdf_link is None:
                            # Try alternative method to find PDF link
                            for link in entry.findall("atom:link", ns):
                                href = link.get("href", "")
                                if "pdf" in href:
                                    pdf_link = href
                                    break
                        
                        papers.append({
                            "id": paper_id,
                            "title": title,
                            "summary": summary,
                            "authors": authors,
                            "pdf_link": pdf_link,
                            "source": "arxiv"
                        })
                        print(f"Added paper: {title} (ID: {paper_id})")
                    except Exception as e:
                        print(f"Error parsing paper entry: {e}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers from arXiv: {e}")
        
        print(f"Total papers fetched: {len(papers)}")
        return papers
    
    def download_arxiv_paper(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """
        Download a paper from arXiv
        """
        if not paper_data.get("pdf_link"):
            print(f"Error: No PDF link for paper '{paper_data.get('title')}'")
            return None
        
        # Create filename from paper ID
        pdf_filename = f"{paper_data['id']}.pdf"
        pdf_path = self.arxiv_dir / pdf_filename
        
        # Download if not exists
        if pdf_path.exists():
            print(f"Paper already downloaded: {pdf_path}")
            return str(pdf_path)
            
        try:
            print(f"Downloading {paper_data['pdf_link']} to {pdf_path}")
            response = requests.get(paper_data["pdf_link"], stream=True, timeout=60)
            response.raise_for_status()  # Check for HTTP errors
            
            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                # Verify file was downloaded correctly
                if os.path.getsize(pdf_path) > 0:
                    print(f"Successfully downloaded {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
                    return str(pdf_path)
                else:
                    print(f"Error: Downloaded file is empty: {pdf_path}")
                    # Clean up empty file
                    os.remove(pdf_path)
                    return None
            else:
                print(f"Error downloading paper: HTTP status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading {paper_data.get('title')}: {e}")
            # Print the stack trace for debugging
            traceback.print_exc()
            return None
    
    def process_pdf(self, pdf_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Process a PDF file and add to vector store
        """
        try:
            # Check if file exists and is readable
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found at {pdf_path}")
                return False
                
            if os.path.getsize(pdf_path) == 0:
                print(f"Error: PDF file is empty: {pdf_path}")
                return False
            
            # Extract text from PDF
            print(f"Extracting text from {pdf_path}")
            text = load_pdf(pdf_path)
            print(f"Extracted {len(text)} characters from {pdf_path}")
            
            # Skip if we got empty text
            if not text.strip():
                print(f"Warning: No text extracted from {pdf_path}")
                return False
                
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            print(f"Split into {len(chunks)} chunks")
            
            if len(chunks) == 0:
                print(f"Warning: No chunks created from {pdf_path}")
                return False
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    print(f"Warning: Empty chunk {i}, skipping")
                    continue
                    
                doc_metadata = metadata.copy()
                doc_metadata["chunk_id"] = i
                doc_metadata["chunk_count"] = len(chunks)
                
                # Try to identify section by looking for common headers
                section_markers = ["abstract", "introduction", "background", 
                                  "method", "approach", "experiment", "result", 
                                  "conclusion", "discussion", "reference"]
                
                for marker in section_markers:
                    if marker in chunk[:50].lower():
                        doc_metadata["section"] = marker.capitalize()
                        break
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            print(f"Created {len(documents)} document objects")
            
            # Add to vector store and immediately save
            if documents:
                print(f"Adding {len(documents)} documents to vectorstore")
                self.vector_store.add_documents(documents)
                print(f"Saving vectorstore to {self.vector_store_path}")
                self.vector_store.save_local(self.vector_store_path)
                print(f"Successfully processed {pdf_path}")
                return True
            else:
                print(f"No valid documents created for {pdf_path}")
                return False
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            traceback.print_exc()
            return False
    
    def ingest_directory(self, directory: str) -> List[str]:
        """
        Ingest all PDFs from a directory
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of successfully processed PDFs
        """
        dir_path = Path(directory)
        processed = []
        
        for pdf_file in dir_path.glob("*.pdf"):
            # Generate paper ID and metadata
            paper_id = f"local-{uuid.uuid4().hex[:8]}"
            metadata = {
                "id": paper_id,
                "title": pdf_file.stem,
                "source": "local",
                "file_path": str(pdf_file)
            }
            
            if self.process_pdf(str(pdf_file), metadata):
                processed.append(str(pdf_file))
        
        return processed
    
    def ingest_arxiv_papers(self, categories: List[str], max_results: int = 10) -> int:
        """
        Fetch and ingest papers from arXiv
        
        Args:
            categories: List of arXiv categories
            max_results: Maximum papers to fetch
            
        Returns:
            Count of successfully ingested papers
        """
        print(f"Fetching papers from categories: {categories}")
        papers = self.fetch_arxiv_papers(categories, max_results)
        print(f"Found {len(papers)} papers")
        ingested_count = 0
        
        for paper in papers:
            print(f"Processing paper: {paper['title']}")
            pdf_path = self.download_arxiv_paper(paper)
            if pdf_path:
                print(f"Downloaded to {pdf_path}")
                if self.process_pdf(pdf_path, paper):
                    ingested_count += 1
                    print(f"Successfully processed paper #{ingested_count}")
                
                # Be nice to arXiv API
                time.sleep(3)
        
        print(f"Finished ingesting {ingested_count} papers")
        return ingested_count

def ingest_papers():
    """
    Command-line interface for paper ingestion
    """
    parser = argparse.ArgumentParser(description="Ingest research papers into the SmartPaperQ system")
    
    # Define arguments
    parser.add_argument("--dir", type=str, help="Directory containing PDF files to ingest")
    parser.add_argument("--arxiv", type=str, nargs="+", help="arXiv categories to fetch papers from")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of papers to fetch from arXiv")
    
    args = parser.parse_args()
    ingestor = PaperIngestor()
    
    # Process based on arguments
    if args.dir:
        print(f"Ingesting papers from directory: {args.dir}")
        processed = ingestor.ingest_directory(args.dir)
        print(f"Successfully processed {len(processed)} papers")
    
    if args.arxiv:
        print(f"Fetching papers from arXiv categories: {', '.join(args.arxiv)}")
        count = ingestor.ingest_arxiv_papers(args.arxiv, args.max)
        print(f"Successfully ingested {count} papers from arXiv")
    
    if not args.dir and not args.arxiv:
        parser.print_help()

if __name__ == "__main__":
    ingest_papers()
