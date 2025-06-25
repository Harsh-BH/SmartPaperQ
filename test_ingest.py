#!/usr/bin/env python3
"""
Test script for paper ingestion
"""

from ingest import PaperIngestor
import os
import argparse

def test_arxiv_fetch(category="cs.AI", max_results=2):
    """Test fetching papers from arXiv"""
    print(f"Testing arXiv paper fetching with category {category}...")
    ingestor = PaperIngestor()
    papers = ingestor.fetch_arxiv_papers([category], max_results)
    
    print(f"Fetched {len(papers)} papers")
    for i, paper in enumerate(papers):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}" + 
              ("..." if len(paper['authors']) > 3 else ""))
        print(f"   ID: {paper['id']}")
        print(f"   PDF Link: {paper['pdf_link']}")
        print(f"   Summary: {paper['summary'][:100]}...")

def test_download_and_process(category="cs.AI", max_results=1):
    """Test downloading and processing papers"""
    print(f"Testing download and process with category {category}...")
    ingestor = PaperIngestor()
    papers = ingestor.fetch_arxiv_papers([category], max_results)
    
    if not papers:
        print("No papers fetched, can't test downloading")
        return
    
    print(f"Attempting to download and process {len(papers)} papers")
    for i, paper in enumerate(papers):
        print(f"\nPaper {i+1}: {paper['title']}")
        pdf_path = ingestor.download_arxiv_paper(paper)
        
        if pdf_path:
            print(f"Successfully downloaded to {pdf_path}")
            print(f"Processing PDF...")
            success = ingestor.process_pdf(pdf_path, paper)
            if success:
                print("Successfully processed and added to vector store")
            else:
                print("Failed to process PDF")
        else:
            print("Failed to download PDF")

def main():
    parser = argparse.ArgumentParser(description="Test paper ingestion functionality")
    parser.add_argument("--fetch", action="store_true", help="Test fetching papers")
    parser.add_argument("--process", action="store_true", help="Test downloading and processing papers")
    parser.add_argument("--category", type=str, default="cs.AI", help="arXiv category to test with")
    parser.add_argument("--max", type=int, default=2, help="Maximum papers to fetch")
    
    args = parser.parse_args()
    
    if args.fetch:
        test_arxiv_fetch(args.category, args.max)
        
    if args.process:
        test_download_and_process(args.category, args.max)
        
    if not args.fetch and not args.process:
        # Run both by default
        test_arxiv_fetch(args.category, args.max)
        test_download_and_process(args.category, 1)  # Only process 1 paper by default

if __name__ == "__main__":
    main()
