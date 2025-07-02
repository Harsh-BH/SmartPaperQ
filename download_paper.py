#!/usr/bin/env python3

"""
Simple command line utility to download papers from arXiv.
This can be used to test download functionality outside the app.

Usage:
  python download_paper.py 2201.00035
  python download_paper.py 1706.03762
"""

import sys
import os
from src.utils.direct_download import direct_download_paper

def main():
    if len(sys.argv) < 2:
        print("Please provide an arXiv paper ID, e.g.: 2201.00035")
        return
        
    paper_id = sys.argv[1]
    print(f"Downloading paper: {paper_id}")
    
    # Create downloads directory in current folder
    download_dir = os.path.join(os.getcwd(), "downloads")
    
    # Try to download
    result = direct_download_paper(paper_id, download_dir)
    
    if result["success"]:
        print(f"✅ Download successful!")
        print(f"  Paper saved to: {result['output_path']}")
    else:
        print(f"❌ Download failed!")
        print(f"  Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
