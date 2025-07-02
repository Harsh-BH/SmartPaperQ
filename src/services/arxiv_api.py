import arxiv
import os
import logging
import re
from typing import List, Dict, Any, Optional
from src.utils.direct_download import direct_download_paper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arxiv_service")

class ArxivService:
    """Service for interacting with the arXiv API."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on arXiv based on the query."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in self.client.results(search):
                # Extract clean arXiv ID - this is critical for downloads to work
                arxiv_id = self._extract_arxiv_id(paper.entry_id)
                
                results.append({
                    'id': paper.entry_id,
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'summary': paper.summary,
                    'published': paper.published,
                    'pdf_url': paper.pdf_url,
                    'entry_id': arxiv_id  # Clean arxiv ID for downloads
                })
                
            return results
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract clean arXiv ID from various formats.
        
        Examples:
        - http://arxiv.org/abs/2201.00035 -> 2201.00035
        - arxiv:2201.00035v1 -> 2201.00035
        - 2201.00035v1 -> 2201.00035
        """
        # First, get the last part after any slashes
        id_part = entry_id.split('/')[-1]
        
        # Remove any 'arxiv:' prefix
        id_part = id_part.replace('arxiv:', '')
        
        # Remove version suffix (vX)
        clean_id = re.sub(r'v\d+$', '', id_part)
        
        logger.info(f"Extracted arXiv ID: {clean_id} from {entry_id}")
        return clean_id
    
    def download_paper(self, paper_id: str, download_dir: str = None) -> Optional[str]:
        """Download a paper PDF by its arXiv ID using direct download."""
        try:
            # Clean the ID first to ensure proper format
            clean_id = self._extract_arxiv_id(paper_id)
            logger.info(f"Downloading paper with clean ID: {clean_id}")
            
            # Use our direct download function
            result = direct_download_paper(clean_id, download_dir)
            
            if result["success"]:
                return result["output_path"]
            else:
                logger.error(f"Failed to download paper {clean_id}: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in download_paper: {e}")
            return None
    
    def get_pdf_bytes(self, paper_id: str, download_dir: str = None) -> Optional[bytes]:
        """Get the PDF file as bytes for direct download."""
        try:
            # First download the paper
            pdf_path = self.download_paper(paper_id, download_dir)
            if not pdf_path or not os.path.exists(pdf_path):
                return None
                
            # Read the bytes
            with open(pdf_path, "rb") as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error getting PDF bytes: {e}")
            return None
