import os
import tempfile
import time
import requests
import urllib.parse
import logging
from typing import Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_handler")

class DownloadHandler:
    """Robust handler for downloading PDF files from various sources."""
    
    def __init__(self, base_download_dir: str = None):
        """Initialize the download handler with a base directory."""
        # Set up download directories
        self.base_download_dir = base_download_dir or os.path.abspath(os.path.join(os.getcwd(), "downloads"))
        self.temp_dir = tempfile.gettempdir()
        
        # Create download directory
        try:
            os.makedirs(self.base_download_dir, exist_ok=True)
            self.is_base_dir_writable = self._check_dir_writable(self.base_download_dir)
        except Exception as e:
            logger.error(f"Error creating download directory: {e}")
            self.is_base_dir_writable = False

    def _check_dir_writable(self, directory: str) -> bool:
        """Check if a directory is writable by attempting to create a test file."""
        try:
            test_file = os.path.join(directory, f"test_write_{int(time.time())}.txt")
            with open(test_file, "w") as f:
                f.write("Test write permission")
            os.remove(test_file)
            return True
        except Exception as e:
            logger.warning(f"Directory {directory} is not writable: {e}")
            return False

    def get_download_dir(self) -> str:
        """Get the best available download directory."""
        if self.is_base_dir_writable:
            return self.base_download_dir
        
        # Try to create a paper-shaper directory in the temp directory
        temp_download_dir = os.path.join(self.temp_dir, "paper-shaper-downloads")
        try:
            os.makedirs(temp_download_dir, exist_ok=True)
            if self._check_dir_writable(temp_download_dir):
                logger.info(f"Using temporary directory for downloads: {temp_download_dir}")
                return temp_download_dir
        except Exception:
            pass
        
        # Fall back to the system temp directory
        logger.info(f"Using system temp directory for downloads: {self.temp_dir}")
        return self.temp_dir

    def download_arxiv_paper(self, paper_id: str) -> Dict[str, Any]:
        """Download a paper from arXiv using multiple methods."""
        download_dir = self.get_download_dir()
        result = {"success": False, "paper_id": paper_id, "download_dir": download_dir}
        
        # Sanitize the paper ID and create a filename
        safe_id = paper_id.replace('/', '_').replace('\\', '_')
        filename = f"{safe_id}.pdf"
        output_path = os.path.join(download_dir, filename)
        
        # Check if file already exists
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Paper already downloaded: {output_path}")
            return {
                "success": True, 
                "path": output_path, 
                "method": "cache", 
                "filename": filename,
                "paper_id": paper_id
            }
        
        # Method 1: Direct download using requests
        try:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            logger.info(f"Attempting to download from: {pdf_url}")
            
            response = requests.get(pdf_url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify the file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully downloaded paper to: {output_path}")
                    return {
                        "success": True, 
                        "path": output_path, 
                        "method": "requests",
                        "filename": filename,
                        "paper_id": paper_id
                    }
        except Exception as e:
            logger.error(f"Error downloading paper using requests: {e}")
        
        # Method 2: Using urllib
        try:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            logger.info(f"Attempting to download using urllib from: {pdf_url}")
            
            with urllib.request.urlopen(pdf_url) as response, open(output_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            
            # Verify the file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully downloaded paper to: {output_path}")
                return {
                    "success": True, 
                    "path": output_path, 
                    "method": "urllib",
                    "filename": filename,
                    "paper_id": paper_id
                }
        except Exception as e:
            logger.error(f"Error downloading paper using urllib: {e}")
        
        # Method 3: Try a different URL format
        try:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}"
            logger.info(f"Attempting to download using alternate URL: {pdf_url}")
            
            response = requests.get(pdf_url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify the file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully downloaded paper to: {output_path}")
                    return {
                        "success": True, 
                        "path": output_path, 
                        "method": "requests-alt",
                        "filename": filename,
                        "paper_id": paper_id
                    }
        except Exception as e:
            logger.error(f"Error downloading paper using alternate URL: {e}")
        
        # All methods failed
        logger.error(f"All download methods failed for paper ID: {paper_id}")
        return {
            "success": False, 
            "error": "Failed to download the paper using all available methods",
            "paper_id": paper_id
        }

    def get_pdf_bytes(self, file_path: str) -> Tuple[bool, bytes]:
        """Get PDF file as bytes for download."""
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "rb") as f:
                    return True, f.read()
            return False, b""
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            return False, b""
