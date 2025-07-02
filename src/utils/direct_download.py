import os
import requests
import time
import logging
import urllib.request
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_download")

def ensure_download_dir(download_dir: str) -> str:
    """Ensure the download directory exists and is writable."""
    if not download_dir:
        download_dir = os.path.abspath(os.path.join(os.getcwd(), "downloads"))
    
    try:
        os.makedirs(download_dir, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(download_dir, f"test_write_{int(time.time())}.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Using download directory: {download_dir}")
        return download_dir
    except Exception as e:
        logger.warning(f"Cannot use {download_dir}: {e}")
        
        # Try to use a temporary directory instead
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "paper-shaper-downloads")
        try:
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"Using temporary directory: {temp_dir}")
            return temp_dir
        except Exception as e2:
            logger.error(f"Cannot use temporary directory: {e2}")
            
            # Last resort: use current working directory
            cwd = os.getcwd()
            logger.warning(f"Falling back to current directory: {cwd}")
            return cwd

def direct_download_paper(paper_id: str, download_dir: Optional[str] = None) -> Dict[str, Any]:
    """Download a paper directly using requests, with multiple fallbacks."""
    # Ensure download directory exists
    download_dir = ensure_download_dir(download_dir)
    
    # Prepare output file path
    safe_id = paper_id.replace('/', '_').replace('\\', '_')
    filename = f"{safe_id}.pdf"
    output_path = os.path.join(download_dir, filename)
    
    # Result dictionary to return
    result = {
        "success": False,
        "paper_id": paper_id,
        "output_path": output_path,
        "download_dir": download_dir,
        "error": None
    }
    
    # Check if file already exists and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"File already exists: {output_path}")
        result["success"] = True
        return result
    
    # Try multiple download methods
    download_methods = [
        # Method 1: Direct download with requests
        {
            "name": "requests-direct",
            "url": f"https://arxiv.org/pdf/{paper_id}.pdf",
            "function": download_with_requests
        },
        # Method 2: Direct download with urllib
        {
            "name": "urllib-direct",
            "url": f"https://arxiv.org/pdf/{paper_id}.pdf",
            "function": download_with_urllib
        },
        # Method 3: Alternative URL format
        {
            "name": "requests-alt",
            "url": f"https://arxiv.org/pdf/{paper_id}",
            "function": download_with_requests
        },
        # Method 4: Export format
        {
            "name": "requests-export",
            "url": f"https://export.arxiv.org/pdf/{paper_id}",
            "function": download_with_requests
        }
    ]
    
    # Try each download method in sequence
    errors = []
    for method in download_methods:
        try:
            logger.info(f"Trying download method: {method['name']} - {method['url']}")
            success = method["function"](method["url"], output_path)
            
            # Check if download was successful
            if success and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Download successful with method {method['name']}")
                result["success"] = True
                result["method"] = method["name"]
                return result
        except Exception as e:
            error_msg = f"Method {method['name']} failed: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
    
    # All methods failed
    error_summary = "\n".join(errors)
    logger.error(f"All download methods failed for {paper_id}")
    result["error"] = error_summary
    return result

def download_with_requests(url: str, output_path: str) -> bool:
    """Download file using requests."""
    try:
        # Set a user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Download the file with streaming
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check if we got PDF content
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and 'pdf' not in content_type:
            logger.warning(f"Content-Type is not PDF: {content_type}")
        
        # Save the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Ensure file was written
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        logger.error(f"Error in requests download: {e}")
        raise

def download_with_urllib(url: str, output_path: str) -> bool:
    """Download file using urllib."""
    try:
        # Set a user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Create request with headers
        request = urllib.request.Request(url, headers=headers)
        
        # Download the file
        with urllib.request.urlopen(request, timeout=30) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        # Ensure file was written
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        logger.error(f"Error in urllib download: {e}")
        raise

# Simple test function
def test_download(paper_id: str = "2201.00035") -> None:
    """Test the download functionality."""
    result = direct_download_paper(paper_id)
    if result["success"]:
        print(f"✅ Download successful: {result['output_path']}")
    else:
        print(f"❌ Download failed: {result['error']}")

if __name__ == "__main__":
    # Can be run directly for testing
    import sys
    test_id = sys.argv[1] if len(sys.argv) > 1 else "2201.00035"
    test_download(test_id)
